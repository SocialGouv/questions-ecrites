"""Routes for /api/questions/* endpoints."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from api.state import _get_state
from qe.assignment import rerank_candidates, retrieve_candidates
from qe.db import get_session
from qe.hashing import stable_question_point_id
from qe.ingestion_an import ingest_questions
from qe.models import Question
from qe.question_fetch import fetch_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["questions"])

QUESTIONS_COLLECTION = "questions_opendata"
ANSWERS_COLLECTION = "answers_opendata"

# Allowlisted collections for the /similar endpoint.
# Maps the public name to (internal_collection, text_field).
_SIMILAR_COLLECTIONS: dict[str, tuple[str, str]] = {
    "questions": (QUESTIONS_COLLECTION, "texte_question"),
    "answers": (ANSWERS_COLLECTION, "texte_reponse"),
}


@router.get("/{question_id}/similar")
def get_similar(
    question_id: str,
    collection: str,
    top_k: int = 10,
    score_threshold: float | None = None,
) -> dict:
    """Return semantically similar items from a vector store collection.

    The source question must already be embedded in ``questions_opendata``.
    Its stored vector is used directly — no embedding API call is made.
    Results are reranked with Albert before being returned.

    Args:
        question_id: Composite question ID, e.g. ``AN-17-QE-12345``.
        collection: Target collection — one of ``questions``, ``answers``.
        top_k: Number of results to return (default 10, max 50).
        score_threshold: Optional minimum cosine similarity (0.0–1.0) applied
            before reranking to drop clearly irrelevant candidates.

    Returns:
        A dict with ``question_id``, ``collection``, and a ``hits`` list sorted
        by descending rerank score.  Each hit has ``id``, ``score``, and
        ``payload`` (collection-specific fields).

    Raises:
        404: Question not found in the vector store (not yet embedded).
        422: ``collection`` is not one of the allowed values, or ``top_k``
            is out of range.
    """
    if collection not in _SIMILAR_COLLECTIONS:
        allowed = ", ".join(sorted(_SIMILAR_COLLECTIONS))
        raise HTTPException(
            status_code=422,
            detail=f"collection must be one of: {allowed}.",
        )
    if not (1 <= top_k <= 50):
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 50.")
    if score_threshold is not None and not (0.0 <= score_threshold <= 1.0):
        raise HTTPException(
            status_code=422, detail="score_threshold must be between 0.0 and 1.0."
        )

    state = _get_state()
    target_collection, text_field = _SIMILAR_COLLECTIONS[collection]

    # Fetch the question's pre-computed vector and text.
    source_point_id = stable_question_point_id(question_id)
    point = state.vector_store.get_point(
        QUESTIONS_COLLECTION, source_point_id, with_vectors=True
    )
    if point is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in the vector store. "
            "Make sure it has been embedded with scripts/embed_questions.py.",
        )

    vector: list[float] = point["vector"]
    texte_question: str = (point.get("payload") or {}).get("texte_question") or ""
    if not texte_question:
        raise HTTPException(
            status_code=422,
            detail=f"Question '{question_id}' has no texte_question in its payload.",
        )

    # When searching within the questions collection, exclude the source point
    # so the question doesn't appear as its own nearest neighbour.
    exclusion_filter: dict | None = None
    if collection == "questions":
        exclusion_filter = {"must_not": [{"has_id": [str(source_point_id)]}]}

    # Retrieve a larger candidate pool before reranking.
    candidates = retrieve_candidates(
        precomputed_vectors=[vector],
        vector_store=state.vector_store,
        collection=target_collection,
        top_k=max(top_k * 3, 20),
        query_filter=exclusion_filter,
        score_threshold=score_threshold,
    )

    # Rerank candidates against the question text.
    scored = rerank_candidates(candidates, state.reranker, texte_question, text_field)

    hits = [
        {
            "id": candidate["id"],
            "score": round(score, 6),
            "payload": candidate.get("payload") or {},
        }
        for candidate, score in scored[:top_k]
    ]

    return {"question_id": question_id, "collection": collection, "hits": hits}


# ---------------------------------------------------------------------------
# GET /{question_id} — on-demand question metadata with auto-fetch
# ---------------------------------------------------------------------------

_RESPONSE_COLS = (
    Question.id,
    Question.source,
    Question.legislature,
    Question.numero_question,
    Question.etat_question,
    Question.objet,
    Question.titre_senat,
    Question.texte_question,
    Question.auteur_nom,
    Question.auteur_prenom,
    Question.ministre_depot_libelle,
    Question.date_publication_jo,
)


def _fetch_question(session: Session, question_id: str) -> Question | None:
    return session.scalar(
        select(Question)
        .where(Question.id == question_id)
        .options(load_only(*_RESPONSE_COLS))
    )


class QuestionResponse(BaseModel):
    id: str
    source: str
    legislature: int
    numero_question: int
    etat_question: str
    objet: str | None
    titre_senat: str | None
    texte_question: str
    auteur_nom: str | None
    auteur_prenom: str | None
    ministre_libelle: str | None
    date_publication_jo: date | None
    fetched_now: bool


def _question_to_response(row: Question, *, fetched_now: bool) -> QuestionResponse:
    return QuestionResponse(
        id=row.id,
        source=row.source,
        legislature=row.legislature,
        numero_question=row.numero_question,
        etat_question=row.etat_question,
        objet=row.objet,
        titre_senat=row.titre_senat,
        texte_question=row.texte_question,
        auteur_nom=row.auteur_nom,
        auteur_prenom=row.auteur_prenom,
        ministre_libelle=row.ministre_depot_libelle,
        date_publication_jo=row.date_publication_jo,
        fetched_now=fetched_now,
    )


@router.get("/{question_id}", response_model=QuestionResponse)
def get_question(question_id: str) -> QuestionResponse:
    """Return metadata for a single question.

    Checks the database first.  For Sénat questions absent from the database,
    the Sénat website is scraped and the result is upserted (``fetched_now=true``).
    AN questions are only available through the daily bulk archive refresh
    (``scripts/download_an.py --ingest``).

    Args:
        question_id: Composite question ID, e.g. ``AN-17-QE-15535``.

    Returns:
        Question metadata.  ``fetched_now`` is true when a Sénat question was
        not in the database and was fetched on this request.

    Raises:
        400: The question ID format is not recognised.
        404: The question is not in the database (and cannot be fetched
            individually for AN questions).
    """
    with get_session() as session:
        row: Question | None = _fetch_question(session, question_id)

    if row is not None:
        return _question_to_response(row, fetched_now=False)

    # AN questions are not individually fetchable — only the bulk archive can
    # populate them.
    if question_id.upper().startswith("AN-"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Question '{question_id}' not found in the database. "
                "AN questions are updated via the daily bulk archive refresh; "
                "run scripts/download_an.py --legislature 17 --ingest."
            ),
        )

    # For Sénat, try scraping the question page.
    state = _get_state()
    try:
        parsed = fetch_question(question_id, state.http)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if parsed is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found at the upstream source.",
        )

    ingest_questions([parsed], ingest_source="ws_polling")

    with get_session() as session:
        row = _fetch_question(session, parsed.id)

    if row is None:
        raise HTTPException(status_code=500, detail="Ingest succeeded but row not found.")

    return _question_to_response(row, fetched_now=True)
