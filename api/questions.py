"""Routes for /api/questions/* endpoints."""

from __future__ import annotations

import logging
import math
import statistics
from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from api.state import _get_state
from qe.assignment import (
    aggregate_matches,
    build_matches,
    rerank_candidates,
    retrieve_candidates,
)
from qe.db import get_session
from qe.hashing import stable_question_point_id
from qe.ingestion_an import ingest_questions
from qe.models import Question
from qe.office_ingestion import OFFICE_COLLECTION
from qe.question_fetch import fetch_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["questions"])

QUESTIONS_COLLECTION = "questions_opendata"
ANSWERS_COLLECTION = "answers_opendata"

# Allowlisted collections for the /similar endpoint.
# Maps the public name to (internal_collection, text_field, dedup_key).
# dedup_key=None means one point per entity (no deduplication needed).
# dedup_key="office_id" means keep only the best-scoring chunk per office.
_SIMILAR_COLLECTIONS: dict[str, tuple[str, str, str | None]] = {
    "questions": (QUESTIONS_COLLECTION, "texte_question", None),
    "answers": (ANSWERS_COLLECTION, "texte_reponse", None),
    "offices": (OFFICE_COLLECTION, "text", "office_id"),
}


def _dedup_by_key(
    scored: list[tuple[dict, float]], dedup_key: str
) -> list[tuple[dict, float]]:
    """Keep the highest-scoring hit per unique value of ``dedup_key`` in payload."""
    seen: dict[str, tuple[dict, float]] = {}
    for candidate, score in scored:
        key = (candidate.get("payload") or {}).get(dedup_key)
        if key is None:
            logger.warning(
                "Chunk %s is missing payload field '%s'; skipping.",
                candidate.get("id"),
                dedup_key,
            )
            continue
        if key not in seen or score > seen[key][1]:
            seen[key] = (candidate, score)
    return sorted(seen.values(), key=lambda x: -x[1])


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid, safe for any magnitude of ``x``.

    ``math.exp`` overflows for very large arguments, so the exponent is kept
    on the negative side regardless of the sign of ``x``.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_relevance(agg_score: float, pool_scores: list[float]) -> float:
    """Relevance of an office for a question, as a percentage.

    ``agg_score`` is the sum of the Albert reranker scores across the top-2
    chunks for this office (responsibilities + keywords).  Using the aggregate
    rather than just the best chunk avoids identical relevance values when
    multiple offices tie on their highest-scoring chunk (a common occurrence).

    Blends two signals:

    - **Absolute** (70 %): ``sigmoid(agg_score) × 100`` — the model's raw
      judgment about how relevant the question is to this office, regardless
      of what other offices were retrieved.

    - **Relative** (30 %): a pool-median-centred linear adjustment — each
      unit above the pool median adds ~6 pp; each unit below subtracts ~6 pp.
      This makes real score gaps between the top offices visible without
      distorting the absolute meaning.

    The blend satisfies both constraints:
    - Tightly clustered raw scores → nearly identical relevance values.
    - Well-separated raw scores → the gap is visible in the output.

    Returns a float in [0.0, 100.0], rounded to one decimal place.
    """
    absolute = 100.0 * _sigmoid(agg_score)

    if len(pool_scores) < 2:
        return round(absolute, 1)

    median_score = statistics.median(pool_scores)
    median_abs = 100.0 * _sigmoid(median_score)

    # Linear relative component: each unit of deviation from the pool median
    # maps to PP_PER_UNIT percentage points.  Clamped to [0, 100].
    PP_PER_UNIT = 20.0
    relative = median_abs + (agg_score - median_score) * PP_PER_UNIT
    relative = max(0.0, min(100.0, relative))

    # 30 % relative weight → effective contribution: 0.3 × 20 = 6 pp per logit.
    return round(0.7 * absolute + 0.3 * relative, 1)


@router.get("/{question_id}/attributions")
def get_attributions(
    question_id: str,
    top_k: int = 3,
    office_ids: list[str] = Query(default=[]),
) -> dict:
    """Return the top-N office attribution suggestions for a question.

    The question must already be embedded in the ``questions_opendata``
    collection.  Its stored vector is used directly for the office search
    so no embedding API call is required.

    Args:
        question_id: Composite question ID, e.g. ``AN-17-QE-12345``.
        top_k: Number of office suggestions to return (default 3).
        office_ids: Optional list of office codes to restrict the search
            (e.g. ``["SD1/1A", "SD2/2B"]``).  Callers that filter by
            top-level direction should resolve direction → office codes
            before calling this endpoint.

    Returns:
        A dict with ``question_id`` and an ``attributions`` list sorted by
        descending relevance, each entry containing ``rank``, ``office_id``,
        ``office_name``, ``direction``, ``score``, and ``relevance``.

    Raises:
        404: Question point not found in the vector store (not yet embedded).
        422: ``top_k`` is less than 1.
    """
    if top_k < 1:
        raise HTTPException(status_code=422, detail="top_k must be at least 1.")

    state = _get_state()

    # 1. Fetch the question's pre-computed vector and text from the vector store.
    point_id = stable_question_point_id(question_id)
    point = state.vector_store.get_point(
        QUESTIONS_COLLECTION, point_id, with_vectors=True
    )
    if point is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in the vector store. "
            "Make sure it has been embedded with scripts/embed_questions.py.",
        )

    vector: list[float] = point["vector"]
    payload = point.get("payload") or {}
    texte_question: str = payload.get("texte_question") or ""

    if not texte_question:
        raise HTTPException(
            status_code=422,
            detail=f"Question '{question_id}' has no texte_question in its payload.",
        )

    # 2. Search the office_responsibilities collection using the stored vector
    #    (no embedding call needed).
    office_filter: dict | None = None
    if office_ids:
        office_filter = {
            "must": [{"key": "office_id", "match": {"any": office_ids}}]
        }
    candidates = retrieve_candidates(
        precomputed_vectors=[vector],
        vector_store=state.vector_store,
        collection=OFFICE_COLLECTION,
        top_k=20,
        query_filter=office_filter,
    )

    # 3. Rerank candidates against the question text.
    matches = build_matches(
        candidates=candidates,
        reranker=state.reranker,
        query=texte_question,
    )

    # 4. Aggregate per-office scores and rank.
    kept_matches, score_by_office = aggregate_matches(matches, max_chunks_per_office=2)

    # 5. Deduplicate to one entry per office and return the top_k.
    #
    # Use the aggregated per-office score (sum of top-2 chunk scores) as the
    # relevance signal.  Using the individual best-chunk score instead causes
    # identical relevance values whenever multiple offices tie on their top
    # chunk — which is common because the reranker often assigns the same score
    # to the highest-ranked chunk across the top-3 offices.  The aggregated
    # score captures both chunks and is already unique per office.
    pool_scores = list(score_by_office.values())

    # Build a metadata lookup (office_name, direction) from kept_matches.
    # kept_matches is sorted by individual chunk score, which can disagree with
    # the aggregated score_by_office ranking when an office's second chunk is
    # strong.  Iterating kept_matches directly would produce attributions whose
    # rank order contradicts their relevance values, so we drive the final loop
    # from score_by_office (sorted descending) instead.
    office_meta: dict[str, dict] = {}
    for m in kept_matches:
        oid = m.get("office_id")
        if oid and oid not in office_meta:
            office_meta[oid] = m

    attributions: list[dict] = []
    for office_id, agg_score in sorted(score_by_office.items(), key=lambda x: -x[1]):
        meta = office_meta.get(office_id)
        if not meta:
            continue
        attributions.append(
            {
                "rank": len(attributions) + 1,
                "office_id": office_id,
                "office_name": meta.get("office_name"),
                "direction": meta.get("direction"),
                "score": round(agg_score, 4),
                "relevance": _to_relevance(agg_score, pool_scores),
            }
        )
        if len(attributions) >= top_k:
            break

    return {"question_id": question_id, "attributions": attributions}


@router.get("/{question_id}/direction-attributions")
def get_direction_attributions(
    question_id: str,
    top_k: int = 3,
) -> dict:
    """Return the top-N direction attribution suggestions for a question.

    Runs the same retrieve → rerank pipeline as :func:`get_direction_attributions`
    but aggregates scores by ``direction`` (one level above offices in the
    French ministerial hierarchy) instead of by ``office_id``.

    Args:
        question_id: Composite question ID, e.g. ``AN-17-QE-12345``.
        top_k: Number of direction suggestions to return (default 3).

    Returns:
        A dict with ``question_id`` and an ``attributions`` list sorted by
        descending relevance, each entry containing ``rank``, ``direction``,
        ``score``, and ``relevance``.

    Raises:
        404: Question point not found in the vector store (not yet embedded).
        422: ``top_k`` is less than 1.
    """
    if top_k < 1:
        raise HTTPException(status_code=422, detail="top_k must be at least 1.")

    state = _get_state()

    point_id = stable_question_point_id(question_id)
    point = state.vector_store.get_point(
        QUESTIONS_COLLECTION, point_id, with_vectors=True
    )
    if point is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in the vector store. "
            "Make sure it has been embedded with scripts/embed_questions.py.",
        )

    vector: list[float] = point["vector"]
    payload = point.get("payload") or {}
    texte_question: str = payload.get("texte_question") or ""

    if not texte_question:
        raise HTTPException(
            status_code=422,
            detail=f"Question '{question_id}' has no texte_question in its payload.",
        )

    candidates = retrieve_candidates(
        precomputed_vectors=[vector],
        vector_store=state.vector_store,
        collection=OFFICE_COLLECTION,
        top_k=20,
    )

    matches = build_matches(
        candidates=candidates,
        reranker=state.reranker,
        query=texte_question,
    )

    # Aggregate by direction: group all reranked chunks by their direction,
    # keep the top-6 highest-scoring chunks per direction (≈ top 3 offices × 2
    # chunks), and sum their scores.  The cap avoids directions with many
    # offices dominating purely by volume.
    max_chunks_per_direction = 6
    chunks_by_direction: dict[str, list[dict]] = {}
    for match in matches:
        direction = match.get("direction")
        if not direction:
            continue
        chunks_by_direction.setdefault(direction, []).append(match)

    score_by_direction: dict[str, float] = {}
    for direction, chunks in chunks_by_direction.items():
        chunks.sort(key=lambda m: -(m.get("score") or 0.0))
        top_chunks = chunks[:max_chunks_per_direction]
        score_by_direction[direction] = sum(
            float(m.get("score") or 0.0) for m in top_chunks
        )

    pool_scores = list(score_by_direction.values())
    attributions: list[dict] = []
    for direction, agg_score in sorted(score_by_direction.items(), key=lambda x: -x[1]):
        attributions.append(
            {
                "rank": len(attributions) + 1,
                "direction": direction,
                "score": round(agg_score, 4),
                "relevance": _to_relevance(agg_score, pool_scores),
            }
        )
        if len(attributions) >= top_k:
            break

    return {"question_id": question_id, "attributions": attributions}


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
        collection: Target collection — one of ``questions``, ``answers``,
            ``offices``.
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
    target_collection, text_field, dedup_key = _SIMILAR_COLLECTIONS[collection]

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

    # For collections with multiple chunks per entity (offices), keep only the
    # best-scoring chunk per entity so each office appears at most once.
    if dedup_key is not None:
        scored = _dedup_by_key(scored, dedup_key)

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
