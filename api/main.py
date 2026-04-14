"""FastAPI server exposing the office attribution pipeline.

Start with:
    poetry run uvicorn api.main:app --reload

Endpoints
---------
GET /api/questions/{question_id}/attributions
    Return the top-N ranked offices for a question that is already embedded
    in the ``questions_opendata`` Qdrant collection.  The question's vector is
    fetched directly from Qdrant — no call to the Socle IA embedding service
    is made.  Only the Albert reranker is called.
"""

from __future__ import annotations

import math
import os
import statistics
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from qe.assignment import aggregate_matches, build_matches, retrieve_candidates
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient
from qe.hashing import stable_question_point_id
from qe.office_ingestion import OFFICE_COLLECTION

QUESTIONS_COLLECTION = "questions_opendata"
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
ALBERT_RERANK_MODEL = "openweight-rerank"

# ---------------------------------------------------------------------------
# Shared client state (initialised once at startup)
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    qdrant: QdrantClient
    reranker: RerankClient


_state: AppState | None = None


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("Application has not started yet.")
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared clients on startup, release on shutdown."""
    global _state

    albert_api_key = os.environ.get("ALBERT_API_KEY", "")
    if not albert_api_key:
        raise RuntimeError("ALBERT_API_KEY environment variable is not set.")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    _state = AppState(
        qdrant=QdrantClient(qdrant_url),
        reranker=RerankClient(
            base_url=ALBERT_BASE_URL,
            model=ALBERT_RERANK_MODEL,
            api_key=albert_api_key,
        ),
    )
    yield
    _state = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QE Attribution API",
    description="Suggests the most relevant ministry offices for a parliamentary question.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    absolute = 100.0 / (1.0 + math.exp(-agg_score))

    if len(pool_scores) < 2:
        return round(absolute, 1)

    median_score = statistics.median(pool_scores)
    median_abs = 100.0 / (1.0 + math.exp(-median_score))

    # Linear relative component: each unit of deviation from the pool median
    # maps to PP_PER_UNIT percentage points.  Clamped to [0, 100].
    PP_PER_UNIT = 20.0
    relative = median_abs + (agg_score - median_score) * PP_PER_UNIT
    relative = max(0.0, min(100.0, relative))

    # 30 % relative weight → effective contribution: 0.3 × 20 = 6 pp per logit.
    return round(0.7 * absolute + 0.3 * relative, 1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/questions/{question_id}/attributions")
def get_attributions(question_id: str, top_k: int = 3) -> dict:
    """Return the top-N office attribution suggestions for a question.

    The question must already be embedded in the ``questions_opendata``
    collection.  Its stored vector is used directly for the office search
    so no embedding API call is required.

    Args:
        question_id: Composite question ID, e.g. ``AN-17-QE-12345``.
        top_k: Number of office suggestions to return (default 3).

    Returns:
        A dict with ``question_id`` and an ``attributions`` list sorted by
        descending relevance, each entry containing ``rank``, ``office_id``,
        ``office_name``, ``direction``, ``score``, and ``relevance``.

    Raises:
        404: Question point not found in Qdrant (not yet embedded).
        422: ``top_k`` is less than 1.
    """
    if top_k < 1:
        raise HTTPException(status_code=422, detail="top_k must be at least 1.")

    state = _get_state()

    # 1. Fetch the question's pre-computed vector and text from Qdrant.
    point_id = stable_question_point_id(question_id)
    point = state.qdrant.get_point(QUESTIONS_COLLECTION, point_id, with_vectors=True)
    if point is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in Qdrant. "
            "Make sure it has been embedded with scripts/embed_questions.py.",
        )

    vector: list[float] = point["vector"]
    payload = point.get("payload") or {}
    texte_question: str = payload.get("texte_question") or ""

    if not texte_question:
        raise HTTPException(
            status_code=422,
            detail=f"Question '{question_id}' has no texte_question in its Qdrant payload.",
        )

    # 2. Search the office_responsibilities collection using the stored vector
    #    (no embedding call needed).
    candidates = retrieve_candidates(
        precomputed_vectors=[vector],
        qdrant=state.qdrant,
        collection=OFFICE_COLLECTION,
        top_k=20,
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
