"""Core assignment logic: retrieval, reranking, and score aggregation."""

from __future__ import annotations

import logging

from qe.clients.embedding import EmbeddingClient
from qe.clients.rerank import RerankClient
from qe.clients.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve_candidates(
    *,
    query_units: list[str] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
    embedder: EmbeddingClient | None = None,
    vector_store: VectorStore,
    collection: str,
    top_k: int,
    query_filter: dict | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Search the vector store and return deduplicated candidates.

    Accepts either raw query texts (which are embedded on the fly) or
    pre-computed vectors (which bypass the embedding step entirely).  The two
    sources can be combined: all resulting vectors are searched and their
    results are merged.

    Each vector is searched independently and results are deduplicated by point
    ID so that the same chunk is never sent to the reranker more than once.

    Args:
        query_units: Texts to embed and use as retrieval queries.  Requires
            ``embedder`` to be provided when non-empty.
        precomputed_vectors: Dense vectors to use directly for search, skipping
            the embedding step.  Useful when the question is already embedded
            in the vector store (e.g. ``questions_opendata`` collection).
        embedder: Client for generating dense embeddings.  Required when
            ``query_units`` is provided; may be ``None`` otherwise.
        vector_store: Vector store client.
        collection: Name of the collection to search.
        top_k: Number of nearest neighbours to retrieve per vector.
        query_filter: Optional filter dict to restrict the search
            (e.g. filter by ``chunk_type``).
        score_threshold: Optional minimum cosine similarity score (0.0–1.0).
            Candidates below this threshold are dropped before reranking.

    Returns:
        Deduplicated list of candidate dicts, each with ``"id"``,
        ``"score"``, and ``"payload"`` keys.

    Raises:
        ValueError: If neither ``query_units`` nor ``precomputed_vectors`` are
            provided, or if ``query_units`` are provided without an
            ``embedder``.
    """
    if not query_units and not precomputed_vectors:
        raise ValueError(
            "At least one of query_units or precomputed_vectors must be provided."
        )
    if query_units and embedder is None:
        raise ValueError("embedder is required when query_units are provided.")

    vectors: list[list[float]] = list(precomputed_vectors or [])
    for query_unit in query_units or []:
        vectors.append(embedder.embed(query_unit))  # type: ignore[union-attr]

    seen_ids: dict[str, dict] = {}
    for vector in vectors:
        candidates = vector_store.search(
            collection,
            vector,
            top_k,
            filter=query_filter,
            score_threshold=score_threshold,
        )
        for candidate in candidates:
            point_id = candidate.get("id")
            if point_id and str(point_id) not in seen_ids:
                seen_ids[str(point_id)] = candidate
    return list(seen_ids.values())


def rerank_candidates(
    candidates: list[dict],
    reranker: RerankClient,
    query: str,
    text_field: str,
) -> list[tuple[dict, float]]:
    """Rerank candidates by relevance and return (candidate, score) pairs.

    Args:
        candidates: Candidate dicts with ``"id"``, ``"score"``,
            ``"payload"`` keys.
        reranker: Albert rerank client.
        query: The rerank query (full question text).
        text_field: Payload key whose value is used as the document text.

    Returns:
        List of ``(candidate, rerank_score)`` tuples sorted by descending
        score.  Empty if ``candidates`` is empty.
    """
    if not candidates:
        return []

    texts = [(c.get("payload") or {}).get(text_field) or "" for c in candidates]
    results = reranker.rerank(query=query, documents=texts, top_n=len(texts))

    scored: list[tuple[dict, float]] = []
    for result in results:
        idx = result.get("index")
        if idx is None or idx >= len(candidates):
            continue
        score = (
            result.get("relevance_score")
            if result.get("relevance_score") is not None
            else result.get("score")
        )
        if score is None:
            logger.warning(
                "Reranker returned no score for result at index %s; skipping.", idx
            )
            continue
        scored.append((candidates[idx], float(score)))

    return sorted(scored, key=lambda x: -x[1])
