"""Core assignment logic: retrieval, reranking, and score aggregation."""

from __future__ import annotations

from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient


def retrieve_candidates(
    *,
    query_units: list[str],
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    collection: str,
    top_k: int,
    query_filter: dict | None = None,
) -> list[dict]:
    """Embed each query unit, search Qdrant, and return deduplicated candidates.

    Each query unit is embedded independently and used to retrieve ``top_k``
    candidates from Qdrant.  Results are deduplicated by point ID so that the
    same chunk is never sent to the reranker more than once, regardless of how
    many query units matched it.

    Args:
        query_units: Texts to use as retrieval queries (typically the full
            question followed by LLM-extracted duty units).
        embedder: Client for generating dense embeddings.
        qdrant: Qdrant REST client.
        collection: Name of the Qdrant collection to search.
        top_k: Number of nearest neighbours to retrieve per query unit.
        query_filter: Optional Qdrant filter dict to restrict the search
            (e.g. filter by ``chunk_type``).

    Returns:
        Deduplicated list of Qdrant candidate dicts, each with ``"id"``,
        ``"score"``, and ``"payload"`` keys.
    """
    seen_ids: dict[str, dict] = {}
    for query_unit in query_units:
        query_vector = embedder.embed(query_unit)
        candidates = qdrant.search(collection, query_vector, top_k, filter=query_filter)
        for candidate in candidates:
            point_id = candidate.get("id")
            if point_id and str(point_id) not in seen_ids:
                seen_ids[str(point_id)] = candidate
    return list(seen_ids.values())


def build_matches(
    *,
    candidates: list[dict],
    reranker: RerankClient,
    query: str,
) -> list[dict]:
    """Rerank candidates against the query and return a flat list of match dicts.

    Uses the full original question text as the rerank query so the cross-encoder
    judges relevance against the complete question rather than a narrower
    duty sub-topic.

    Args:
        candidates: Deduplicated Qdrant candidates from :func:`retrieve_candidates`.
        reranker: Albert rerank client.
        query: The rerank query — should be the full question text.

    Returns:
        List of match dicts sorted by rerank position (best first).  Each dict
        contains ``rerank_position``, ``score``, and office payload fields.
        Returns an empty list if ``candidates`` is empty.
    """
    if not candidates:
        return []

    candidate_texts: list[str] = [
        (c.get("payload") or {}).get("text") or "" for c in candidates
    ]
    rerank_results = reranker.rerank(
        query=query,
        documents=candidate_texts,
        top_n=len(candidate_texts),
    )

    matches: list[dict] = []
    for rerank_position, result in enumerate(rerank_results, start=1):
        candidate_index = result.get("index")
        if candidate_index is None or candidate_index >= len(candidates):
            continue
        candidate = candidates[candidate_index]
        payload = candidate.get("payload") or {}

        office_id = payload.get("office_id")
        if not office_id:
            continue
        matches.append(
            {
                "rerank_position": rerank_position,
                "score": result.get("relevance_score")
                if result.get("relevance_score") is not None
                else result.get("score"),
                "office_id": office_id,
                "office_name": payload.get("office_name"),
                "direction": payload.get("direction"),
                "chunk_type": payload.get("chunk_type"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_preview": payload.get("chunk_preview"),
            }
        )
    return matches


def aggregate_matches(
    matches: list[dict],
    *,
    max_chunks_per_office: int,
) -> tuple[list[dict], dict[str, float]]:
    """Aggregate reranked matches into per-office scores using sum-of-top-N.

    Groups matches by office, keeps the ``max_chunks_per_office`` highest-scoring
    chunks per office, and sums their scores.  This rewards offices that cover
    multiple aspects of the question (breadth signal) while the cap prevents
    an office with many chunks from winning on volume alone.

    Args:
        matches: Flat list of match dicts from :func:`build_matches`.
        max_chunks_per_office: Maximum number of chunks to count per office.

    Returns:
        A ``(kept_matches, score_by_office)`` tuple where:

        - ``kept_matches`` is the capped list sorted by descending score with
          sequential ``"rank"`` values assigned (1-based).
        - ``score_by_office`` maps each ``office_id`` to its cumulative score
          (sum of its top-N chunk scores).
    """
    chunks_by_office: dict[str, list[dict]] = {}
    for match in matches:
        key = str(match.get("office_id") or "")
        if not key:
            continue
        chunks_by_office.setdefault(key, []).append(match)

    kept_matches: list[dict] = []
    score_by_office: dict[str, float] = {}
    for office_chunks in chunks_by_office.values():
        office_chunks.sort(key=lambda m: -(m.get("score") or 0.0))
        top_chunks = office_chunks[:max_chunks_per_office]
        kept_matches.extend(top_chunks)
        office_id = top_chunks[0].get("office_id")
        if office_id:
            score_by_office[office_id] = sum(
                float(m.get("score") or 0.0) for m in top_chunks
            )

    kept_matches.sort(key=lambda m: -(m.get("score") or 0.0))
    for idx, m in enumerate(kept_matches, start=1):
        m["rank"] = idx

    return kept_matches, score_by_office


def match_question_to_offices(
    question: str,
    *,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    reranker: RerankClient,
    collection: str,
    top_k: int,
    query_filter: dict | None = None,
    max_chunks_per_office: int = 2,
) -> tuple[list[dict], dict[str, float]]:
    """Embed, search, rerank, and aggregate a single question against offices.

    Convenience wrapper around :func:`retrieve_candidates`,
    :func:`build_matches`, and :func:`aggregate_matches`.  Returns the same
    ``(kept_matches, score_by_office)`` tuple so callers can do their own
    post-processing.
    """
    candidates = retrieve_candidates(
        query_units=[question],
        embedder=embedder,
        qdrant=qdrant,
        collection=collection,
        top_k=top_k,
        query_filter=query_filter,
    )
    matches = build_matches(candidates=candidates, reranker=reranker, query=question)
    return aggregate_matches(matches, max_chunks_per_office=max_chunks_per_office)
