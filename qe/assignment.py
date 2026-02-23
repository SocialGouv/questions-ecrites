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

    Returns:
        Deduplicated list of Qdrant candidate dicts, each with ``"id"``,
        ``"score"``, and ``"payload"`` keys.
    """
    seen_ids: dict[str, dict] = {}
    for query_unit in query_units:
        query_vector = embedder.embed(query_unit)
        candidates = qdrant.search(collection, query_vector, top_k)
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
    judges "does this chunk help answer this question?" rather than a narrower
    duty sub-topic.

    Args:
        candidates: Deduplicated Qdrant candidates from :func:`retrieve_candidates`.
        reranker: Albert rerank client.
        query: The rerank query — should be the full question text.

    Returns:
        List of match dicts sorted by rerank position (best first).  Each dict
        contains ``rerank_position``, ``score``, and all relevant payload fields.
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

        job_id = payload.get("job_id") or payload.get("job_path")
        if not job_id:
            continue
        matches.append(
            {
                "rerank_position": rerank_position,
                "score": result.get("score") or result.get("relevance_score"),
                "job_id": job_id,
                "job_title": payload.get("job_title"),
                "user": payload.get("user"),
                "job_path": payload.get("job_path") or payload.get("path"),
                "section_title": payload.get("section_title"),
                "section_index": payload.get("section_index"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_preview": payload.get("chunk_preview"),
                "chunk_type": payload.get("chunk_type"),
                "duty_index": payload.get("duty_index"),
            }
        )
    return matches


def aggregate_matches(
    matches: list[dict],
    *,
    max_chunks_per_job: int,
) -> tuple[list[dict], dict[str, float]]:
    """Aggregate reranked matches into per-job scores using sum-of-top-N.

    Groups matches by job, keeps the ``max_chunks_per_job`` highest-scoring
    chunks per job, and sums their scores.  This rewards jobs that cover
    multiple aspects of the question (breadth signal) while the cap prevents
    a job with many chunks from winning on volume alone.

    Args:
        matches: Flat list of match dicts from :func:`build_matches`.
        max_chunks_per_job: Maximum number of chunks to count per job.

    Returns:
        A ``(kept_matches, score_by_user)`` tuple where:

        - ``kept_matches`` is the capped list sorted by descending score with
          sequential ``"rank"`` values assigned (1-based).
        - ``score_by_user`` maps each ``user`` to their cumulative score (sum
          of their top-N chunk scores).
    """
    chunks_by_job: dict[str, list[dict]] = {}
    for match in matches:
        key = str(match.get("job_id") or match.get("user") or "")
        if not key:
            continue
        chunks_by_job.setdefault(key, []).append(match)

    kept_matches: list[dict] = []
    score_by_user: dict[str, float] = {}
    for job_chunks in chunks_by_job.values():
        job_chunks.sort(key=lambda m: -(m.get("score") or 0.0))
        top_chunks = job_chunks[:max_chunks_per_job]
        kept_matches.extend(top_chunks)
        user = top_chunks[0].get("user")
        if user:
            score_by_user[user] = sum(float(m.get("score") or 0.0) for m in top_chunks)

    kept_matches.sort(key=lambda m: -(m.get("score") or 0.0))
    for idx, m in enumerate(kept_matches, start=1):
        m["rank"] = idx

    return kept_matches, score_by_user
