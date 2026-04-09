#!/usr/bin/env python
"""Assign a parliamentary question to the most relevant ministry office."""

from __future__ import annotations

import argparse
import json

from qe.assignment import match_question_to_offices
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.office_ingestion import OFFICE_COLLECTION

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
DEFAULT_ALBERT_MODEL = "openweight-rerank"
DEFAULT_TOP_K = 20
DEFAULT_TOP_OFFICES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign a parliamentary question to the most relevant office."
    )
    parser.add_argument(
        "--question", required=True, help="The question text to assign."
    )
    parser.add_argument(
        "--collection",
        default=OFFICE_COLLECTION,
        help=f"Qdrant collection to search (default: {OFFICE_COLLECTION}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Candidates to retrieve per query unit (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--top-offices",
        type=int,
        default=DEFAULT_TOP_OFFICES,
        help=f"Number of top offices to return (default: {DEFAULT_TOP_OFFICES}).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant base URL (default: {DEFAULT_QDRANT_URL}).",
    )
    parser.add_argument(
        "--chunks",
        choices=["all", "responsibilities", "keywords"],
        default="all",
        help="Which chunk types to search (default: all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    embedder = EmbeddingClient(
        url=settings.embeddings_url,
        api_key=settings.socle_api_key,
        model=settings.embedding_model,
    )
    qdrant = QdrantClient(args.qdrant_url)
    reranker = RerankClient(
        base_url=DEFAULT_ALBERT_BASE_URL,
        model=DEFAULT_ALBERT_MODEL,
        api_key=settings.albert_api_key,
    )

    question = args.question.strip()

    query_filter: dict | None = None
    if args.chunks != "all":
        query_filter = {
            "must": [{"key": "chunk_type", "match": {"value": args.chunks}}]
        }

    kept_matches, score_by_office = match_question_to_offices(
        question,
        embedder=embedder,
        qdrant=qdrant,
        reranker=reranker,
        collection=args.collection,
        top_k=args.top_k,
        query_filter=query_filter,
    )

    # Linear normalization: each office's share of the total relevance signal.
    total_score = sum(score_by_office.values())
    pct_by_office = {
        office_id: s / total_score for office_id, s in score_by_office.items()
    }

    # Deduplicate to one result row per office, sorted by aggregated score.
    seen: set[str] = set()
    results: list[dict] = []
    for m in sorted(
        kept_matches, key=lambda x: -(score_by_office.get(x.get("office_id", ""), 0.0))
    ):
        office_id = m.get("office_id", "")
        if office_id in seen:
            continue
        seen.add(office_id)
        results.append(
            {
                "rank": len(results) + 1,
                "office_id": office_id,
                "office_name": m.get("office_name"),
                "direction": m.get("direction"),
                "relevance_pct": round(pct_by_office.get(office_id, 0.0) * 100, 1),
            }
        )
        if len(results) >= args.top_offices:
            break

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
