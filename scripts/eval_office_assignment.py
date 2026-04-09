#!/usr/bin/env python
"""Evaluate office assignment quality against a ground-truth attribution file."""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
from tqdm import tqdm

from qe.assignment import match_question_to_offices
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.office_ingestion import OFFICE_COLLECTION

DEFAULT_INPUT = Path("data/qe_attributions_DGCS.xlsx")
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
DEFAULT_ALBERT_MODEL = "openweight-rerank"
DEFAULT_TOP_K = 20
DEFAULT_TOP_OFFICES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate office assignment against a ground-truth XLSX file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Ground-truth XLSX file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Qdrant candidates to retrieve per question (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--top-offices",
        type=int,
        default=DEFAULT_TOP_OFFICES,
        help=f"Max offices to rank per question (default: {DEFAULT_TOP_OFFICES}).",
    )
    parser.add_argument(
        "--collection",
        default=OFFICE_COLLECTION,
        help=f"Qdrant collection to search (default: {OFFICE_COLLECTION}).",
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


def _load_ground_truth(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (question_id, question_text, expected_office_id) from XLSX."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue  # skip header
        question_id, question_text, office = row[0], row[1], row[2]
        if not question_text or not office:
            continue
        rows.append(
            (str(question_id or ""), str(question_text).strip(), str(office).strip())
        )
    wb.close()
    return rows


def main() -> int:  # noqa: C901
    args = parse_args()
    settings = get_settings()

    rows = _load_ground_truth(args.input)
    if not rows:
        print(f"No rows found in {args.input}.")
        return 1

    query_filter: dict | None = None
    if args.chunks != "all":
        query_filter = {
            "must": [{"key": "chunk_type", "match": {"value": args.chunks}}]
        }

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

    hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []

    for _question_id, question_text, expected_office in tqdm(rows, desc="Evaluating"):
        kept_matches, score_by_office = match_question_to_offices(
            question_text,
            embedder=embedder,
            qdrant=qdrant,
            reranker=reranker,
            collection=args.collection,
            top_k=args.top_k,
            query_filter=query_filter,
        )
        seen: set[str] = set()
        ranked: list[str] = []
        for m in sorted(
            kept_matches,
            key=lambda x: -(score_by_office.get(x.get("office_id", ""), 0.0)),
        ):
            office_id = m.get("office_id", "")
            if office_id and office_id not in seen:
                seen.add(office_id)
                ranked.append(office_id)
                if len(ranked) >= args.top_offices:
                    break

        rank: int | None = None
        for i, office_id in enumerate(ranked, 1):
            if office_id == expected_office:
                rank = i
                break

        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1

        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    n = len(rows)
    mrr = sum(reciprocal_ranks) / n

    print(f"\nResults ({args.input.name})")
    print("-" * 40)
    print(f"{'Total questions':<20}: {n}")
    for k in sorted(hits_at):
        print(f"{'Hit@' + str(k):<20}: {hits_at[k] / n * 100:.1f}%")
    print(f"{'MRR':<20}: {mrr:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
