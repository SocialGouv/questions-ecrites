#!/usr/bin/env python
"""Ingest office responsibility XLSX files into the Qdrant office_responsibilities collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings
from qe.office_ingestion import OFFICE_COLLECTION, ingest_office_xlsx

DEFAULT_INPUT_DIR = Path("data/office_responsibilities")
DEFAULT_QDRANT_URL = "http://localhost:6333"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest office responsibility XLSX files into Qdrant."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing XLSX files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--collection",
        default=OFFICE_COLLECTION,
        help=f"Qdrant collection name (default: {OFFICE_COLLECTION}).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant base URL (default: {DEFAULT_QDRANT_URL}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    xlsx_files = sorted(args.dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"No XLSX files found in {args.dir}.")
        return 1

    embedder = EmbeddingClient(
        url=settings.embeddings_url,
        api_key=settings.socle_api_key,
        model=settings.embedding_model,
    )
    qdrant = QdrantClient(args.qdrant_url)

    for xlsx_path in xlsx_files:
        ingest_office_xlsx(
            xlsx_path=xlsx_path,
            collection=args.collection,
            embedder=embedder,
            qdrant=qdrant,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
