#!/usr/bin/env python3
"""Dump all Qdrant collections to a JSONL file for import into pgvector.

Each line of the output file is a JSON object:
    {"collection": "<name>", "id": "<uuid>", "vector": [...], "payload": {...}}

The file can be shared with coworkers and loaded with scripts/load_pgvector.py.

Usage:
    poetry run python scripts/dump_qdrant.py --output data/qdrant_dump.jsonl
    poetry run python scripts/dump_qdrant.py --output data/qdrant_dump.jsonl \\
        --qdrant-url http://qdrant.internal:6333 \\
        --collections office_responsibilities questions_opendata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qe.clients.qdrant import QdrantClient

ALL_COLLECTIONS = [
    "office_responsibilities",
    "questions_opendata",
    "answers_opendata",
]
DEFAULT_QDRANT_URL = "http://localhost:6333"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump Qdrant collections to a JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Destination JSONL file (will be created or overwritten).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        metavar="URL",
        help=f"Qdrant base URL (default: {DEFAULT_QDRANT_URL}).",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=ALL_COLLECTIONS,
        metavar="NAME",
        help=f"Collections to dump (default: all three — {', '.join(ALL_COLLECTIONS)}).",
    )
    args = parser.parse_args()

    client = QdrantClient(args.qdrant_url)
    total = 0

    with args.output.open("w", encoding="utf-8") as fh:
        for coll in args.collections:
            if not client.collection_exists(coll):
                print(
                    f"  Collection '{coll}' not found in Qdrant — skipping.",
                    file=sys.stderr,
                )
                continue
            print(f"Scrolling '{coll}'...", file=sys.stderr)
            points = client.scroll_all(coll, with_vectors=True)
            print(f"  {len(points)} point(s).", file=sys.stderr)
            for pt in points:
                line = {"collection": coll, **pt}
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                total += 1

    print(f"Done — {total} point(s) written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
