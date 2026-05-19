#!/usr/bin/env python3
"""Load a JSONL dump (produced by scripts/dump_qdrant.py) into pgvector tables.

Idempotent: uses INSERT ... ON CONFLICT DO UPDATE, so it is safe to run
multiple times or resume after interruption.

Prerequisites:
    poetry run alembic upgrade head   # creates the vec_* tables

Usage:
    poetry run python scripts/load_pgvector.py --input data/qdrant_dump.jsonl
    poetry run python scripts/load_pgvector.py --input data/qdrant_dump.jsonl \\
        --batch-size 200
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

from qe.clients.pgvector_client import PgvectorClient

DEFAULT_BATCH_SIZE = 500


def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a JSONL Qdrant dump into pgvector tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSONL file produced by scripts/dump_qdrant.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Points per INSERT statement (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    client = PgvectorClient()

    # Group all points by collection before upserting so we make one pass
    # over the file and emit collection-grouped batches.
    points_by_collection: dict[str, list[dict]] = {}
    total_lines = 0

    print(f"Reading {args.input}...", file=sys.stderr)
    with args.input.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            coll = obj.pop("collection")
            points_by_collection.setdefault(coll, []).append(obj)
            total_lines += 1

    print(
        f"Read {total_lines} point(s) across {len(points_by_collection)} collection(s).",
        file=sys.stderr,
    )

    upserted = 0
    for coll, points in points_by_collection.items():
        print(f"Loading {len(points)} point(s) into '{coll}'...", file=sys.stderr)
        for batch in _batched(points, args.batch_size):
            client.upsert_points(coll, batch)
            upserted += len(batch)
            print(f"  {upserted}/{total_lines}", end="\r", file=sys.stderr)

    print(f"\nDone — {upserted} point(s) upserted.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
