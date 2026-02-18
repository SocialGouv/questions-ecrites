#!/usr/bin/env python
"""Reset the Qdrant collection and ingest manifest for job descriptions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qe import db
from qe.clients.qdrant import QdrantClient

DEFAULT_INPUT_DIR = Path("data/job_descriptions")
DEFAULT_COLLECTION = "job_descriptions"
DEFAULT_QDRANT_URL = "http://localhost:6333"
MANIFEST_FILENAME = ".ingest_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete a Qdrant collection and the corresponding ingest manifest."
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name to delete.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Base URL for Qdrant (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory whose manifest will be removed.",
    )
    return parser.parse_args()


def _remove_manifest(input_dir: Path) -> None:
    manifest_path = input_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"Removed ingest manifest at {manifest_path}.")
    else:
        print(f"No manifest found at {manifest_path}; nothing to remove.")


def _path_prefix_variants(input_dir: Path) -> list[str]:
    """Return likely path-prefix strings for matching ingest_manifest.path."""
    candidates: list[str] = []

    raw = str(input_dir)
    candidates.append(raw)
    candidates.append(raw.rstrip("/"))
    candidates.append(raw.rstrip("/") + "/")

    try:
        rel = str(input_dir.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel = None
    if rel:
        candidates.append(rel)
        candidates.append(rel.rstrip("/"))
        candidates.append(rel.rstrip("/") + "/")

    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _clear_postgres_state(input_dir: Path) -> None:
    """Clear Postgres ingest state for documents under input_dir."""
    variants = _path_prefix_variants(input_dir)
    manifest_entries: dict[str, str] = {}
    chosen_prefix: str | None = None

    for prefix in variants:
        entries = db.get_manifest_entries_under_prefix(prefix)
        if entries:
            manifest_entries = entries
            chosen_prefix = prefix
            break

    if not manifest_entries:
        print(
            "No Postgres manifest entries matched the input directory; nothing to remove."
        )
        return

    hashes = list(manifest_entries.values())
    deleted_cache = db.delete_chunk_cache_for_document_hashes(hashes)
    deleted_manifest = 0
    if chosen_prefix is not None:
        deleted_manifest = db.delete_manifest_under_prefix(chosen_prefix)

    print(
        f"Cleared Postgres ingest state for '{input_dir}': "
        f"deleted {deleted_manifest} manifest rows and {deleted_cache} chunk_cache rows."
    )


def main() -> int:
    args = parse_args()
    qdrant = QdrantClient(args.qdrant_url)

    try:
        deleted = qdrant.delete_collection(args.collection)
        if deleted:
            print(
                f"Deleted collection '{args.collection}'"
                f" from Qdrant at {args.qdrant_url}."
            )
        else:
            print(f"Collection '{args.collection}' was already absent in Qdrant.")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to delete collection: {exc}", file=sys.stderr)
        return 1

    try:
        _remove_manifest(args.input_dir)
    except OSError as exc:
        print(f"Failed to remove manifest: {exc}", file=sys.stderr)
        return 1

    try:
        _clear_postgres_state(args.input_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to clear Postgres ingest state: {exc}", file=sys.stderr)
        return 1

    print("Reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
