#!/usr/bin/env python
"""Reset the Qdrant collection and ingest manifest for job descriptions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

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


def delete_collection(base_url: str, collection: str) -> None:
    base = base_url.rstrip("/")
    response = requests.delete(
        f"{base}/collections/{collection}",
        timeout=30,
    )
    if response.status_code == 404:
        print(f"Collection '{collection}' was already absent in Qdrant.")
        return
    response.raise_for_status()
    print(f"Deleted collection '{collection}' from Qdrant at {base}.")


def remove_manifest(input_dir: Path) -> None:
    manifest_path = input_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"Removed ingest manifest at {manifest_path}.")
    else:
        print(f"No manifest found at {manifest_path}; nothing to remove.")


def main() -> int:
    args = parse_args()
    try:
        delete_collection(args.qdrant_url, args.collection)
    except requests.RequestException as exc:
        print(f"Failed to delete collection: {exc}", file=sys.stderr)
        return 1

    try:
        remove_manifest(args.input_dir)
    except OSError as exc:
        print(f"Failed to remove manifest: {exc}", file=sys.stderr)
        return 1

    print("Reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
