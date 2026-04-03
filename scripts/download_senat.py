#!/usr/bin/env python3
"""Download the Sénat full-database SQL dump (questions.zip).

The Sénat publishes a complete PostgreSQL dump of all written questions
(1978–present) at https://data.senat.fr/data/questions/questions.zip.
This archive is updated regularly.  The ingestion script filters to
legislatures 14–17.

By default this script skips the download if questions.zip already exists
locally.  Use --force to re-download and replace the local copy.

Usage:
    poetry run python scripts/download_senat.py --dir data/senat/
    poetry run python scripts/download_senat.py --dir data/senat/ --force
    poetry run python scripts/download_senat.py --dir data/senat/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_URL = "https://data.senat.fr/data/questions/questions.zip"
_FILENAME = "questions.zip"


def _download(url: str, dest: Path, http: requests.Session) -> bool:
    """Stream *url* to *dest*. Returns True on success."""
    try:
        with http.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            downloaded = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(
                            f"\r  {pct:3d}%  {downloaded // 1_000_000} MB",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress
            tmp.rename(dest)
        return True
    except requests.RequestException as exc:
        print()
        logger.error("Failed to download %s: %s", url, exc)
        tmp = dest.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        return False


def run(dest_dir: Path, force: bool, dry_run: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _FILENAME

    if dest.exists() and not force:
        logger.info("Already present: %s (use --force to re-download)", dest)
        return

    if dry_run:
        action = "re-download" if dest.exists() else "download"
        logger.info("[dry-run] would %s: %s", action, _FILENAME)
        logger.info("  from: %s", _URL)
        return

    action = "Re-downloading" if dest.exists() else "Downloading"
    logger.info("%s %s", action, _FILENAME)
    logger.info("  from: %s", _URL)

    http = requests.Session()
    http.headers["User-Agent"] = "qe-ingestion/1.0"

    ok = _download(_URL, dest, http)
    if ok:
        size_mb = dest.stat().st_size / 1_000_000
        logger.info("  saved: %s (%.1f MB)", dest, size_mb)
        logger.info(
            "Ingest with: poetry run python scripts/ingest_senat.py --file %s",
            dest,
        )
    else:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Sénat full questions SQL dump.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The dump covers all Sénat questions from 1978 to present.\n"
            "The ingestion script filters to legislatures 14–17."
        ),
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Local directory to save questions.zip into",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if questions.zip already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(dest_dir=args.dir, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
