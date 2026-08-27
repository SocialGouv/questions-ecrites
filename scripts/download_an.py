#!/usr/bin/env python3
"""Download Assemblée Nationale question archives (XML ZIP per legislature).

The AN publishes consolidated ZIP archives of written questions per legislature
on their open data portal, one directory per legislature:

    https://data.assemblee-nationale.fr/static/openData/repository/{N}/
        questions/questions_ecrites/...

Legislatures 14 and 15 use a roman-numeral filename
(``Questions_ecrites_XIV.xml.zip``); legislature 16 onward uses a generic one
(``Questions_ecrites.xml.zip``, distinguished only by the ``{N}`` in the path).
Whichever legislature is highest is the live, ongoing one — re-download it
regularly to pick up new questions and answers. The current legislature is
detected automatically (see ``discover_current_legislature``): there is no
fixed number to keep up to date in this script.

Usage:
    # Download only the live legislature and immediately ingest (recommended daily run)
    poetry run python scripts/download_an.py --dir data/an_archives/ --legislature current --ingest

    # Download all legislatures from 14 through the current one (default, no ingest)
    poetry run python scripts/download_an.py --dir data/an_archives/

    # Download only a specific legislature
    poetry run python scripts/download_an.py --dir data/an_archives/ --legislature 17

    # List what would be downloaded without fetching
    poetry run python scripts/download_an.py --dir data/an_archives/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

from qe.downloads import download_with_retries
from qe.hashing import hash_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_BASE = "https://data.assemblee-nationale.fr/static/openData/repository"

# Legislatures 14 and 15 use a roman-numeral filename; 16 onward doesn't.
_ROMAN = {14: "XIV", 15: "XV"}

# Oldest legislature this pipeline supports.
_MIN_LEGISLATURE = 14

# Scan-start optimization only, not a correctness bound — discovery still
# finds the right answer if this is left stale for years, it just costs a
# couple more HEAD requests.
_KNOWN_LEGISLATURE_FLOOR = 17

_CURRENT = "current"


def _url_for(n: int) -> str:
    if n in _ROMAN:
        return f"{_BASE}/{n}/questions/questions_ecrites/Questions_ecrites_{_ROMAN[n]}.xml.zip"
    return f"{_BASE}/{n}/questions/questions_ecrites/Questions_ecrites.xml.zip"


def _archive_filename(n: int) -> str:
    return f"Questions_ecrites_{_ROMAN.get(n, str(n))}.xml.zip"


def discover_current_legislature(
    http: requests.Session, floor: int = _KNOWN_LEGISLATURE_FLOOR
) -> int:
    """Return the highest legislature number whose archive exists on the AN portal."""
    current = floor
    while True:
        candidate = current + 1
        try:
            resp = http.head(_url_for(candidate), timeout=30, allow_redirects=True)
        except requests.RequestException as exc:
            logger.warning(
                "Legislature discovery failed probing %d, assuming %d is current: %s",
                candidate,
                current,
                exc,
            )
            return current
        if resp.status_code != 200:
            return current
        current = candidate


def _ingest(dest_dir: Path, legislatures: list[int]) -> None:
    """Ingest downloaded ZIP archives into PostgreSQL."""
    from qe import db
    from qe.ingestion_an import ingest_an_zip_file

    manifest = db.get_manifest_entries()
    for leg in sorted(legislatures):
        zip_path = dest_dir / _archive_filename(leg)
        if not zip_path.exists():
            logger.warning(
                "Legislature %d — archive not found, skipping ingest: %s", leg, zip_path
            )
            continue
        file_hash = hash_file(zip_path)
        if manifest.get(str(zip_path)) == file_hash:
            logger.info(
                "Legislature %d — already ingested (hash unchanged), skipping", leg
            )
            continue
        logger.info("Legislature %d — ingesting %s", leg, zip_path.name)
        stats = ingest_an_zip_file(zip_path)
        db.upsert_manifest(str(zip_path), file_hash)
        logger.info(
            "  ingested: %d questions (%d inserted, %d updated)",
            stats.questions_parsed,
            stats.questions_inserted,
            stats.questions_updated,
        )


def run(
    dest_dir: Path,
    legislatures: list[int],
    live_legislature: int,
    dry_run: bool,
    ingest: bool,
    http: requests.Session,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    downloaded: list[int] = []

    for leg in sorted(legislatures):
        url = _url_for(leg)
        filename = _archive_filename(leg)
        dest = dest_dir / filename
        is_live = leg == live_legislature

        if dest.exists() and not is_live:
            logger.info("Legislature %d — already present: %s", leg, filename)
            downloaded.append(leg)
            continue

        if dest.exists() and is_live:
            logger.info(
                "Legislature %d — re-downloading live archive: %s", leg, filename
            )

        if dry_run:
            logger.info("Legislature %d — [dry-run] would download: %s", leg, filename)
        else:
            logger.info("Legislature %d — downloading %s", leg, filename)
            logger.info("  from: %s", url)
            ok = download_with_retries(url, dest, http)
            if ok:
                size_mb = dest.stat().st_size / 1_000_000
                logger.info("  saved: %s (%.1f MB)", dest, size_mb)
                downloaded.append(leg)
            else:
                errors.append(filename)

    if errors:
        logger.warning("Failed downloads (%d): %s", len(errors), ", ".join(errors))
        sys.exit(1)

    if ingest and not dry_run:
        _ingest(dest_dir, downloaded)
    elif not dry_run and not ingest:
        logger.info(
            "Done. Ingest with: poetry run python scripts/ingest_an.py --dir %s",
            dest_dir,
        )


def _legislature_arg(value: str) -> int | str:
    if value == _CURRENT:
        return _CURRENT
    n = int(value)
    if n < _MIN_LEGISLATURE:
        raise argparse.ArgumentTypeError(f"legislature must be >= {_MIN_LEGISLATURE}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AN question archives (legislature 14 onward).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The current legislature is auto-detected and always re-downloaded\n"
            "(it's the live, ongoing one). Re-run without --legislature to refresh\n"
            "it alongside every static archive."
        ),
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Local directory to save ZIP archives into",
    )
    parser.add_argument(
        "--legislature",
        type=_legislature_arg,
        metavar=f"N|{_CURRENT}",
        help="Download only a specific legislature, or 'current' for the live one (default: 14 through current)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without fetching",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest downloaded archives into PostgreSQL immediately after download",
    )
    args = parser.parse_args()

    http = requests.Session()
    http.headers["User-Agent"] = "qe-ingestion/1.0"

    live_legislature = discover_current_legislature(http)

    if args.legislature == _CURRENT:
        legislatures = [live_legislature]
    elif args.legislature is not None:
        legislatures = [args.legislature]
    else:
        legislatures = list(range(_MIN_LEGISLATURE, live_legislature + 1))

    run(
        dest_dir=args.dir,
        legislatures=legislatures,
        live_legislature=live_legislature,
        dry_run=args.dry_run,
        ingest=args.ingest,
        http=http,
    )


if __name__ == "__main__":
    main()
