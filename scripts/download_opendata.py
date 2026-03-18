#!/usr/bin/env python3
"""Download DILA open data .taz archives for written questions.

Scrapes the DILA open data directory listing and downloads any .taz files
not already present in the local destination directory.

Directory layout on the DILA server:
    AN/Annee_en_cours/   — current year (rolling, new files added each JO)
    AN/2025/             — full year archive
    AN/2024/             — etc.
    SENAT/Annee_en_cours/
    SENAT/2025/
    ...

Usage:
    # Current year only (default)
    poetry run python scripts/download_opendata.py --dir data/opendata/

    # Current year + 2 prior years
    poetry run python scripts/download_opendata.py --dir data/opendata/ --years 2

    # List what would be downloaded without fetching
    poetry run python scripts/download_opendata.py --dir data/opendata/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://echanges.dila.gouv.fr/OPENDATA/Questions-Reponses"

# Apache directory listing: links look like href="ANQ20260010.taz"
_TAZ_HREF_RE = re.compile(r'href="([^"]+\.taz)"', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------


def _list_directory(url: str, http: requests.Session) -> list[str]:
    """Fetch an Apache directory listing and return all .taz filenames found."""
    try:
        resp = http.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch directory listing %s: %s", url, exc)
        return []

    filenames = _TAZ_HREF_RE.findall(resp.text)
    # Strip any leading path components — keep the basename only
    return [f.split("/")[-1] for f in filenames]


def _directory_urls(years: int) -> list[tuple[str, str]]:
    """Return a list of (label, url) pairs to scrape.

    Always includes Annee_en_cours for both AN and SENAT.
    If years > 0, also includes that many prior calendar years.
    """
    current_year = date.today().year
    dirs: list[tuple[str, str]] = []

    for source in ("AN", "SENAT"):
        dirs.append(
            (
                f"{source}/Annee_en_cours",
                f"{BASE_URL}/{source}/Annee_en_cours/",
            )
        )
        for offset in range(1, years + 1):
            yr = current_year - offset
            dirs.append(
                (
                    f"{source}/{yr}",
                    f"{BASE_URL}/{source}/{yr}/",
                )
            )

    return dirs


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, http: requests.Session) -> bool:
    """Download url to dest. Returns True on success."""
    try:
        with http.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            tmp.rename(dest)
        return True
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        if dest.with_suffix(".tmp").exists():
            dest.with_suffix(".tmp").unlink()
        return False


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run(dest_dir: Path, years: int, dry_run: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    http = requests.Session()
    http.headers["User-Agent"] = "qe-ingestion/1.0"

    total_found = 0
    total_downloaded = 0
    total_skipped = 0
    errors: list[str] = []

    for label, url in _directory_urls(years):
        filenames = _list_directory(url, http)
        if not filenames:
            logger.warning("No .taz files found at %s", url)
            continue

        logger.info("%s — %d file(s) found", label, len(filenames))
        total_found += len(filenames)

        for filename in sorted(filenames):
            dest = dest_dir / filename
            if dest.exists():
                logger.debug("  already present: %s", filename)
                total_skipped += 1
                continue

            file_url = url + filename
            if dry_run:
                logger.info("  [dry-run] would download: %s", filename)
                total_downloaded += 1
            else:
                logger.info("  downloading: %s", filename)
                ok = _download_file(file_url, dest, http)
                if ok:
                    total_downloaded += 1
                else:
                    errors.append(filename)

    action = "would download" if dry_run else "downloaded"
    logger.info(
        "Done — %d file(s) found, %d %s, %d already present",
        total_found,
        total_downloaded,
        action,
        total_skipped,
    )
    if errors:
        logger.warning("Failed downloads (%d): %s", len(errors), ", ".join(errors))
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DILA open data .taz archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Local directory to save .taz files into",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of prior calendar years to include in addition to the "
            "current year (default: 0 — current year only)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without fetching them",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging (shows already-present files)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(dest_dir=args.dir, years=args.years, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
