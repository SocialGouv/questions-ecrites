#!/usr/bin/env python3
"""Download Assemblée Nationale legacy question archives (XML ZIP per legislature).

The AN publishes consolidated ZIP archives of written questions per completed
legislature on their open data portal.  Each ZIP contains one XML file per
question.

Available archives (static, one per completed legislature):

    XIV (2012–2017)  https://data.assemblee-nationale.fr/static/openData/
                         repository/14/questions/questions_ecrites/
                         Questions_ecrites_XIV.xml.zip          (~133 MB)

    XV  (2017–2022)  https://data.assemblee-nationale.fr/static/openData/
                         repository/15/questions/questions_ecrites/
                         Questions_ecrites_XV.xml.zip           (~97 MB)

    XVI (2022–2024)  https://data.assemblee-nationale.fr/static/openData/
                         repository/16/questions/questions_ecrites/
                         Questions_ecrites.xml.zip              (~47 MB)
                         Note: no roman numeral in the server-side filename.
                         Last updated June 28, 2024 (covers the legislature
                         up to ~2 weeks before dissolution).

XVII (2024–present) questions are covered by the DILA REDIF archives
downloaded with download_opendata.py.

Usage:
    # Download XIV, XV, and XVI archives (default)
    poetry run python scripts/download_an_legacy.py --dir data/an_archives/

    # Download only a specific legislature
    poetry run python scripts/download_an_legacy.py --dir data/an_archives/ --legislature 16

    # List what would be downloaded without fetching
    poetry run python scripts/download_an_legacy.py --dir data/an_archives/ --dry-run
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

_BASE = "https://data.assemblee-nationale.fr/static/openData/repository"

_ROMAN = {14: "XIV", 15: "XV", 16: "XVI"}

# XVI uses a different server-side filename (no roman numeral suffix).
_ARCHIVES: dict[int, str] = {
    14: f"{_BASE}/14/questions/questions_ecrites/Questions_ecrites_XIV.xml.zip",
    15: f"{_BASE}/15/questions/questions_ecrites/Questions_ecrites_XV.xml.zip",
    16: f"{_BASE}/16/questions/questions_ecrites/Questions_ecrites.xml.zip",
}


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


def run(dest_dir: Path, legislatures: list[int], dry_run: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    http.headers["User-Agent"] = "qe-ingestion/1.0"

    errors: list[str] = []

    for leg in sorted(legislatures):
        url = _ARCHIVES[leg]
        filename = f"Questions_ecrites_{_ROMAN[leg]}.xml.zip"
        dest = dest_dir / filename

        if dest.exists():
            logger.info("Legislature %d — already present: %s", leg, filename)
            continue

        if dry_run:
            logger.info("Legislature %d — [dry-run] would download: %s", leg, filename)
        else:
            logger.info("Legislature %d — downloading %s", leg, filename)
            logger.info("  from: %s", url)
            ok = _download(url, dest, http)
            if ok:
                size_mb = dest.stat().st_size / 1_000_000
                logger.info("  saved: %s (%.1f MB)", dest, size_mb)
            else:
                errors.append(filename)

    if errors:
        logger.warning("Failed downloads (%d): %s", len(errors), ", ".join(errors))
        sys.exit(1)

    if not dry_run:
        logger.info(
            "Done. Ingest with: poetry run python scripts/ingest_an_legacy.py --dir %s",
            dest_dir,
        )


def main() -> None:
    available = sorted(_ARCHIVES)
    parser = argparse.ArgumentParser(
        description="Download AN legacy question archives (XIV, XV, and XVI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Note: XVII (2024–present) questions are covered by the DILA\n"
            "REDIF archives — use download_opendata.py for those."
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
        type=int,
        choices=available,
        metavar="|".join(str(legi) for legi in available),
        help="Download only a specific legislature (default: all available)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without fetching",
    )
    args = parser.parse_args()

    legislatures = [args.legislature] if args.legislature else available
    run(dest_dir=args.dir, legislatures=legislatures, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
