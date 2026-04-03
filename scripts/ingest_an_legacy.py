#!/usr/bin/env python3
"""Ingest AN legacy question ZIP archives (XIV and XV legislatures).

Processes the ZIP archives downloaded by download_an_legacy.py and upserts
questions into PostgreSQL.

Usage:
    poetry run python scripts/ingest_an_legacy.py --dir data/an_archives/
    poetry run python scripts/ingest_an_legacy.py --dir data/an_archives/ --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import zipfile
from pathlib import Path

from qe import db
from qe.ingestion_an import (
    ingest_an_zip_file,
    parse_an_archive_question_xml,
    parse_an_bulk_xml,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _dry_run_zip(zip_path: Path) -> int:
    """Parse a ZIP archive without DB writes. Returns question count."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            per_file = [n for n in names if n.startswith("xml/") and n.endswith(".xml")]
            bulk = [n for n in names if n.endswith(".xml") and "/" not in n]

            if per_file:
                questions = [
                    pq
                    for name in per_file
                    if (pq := parse_an_archive_question_xml(zf.read(name))) is not None
                ]
            elif bulk:
                questions = parse_an_bulk_xml(zf.read(bulk[0]))
            else:
                questions = []
    except zipfile.BadZipFile as exc:
        logger.error("[dry-run] Failed to open %s: %s", zip_path.name, exc)
        return 0

    total = len(questions)
    repondues = sum(1 for pq in questions if pq.etat_question == "REPONDU")
    logger.info(
        "[dry-run] %-40s -> %d questions (%d EN_COURS, %d REPONDU)",
        zip_path.name,
        total,
        total - repondues,
        repondues,
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest AN legacy question ZIP archives into PostgreSQL.",
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing the .xml.zip archives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse archives only, do not write to the database",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    zip_dir: Path = args.dir
    if not zip_dir.exists():
        logger.error("Directory not found: %s", zip_dir)
        sys.exit(1)

    zip_files = sorted(zip_dir.glob("*.xml.zip"))
    if not zip_files:
        logger.warning("No .xml.zip files found in %s", zip_dir)
        return

    logger.info("Found %d archive(s) in %s", len(zip_files), zip_dir)

    if args.dry_run:
        logger.info("Dry-run mode — no database writes")
        total = sum(_dry_run_zip(p) for p in zip_files)
        logger.info("Dry-run total: %d questions parsed", total)
        return

    total_questions = 0
    total_ministeres = 0
    errors: list[str] = []

    manifest = db.get_manifest_entries()

    for zip_path in zip_files:
        file_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if manifest.get(str(zip_path)) == file_hash:
            logger.info("  %-40s -> already ingested, skipping", zip_path.name)
            continue
        try:
            stats = ingest_an_zip_file(zip_path)
            total_questions += stats.questions_inserted
            total_ministeres += stats.ministeres_created
            db.upsert_manifest(str(zip_path), file_hash)
        except Exception as exc:
            logger.error("Error processing %s: %s", zip_path.name, exc)
            errors.append(zip_path.name)

    logger.info(
        "Done: %d questions upserted, %d new ministries created",
        total_questions,
        total_ministeres,
    )
    if errors:
        logger.warning("Failed files (%d): %s", len(errors), ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
