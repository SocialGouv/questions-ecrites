#!/usr/bin/env python3
"""Ingest Sénat questions from the full-database SQL dump.

Processes the ZIP downloaded by download_senat.py and upserts questions
écrites from legislatures 14–17 into PostgreSQL.

Usage:
    poetry run python scripts/ingest_senat.py --file data/senat/questions.zip
    poetry run python scripts/ingest_senat.py --file data/senat/questions.zip --dry-run
    poetry run python scripts/ingest_senat.py --file data/senat/questions.zip --force
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import logging
import sys
import zipfile
from pathlib import Path

from qe import db
from qe.ingestion_senat import ingest_senat_dump, parse_senat_sql_dump

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _dry_run(zip_path: Path) -> None:
    """Parse the dump without DB writes and log a breakdown."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        logger.error("Failed to open %s: %s", zip_path.name, exc)
        return

    with zf:
        sql_names = [n for n in zf.namelist() if n.endswith(".sql")]
        if not sql_names:
            logger.error("No .sql file found in %s", zip_path.name)
            return
        with zf.open(sql_names[0]) as f:
            questions = parse_senat_sql_dump(f)

    total = len(questions)
    by_leg: dict[int, int] = collections.Counter(q.legislature for q in questions)
    by_etat: dict[str, int] = collections.Counter(q.etat_question for q in questions)

    logger.info("[dry-run] %d questions parsed (Nature=QE, legislature 14–17)", total)
    logger.info("[dry-run] By legislature: %s", dict(sorted(by_leg.items())))
    logger.info("[dry-run] By état: %s", dict(sorted(by_etat.items())))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Sénat questions from the full SQL dump into PostgreSQL.",
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the Sénat questions.zip SQL dump",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only, do not write to the database",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-ingest even if the file hash is already in the manifest",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    zip_path: Path = args.file
    if not zip_path.exists():
        logger.error("File not found: %s", zip_path)
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry-run mode — no database writes")
        _dry_run(zip_path)
        return

    file_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    manifest = db.get_manifest_entries()

    if not args.force and manifest.get(str(zip_path)) == file_hash:
        logger.info(
            "Already ingested (hash matches manifest). Use --force to re-ingest."
        )
        return

    try:
        stats = ingest_senat_dump(zip_path)
        db.upsert_manifest(str(zip_path), file_hash)
    except Exception as exc:
        logger.error("Error processing %s: %s", zip_path.name, exc)
        sys.exit(1)

    logger.info(
        "Done: %d questions upserted, %d new ministries created",
        stats.questions_inserted,
        stats.ministeres_created,
    )


if __name__ == "__main__":
    main()
