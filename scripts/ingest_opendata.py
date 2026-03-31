#!/usr/bin/env python3
"""Ingest DILA open data .taz archives (written questions).

Processes local .taz files and upserts questions into PostgreSQL.
The assembly source (AN / SENAT) is read from the XML inside each archive,
so AN and SENAT archives can be mixed freely in the same directory.

Usage:
    # Process all .taz files already downloaded to data/opendata/
    poetry run python scripts/ingest_opendata.py --dir data/opendata/

    # Dry-run (parse only, no DB writes)
    poetry run python scripts/ingest_opendata.py --dir data/opendata/ --dry-run

Typical workflow for the historical ingest:
    1. Download archives from
       https://echanges.dila.gouv.fr/OPENDATA/Questions-Reponses/AN/Annee_en_cours/
       (and prior years as needed)
    2. Place them in data/opendata/
    3. Run this script
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
from pathlib import Path

from qe import db
from qe.ingestion_opendata import ingest_taz_file, parse_redif_xml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest DILA open data .taz archives into PostgreSQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing the .taz archives to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse archives only, do not write to the database",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def _dry_run_taz(taz_path: Path) -> int:
    """Parse a .taz archive without writing to the DB. Returns question count."""
    try:
        with tarfile.open(taz_path) as tar:
            redif_file_member = next(
                (
                    file_member
                    for file_member in tar.getmembers()
                    if file_member.name.startswith("REDIF_")
                ),
                None,
            )
            if redif_file_member is None:
                logger.warning("[dry-run] No REDIF_*.xml in %s", taz_path.name)
                return 0
            file = tar.extractfile(redif_file_member)
            if file is None:
                return 0
            xml_bytes = file.read()
    except Exception as exc:
        logger.error("[dry-run] Failed to read %s: %s", taz_path.name, exc)
        return 0

    questions = parse_redif_xml(xml_bytes)
    en_cours = sum(1 for q in questions if q.etat_question == "EN_COURS")
    repondues = sum(1 for q in questions if q.etat_question == "REPONDU")
    logger.info(
        "[dry-run] %-32s -> %3d questions (%d EN_COURS, %d REPONDU)",
        taz_path.name,
        len(questions),
        en_cours,
        repondues,
    )
    return len(questions)


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    taz_dir: Path = args.dir
    if not taz_dir.exists():
        logger.error("Directory not found: %s", taz_dir)
        sys.exit(1)

    taz_files = sorted(taz_dir.glob("*.taz"))
    if not taz_files:
        logger.warning("No .taz files found in %s", taz_dir)
        return

    logger.info("Found %d .taz file(s) in %s", len(taz_files), taz_dir)

    if args.dry_run:
        logger.info("Dry-run mode — no database writes")
        total = sum(_dry_run_taz(p) for p in taz_files)
        logger.info("Dry-run total: %d questions parsed", total)
        return

    total_questions = 0
    total_ministeres = 0
    errors: list[str] = []

    manifest = db.get_manifest_entries()

    for taz_path in taz_files:
        file_hash = hashlib.sha256(taz_path.read_bytes()).hexdigest()
        if manifest.get(str(taz_path)) == file_hash:
            logger.info("  %-32s -> already ingested, skipping", taz_path.name)
            continue
        try:
            stats = ingest_taz_file(taz_path)
            logger.info(
                "  %-32s -> %d questions upserted, %d new ministries",
                stats.taz_file,
                stats.questions_inserted,
                stats.ministeres_created,
            )
            total_questions += stats.questions_inserted
            total_ministeres += stats.ministeres_created
            db.upsert_manifest(str(taz_path), file_hash)
        except Exception as exc:
            logger.error("Error processing %s: %s", taz_path.name, exc)
            errors.append(taz_path.name)

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
