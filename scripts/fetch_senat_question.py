#!/usr/bin/env python3
"""Fetch individual Sénat questions by ID from the Sénat website.

Use this when a question is missing from the bulk SQL dump (the dump is updated
periodically, so recently published questions may not be in it yet).

Unlike the AN (which hosts individual XML files), the Sénat only publishes a
full PostgreSQL dump.  This script falls back to scraping the Sénat's public
HTML question page and mapping the fields to the shared ParsedQuestion format.

Page URL pattern:
    https://www.senat.fr/questions/base/{legislature}/q{legislature}{numero:05d}.html

    Example for SENAT-17-QE-1234:
    https://www.senat.fr/questions/base/17/q1701234.html

Usage:
    poetry run python scripts/fetch_senat_question.py SENAT-17-QE-1234
    poetry run python scripts/fetch_senat_question.py SENAT-17-QE-1234 SENAT-17-QE-1240
    poetry run python scripts/fetch_senat_question.py SENAT-17-QE-1234 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

import requests

from qe.ingestion_an import ParsedQuestion, ingest_questions
from qe.question_fetch import fetch_senat_question as fetch_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch individual Sénat questions by ID and upsert into PostgreSQL.",
    )
    parser.add_argument(
        "ids",
        nargs="+",
        metavar="QUESTION_ID",
        help="One or more question IDs in SENAT-NN-QE-NNNNN format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse only; do not write to the database",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    http = requests.Session()
    http.headers["User-Agent"] = "qe-ingestion/1.0"

    questions: list[ParsedQuestion] = []
    errors: list[str] = []

    for qid in args.ids:
        try:
            pq = fetch_question(qid, http)
        except ValueError as exc:
            logger.error("%s", exc)
            errors.append(qid)
            continue
        if pq is None:
            errors.append(qid)
        else:
            questions.append(pq)
            logger.info(
                "  parsed: %s  etat=%s  titre=%s",
                pq.id,
                pq.etat_question,
                (pq.titre_senat or "")[:60],
            )

    if args.dry_run:
        logger.info("[dry-run] Would upsert %d question(s)", len(questions))
    elif questions:
        stats = ingest_questions(questions, ingest_source="ws_polling")
        logger.info(
            "Done: %d upserted (%d inserted, %d updated), %d new ministries",
            stats.questions_parsed,
            stats.questions_inserted,
            stats.questions_updated,
            stats.ministeres_created,
        )

    if errors:
        logger.warning("Failed IDs (%d): %s", len(errors), ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
