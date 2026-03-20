#!/usr/bin/env python3
"""Poll the DILA Réponse WS for new questions and state changes.

Reads WS credentials from environment variables and runs three polling loops:

1. rechercherDossier    — new / answered questions in a sliding date window
2. chercherChangementDEtatQuestions — jeton-based state-change queue
3. chercherAttributionsDate         — jeton-based attribution queue

Meant to be called by a cron job every 30 minutes::

    */30 * * * * poetry run python scripts/poll_questions.py >> /var/log/qe_poll.log 2>&1

Environment variables (all required unless a default is shown):
    WS_BASE_URL       Base URL of the Réponse WS, e.g. https://reponses.dila.gouv.fr/ws
    WS_USERNAME       HTTP Basic Auth username
    WS_PASSWORD       HTTP Basic Auth password
    WS_LOOKBACK_DAYS  (optional, default 7) Lookback window for rechercherDossier

Usage:
    # Ingest all questions (no ministry filter)
    poetry run python scripts/poll_questions.py

    # Restrict to a specific ministry (substring match on titre_jo)
    poetry run python scripts/poll_questions.py --ministry "cohésion sociale"

    # Dry-run: connect and test, but do not write to the DB
    poetry run python scripts/poll_questions.py --dry-run

    # Skip specific loops
    poetry run python scripts/poll_questions.py --skip-dossier
    poetry run python scripts/poll_questions.py --skip-state-changes
    poetry run python scripts/poll_questions.py --skip-attributions

    # Filter by source
    poetry run python scripts/poll_questions.py --source AN
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from qe.clients.reponse_ws import ReponseWSClient, WSError
from qe.ingestion_ws_polling import run_full_poll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Return the value of an environment variable or exit with a helpful message."""
    val = os.getenv(name)
    if not val:
        logger.error(
            "Missing required environment variable: %s.  "
            "Set it before running this script.",
            name,
        )
        sys.exit(1)
    return val


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll the DILA Réponse WS for new questions and state changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test connectivity only; do not write anything to the database.",
    )
    parser.add_argument(
        "--skip-dossier",
        action="store_true",
        help="Skip the rechercherDossier loop (new / answered questions).",
    )
    parser.add_argument(
        "--skip-state-changes",
        action="store_true",
        help="Skip the chercherChangementDEtatQuestions loop.",
    )
    parser.add_argument(
        "--skip-attributions",
        action="store_true",
        help="Skip the chercherAttributionsDate loop.",
    )
    parser.add_argument(
        "--ministry",
        default=None,
        metavar="TEXT",
        help=(
            "Case-insensitive substring filter on the ministry label "
            "(titre_jo).  Only questions attributed to a matching ministry "
            "are ingested.  Omit to ingest all ministries."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["AN", "SENAT"],
        action="append",
        dest="sources",
        metavar="SOURCE",
        help=(
            'Restrict the dossier loop to this source ("AN" or "SENAT").  '
            "Can be repeated.  Defaults to both when omitted."
        ),
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        metavar="DAYS",
        help=(
            "Number of past days to include in the rechercherDossier window.  "
            "Defaults to the WS_LOOKBACK_DAYS env var (fallback: 7)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_url = _require_env("WS_BASE_URL")
    username = _require_env("WS_USERNAME")
    password = _require_env("WS_PASSWORD")

    lookback_days = args.lookback
    if lookback_days is None:
        lookback_days = int(os.getenv("WS_LOOKBACK_DAYS", "7"))

    client = ReponseWSClient(base_url=base_url, username=username, password=password)

    # --- connectivity test ---
    logger.info("Testing WS connectivity at %s …", base_url)
    try:
        reachability = client.test_connectivity()
    except Exception as exc:
        logger.error("Connectivity test failed: %s", exc)
        sys.exit(1)

    for service, ok in reachability.items():
        status = "OK" if ok else "UNREACHABLE"
        logger.info("  %s/test → %s", service, status)

    if not all(reachability.values()):
        logger.error("One or more WS endpoints are unreachable — aborting.")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry-run mode: connectivity test passed, no DB writes.")
        return

    # --- run the polling loops ---
    try:
        stats = run_full_poll(
            client,
            lookback_days=lookback_days,
            ministry_filter=args.ministry or None,
            sources=args.sources or None,
            skip_dossier=args.skip_dossier,
            skip_state_changes=args.skip_state_changes,
            skip_attributions=args.skip_attributions,
        )
    except WSError as exc:
        logger.error("WS returned an error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during polling: %s", exc)
        sys.exit(1)

    logger.info(
        "Poll complete — questions upserted: %d, skipped (out of scope): %d, "
        "state changes: %d, attributions: %d, errors: %d",
        stats.questions_upserted,
        stats.questions_skipped,
        stats.state_changes_processed,
        stats.attributions_processed,
        stats.errors,
    )

    if stats.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
