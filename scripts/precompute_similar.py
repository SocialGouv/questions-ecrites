"""Fill qe-front's `/similar` stage caches until done or out of time.

Run nightly after ingestion. Exits non-zero when it cannot run at all
(unconfigured, or qe-front refusing the call) or makes no progress while
errors occur, so a broken setup fails the job instead of passing silently.
"""

from __future__ import annotations

import argparse
import logging
import sys

from qe.clients.qe_front_client import QeFrontClient
from qe.config import get_settings

logger = logging.getLogger("precompute_similar")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-duration", type=float, default=3.5 * 3600, help="seconds, whole run"
    )
    parser.add_argument(
        "--budget", type=int, default=120, help="seconds, per qe-front call"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="sources per stage and call"
    )
    return parser.parse_args()


def run(client: QeFrontClient, args: argparse.Namespace) -> int:
    totals = client.precompute_until_done(
        limit=args.limit,
        budget_seconds=args.budget,
        max_duration_seconds=args.max_duration,
    )
    logger.info("qe-front precompute: %s", totals)
    if totals["failed"]:
        return 1
    if (
        not totals["done"]
        and totals["processed"] == 0
        and (totals["errors"] or totals["throttled"])
    ):
        return 1
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = get_settings()
    if not (settings.qe_front_base_url and settings.qe_front_internal_token):
        logger.error("QE_FRONT_BASE_URL and INTERNAL_API_TOKEN must both be set.")
        return 2
    client = QeFrontClient(
        base_url=settings.qe_front_base_url, token=settings.qe_front_internal_token
    )
    return run(client, _parse_args())


if __name__ == "__main__":
    sys.exit(main())
