"""Retry helper for long benchmark runs over a fragile connection.

Benchmarks against preprod go through ``kubectl port-forward``, which
drops under sustained load — a single tunnel reset would otherwise throw
away an hour of work. Retries only ``OperationalError`` (the connection
died); a programming or data error still fails immediately, because
retrying it would just produce the same wrong answer more slowly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import OperationalError

from qe import db

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_reconnect(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
) -> T:
    """Run ``fn``, re-establishing the pool if the connection drops.

    ``dispose()`` matters: after a tunnel reset the pool still holds
    sockets to the old one, and ``pool_pre_ping`` only catches those
    before a query starts, not a connection cut mid-query.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except OperationalError as error:
            last = error
            db.get_engine().dispose()
            if attempt == attempts:
                break
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "connection lost (attempt %d/%d), retrying in %.0fs",
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"connection kept failing after {attempts} attempts") from last
