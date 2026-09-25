"""HTTP client for qe-front's internal precompute route.

The route fills qe-front's `/similar` stage caches (neighbours, rerank
scores, judge verdicts) for a time budget per call and reports what is
left; `precompute_until_done` keeps calling it until nothing is.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import requests

logger = logging.getLogger(__name__)

COUNTERS = (
    "processed",
    "queue",
    "neighbors",
    "completed",
    "reranked",
    "judged",
    "throttled",
    "errors",
)


class QeFrontClient:
    """Calls qe-front's ``POST /api/internal/similar-cache/precompute``."""

    def __init__(self, base_url: str, token: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, params: dict) -> requests.Response:
        """Transport seam — overridden in tests instead of mocking `requests`."""
        return requests.post(
            f"{self.base_url}/api/internal/similar-cache/precompute",
            headers={"Authorization": f"Bearer {self.token}"},
            params=params,
            timeout=self.timeout,
        )

    def precompute(self, *, limit: int, budget_seconds: int) -> dict | None:
        """One precompute call. None on any transport/HTTP/body failure (logged)."""
        try:
            response = self._post({"limit": limit, "budgetSeconds": budget_seconds})
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except requests.RequestException as exc:
            logger.warning("qe-front precompute call failed: %s", exc)
            return None

    def precompute_until_done(
        self,
        *,
        limit: int,
        budget_seconds: int,
        max_duration_seconds: float,
        _clock: Callable[[], float] = time.monotonic,
    ) -> dict:
        """Call the route until it reports `done`, stops making progress,
        fails, or `max_duration_seconds` elapses.

        Returns the summed counters plus `calls`, the last `pending_left`,
        `done`, and `failed` (a call itself failed).
        """
        started = _clock()
        totals: dict = dict.fromkeys(COUNTERS, 0)
        totals.update(calls=0, pending_left=None, done=False, failed=False)
        while _clock() - started < max_duration_seconds:
            result = self.precompute(limit=limit, budget_seconds=budget_seconds)
            if result is None:
                totals["failed"] = True
                break
            totals["calls"] += 1
            for key in COUNTERS:
                value = result.get(key, 0)
                totals[key] += value if isinstance(value, int) else 0
            totals["pending_left"] = result.get("pending_left")
            if result.get("done") is True:
                totals["done"] = True
                break
            if result.get("processed", 0) == 0:
                # Same work would be selected again: whatever blocks it
                # (quota, Albert down) won't clear within this run.
                logger.warning(
                    "qe-front precompute made no progress (%s); stopping until next run.",
                    {
                        key: result.get(key)
                        for key in ("throttled", "errors", "pending_left")
                    },
                )
                break
        else:
            logger.info(
                "qe-front precompute: time budget (%.0fs) reached, %s left for the next run.",
                max_duration_seconds,
                totals["pending_left"],
            )
        return totals
