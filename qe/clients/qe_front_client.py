"""HTTP client for qe-front's internal precompute route.

Phase 2 of docs/llm-judge-caching-plan.md (qe-front repo): after an
embedding run, the ingestion pipeline triggers qe-front to precompute
rerank+judge results for newly-embedded EN_COURS questions into
`question_similar_cache`, off qe-front's request path. Best-effort by
design — a failure here must never fail the ingestion run that
triggered it, since the live `/similar` route still computes on demand
without a warm cache (Phase 1).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import requests

logger = logging.getLogger(__name__)


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

    def precompute_similar_cache(self, limit: int) -> dict | None:
        """Trigger one precompute batch of up to `limit` questions.

        Returns the parsed response body, or None on any transport/HTTP
        failure (logged, never raised) — the caller treats this as
        "nothing precomputed this run", not as an ingestion failure.
        """
        try:
            response = self._post({"limit": limit})
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except requests.RequestException as exc:
            logger.warning("qe-front precompute call failed: %s", exc)
            return None

    def precompute_similar_cache_batches(
        self,
        *,
        batch_limit: int,
        max_duration_seconds: float,
        _clock: Callable[[], float] = time.monotonic,
    ) -> dict:
        """Drain the precompute backlog in batches of `batch_limit`.

        The route always selects the oldest still-missing questions first
        (a NOT EXISTS anti-join on question_similar_cache), so each batch
        makes forward progress and never repeats work a previous batch (or a
        previous run) already finished. That makes it safe to just keep
        calling: a backlog bigger than one batch — the very first run,
        backfilling every pre-existing EN_COURS question, or an ordinary day
        with hundreds of newly-embedded ones — gets fully cleared within this
        run if it fits the time budget, or picked up again by the next run
        otherwise. Nothing is lost or double-counted between runs.

        Stops when: the backlog is drained (a batch reports `processed ==
        0`), a batch made no cached progress (every attempted question
        errored — `errors >= processed` — so the anti-join would just
        re-select the same failing questions forever), a batch call fails
        outright (already logged by `precompute_similar_cache` — best-effort,
        try again next run), or `max_duration_seconds` elapses (a safety
        bound so one unusually large backlog can't consume the whole
        ingestion job's time budget).
        """
        started = _clock()
        totals = {"batches": 0, "processed": 0, "cached": 0, "reciprocal": 0, "pruned": 0, "errors": 0}
        while _clock() - started < max_duration_seconds:
            result = self.precompute_similar_cache(limit=batch_limit)
            if result is None:
                break
            totals["batches"] += 1
            for key in ("processed", "cached", "reciprocal", "pruned", "errors"):
                value = result.get(key, 0)
                totals[key] += value if isinstance(value, int) else 0
            processed = result.get("processed", 0)
            errors = result.get("errors", 0)
            if not isinstance(processed, int) or processed == 0:
                break
            if isinstance(errors, int) and errors >= processed:
                break
        else:
            logger.warning(
                "qe-front precompute: time budget (%.0fs) reached after %d batch(es); "
                "backlog may not be fully drained, continuing next run.",
                max_duration_seconds,
                totals["batches"],
            )
        return totals
