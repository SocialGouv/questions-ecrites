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
