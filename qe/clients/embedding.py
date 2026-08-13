"""Albert API embeddings client."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

# Markers that identify a guardrail rejection (e.g. the platform's
# EU-AI-Act content filters) in a 403 response body. Matched
# case-insensitively, and deliberately specific ("content blocked" is
# the platform's error message prefix, "guardrail_name" its JSON field)
# — a generic word like "guardrail" alone could match an unrelated 403
# body and silently convert an outage into mass per-item skips. Auth
# 403s carry none of these and must keep raising immediately.
_CONTENT_BLOCK_MARKERS = ("content blocked", "guardrail_name")


def _is_content_block(response: requests.Response) -> bool:
    if response.status_code != 403:
        return False
    body = response.text[:2000].lower()
    return any(marker in body for marker in _CONTENT_BLOCK_MARKERS)


class ContentBlockedError(Exception):
    """A text was rejected by the platform's content guardrails."""


class EmbeddingClient:
    """Generate text embeddings via the Albert API."""

    def __init__(
        self, *, url: str, model: str, api_key: str, timeout: int = 60
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, texts: list[str]) -> requests.Response:
        """Single API round-trip. Seam for tests (subclass and override)."""
        return requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._post(texts)
        if not response.ok:
            logger.error(
                "Embedding API error %d for %d text(s): %s",
                response.status_code,
                len(texts),
                response.text[:500],
            )
            if _is_content_block(response):
                raise ContentBlockedError(response.text[:500])
            response.raise_for_status()
        data = response.json()
        return [
            item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
        ]

    def embed_batch_partial(self, texts: list[str]) -> list[list[float] | None]:
        """Like embed_batch, but survives guardrail rejections.

        When the whole batch is accepted, behaves exactly like
        embed_batch. When the batch is rejected by a content guardrail
        (403 with a content-block body), falls back to embedding each
        text individually; blocked texts yield None at their position so
        the result stays aligned 1:1 with the input.

        Auth failures and other HTTP errors raise as usual — a broken
        key must not degrade into one failing API call per text.
        """
        try:
            return list(self.embed_batch(texts))
        except ContentBlockedError:
            logger.warning(
                "Batch of %d text(s) rejected by content guardrail — "
                "retrying one by one to isolate the blocked text(s).",
                len(texts),
            )

        results: list[list[float] | None] = []
        blocked = 0
        for text in texts:
            try:
                results.append(self.embed_batch([text])[0])
            except ContentBlockedError:
                results.append(None)
                blocked += 1
        logger.warning(
            "Guardrail fallback done: %d/%d text(s) blocked and skipped.",
            blocked,
            len(texts),
        )
        return results
