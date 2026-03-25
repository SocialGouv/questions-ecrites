"""Socle IA embeddings API client."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generate text embeddings via the Socle IA API."""

    def __init__(
        self, *, url: str, model: str, api_key: str, timeout: int = 60
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
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
        if not response.ok:
            logger.error(
                "Embedding API error %d for %d text(s): %s",
                response.status_code,
                len(texts),
                response.text[:500],
            )
            response.raise_for_status()
        data = response.json()
        return [
            item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
        ]
