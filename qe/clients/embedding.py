"""Socle IA embeddings API client."""

from __future__ import annotations

import requests


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
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
