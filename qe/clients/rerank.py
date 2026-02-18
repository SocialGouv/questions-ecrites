"""Albert reranking API client."""

from __future__ import annotations

from typing import Sequence

import requests


class RerankClient:
    """Rerank candidate documents against a query via the Albert API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
        }
        response = requests.post(
            f"{self.base_url}/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data") or data.get("results") or []
