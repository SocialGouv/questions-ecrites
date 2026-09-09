"""Albert reranking API client."""

from __future__ import annotations

import os
from typing import Sequence

import requests


class RerankClient:
    """Rerank candidate documents against a query via the Albert API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    # Albert caps the number of documents per call, not the payload size:
    # 100 texts of 500 chars are refused (413), 50 texts of 4 000 chars are
    # accepted. Same limit and same knob as qe-front's rerank client.
    # Scores are independent across batches, so the merged ranking equals
    # a single-call ranking.
    BATCH_SIZE = int(os.environ.get("ALBERT_RERANK_BATCH_SIZE", "64"))

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict]:
        if not documents:
            return []
        docs = list(documents)
        merged: list[dict] = []
        for start in range(0, len(docs), self.BATCH_SIZE):
            batch = docs[start : start + self.BATCH_SIZE]
            for item in self._rerank_batch(query, batch, top_n=len(batch)):
                merged.append({**item, "index": start + int(item["index"])})
        merged.sort(key=_score, reverse=True)
        return merged[:top_n]

    def _rerank_batch(
        self, query: str, documents: list[str], top_n: int
    ) -> list[dict]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
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


def _score(item: dict) -> float:
    return float(item.get("relevance_score", item.get("score", 0.0)))
