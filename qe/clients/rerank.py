"""Albert reranking API client."""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import requests

logger = logging.getLogger(__name__)


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

    # Wall-clock budget across ALL batches of one rerank() call. Each batch
    # already has its own socket timeout, but a large candidate pool (e.g.
    # the eval's 2 000-document pool -> 32 batches) would otherwise have no
    # bound on total latency.
    TOTAL_TIMEOUT = float(os.environ.get("ALBERT_RERANK_TOTAL_TIMEOUT", "60"))

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
        deadline = time.monotonic() + self.TOTAL_TIMEOUT
        for start in range(0, len(docs), self.BATCH_SIZE):
            if time.monotonic() >= deadline:
                logger.warning(
                    "Rerank overall deadline (%.0fs) reached after %d/%d document(s); "
                    "returning the ranking from completed batches only.",
                    self.TOTAL_TIMEOUT,
                    start,
                    len(docs),
                )
                break
            batch = docs[start : start + self.BATCH_SIZE]
            try:
                batch_results = self._rerank_batch(query, batch, top_n=len(batch))
            except requests.RequestException:
                logger.warning(
                    "Rerank batch [%d:%d] failed; keeping %d already-scored "
                    "document(s) from other batches.",
                    start,
                    start + len(batch),
                    len(merged),
                    exc_info=True,
                )
                continue
            for item in batch_results:
                idx = item.get("index")
                if idx is None:
                    continue
                merged.append({**item, "index": start + int(idx)})
        merged.sort(key=_score, reverse=True)
        return merged[:top_n]

    def _rerank_batch(self, query: str, documents: list[str], top_n: int) -> list[dict]:
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
    score = item.get("relevance_score")
    if score is None:
        score = item.get("score")
    return float(score) if score is not None else 0.0
