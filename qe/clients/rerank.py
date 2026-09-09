"""Albert reranking API client."""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import requests

logger = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
    """Read an int env var, clamped to >= 1 so a bad value (0, negative)
    can't turn `range(0, n, BATCH_SIZE)` into an infinite loop or a
    silently-empty one instead of a loud misconfiguration."""
    return max(1, int(os.environ.get(name, str(default))))


def _positive_float_env(name: str, default: float) -> float:
    return max(0.001, float(os.environ.get(name, str(default))))


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
    BATCH_SIZE = _positive_int_env("ALBERT_RERANK_BATCH_SIZE", 64)

    # Wall-clock budget across ALL batches of one rerank() call, comfortably
    # above the documented worst case (the eval's 2 000-document pool ->
    # 32 batches). Each batch's own socket timeout is derived from whatever
    # of this budget remains (see _rerank_batch call below), so a single
    # slow batch can't by itself blow past this deadline.
    TOTAL_TIMEOUT = _positive_float_env("ALBERT_RERANK_TOTAL_TIMEOUT", 300.0)

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
        last_error: requests.RequestException | None = None
        deadline = time.monotonic() + self.TOTAL_TIMEOUT
        for start in range(0, len(docs), self.BATCH_SIZE):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
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
                batch_results = self._rerank_batch(
                    query, batch, top_n=len(batch), timeout=min(60.0, remaining)
                )
            except requests.RequestException as exc:
                last_error = exc
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
        if not merged and last_error is not None:
            # Every batch failed: raise rather than pass a plausible-looking
            # empty ranking to callers built around rerank() raising on
            # total failure (eval_realistic_encours.py's cosine-order
            # fallback, api/questions.py's 5xx) — see qe/assignment.py's
            # rerank_candidates(), the only caller of this method.
            raise last_error
        merged.sort(key=_score, reverse=True)
        return merged[:top_n]

    def _rerank_batch(
        self, query: str, documents: list[str], top_n: int, timeout: float = 60.0
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
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data") or data.get("results") or []


def _score(item: dict) -> float:
    score = item.get("relevance_score")
    if score is None:
        score = item.get("score")
    return float(score) if score is not None else 0.0
