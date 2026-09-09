"""Tests for RerankClient's batching, partial-failure, and score-fallback behaviour.

Albert caps documents per call, so `rerank()` splits large candidate pools
into batches and merges results by remapping each batch's local `index`
back into the global list. That remapping is exactly the kind of change
that fails silently (a wrong offset attaches one candidate's score to
another, producing a plausible but wrong ranking). Albert responses can
also carry a null `relevance_score` or omit `index` altogether --
`qe/assignment.py` already guards both when reading `rerank()`'s output,
so `rerank()` itself must not crash on them first.
"""

from __future__ import annotations

import pytest
import requests

from qe.clients.rerank import RerankClient, _positive_float_env, _positive_int_env


class _ScriptedClient(RerankClient):
    """Overrides the transport seam with one scripted response/exception per batch call."""

    BATCH_SIZE = 2  # small, deterministic batch boundaries for tests

    def __init__(self, script):
        super().__init__(base_url="http://test", model="m", api_key="k")
        self._script = list(script)
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []

    def _rerank_batch(self, query, documents, top_n, timeout=60.0):
        self.calls.append(list(documents))
        self.timeouts.append(timeout)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _item(index, relevance_score=None, score=None):
    item: dict = {"index": index}
    if relevance_score is not None:
        item["relevance_score"] = relevance_score
    if score is not None:
        item["score"] = score
    return item


def test_merges_and_sorts_across_batches():
    # 5 documents, BATCH_SIZE=2 -> batches [0:2), [2:4), [4:5)
    client = _ScriptedClient(
        [
            [_item(0, relevance_score=0.2), _item(1, relevance_score=0.9)],
            [_item(0, relevance_score=0.5), _item(1, relevance_score=0.1)],
            [_item(0, relevance_score=0.7)],
        ]
    )
    result = client.rerank("q", ["a", "b", "c", "d", "e"], top_n=5)
    assert [r["index"] for r in result] == [1, 4, 2, 0, 3]
    assert client.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_global_index_correct_at_batch_boundary():
    client = _ScriptedClient(
        [
            [_item(0, relevance_score=1.0), _item(1, relevance_score=1.0)],
            [_item(0, relevance_score=1.0), _item(1, relevance_score=1.0)],
            [_item(0, relevance_score=1.0)],
        ]
    )
    result = client.rerank("q", list("abcde"), top_n=5)
    assert sorted(r["index"] for r in result) == [0, 1, 2, 3, 4]


def test_skips_items_with_missing_index():
    client = _ScriptedClient(
        [[{"relevance_score": 0.9}, _item(1, relevance_score=0.1)]]
    )
    result = client.rerank("q", ["a", "b"], top_n=2)
    assert [r["index"] for r in result] == [1]


def test_null_relevance_score_falls_back_to_score():
    client = _ScriptedClient([[{"index": 0, "relevance_score": None, "score": 0.7}]])
    result = client.rerank("q", ["a"], top_n=1)
    assert result[0]["index"] == 0


def test_missing_both_scores_defaults_to_zero_without_raising():
    client = _ScriptedClient([[{"index": 0}]])
    result = client.rerank("q", ["a"], top_n=1)
    assert result == [{"index": 0}]


def test_failed_batch_keeps_results_from_other_batches():
    # 5 documents, BATCH_SIZE=2 -> the middle batch [2:4) fails outright
    client = _ScriptedClient(
        [
            [_item(0, relevance_score=0.9)],
            requests.ConnectionError("boom"),
            [_item(0, relevance_score=0.5)],
        ]
    )
    result = client.rerank("q", ["a", "b", "c", "d", "e"], top_n=5)
    assert sorted(r["index"] for r in result) == [0, 4]


def test_overall_deadline_stops_remaining_batches(monkeypatch):
    ticks = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr("qe.clients.rerank.time.monotonic", lambda: next(ticks))
    client = _ScriptedClient([[_item(0, relevance_score=0.9)]])
    client.TOTAL_TIMEOUT = 1.0
    result = client.rerank("q", list("abcd"), top_n=4)
    assert client.calls == [["a", "b"]]
    assert [r["index"] for r in result] == [0]


def test_all_batches_failing_reraises_instead_of_returning_empty():
    # Callers (eval_realistic_encours.py's cosine-order fallback,
    # api/questions.py's 5xx) are built around rerank() raising when it
    # cannot produce a ranking at all -- a silent [] would look like "no
    # relevant candidates" instead of "the reranker is unreachable".
    client = _ScriptedClient(
        [requests.ConnectionError("boom"), requests.ConnectionError("boom again")]
    )
    with pytest.raises(requests.ConnectionError):
        client.rerank("q", ["a", "b", "c"], top_n=3)


def test_partial_failure_does_not_raise_even_though_some_batches_failed():
    client = _ScriptedClient(
        [[_item(0, relevance_score=0.9)], requests.ConnectionError("boom")]
    )
    result = client.rerank("q", ["a", "b", "c"], top_n=3)
    assert [r["index"] for r in result] == [0]


def test_per_batch_timeout_is_capped_by_remaining_budget(monkeypatch):
    # deadline computed once, then one remaining-budget check per batch
    ticks = iter([0.0, 0.0, 45.0])
    monkeypatch.setattr("qe.clients.rerank.time.monotonic", lambda: next(ticks))
    client = _ScriptedClient(
        [[_item(0, relevance_score=0.9)], [_item(0, relevance_score=0.5)]]
    )
    client.TOTAL_TIMEOUT = 50.0
    client.rerank("q", ["a", "b", "c", "d"], top_n=4)
    # First batch: full 50s budget remaining, capped at the 60s socket ceiling -> 50.
    # Second batch: only 5s left (50 - 45) -> capped at 5, not the 60s ceiling.
    assert client.timeouts == [50.0, 5.0]


def test_batch_size_env_var_clamped_to_at_least_one(monkeypatch):
    monkeypatch.setenv("X_TEST_BATCH_SIZE", "-1")
    assert _positive_int_env("X_TEST_BATCH_SIZE", 64) == 1
    monkeypatch.setenv("X_TEST_BATCH_SIZE", "0")
    assert _positive_int_env("X_TEST_BATCH_SIZE", 64) == 1
    monkeypatch.setenv("X_TEST_BATCH_SIZE", "8")
    assert _positive_int_env("X_TEST_BATCH_SIZE", 64) == 8


def test_total_timeout_env_var_clamped_to_positive(monkeypatch):
    monkeypatch.setenv("X_TEST_TIMEOUT", "-5")
    assert _positive_float_env("X_TEST_TIMEOUT", 300.0) > 0
    monkeypatch.setenv("X_TEST_TIMEOUT", "0")
    assert _positive_float_env("X_TEST_TIMEOUT", 300.0) > 0
