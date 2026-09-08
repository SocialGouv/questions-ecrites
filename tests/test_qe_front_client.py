"""Tests for QeFrontClient's fail-open contract.

The precompute call is best-effort (docs/llm-judge-caching-plan.md Phase 2):
a transport/HTTP failure must be swallowed and logged, never raised, since
the ingestion run that triggers it must succeed regardless of whether
qe-front's cache warmed up.
"""

from __future__ import annotations

import requests

from qe.clients.qe_front_client import QeFrontClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: object = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _ScriptedClient(QeFrontClient):
    """Overrides the transport seam with a scripted response/exception."""

    def __init__(self, script):
        super().__init__(base_url="http://test", token="secret")  # noqa: S106 -- test double, not a real credential
        self._script = list(script)
        self.calls: list[dict] = []

    def _post(self, params):
        self.calls.append(params)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def test_returns_the_parsed_body_on_success():
    client = _ScriptedClient([_FakeResponse(200, {"processed": 3, "cached": 12})])
    result = client.precompute_similar_cache(limit=50)
    assert result == {"processed": 3, "cached": 12}
    assert client.calls == [{"limit": 50}]


def test_returns_none_on_http_error_without_raising():
    client = _ScriptedClient([_FakeResponse(403)])
    result = client.precompute_similar_cache(limit=50)
    assert result is None


def test_returns_none_on_transport_error_without_raising():
    client = _ScriptedClient([requests.ConnectionError("connection refused")])
    result = client.precompute_similar_cache(limit=50)
    assert result is None


def test_returns_none_when_the_response_body_is_not_a_json_object():
    # A 200 whose body is a JSON array/string/number is not a RequestException
    # and would otherwise pass through as-is, breaking the caller's `.get()`
    # calls (embed_questions.py:505) with an AttributeError the fail-open
    # contract is supposed to prevent.
    client = _ScriptedClient([_FakeResponse(200, ["not", "a", "dict"])])
    result = client.precompute_similar_cache(limit=50)
    assert result is None


def _clock_from(ticks):
    """A fake monotonic clock that advances one value per call."""
    ticks = iter(ticks)
    return lambda: next(ticks)


def test_batches_drains_the_backlog_across_multiple_calls():
    # Three batches: two full pages, then an empty one signalling "done" —
    # the backlog is smaller than batch_limit * 2 but bigger than one batch.
    client = _ScriptedClient(
        [
            _FakeResponse(200, {"processed": 50, "cached": 40, "reciprocal": 5, "pruned": 1, "errors": 0}),
            _FakeResponse(200, {"processed": 50, "cached": 30, "reciprocal": 2, "pruned": 0, "errors": 1}),
            _FakeResponse(200, {"processed": 0, "cached": 0, "reciprocal": 0, "pruned": 0, "errors": 0}),
        ]
    )
    result = client.precompute_similar_cache_batches(
        batch_limit=50, max_duration_seconds=1000, _clock=_clock_from([0, 0, 0, 0])
    )
    assert result == {"batches": 3, "processed": 100, "cached": 70, "reciprocal": 7, "pruned": 1, "errors": 1}
    assert len(client.calls) == 3


def test_batches_stops_once_the_time_budget_elapses():
    # Every batch is still full (processed == limit), i.e. the backlog is
    # NOT drained — only the time budget makes the loop stop.
    client = _ScriptedClient(
        [
            _FakeResponse(200, {"processed": 50, "cached": 50, "reciprocal": 0, "pruned": 0, "errors": 0}),
            _FakeResponse(200, {"processed": 50, "cached": 50, "reciprocal": 0, "pruned": 0, "errors": 0}),
        ]
    )
    # Clock ticks: 0 (start), 0 (< 10, loop), 5 (< 10, loop), 15 (>= 10, stop).
    result = client.precompute_similar_cache_batches(
        batch_limit=50, max_duration_seconds=10, _clock=_clock_from([0, 0, 5, 15])
    )
    assert result["batches"] == 2
    assert result["processed"] == 100
    assert len(client.calls) == 2


def test_batches_stops_immediately_on_transport_failure_without_looping_forever():
    client = _ScriptedClient([requests.ConnectionError("connection refused")])
    result = client.precompute_similar_cache_batches(
        batch_limit=50, max_duration_seconds=1000, _clock=_clock_from([0, 0])
    )
    assert result == {"batches": 0, "processed": 0, "cached": 0, "reciprocal": 0, "pruned": 0, "errors": 0}
    assert len(client.calls) == 1
