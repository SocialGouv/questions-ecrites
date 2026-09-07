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
