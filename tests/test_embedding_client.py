"""Tests for the embedding client's guardrail-resilience fallback.

The platform's content guardrails (e.g. EU-AI-Act filters) can reject a
batch with HTTP 403 + a "Content blocked" body. `embed_batch_partial`
must isolate the blocked text(s) by retrying one by one, while plain
auth failures must keep raising immediately (no 272k-item retry storm
on a revoked key).
"""

from __future__ import annotations

import json

import pytest
import requests

from qe.clients.embedding import ContentBlockedError, EmbeddingClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def _ok_response(n_texts: int) -> _FakeResponse:
    return _FakeResponse(
        200,
        {"data": [{"index": i, "embedding": [float(i)]} for i in range(n_texts)]},
    )


def _blocked() -> _FakeResponse:
    """Fresh instance per use — no state shared between tests."""
    return _FakeResponse(
        403,
        text='{"error":{"message":"Content blocked: eu_ai_act_art5 match","code":"403"}}',
    )


def _auth_denied() -> _FakeResponse:
    return _FakeResponse(403, text='{"detail":"Forbidden"}')


class _ScriptedClient(EmbeddingClient):
    """Overrides the transport seam with a scripted response per call."""

    def __init__(self, script):
        super().__init__(url="http://test", model="m", api_key="k")
        self._script = list(script)
        self.calls: list[list[str]] = []

    def _post(self, texts):
        self.calls.append(list(texts))
        action = self._script.pop(0)
        return action(texts) if callable(action) else action


def test_partial_happy_path_is_a_single_batch_call():
    client = _ScriptedClient([lambda texts: _ok_response(len(texts))])
    result = client.embed_batch_partial(["a", "b", "c"])
    assert result == [[0.0], [1.0], [2.0]]
    assert len(client.calls) == 1


def test_partial_isolates_the_blocked_text():
    client = _ScriptedClient(
        [
            _blocked(),  # whole batch rejected
            lambda texts: _ok_response(1),  # "a" alone → OK
            _blocked(),  # "b" alone → blocked
            lambda texts: _ok_response(1),  # "c" alone → OK
        ]
    )
    result = client.embed_batch_partial(["a", "b", "c"])
    assert result == [[0.0], None, [0.0]]
    assert client.calls == [["a", "b", "c"], ["a"], ["b"], ["c"]]


def test_auth_403_raises_without_per_item_fallback():
    client = _ScriptedClient([_auth_denied()])
    with pytest.raises(requests.HTTPError):
        client.embed_batch_partial(["a", "b"])
    assert len(client.calls) == 1  # no retry storm on a revoked key


def test_server_error_raises_without_fallback():
    client = _ScriptedClient([_FakeResponse(500, text="oops")])
    with pytest.raises(requests.HTTPError):
        client.embed_batch_partial(["a"])
    assert len(client.calls) == 1


def test_embed_batch_raises_typed_error_on_content_block():
    client = _ScriptedClient([_blocked()])
    with pytest.raises(ContentBlockedError):
        client.embed_batch(["a"])
