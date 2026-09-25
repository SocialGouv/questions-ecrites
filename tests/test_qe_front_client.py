"""Tests for QeFrontClient and the precompute job's exit contract."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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


def _report(**overrides):
    base = {
        "processed": 1,
        "judged": 5,
        "errors": 0,
        "throttled": 0,
        "pending_left": 10,
        "done": False,
    }
    return _FakeResponse(200, {**base, **overrides})


def _clock_from(ticks):
    ticks = iter(ticks)
    return lambda: next(ticks)


def _until_done(client, clock=None, max_duration=3600):
    kwargs = {"_clock": clock} if clock else {}
    return client.precompute_until_done(
        limit=5, budget_seconds=60, max_duration_seconds=max_duration, **kwargs
    )


def test_one_call_passes_limit_and_budget_and_returns_the_body():
    client = _ScriptedClient([_report()])
    assert client.precompute(limit=5, budget_seconds=60)["judged"] == 5
    assert client.calls == [{"limit": 5, "budgetSeconds": 60}]


def test_one_call_swallows_http_transport_and_body_failures():
    client = _ScriptedClient(
        [
            _FakeResponse(403),
            requests.ConnectionError("refused"),
            _FakeResponse(200, ["x"]),
        ]
    )
    assert [client.precompute(limit=5, budget_seconds=60) for _ in range(3)] == [
        None,
        None,
        None,
    ]


def test_loops_until_done_and_sums_counters():
    client = _ScriptedClient([_report(), _report(), _report(done=True, pending_left=0)])
    totals = _until_done(client)
    assert totals["calls"] == 3
    assert totals["judged"] == 15
    assert totals["done"] is True
    assert totals["pending_left"] == 0


def test_stops_when_a_call_makes_no_progress():
    client = _ScriptedClient([_report(), _report(processed=0, throttled=3), _report()])
    totals = _until_done(client)
    assert totals["calls"] == 2
    assert totals["done"] is False
    assert totals["throttled"] == 3


def test_stops_and_flags_a_failed_call():
    client = _ScriptedClient([_report(), _FakeResponse(403)])
    totals = _until_done(client)
    assert totals["failed"] is True
    assert totals["calls"] == 1


def test_stops_at_the_time_budget_with_work_left():
    client = _ScriptedClient([_report(), _report(), _report()])
    totals = _until_done(client, clock=_clock_from([0, 0, 50, 150]), max_duration=100)
    assert totals["calls"] == 2
    assert totals["done"] is False


def _load_job():
    path = Path(__file__).resolve().parents[1] / "scripts" / "precompute_similar.py"
    spec = importlib.util.spec_from_file_location("precompute_similar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARGS = argparse.Namespace(limit=5, budget=60, max_duration=3600)


def test_job_fails_when_qe_front_refuses_the_call():
    assert _load_job().run(_ScriptedClient([_FakeResponse(403)]), _ARGS) == 1


def test_job_fails_when_stuck_on_errors():
    assert (
        _load_job().run(_ScriptedClient([_report(processed=0, errors=4)]), _ARGS) == 1
    )


def test_job_succeeds_when_done_or_out_of_time_with_progress():
    job = _load_job()
    assert job.run(_ScriptedClient([_report(done=True, pending_left=0)]), _ARGS) == 0
    assert job.run(_ScriptedClient([_report(processed=0)]), _ARGS) == 0


def test_job_exits_2_when_not_configured(monkeypatch):
    job = _load_job()

    class _Unconfigured:
        qe_front_base_url = ""
        qe_front_internal_token = ""

    monkeypatch.setattr(job, "get_settings", lambda: _Unconfigured())
    assert job.main() == 2
