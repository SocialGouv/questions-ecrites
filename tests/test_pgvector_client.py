"""Tests for PgvectorClient's pgvector-version guard around hnsw.iterative_scan.

pgvector reserves the `hnsw` GUC prefix, so `SET LOCAL hnsw.iterative_scan`
raises "unrecognized configuration parameter" (aborting the transaction) on
an extension older than 0.8 -- some environments are still on 0.6.0 (see
migration 087d1c73ddbc). `_pgvector_supports_iterative_scan` probes the
installed extension version once and caches it, so `search()` never issues
that SET LOCAL against a version that doesn't support it.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import qe.clients.pgvector_client as pgvector_client


class _FakeResult:
    def __init__(self, value=None, rows=None):
        self._value = value
        self._rows = rows if rows is not None else []

    def scalar(self):
        return self._value

    def all(self):
        return self._rows


class _FakeSession:
    """Fake session distinguishing the version probe / SET LOCAL / actual
    search statement by their rendered SQL, so `search()` can run end to
    end without a real database."""

    def __init__(self, version: str | None):
        self.version = version
        self.executed: list[str] = []

    def execute(self, stmt):
        sql = str(stmt)
        self.executed.append(sql)
        if "extversion" in sql:
            return _FakeResult(value=self.version)
        return _FakeResult()


@pytest.fixture(autouse=True)
def _reset_cache():
    pgvector_client._supports_iterative_scan = None
    yield
    pgvector_client._supports_iterative_scan = None


def test_supports_iterative_scan_on_0_8_2():
    session = _FakeSession("0.8.2")
    assert pgvector_client._pgvector_supports_iterative_scan(session) is True


def test_does_not_support_iterative_scan_on_0_6_0():
    session = _FakeSession("0.6.0")
    assert pgvector_client._pgvector_supports_iterative_scan(session) is False


def test_exactly_0_8_0_supports_iterative_scan():
    session = _FakeSession("0.8.0")
    assert pgvector_client._pgvector_supports_iterative_scan(session) is True


def test_result_is_cached_after_first_probe():
    session = _FakeSession("0.6.0")
    assert pgvector_client._pgvector_supports_iterative_scan(session) is False
    # Second call must not re-query, even against a session that would now say otherwise.
    session2 = _FakeSession("0.8.2")
    assert pgvector_client._pgvector_supports_iterative_scan(session2) is False
    assert len(session2.executed) == 0


@contextmanager
def _fake_get_session(session):
    yield session


def test_search_skips_set_local_on_pre_0_8_extension(monkeypatch):
    session = _FakeSession("0.6.0")
    monkeypatch.setattr(
        pgvector_client.db, "get_session", lambda: _fake_get_session(session)
    )
    pgvector_client.PgvectorClient().search("questions_opendata", [0.1, 0.2], top_k=10)
    assert not any("SET LOCAL" in sql for sql in session.executed)


def test_search_sets_ef_search_and_iterative_scan_on_supported_extension(monkeypatch):
    session = _FakeSession("0.8.2")
    monkeypatch.setattr(
        pgvector_client.db, "get_session", lambda: _fake_get_session(session)
    )
    pgvector_client.PgvectorClient().search("questions_opendata", [0.1, 0.2], top_k=10)
    assert any("hnsw.iterative_scan" in sql for sql in session.executed)
    assert any("hnsw.ef_search" in sql for sql in session.executed)
