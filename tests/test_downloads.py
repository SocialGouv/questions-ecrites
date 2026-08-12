from pathlib import Path

import requests

from qe.downloads import download_with_retries


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        yield from self._chunks

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        pass


class _FailThenSucceedSession:
    """Fails on the first *fail_count* calls, then succeeds."""

    def __init__(self, fail_count: int, chunks: list[bytes] | None = None) -> None:
        self.fail_count = fail_count
        self.chunks = chunks if chunks is not None else [b"hello "]
        self.calls = 0

    def get(self, url: str, stream: bool, timeout: int) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise requests.ConnectionError("Connection broken: IncompleteRead")
        return _FakeResponse(self.chunks)


class _AlwaysFailSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, stream: bool, timeout: int) -> _FakeResponse:
        self.calls += 1
        raise requests.ConnectionError("Connection broken: IncompleteRead")


def test_succeeds_on_first_attempt(tmp_path: Path) -> None:
    session = _FailThenSucceedSession(fail_count=0)
    dest = tmp_path / "archive.zip"

    ok = download_with_retries("https://example.test/a.zip", dest, session, backoff=0)

    assert ok is True
    assert session.calls == 1
    assert dest.read_bytes() == b"hello "


def test_succeeds_after_transient_failures(tmp_path: Path) -> None:
    session = _FailThenSucceedSession(fail_count=2)
    dest = tmp_path / "archive.zip"

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=3, backoff=0
    )

    assert ok is True
    assert session.calls == 3
    assert dest.read_bytes() == b"hello "


def test_gives_up_after_exhausting_retries(tmp_path: Path) -> None:
    session = _AlwaysFailSession()
    dest = tmp_path / "archive.zip"

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=3, backoff=0
    )

    assert ok is False
    assert session.calls == 3
    assert not dest.exists()
    assert not dest.with_suffix(".tmp").exists()
