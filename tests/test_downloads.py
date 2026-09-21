from pathlib import Path

import requests

from qe.downloads import download_with_retries


class _FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self.headers: dict[str, str] = headers or {}

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

    def get(
        self, url: str, stream: bool, timeout: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise requests.ConnectionError("Connection broken: IncompleteRead")
        return _FakeResponse(self.chunks)


class _AlwaysFailSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(
        self, url: str, stream: bool, timeout: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        self.calls += 1
        raise requests.ConnectionError("Connection broken: IncompleteRead")


class _CutThenResumeSession:
    """Serves the archive in two halves, dropping the connection in between.

    First call: sends the first half, then raises — exactly the portal's
    ``IncompleteRead``. Second call: expects a Range request and serves the
    remainder with 206, the way data.assemblee-nationale.fr does.
    """

    def __init__(self, body: bytes, cut_at: int, etag: str = '"abc123"') -> None:
        self.body = body
        self.cut_at = cut_at
        self.etag = etag
        self.calls = 0
        self.seen_headers: list[dict[str, str]] = []

    def get(
        self, url: str, stream: bool, timeout: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        self.calls += 1
        self.seen_headers.append(dict(headers or {}))
        if self.calls == 1:

            def _cut():
                yield self.body[: self.cut_at]
                raise requests.ConnectionError("Connection broken: IncompleteRead")

            resp = _FakeResponse(
                [],
                headers={"ETag": self.etag, "content-length": str(len(self.body))},
            )
            resp.iter_content = lambda chunk_size: _cut()  # type: ignore[method-assign]
            return resp
        start = int(headers["Range"].removeprefix("bytes=").rstrip("-"))
        return _FakeResponse(
            [self.body[start:]],
            status_code=206,
            headers={
                "ETag": self.etag,
                "content-length": str(len(self.body) - start),
            },
        )


class _RangeIgnoredSession:
    """Answers a Range request with a full 200 body, as If-Range does when the
    resource changed server-side."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def get(
        self, url: str, stream: bool, timeout: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(
            [self.body],
            status_code=200,
            headers={"ETag": '"new"', "content-length": str(len(self.body))},
        )


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


def test_resumes_from_partial_file_instead_of_restarting(tmp_path: Path) -> None:
    """The reason this helper exists: a 133 MB archive the portal keeps cutting.

    Restarting at byte 0 never finished it; continuing from the bytes already
    on disk does. The second call must ask for the remainder only, and the
    assembled file must equal the original.
    """
    body = bytes(range(256)) * 40  # 10 240 bytes
    session = _CutThenResumeSession(body, cut_at=4096)
    dest = tmp_path / "archive.zip"

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=2, backoff=0
    )

    assert ok is True
    assert dest.read_bytes() == body
    assert "Range" not in session.seen_headers[0]
    assert session.seen_headers[1]["Range"] == "bytes=4096-"
    assert session.seen_headers[1]["If-Range"] == '"abc123"'
    assert not dest.with_suffix(".tmp").exists()


def test_partial_file_survives_a_failed_run_for_the_next_process(
    tmp_path: Path,
) -> None:
    """The CronJob and the backfill Job both retry by re-running the script, so
    giving up must leave the partial file and its validator behind."""
    body = b"x" * 10_000
    session = _CutThenResumeSession(body, cut_at=6_000)
    dest = tmp_path / "archive.zip"

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=1, backoff=0
    )

    assert ok is False
    tmp = dest.with_suffix(".tmp")
    assert tmp.read_bytes() == body[:6_000]
    assert Path(f"{tmp}.meta").read_text() == '"abc123"'


def test_stale_partial_is_discarded_when_the_server_ignores_the_range(
    tmp_path: Path,
) -> None:
    """If-Range answered with 200 means the archive changed: splicing the old
    bytes onto the new ones would build a corrupt file."""
    body = b"fresh-archive-contents"
    dest = tmp_path / "archive.zip"
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(b"bytes-from-a-previous-version")
    Path(f"{tmp}.meta").write_text('"old"')
    session = _RangeIgnoredSession(body)

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=1, backoff=0
    )

    assert ok is True
    assert dest.read_bytes() == body


def test_partial_without_validator_is_not_resumed(tmp_path: Path) -> None:
    """No validator on disk means nothing proves the bytes are still current."""
    dest = tmp_path / "archive.zip"
    dest.with_suffix(".tmp").write_bytes(b"unprovable")
    session = _FailThenSucceedSession(fail_count=0)

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=1, backoff=0
    )

    assert ok is True
    assert dest.read_bytes() == b"hello "


def test_short_stream_is_not_renamed_into_place(tmp_path: Path) -> None:
    """A stream that ends early without raising must not leave a truncated
    archive behind looking like a complete one."""
    session = _FailThenSucceedSession(fail_count=0, chunks=[b"half"])
    dest = tmp_path / "archive.zip"

    class _ShortSession(_FailThenSucceedSession):
        def get(self, url, stream, timeout, headers=None):  # type: ignore[override]
            self.calls += 1
            return _FakeResponse([b"half"], headers={"content-length": "999"})

    session = _ShortSession(fail_count=0)

    ok = download_with_retries(
        "https://example.test/a.zip", dest, session, retries=1, backoff=0
    )

    assert ok is False
    assert not dest.exists()
