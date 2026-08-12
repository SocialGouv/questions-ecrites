import requests

from scripts.download_an import (
    _archive_filename,
    _url_for,
    discover_current_legislature,
)


def test_url_for_roman_numeral_legislatures() -> None:
    assert _url_for(14).endswith("Questions_ecrites_XIV.xml.zip")
    assert _url_for(15).endswith("Questions_ecrites_XV.xml.zip")


def test_url_for_generic_legislatures() -> None:
    assert _url_for(16).endswith(
        "/16/questions/questions_ecrites/Questions_ecrites.xml.zip"
    )
    assert _url_for(17).endswith(
        "/17/questions/questions_ecrites/Questions_ecrites.xml.zip"
    )
    # Legislatures beyond the ones known today must resolve without a lookup error.
    assert _url_for(25).endswith(
        "/25/questions/questions_ecrites/Questions_ecrites.xml.zip"
    )


def test_archive_filename_roman_and_generic() -> None:
    assert _archive_filename(14) == "Questions_ecrites_XIV.xml.zip"
    assert _archive_filename(25) == "Questions_ecrites_25.xml.zip"


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHeadSession:
    """Reports 200 up to *last_existing*, 404 beyond it."""

    def __init__(self, last_existing: int) -> None:
        self.last_existing = last_existing
        self.probed: list[int] = []

    def head(self, url: str, timeout: int, allow_redirects: bool) -> _FakeResponse:
        n = int(url.split("/")[-4])  # .../repository/{n}/questions/...
        self.probed.append(n)
        return _FakeResponse(200 if n <= self.last_existing else 404)


class _FailingHeadSession:
    def head(self, url: str, timeout: int, allow_redirects: bool) -> _FakeResponse:
        raise requests.ConnectionError("boom")


def test_discover_current_legislature_stops_at_first_missing() -> None:
    session = _FakeHeadSession(last_existing=19)

    current = discover_current_legislature(session, floor=17)

    assert current == 19
    assert session.probed == [18, 19, 20]


def test_discover_current_legislature_returns_floor_when_none_newer() -> None:
    session = _FakeHeadSession(last_existing=17)

    current = discover_current_legislature(session, floor=17)

    assert current == 17
    assert session.probed == [18]


def test_discover_current_legislature_falls_back_on_request_error() -> None:
    session = _FailingHeadSession()

    current = discover_current_legislature(session, floor=17)

    assert current == 17
