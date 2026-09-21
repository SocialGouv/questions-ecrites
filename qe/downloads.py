"""Shared HTTP streaming-download helper with resumable retries.

data.assemblee-nationale.fr and data.senat.fr intermittently drop the
connection mid-stream (``IncompleteRead``) on large archives — a transient
network blip, not a permanent failure. Retrying the whole download a few
times, with a short backoff, resolves it without operator intervention.

Retrying from byte 0 does not, once the archive is big enough. The AN's
closed-legislature archives are ~133 MB (vs ~50 MB for the live one) and the
portal cuts them so regularly that whole-file attempts kept dying at 14 MB,
then 33 MB, never reaching the end: the backfill Job could not load
legislature 14 at all. The same file downloads fine with ``curl -C -``, so
the missing piece was resumption, not luck.

Each attempt therefore continues the partial file with a ``Range`` request
instead of restarting it. Splicing bytes from two different versions of an
archive would produce a corrupt file that no checksum here would catch, so a
partial file is only ever resumed when the server confirms, through
``If-Range`` against the validator saved next to it, that the resource has
not changed since — otherwise the server answers 200 and the partial file is
discarded. The validator is persisted rather than kept in memory because the
retries that matter span processes: the ingestion CronJob and the backfill
Job both re-run the script in a shell loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def _resume_state(tmp: Path, meta: Path) -> tuple[int, str | None]:
    """Bytes already downloaded, and the validator proving they're still current.

    Both or neither: a partial file whose validator is missing cannot be
    proven to match what the server would send now, so it is not resumable.
    """
    if not tmp.exists() or not meta.exists():
        return 0, None
    validator = meta.read_text().strip()
    if not validator:
        return 0, None
    return tmp.stat().st_size, validator


def download_with_retries(
    url: str,
    dest: Path,
    http: requests.Session,
    *,
    retries: int = 3,
    backoff: float = 2.0,
) -> bool:
    """Stream *url* to *dest*, resuming and retrying transient failures.

    Returns True on success. On failure the partial file and its validator are
    left in place so a later call — including one from a later process — picks
    up where this one stopped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    meta = Path(f"{tmp}.meta")

    for attempt in range(1, retries + 1):
        resume_from, validator = _resume_state(tmp, meta)
        headers = (
            {"Range": f"bytes={resume_from}-", "If-Range": validator}
            if resume_from and validator
            else {}
        )
        try:
            with http.get(url, stream=True, timeout=120, headers=headers) as resp:
                resp.raise_for_status()
                # 206 means the server honoured the range: our bytes are still
                # valid and it is sending the remainder. Anything else (200,
                # typically, when If-Range no longer matches) is a full body,
                # so the partial file is stale and gets overwritten.
                resumed = resp.status_code == 206 and bool(headers)
                if not resumed:
                    resume_from = 0
                new_validator = resp.headers.get("ETag") or resp.headers.get(
                    "Last-Modified"
                )
                if new_validator:
                    meta.write_text(new_validator)
                elif meta.exists():
                    # Without a validator a later resume could splice versions.
                    meta.unlink()
                total = int(resp.headers.get("content-length", 0)) + resume_from
                downloaded = resume_from
                with tmp.open("ab" if resumed else "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(
                                f"\r  {pct:3d}%  {downloaded // 1_000_000} MB",
                                end="",
                                flush=True,
                            )
                print()  # newline after progress
                if total and downloaded != total:
                    # A stream that ends short without raising would otherwise
                    # be renamed into place as a truncated archive.
                    raise requests.ConnectionError(
                        f"Incomplete download: {downloaded} of {total} bytes"
                    )
                tmp.rename(dest)
                meta.unlink(missing_ok=True)
            return True
        except requests.RequestException as exc:
            print()
            if attempt < retries:
                logger.warning(
                    "Download failed (attempt %d/%d), resuming: %s: %s",
                    attempt,
                    retries,
                    url,
                    exc,
                )
                time.sleep(backoff * attempt)
            else:
                logger.error("Failed to download %s: %s", url, exc)
                return False
    return False
