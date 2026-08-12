"""Shared HTTP streaming-download helper with retries.

data.assemblee-nationale.fr and data.senat.fr intermittently drop the
connection mid-stream (``IncompleteRead``) on large archives — a transient
network blip, not a permanent failure. Retrying the whole download a few
times, with a short backoff, resolves it without operator intervention.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def download_with_retries(
    url: str,
    dest: Path,
    http: requests.Session,
    *,
    retries: int = 3,
    backoff: float = 2.0,
) -> bool:
    """Stream *url* to *dest*, retrying transient failures. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            with http.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                downloaded = 0
                with tmp.open("wb") as fh:
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
                tmp.rename(dest)
            return True
        except requests.RequestException as exc:
            print()
            tmp = dest.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                logger.warning(
                    "Download failed (attempt %d/%d), retrying: %s: %s",
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
