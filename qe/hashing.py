"""Hashing and stable ID utilities."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import UUID


def stable_point_id(path: Path) -> str:
    """Deterministic UUID for a document path."""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(path: Path, section_index: int, chunk_index: int) -> str:
    """Deterministic UUID for a specific chunk within a document."""
    digest_source = f"{path.resolve()}::{section_index}::{chunk_index}".encode("utf-8")
    digest = hashlib.sha256(digest_source).hexdigest()
    return str(UUID(digest[:32]))


def stable_question_point_id(question_id: str) -> str:
    """Deterministic Qdrant point UUID for a question, derived from its string ID.

    Mirrors the ``_question_point_id`` logic used in ``scripts/embed_questions.py``
    so that any code needing to look up a question point can share a single
    implementation.
    """
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


def make_preview(text: str, max_chars: int = 240) -> str:
    """Produce a short single-line preview of *text*."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"
