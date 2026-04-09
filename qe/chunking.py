"""Chunk data model and protocol for document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    title: str
    text: str
    section_index: int
    chunk_index: int
    metadata: dict[str, object]


class Chunker(Protocol):
    def chunk(self, *, text: str) -> list[Chunk]: ...
