"""Chunking strategies, models, and caching for document ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from qe import db
from qe.hashing import compute_content_hash, make_preview
from qe.llm_duties import LLMJobDescriptionDutyExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_CHAR_LIMIT = 1800
MIN_CHUNK_CHAR_LENGTH = 350
CHUNK_CHAR_OVERLAP = 200
DEFAULT_SECTION_TITLE = "Overview"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    title: str
    text: str
    section_index: int
    chunk_index: int
    metadata: dict[str, object]


class Chunker(Protocol):
    def chunk(self, *, text: str) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Chunk cache (backed by Postgres)
# ---------------------------------------------------------------------------


class ChunkCache:
    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name

    def load(self, strategy: str, document_hash: str) -> list[Chunk] | None:
        cached = db.fetch_chunk_cache(strategy, document_hash)
        if cached is None:
            return None
        return [Chunk(**chunk_dict) for chunk_dict in cached]

    def save(self, strategy: str, document_hash: str, chunks: list[Chunk]) -> None:
        payload = [
            {
                "title": c.title,
                "text": c.text,
                "section_index": c.section_index,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        db.save_chunk_cache(strategy, document_hash, payload)


# ---------------------------------------------------------------------------
# Heuristic chunking
# ---------------------------------------------------------------------------


def looks_like_heading(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if line.endswith(":"):
        return True
    if line.isupper() and len(line.split()) <= 12:
        return True
    if re.match(r"^[0-9]+(?:\.[0-9]+)*[\).]?\s+.+", line):
        return True
    return False


def extract_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = DEFAULT_SECTION_TITLE
    current_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and looks_like_heading(stripped):
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_title, section_text))
            current_title = stripped.rstrip(":").strip() or DEFAULT_SECTION_TITLE
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_title, section_text))
    if not sections and text.strip():
        sections.append((DEFAULT_SECTION_TITLE, text.strip()))
    return sections


def iter_section_chunks(
    section_text: str,
    max_chars: int = DEFAULT_CHUNK_CHAR_LIMIT,
    min_chars: int = MIN_CHUNK_CHAR_LENGTH,
    overlap_chars: int = CHUNK_CHAR_OVERLAP,
) -> Iterator[str]:
    clean = section_text.strip()
    if not clean:
        return

    start = 0
    length = len(clean)
    while start < length:
        end = min(length, start + max_chars)
        raw_chunk = clean[start:end]
        chunk = raw_chunk.strip()
        if not chunk:
            start = end
            continue

        if len(chunk) < min_chars and end != length:
            extend_by = min(min_chars - len(chunk), length - end)
            end += extend_by
            raw_chunk = clean[start:end]
            chunk = raw_chunk.strip()

        yield chunk

        if end >= length:
            break

        chunk_span = max(1, end - start)
        overlap = min(overlap_chars, chunk_span - 1) if chunk_span > 1 else 0
        start = start + chunk_span - overlap


class HeuristicChunker:
    def __init__(
        self,
        *,
        max_chars: int = DEFAULT_CHUNK_CHAR_LIMIT,
        min_chars: int = MIN_CHUNK_CHAR_LENGTH,
        overlap_chars: int = CHUNK_CHAR_OVERLAP,
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    def chunk(self, *, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        sections = extract_sections(text)
        chunk_index = 0
        for section_index, (section_title, section_text) in enumerate(sections):
            for section_chunk_text in iter_section_chunks(
                section_text,
                max_chars=self.max_chars,
                min_chars=self.min_chars,
                overlap_chars=self.overlap_chars,
            ):
                chunks.append(
                    Chunk(
                        title=section_title,
                        text=section_chunk_text,
                        section_index=section_index,
                        chunk_index=chunk_index,
                        metadata={},
                    )
                )
                chunk_index += 1
        return chunks


# ---------------------------------------------------------------------------
# LLM duty chunking
# ---------------------------------------------------------------------------


class LLMDutyChunker:
    def __init__(
        self,
        *,
        llm_client: LLMJobDescriptionDutyExtractor,
        cache: ChunkCache,
        strategy_name: str = "llm_duty",
    ) -> None:
        self.llm_client = llm_client
        self.cache = cache
        self.strategy_name = strategy_name

    def chunk(self, *, text: str) -> list[Chunk]:
        document_hash = compute_content_hash(text)
        cached = self.cache.load(self.strategy_name, document_hash)
        if cached is not None:
            return cached

        duties = self.llm_client.request_duties(text)
        if not duties:
            raise ValueError("LLM returned no duties")

        chunks: list[Chunk] = []
        for idx, duty_text in enumerate(duties):
            if not isinstance(duty_text, str):
                continue
            cleaned = duty_text.strip()
            if not cleaned:
                continue
            metadata: dict[str, object] = {
                "chunk_type": "llm_duty",
                "duty_index": idx,
            }
            chunks.append(
                Chunk(
                    title="Duties",
                    text=cleaned,
                    section_index=0,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )

        if not chunks:
            raise ValueError("LLM did not produce any usable duty chunks")

        self.cache.save(self.strategy_name, document_hash, chunks)
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def chunker_factory(
    *,
    strategy: str,
    llm_extractor: LLMJobDescriptionDutyExtractor | None = None,
    chunk_cache: ChunkCache | None = None,
) -> Chunker:
    """Create a Chunker based on the chosen strategy."""
    if strategy == "heuristic":
        return HeuristicChunker()
    if strategy in {"llm_duty", "llm_responsibility"}:
        if llm_extractor is None or chunk_cache is None:
            raise ValueError(
                "llm_extractor and chunk_cache are required for LLM chunking strategies"
            )
        return LLMDutyChunker(
            llm_client=llm_extractor,
            cache=chunk_cache,
            strategy_name="llm_duty",
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


# ---------------------------------------------------------------------------
# Payload building (depends on Chunk, so lives here)
# ---------------------------------------------------------------------------


def build_chunk_payload(
    path: Path,
    job_id: str,
    chunk: Chunk,
    document_hash: str,
) -> dict:
    """Build the Qdrant point payload for a single chunk."""
    base_payload = {
        "kind": "chunk",
        "job_id": job_id,
        "job_filename": path.name,
        "job_path": str(path),
        "file_extension": path.suffix.lower(),
        "section_title": chunk.title,
        "section_index": chunk.section_index,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "char_count": len(chunk.text),
        "chunk_preview": make_preview(chunk.text),
        "document_hash": document_hash,
    }
    base_payload.update(chunk.metadata)
    return base_payload
