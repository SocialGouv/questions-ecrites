#!/usr/bin/env python
"""Ingest job description files into Qdrant using Socle IA embeddings.

Supported extensions: .txt, .pdf, .doc, .docx

Defaults:
  - Input folder: data/job_descriptions
  - Qdrant collection: job_descriptions
  - Qdrant URL: http://localhost:6333

Usage:
  python scripts/ingest_job_descriptions.py

Requires:
  - SOCLE_IA_API_KEY environment variable set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol
from uuid import UUID

import requests

from qe import db

SOCLE_BASE_URL = (
    "https://pliage-prod.socle-ia.data-ia.prod.atlas.fabrique.social.gouv.fr"
)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_INPUT_DIR = Path("data/job_descriptions")
DEFAULT_COLLECTION = "job_descriptions"
DEFAULT_QDRANT_URL = "http://localhost:6333"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx"}
IGNORED_PREFIXES = {"~$", "."}

DEFAULT_CHUNK_CHAR_LIMIT = 1800
MIN_CHUNK_CHAR_LENGTH = 350
CHUNK_CHAR_OVERLAP = 200
DEFAULT_SECTION_TITLE = "Overview"
DEFAULT_CHUNK_STRATEGY = "llm_responsibility"
# Legacy default kept for CLI compatibility (no longer used for filesystem cache)
DEFAULT_CHUNK_CACHE_DIR = Path(".chunk_cache")
SOCLE_LLM_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
SOCLE_LLM_BASE_URL = SOCLE_BASE_URL
SOCLE_CHAT_COMPLETIONS_PATH = "/api/v1/chat/completions"


@dataclass(frozen=True)
class IngestionConfig:
    input_dir: Path
    collection: str
    qdrant_url: str
    embedding_model: str
    embedding_base_url: str
    chunk_strategy: str
    chunk_cache_dir: Path
    llm_base_url: str
    llm_model: str


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


class QdrantClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def collection_exists(self, name: str) -> bool:
        response = requests.get(f"{self.base_url}/collections/{name}", timeout=30)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def create_collection(self, name: str, vector_size: int) -> None:
        payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        }
        response = requests.put(
            f"{self.base_url}/collections/{name}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    def get_point(self, name: str, point_id: str) -> dict | None:
        response = requests.get(
            f"{self.base_url}/collections/{name}/points/{point_id}",
            timeout=30,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("result")

    def upsert_points(self, name: str, points: list[dict]) -> None:
        response = requests.put(
            f"{self.base_url}/collections/{name}/points",
            json={"points": points},
            timeout=60,
        )
        response.raise_for_status()

    def delete_points_by_filter(self, name: str, filter_payload: dict) -> None:
        response = requests.post(
            f"{self.base_url}/collections/{name}/points/delete",
            json={"filter": filter_payload},
            timeout=60,
        )
        response.raise_for_status()


def parse_args() -> IngestionConfig:
    parser = argparse.ArgumentParser(
        description="Ingest job description files into Qdrant."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing job description files.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Base URL for Qdrant (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Socle IA embedding model name.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=SOCLE_BASE_URL,
        help="Base URL for the Socle IA API.",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=DEFAULT_CHUNK_STRATEGY,
        choices=["heuristic", "llm_responsibility"],
        help="Chunking strategy to use (heuristic | llm_responsibility).",
    )
    parser.add_argument(
        "--chunk-cache-dir",
        type=Path,
        default=DEFAULT_CHUNK_CACHE_DIR,
        help="Directory where chunker caches intermediate outputs.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=SOCLE_LLM_BASE_URL,
        help="Base URL for LLM chunking API.",
    )
    parser.add_argument(
        "--llm-model",
        default=SOCLE_LLM_MODEL,
        help="Model name for LLM chunking (e.g. mistralai/Ministral-3-8B-Instruct-2512).",
    )
    args = parser.parse_args()
    return IngestionConfig(
        input_dir=args.input_dir,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        chunk_strategy=args.chunking_strategy,
        chunk_cache_dir=args.chunk_cache_dir,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
    )


@dataclass
class Chunk:
    title: str
    text: str
    section_index: int
    chunk_index: int
    metadata: dict[str, object]


class Chunker(Protocol):
    def chunk(self, *, text: str) -> list[Chunk]: ...


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
                "title": chunk.title,
                "text": chunk.text,
                "section_index": chunk.section_index,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        db.save_chunk_cache(strategy, document_hash, payload)


class LLMClient:
    def __init__(
        self, base_url: str, model: str, api_key: str, timeout: int = 120
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.chat_endpoint = f"{self.base_url}{SOCLE_CHAT_COMPLETIONS_PATH}"

    def request_responsibilities(self, text: str) -> list[str]:  # noqa: C901
        system_message = (
            "Vous êtes un analyste expert en ressources humaines. À partir d’une description de poste, "
            "identifiez des responsabilités distinctes. Chaque responsabilité doit être exactement une "
            "phrase complète, détaillée, autonome et en français. Développez chaque acronyme lors de sa première "
            "apparition en écrivant « Forme longue (ACRONYME) ». Répondez uniquement avec du JSON strict, "
            'en respectant le schéma suivant : {"responsibilities": ["responsabilité 1", "responsabilité 2", ...]}'
        )
        user_message = (
            "Description du poste :\n{text}\n\n"
            "Règles :\n"
            "1. Produisez entre 5 et 25 responsabilités selon le contenu.\n"
            "2. Chaque responsabilité doit décrire une seule action ou attente.\n"
            "3. Les acronymes doivent être développés lorsqu’ils apparaissent.\n"
            "4. Écrivez en français.\n"
            "5. Fournissez uniquement du JSON, sans commentaire."
        ).format(text=text.strip())

        try:
            response = requests.post(
                self.chat_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network failure path
            status = exc.response.status_code if exc.response else "unknown"
            detail = exc.response.text.strip() if exc.response else ""
            snippet = detail[:500] + ("…" if len(detail) > 500 else "")
            raise RuntimeError(
                "LLM chunking request failed "
                f"(status {status} from {self.chat_endpoint}). "
                "Response body snippet: "
                f"{snippet or '[empty response]'}"
            ) from exc
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.strip("`")
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
        content = content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM response is not valid JSON. Snippet: "
                f"{content[:200]}{'…' if len(content) > 200 else ''}"
            ) from exc

        responsibilities_field = parsed.get("responsibilities")
        responsibilities: list[str] = []

        def _clean_line(line: str) -> str:
            stripped = line.strip()
            stripped = stripped.lstrip("-•*0123456789.) ")
            return stripped.strip()

        if isinstance(responsibilities_field, str):
            for line in responsibilities_field.splitlines():
                cleaned = _clean_line(line)
                if cleaned:
                    responsibilities.append(cleaned)
        elif isinstance(responsibilities_field, list):
            for item in responsibilities_field:
                if isinstance(item, str):
                    cleaned = _clean_line(item)
                    if cleaned:
                        responsibilities.append(cleaned)
                elif isinstance(item, dict):
                    value = (
                        item.get("responsibility")
                        or item.get("text")
                        or item.get("description")
                    )
                    if isinstance(value, str):
                        cleaned = _clean_line(value)
                        if cleaned:
                            responsibilities.append(cleaned)

        if not responsibilities:
            raise ValueError(
                "LLM response missing usable 'responsibilities'. Parsed payload: "
                f"{json.dumps(parsed)[:300]}{'…' if len(json.dumps(parsed)) > 300 else ''}"
            )
        return responsibilities


class LLMResponsibilityChunker:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        cache: ChunkCache,
        strategy_name: str = "llm_responsibility",
    ) -> None:
        self.llm_client = llm_client
        self.cache = cache
        self.strategy_name = strategy_name

    def chunk(self, *, text: str) -> list[Chunk]:
        document_hash = compute_content_hash(text)
        cached = self.cache.load(self.strategy_name, document_hash)
        if cached is not None:
            return cached

        responsibilities = self.llm_client.request_responsibilities(text)
        if not responsibilities:
            raise ValueError("LLM returned no responsibilities")

        chunks: list[Chunk] = []
        for idx, responsibility_text in enumerate(responsibilities):
            if not isinstance(responsibility_text, str):
                continue
            cleaned = responsibility_text.strip()
            if not cleaned:
                continue
            metadata: dict[str, object] = {
                "chunk_type": "llm_responsibility",
                "responsibility_index": idx,
            }
            chunks.append(
                Chunk(
                    title="Responsibilities",
                    text=cleaned,
                    section_index=0,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )

        if not chunks:
            raise ValueError("LLM did not produce any usable responsibility chunks")

        self.cache.save(self.strategy_name, document_hash, chunks)
        return chunks


def load_documents(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Input directory '{folder}' does not exist. Create it and add files."
        )
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(path.name.startswith(prefix) for prefix in IGNORED_PREFIXES)
    )


def read_document(path: Path) -> str:
    extension = path.suffix.lower()
    if extension == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".pdf":
        return read_pdf(path)
    if extension == ".docx":
        return read_docx(path)
    if extension == ".doc":
        return read_doc(path)
    raise ValueError(f"Unsupported file extension: {extension}")


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def read_doc(path: Path) -> str:
    import textract

    content = textract.process(str(path))
    return content.decode("utf-8", errors="ignore")


def stable_point_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def make_preview(text: str, max_chars: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def stable_chunk_id(path: Path, section_index: int, chunk_index: int) -> str:
    digest_source = f"{path.resolve()}::{section_index}::{chunk_index}".encode("utf-8")
    digest = hashlib.sha256(digest_source).hexdigest()
    return str(UUID(digest[:32]))


def build_chunk_payload(
    path: Path,
    job_id: str,
    chunk: Chunk,
    document_hash: str,
) -> dict:
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


def load_manifest_entries() -> dict[str, str]:
    return db.get_manifest_entries()


def delete_job_chunks(qdrant: QdrantClient, collection: str, job_id: str) -> None:
    filter_payload = {"must": [{"key": "job_id", "match": {"value": job_id}}]}
    qdrant.delete_points_by_filter(collection, filter_payload)


def chunker_factory(
    config: IngestionConfig,
    api_key: str,
    chunk_cache: ChunkCache,
) -> Chunker:
    if config.chunk_strategy == "heuristic":
        return HeuristicChunker()
    if config.chunk_strategy == "llm_responsibility":
        llm_client = LLMClient(
            base_url=config.llm_base_url,
            model=config.llm_model,
            api_key=api_key,
        )
        return LLMResponsibilityChunker(
            llm_client=llm_client,
            cache=chunk_cache,
            strategy_name=config.chunk_strategy,
        )
    raise ValueError(f"Unknown chunking strategy: {config.chunk_strategy}")


def ingest_files(  # noqa: C901
    config: IngestionConfig,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    chunker: Chunker,
) -> None:
    files = list(load_documents(config.input_dir))
    if not files:
        print(f"No supported files found in {config.input_dir}.")
        return

    non_empty_text = None
    for file_path in files:
        text = read_document(file_path).strip()
        if text:
            non_empty_text = text
            break

    if not non_empty_text:
        print("No non-empty documents found to ingest.")
        return

    collection_exists = qdrant.collection_exists(config.collection)
    if not collection_exists:
        embedding = embedder.embed(non_empty_text)
        qdrant.create_collection(config.collection, vector_size=len(embedding))
        print(f"Created collection '{config.collection}' with size {len(embedding)}.")
        collection_exists = True

    manifest = load_manifest_entries()
    updated_manifest = dict(manifest)

    indexed_paths = {str(path) for path in files}
    removed_paths = [path_str for path_str in manifest if path_str not in indexed_paths]
    if removed_paths and collection_exists:
        for removed_path in removed_paths:
            job_id = stable_point_id(Path(removed_path))
            delete_job_chunks(qdrant, config.collection, job_id)
            updated_manifest.pop(removed_path, None)
            db.delete_manifest(removed_path)
            print(f"Removed stale entries for {removed_path}")

    for file_path in files:
        text = read_document(file_path).strip()
        if not text:
            print(f"Skipping empty file: {file_path}")
            continue

        document_hash = compute_content_hash(text)
        path_key = str(file_path)
        if manifest.get(path_key) == document_hash:
            print(f"Skipping unchanged file: {file_path}")
            continue

        job_id = stable_point_id(file_path)
        if collection_exists:
            delete_job_chunks(qdrant, config.collection, job_id)

        chunks = chunker.chunk(text=text)
        chunk_points: list[dict] = []
        chunk_counter = 0
        for chunk in chunks:
            chunk_id = stable_chunk_id(
                file_path, chunk.section_index, chunk.chunk_index
            )
            embedding = embedder.embed(chunk.text)
            payload = build_chunk_payload(
                path=file_path,
                job_id=job_id,
                chunk=chunk,
                document_hash=document_hash,
            )
            chunk_points.append(
                {"id": chunk_id, "vector": embedding, "payload": payload}
            )
            chunk_counter += 1

        if not chunk_points:
            print(f"No valid chunks produced for {file_path}; skipping.")
            continue

        qdrant.upsert_points(config.collection, chunk_points)
        updated_manifest[path_key] = document_hash
        db.upsert_manifest(path_key, document_hash)
        print(
            f"Upserted {chunk_counter} chunks for '{file_path.name}' into '{config.collection}'."
        )


def main() -> None:
    config = parse_args()
    api_key = os.environ.get("SOCLE_IA_API_KEY")
    if not api_key:
        raise ValueError("SOCLE_IA_API_KEY environment variable is not set")

    embedder = EmbeddingClient(
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        api_key=api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)
    chunker = chunker_factory(config, api_key, ChunkCache(config.chunk_strategy))
    ingest_files(config, embedder, qdrant, chunker)


if __name__ == "__main__":
    main()
