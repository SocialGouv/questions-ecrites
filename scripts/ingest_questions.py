#!/usr/bin/env python
"""Ingest parliamentary question files into Qdrant for similarity search.

One embedding is produced per question file (no chunking). Each question is
stored as a single Qdrant point keyed by a deterministic UUID derived from the
file path.

Supports incremental updates: unchanged files (same content hash as the stored
manifest entry) are skipped. Modified or new files are re-embedded and upserted.
Files removed from disk are deleted from Qdrant and the manifest.

Defaults:
  - Input folder: data/qe_no_answers
  - Qdrant collection: questions
  - Qdrant URL: http://localhost:6333

Usage:
  python scripts/ingest_questions.py
  python scripts/ingest_questions.py --input-dir data/qe_with_answers

Requires:
  - SOCLE_IA_API_KEY environment variable set
  - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from qe import db
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings, require_api_key
from qe.documents import load_documents, read_document
from qe.hashing import compute_content_hash, make_preview, stable_point_id

DEFAULT_INPUT_DIR = Path("data/qe_no_answers")
DEFAULT_COLLECTION = "questions"
DEFAULT_QDRANT_URL = "http://localhost:6333"


@dataclass(frozen=True)
class IngestionConfig:
    input_dir: Path
    collection: str
    qdrant_url: str
    embedding_model: str
    embeddings_url: str
    api_key: str


def parse_args() -> IngestionConfig:
    parser = argparse.ArgumentParser(
        description="Ingest parliamentary question files into Qdrant."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing question files.",
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
        default=None,
        help="Socle IA embedding model name.",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = require_api_key("SOCLE_IA_API_KEY")

    return IngestionConfig(
        input_dir=args.input_dir,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model or settings.embedding_model,
        embeddings_url=settings.embeddings_url,
        api_key=api_key,
    )


def _delete_question_point(
    qdrant: QdrantClient, collection: str, question_id: str
) -> None:
    filter_payload = {"must": [{"key": "question_id", "match": {"value": question_id}}]}
    qdrant.delete_points_by_filter(collection, filter_payload)


def ingest_questions(  # noqa: C901
    *,
    input_dir: Path,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
) -> None:
    """Ingest question files from ``input_dir`` into a Qdrant collection.

    One embedding per question file is produced. Unchanged files are skipped.
    Files removed from disk are cleaned up from Qdrant and the manifest.

    Args:
        input_dir: Directory containing question files.
        collection: Qdrant collection name.
        embedder: Client for generating dense embeddings.
        qdrant: Qdrant REST client.
    """
    files = list(load_documents(input_dir))
    if not files:
        print(f"No supported files found in {input_dir}.")
        return

    # Probe the first non-empty file to determine embedding dimension.
    non_empty_text: str | None = None
    for file_path in files:
        text = read_document(file_path).strip()
        if text:
            non_empty_text = text
            break

    if not non_empty_text:
        print("No non-empty documents found to ingest.")
        return

    collection_exists = qdrant.collection_exists(collection)
    if not collection_exists:
        embedding = embedder.embed(non_empty_text)
        qdrant.create_collection(collection, vector_size=len(embedding))
        print(f"Created collection '{collection}' with size {len(embedding)}.")
        collection_exists = True

    # Scope the manifest query to this input directory to avoid interfering
    # with entries from other collections (e.g. job_descriptions).
    path_prefix = str(input_dir)
    manifest = db.get_manifest_entries_under_prefix(path_prefix)
    indexed_paths = {str(path) for path in files}

    # Remove stale entries for files that no longer exist on disk.
    removed_paths = [p for p in manifest if p not in indexed_paths]
    for removed_path in removed_paths:
        question_id = stable_point_id(Path(removed_path))
        _delete_question_point(qdrant, collection, question_id)
        db.delete_manifest(removed_path)
        print(f"Removed stale entry for {removed_path}")

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

        question_id = stable_point_id(file_path)
        if collection_exists:
            _delete_question_point(qdrant, collection, question_id)

        embedding = embedder.embed(text)
        point = {
            "id": question_id,
            "vector": embedding,
            "payload": {
                "kind": "question",
                "question_id": file_path.stem.lower(),
                "question_path": str(file_path),
                "question_text": text[:2000],
                "question_preview": make_preview(text),
                "document_hash": document_hash,
            },
        }

        qdrant.upsert_points(collection, [point])
        db.upsert_manifest(path_key, document_hash)
        print(f"Upserted '{file_path.name}' into '{collection}'.")


def main() -> None:
    config = parse_args()

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=config.api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)

    ingest_questions(
        input_dir=config.input_dir,
        collection=config.collection,
        embedder=embedder,
        qdrant=qdrant,
    )


if __name__ == "__main__":
    main()
