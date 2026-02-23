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
from dataclasses import dataclass
from pathlib import Path

from qe.chunking import ChunkCache, chunker_factory
from qe.clients.embedding import EmbeddingClient
from qe.clients.llm import SocleLLMClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings, require_api_key
from qe.ingestion import ingest_files
from qe.llm_duties import LLMJobDescriptionDutyExtractor

DEFAULT_INPUT_DIR = Path("data/job_descriptions")
DEFAULT_COLLECTION = "job_descriptions"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_CHUNK_STRATEGY = "llm_duty"


@dataclass(frozen=True)
class IngestionConfig:
    input_dir: Path
    collection: str
    qdrant_url: str
    embedding_model: str
    embeddings_url: str
    chunk_strategy: str
    chat_completions_url: str
    llm_model: str
    api_key: str


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
        default=None,
        help="Socle IA embedding model name.",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=DEFAULT_CHUNK_STRATEGY,
        choices=["heuristic", "llm_duty", "llm_responsibility"],
        help=(
            "Chunking strategy to use (heuristic | llm_duty). "
            "'llm_responsibility' is deprecated and treated as llm_duty."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name for LLM chunking.",
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
        chunk_strategy=args.chunking_strategy,
        chat_completions_url=settings.chat_completions_url,
        llm_model=args.llm_model or settings.llm_model,
        api_key=api_key,
    )


def main() -> None:
    config = parse_args()

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=config.api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)
    chat_client = SocleLLMClient(
        url=config.chat_completions_url,
        model=config.llm_model,
        api_key=config.api_key,
    )
    llm_extractor = LLMJobDescriptionDutyExtractor(client=chat_client)
    chunk_cache = ChunkCache(config.chunk_strategy)
    chunker = chunker_factory(
        strategy=config.chunk_strategy,
        llm_extractor=llm_extractor,
        chunk_cache=chunk_cache,
    )

    ingest_files(
        input_dir=config.input_dir,
        collection=config.collection,
        embedder=embedder,
        qdrant=qdrant,
        chunker=chunker,
    )


if __name__ == "__main__":
    main()
