#!/usr/bin/env python
"""Test script for generating embeddings using the Socle IA API."""

from pathlib import Path

from qe.clients.embedding import EmbeddingClient
from qe.config import get_settings, require_api_key


def main() -> None:
    settings = get_settings()
    api_key = require_api_key("SOCLE_IA_API_KEY")

    embedder = EmbeddingClient(
        url=settings.embeddings_url,
        model=settings.embedding_model,
        api_key=api_key,
    )

    data_dir = Path(__file__).parent.parent / "data"
    sample_file = data_dir / "sample.txt"

    text = sample_file.read_text().strip()
    print(f"Text content ({len(text)} chars):")
    print(f"  {text[:100]}...")
    print()

    embedding = embedder.embed(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")


if __name__ == "__main__":
    main()
