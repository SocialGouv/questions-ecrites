#!/usr/bin/env python
"""Test script for generating embeddings using the Socle IA API."""

import os
from pathlib import Path

import requests

BASE_URL = "https://pliage-prod.socle-ia.data-ia.prod.atlas.fabrique.social.gouv.fr"
EMBEDDING_MODEL = "BAAI/bge-m3"


def get_embedding(text: str) -> list[float]:
    """Generate embedding for the given text using the Socle IA API."""
    api_key = os.environ.get("SOCLE_IA_API_KEY")
    if not api_key:
        raise ValueError("SOCLE_IA_API_KEY environment variable is not set")

    response = requests.post(
        f"{BASE_URL}/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def main() -> None:
    """Read sample text file and generate its embedding."""
    data_dir = Path(__file__).parent.parent / "data"
    sample_file = data_dir / "sample.txt"

    text = sample_file.read_text().strip()
    print(f"Text content ({len(text)} chars):")
    print(f"  {text[:100]}...")
    print()

    embedding = get_embedding(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")


if __name__ == "__main__":
    main()
