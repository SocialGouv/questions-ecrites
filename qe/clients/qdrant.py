"""Qdrant vector database REST client."""

from __future__ import annotations

from typing import Sequence

import requests


class QdrantClient:
    """Unified client for all Qdrant REST operations."""

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

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True if deleted, False if already absent."""
        response = requests.delete(
            f"{self.base_url}/collections/{name}",
            timeout=30,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

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

    def search(
        self, collection: str, vector: Sequence[float], top_k: int
    ) -> list[dict]:
        payload = {
            "vector": vector,
            "top": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        response = requests.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])
