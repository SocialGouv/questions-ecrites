"""VectorStore Protocol — common interface for vector database backends."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Structural protocol satisfied by any vector database client.

    PgvectorClient (and the legacy QdrantClient) implement this interface, so
    call sites typed against VectorStore remain backend-agnostic.

    Point dict shapes
    -----------------
    Upsert input:  {"id": str, "vector": list[float], "payload": dict}
    Search output: {"id": str, "score": float, "payload": dict}
                   score is cosine similarity in [0, 1] (higher = more similar)
    Scroll/get:    {"id": str, "vector": list[float], "payload": dict}
                   (vector absent when with_vectors=False)
    """

    def collection_exists(self, name: str) -> bool: ...

    def get_vector_size(self, name: str) -> int | None: ...

    def create_collection(self, name: str, vector_size: int) -> None: ...

    def delete_collection(self, name: str) -> bool: ...

    def get_point(
        self, name: str, point_id: str, *, with_vectors: bool = False
    ) -> dict | None: ...

    def upsert_points(self, name: str, points: list[dict]) -> None: ...

    def delete_points_by_filter(self, name: str, filter_payload: dict) -> None: ...

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        top_k: int,
        *,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]: ...

    def get_points_by_ids(
        self, name: str, ids: list[str], *, with_vectors: bool = True
    ) -> list[dict]: ...

    def scroll_all(
        self,
        collection: str,
        *,
        filter: dict | None = None,
        with_vectors: bool = True,
        batch_size: int = 100,
    ) -> list[dict]: ...
