"""Shared application state initialised at startup and accessed by route modules."""

from __future__ import annotations

from dataclasses import dataclass

from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient

ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
ALBERT_RERANK_MODEL = "openweight-rerank"


@dataclass
class AppState:
    qdrant: QdrantClient
    reranker: RerankClient


_state: AppState | None = None


def set_state(state: AppState | None) -> None:
    global _state
    _state = state


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("Application has not started yet.")
    return _state
