"""Centralized configuration for the QE project.

All environment variable reads are deferred to function calls -- importing
this module has zero side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    albert_api_key: str
    albert_base_url: str
    albert_rerank_model: str
    albert_embedding_model: str
    albert_embeddings_url: str


def get_settings() -> Settings:
    """Read env vars and build a Settings instance.

    All LLM/embedding/reranking calls in this repo go through the Albert API.
    PLIAGE (the OpenWebUI instance behind LLM_BASE_URL/PLIAGE_API_KEY) is a
    separate provider used only by qe-front's correction feature — it must
    never be wired into this repo's settings.

    URL derivation rule:
      - ALBERT_EMBEDDINGS_URL defaults to {ALBERT_BASE_URL}/v1/embeddings.

    Raises ValueError if required variables are missing.
    """
    albert_api_key = os.environ.get("ALBERT_API_KEY", "")
    albert_base_url = os.environ.get("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr")
    albert_rerank_model = os.environ.get("ALBERT_RERANK_MODEL", "openweight-rerank")
    albert_embedding_model = os.environ.get("ALBERT_EMBEDDING_MODEL", "BAAI/bge-m3")
    albert_embeddings_url = os.environ.get("ALBERT_EMBEDDINGS_URL", "") or (
        f"{albert_base_url.rstrip('/')}/v1/embeddings"
    )

    missing: list[str] = []
    if not albert_api_key:
        missing.append("ALBERT_API_KEY")

    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return Settings(
        albert_api_key=albert_api_key,
        albert_base_url=albert_base_url,
        albert_rerank_model=albert_rerank_model,
        albert_embedding_model=albert_embedding_model,
        albert_embeddings_url=albert_embeddings_url,
    )


def require_api_key(env_var: str) -> str:
    """Read a single API key env var and raise if it is not set."""
    value = os.environ.get(env_var, "")
    if not value:
        raise ValueError(f"{env_var} environment variable is not set")
    return value
