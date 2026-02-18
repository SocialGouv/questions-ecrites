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

    socle_api_key: str
    albert_api_key: str
    llm_base_url: str
    chat_completions_url: str
    embeddings_url: str
    llm_model: str
    embedding_model: str


def get_settings() -> Settings:
    """Read env vars and build a Settings instance.

    URL derivation rules:
      - CHAT_COMPLETIONS_URL defaults to {LLM_BASE_URL}/api/v1/chat/completions
      - EMBEDDINGS_URL defaults to {LLM_BASE_URL}/api/embeddings
      - Explicit env vars always take precedence over derived values.

    Raises ValueError if required variables are missing.
    """
    llm_base_url = os.environ.get("LLM_BASE_URL", "")

    chat_completions_url = os.environ.get("CHAT_COMPLETIONS_URL", "")
    if not chat_completions_url and llm_base_url:
        chat_completions_url = f"{llm_base_url.rstrip('/')}/api/v1/chat/completions"

    embeddings_url = os.environ.get("EMBEDDINGS_URL", "")
    if not embeddings_url and llm_base_url:
        embeddings_url = f"{llm_base_url.rstrip('/')}/api/embeddings"

    socle_api_key = os.environ.get("SOCLE_IA_API_KEY", "")
    albert_api_key = os.environ.get("ALBERT_API_KEY", "")
    llm_model = os.environ.get("LLM_MODEL", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

    missing: list[str] = []
    if not socle_api_key:
        missing.append("SOCLE_IA_API_KEY")
    if not llm_base_url and not (chat_completions_url and embeddings_url):
        missing.append("LLM_BASE_URL (or both CHAT_COMPLETIONS_URL and EMBEDDINGS_URL)")
    if not llm_model:
        missing.append("LLM_MODEL")

    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return Settings(
        socle_api_key=socle_api_key,
        albert_api_key=albert_api_key,
        llm_base_url=llm_base_url,
        chat_completions_url=chat_completions_url,
        embeddings_url=embeddings_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


def require_api_key(env_var: str) -> str:
    """Read a single API key env var and raise if it is not set."""
    value = os.environ.get(env_var, "")
    if not value:
        raise ValueError(f"{env_var} environment variable is not set")
    return value
