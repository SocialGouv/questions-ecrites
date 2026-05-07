"""FastAPI server exposing the office attribution pipeline.

Start with:
    poetry run uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import api.state as _app_state
from api.questions import router as questions_router
from api.state import ALBERT_BASE_URL, ALBERT_RERANK_MODEL, AppState
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared clients on startup, release on shutdown."""
    albert_api_key = os.environ.get("ALBERT_API_KEY", "")
    if not albert_api_key:
        raise RuntimeError("ALBERT_API_KEY environment variable is not set.")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    _app_state._state = AppState(
        qdrant=QdrantClient(qdrant_url),
        reranker=RerankClient(
            base_url=ALBERT_BASE_URL,
            model=ALBERT_RERANK_MODEL,
            api_key=albert_api_key,
        ),
    )
    yield
    _app_state._state = None


app = FastAPI(
    title="QE Attribution API",
    description="Suggests the most relevant ministry offices for a parliamentary question.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(questions_router)
