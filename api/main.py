"""FastAPI server exposing the office attribution pipeline.

Start with:
    poetry run uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import requests
from api.questions import router as questions_router
from api.state import AppState, set_state
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.db import check_db_connection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared clients on startup, release on shutdown."""
    settings = get_settings()
    if not settings.albert_api_key:
        raise RuntimeError("ALBERT_API_KEY environment variable is not set.")

    http = requests.Session()
    http.headers["User-Agent"] = "qe-api/1.0"
    set_state(
        AppState(
            vector_store=PgvectorClient(),
            reranker=RerankClient(
                base_url=settings.albert_base_url,
                model=settings.albert_rerank_model,
                api_key=settings.albert_api_key,
            ),
            http=http,
        )
    )
    yield
    set_state(None)
    http.close()


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


@app.get("/healthz")
def healthz():
    if not check_db_connection():
        raise HTTPException(status_code=503, detail="database unreachable")
    return {"status": "ok"}
