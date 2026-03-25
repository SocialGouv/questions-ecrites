from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from qe.models import (
    ChunkCache,
    IngestManifest,
    QuestionCluster,
)

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------


def _build_database_url() -> str:
    """Build a psycopg3 SQLAlchemy URL from PG* environment variables."""
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5433")
    user = os.getenv("PGUSER", "qe")
    password = os.getenv("PGPASSWORD", "qe")
    dbname = os.getenv("PGDATABASE", "qe")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"


_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine (created once, reused across calls)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_database_url(),
            pool_pre_ping=True,  # discard stale connections before use
            pool_size=5,
            max_overflow=10,
        )
    return _engine


_SessionLocal: sessionmaker[Session] | None = None


def _get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager that yields a SQLAlchemy Session and handles commit/rollback."""
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except:  # noqa: E722
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ingest_manifest
# ---------------------------------------------------------------------------


def fetch_manifest(path: str) -> str | None:
    with get_session() as session:
        row = session.get(IngestManifest, path)
        return row.document_hash if row else None


def upsert_manifest(path: str, document_hash: str) -> None:
    with get_session() as session:
        stmt = (
            pg_insert(IngestManifest)
            .values(path=path, document_hash=document_hash)
            .on_conflict_do_update(
                index_elements=["path"],
                set_={"document_hash": document_hash, "updated_at": func.now()},
            )
        )
        session.execute(stmt)


def delete_manifest(path: str) -> None:
    with get_session() as session:
        session.execute(delete(IngestManifest).where(IngestManifest.path == path))


def get_manifest_entries() -> dict[str, str]:
    with get_session() as session:
        rows = session.execute(select(IngestManifest)).scalars().all()
        return {row.path: row.document_hash for row in rows}


def get_manifest_entries_under_prefix(path_prefix: str) -> dict[str, str]:
    """Return manifest entries whose path starts with the given prefix."""
    with get_session() as session:
        rows = (
            session.execute(
                select(IngestManifest).where(
                    IngestManifest.path.like(f"{path_prefix}%")
                )
            )
            .scalars()
            .all()
        )
        return {row.path: row.document_hash for row in rows}


def delete_manifest_under_prefix(path_prefix: str) -> int:
    """Delete manifest rows whose path starts with the given prefix.

    Returns the number of deleted rows.
    """
    with get_session() as session:
        result = session.execute(
            delete(IngestManifest).where(IngestManifest.path.like(f"{path_prefix}%"))
        )
        return result.rowcount


# ---------------------------------------------------------------------------
# chunk_cache
# ---------------------------------------------------------------------------


def fetch_chunk_cache(strategy: str, document_hash: str) -> list[dict] | None:
    with get_session() as session:
        row = session.get(ChunkCache, (strategy, document_hash))
        return row.chunks if row else None


def save_chunk_cache(strategy: str, document_hash: str, chunks: Sequence[dict]) -> None:
    chunks_list = list(chunks)
    with get_session() as session:
        stmt = (
            pg_insert(ChunkCache)
            .values(strategy=strategy, document_hash=document_hash, chunks=chunks_list)
            .on_conflict_do_update(
                index_elements=["strategy", "document_hash"],
                set_={"chunks": chunks_list, "updated_at": func.now()},
            )
        )
        session.execute(stmt)


def delete_chunk_cache(strategy: str, document_hash: str) -> None:
    with get_session() as session:
        session.execute(
            delete(ChunkCache).where(
                ChunkCache.strategy == strategy,
                ChunkCache.document_hash == document_hash,
            )
        )


# ---------------------------------------------------------------------------
# question_clusters
# ---------------------------------------------------------------------------


def save_clusters(clusters: list[dict]) -> None:
    """Replace all question clusters with the new results."""
    with get_session() as session:
        session.execute(delete(QuestionCluster))
        rows = [
            QuestionCluster(
                question_id=q["question_id"],
                cluster_id=cluster["cluster_id"],
                similarity_to_centroid=q["similarity_to_centroid"],
            )
            for cluster in clusters
            for q in cluster["questions"]
        ]
        session.add_all(rows)


def delete_chunk_cache_for_document_hashes(document_hashes: Sequence[str]) -> int:
    """Delete chunk_cache rows for the provided document hashes (all strategies).

    Returns the number of deleted rows.
    """
    hashes = list(dict.fromkeys(document_hashes))  # deduplicate, preserve order
    if not hashes:
        return 0
    with get_session() as session:
        result = session.execute(
            delete(ChunkCache).where(ChunkCache.document_hash.in_(hashes))
        )
        return result.rowcount
