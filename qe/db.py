from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import psycopg
from psycopg.types.json import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# SQLAlchemy engine + session
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
            pool_pre_ping=True,  # drop stale connections before use
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
# Raw psycopg connection (kept for existing ingest_manifest / chunk_cache code)
# ---------------------------------------------------------------------------


def _connection_kwargs() -> dict:
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5433"))
    user = os.getenv("PGUSER", "qe")
    password = os.getenv("PGPASSWORD", "qe")
    dbname = os.getenv("PGDATABASE", "qe")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "dbname": dbname,
    }


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(**_connection_kwargs(), autocommit=False)
    try:
        yield conn
        conn.commit()
    except:  # noqa: E722
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_manifest(path: str) -> str | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_hash FROM ingest_manifest WHERE path = %s",
                (path,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def upsert_manifest(path: str, document_hash: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_manifest(path, document_hash)
                VALUES (%s, %s)
                ON CONFLICT (path)
                DO UPDATE SET document_hash = EXCLUDED.document_hash,
                              updated_at = now()
                """,
                (path, document_hash),
            )


def delete_manifest(path: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ingest_manifest WHERE path = %s", (path,))


def get_manifest_entries() -> dict[str, str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT path, document_hash FROM ingest_manifest")
            rows = cur.fetchall()
            return {row[0]: row[1] for row in rows}


def fetch_chunk_cache(strategy: str, document_hash: str) -> list[dict] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunks
                FROM chunk_cache
                WHERE strategy = %s AND document_hash = %s
                """,
                (strategy, document_hash),
            )
            row = cur.fetchone()
            return row[0] if row else None


def save_chunk_cache(strategy: str, document_hash: str, chunks: Sequence[dict]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunk_cache(strategy, document_hash, chunks)
                VALUES (%s, %s, %s)
                ON CONFLICT (strategy, document_hash)
                DO UPDATE SET chunks = EXCLUDED.chunks,
                              updated_at = now()
                """,
                (strategy, document_hash, Json(list(chunks))),
            )


def delete_chunk_cache(strategy: str, document_hash: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunk_cache WHERE strategy = %s AND document_hash = %s",
                (strategy, document_hash),
            )


def get_manifest_entries_under_prefix(path_prefix: str) -> dict[str, str]:
    """Return manifest entries whose path starts with the given prefix.

    Notes:
      - `ingest_manifest.path` stores whatever string was passed by the ingest script.
        In practice it is the stringified `Path` (often relative to repo root).
      - We do a simple SQL prefix match here (`LIKE prefix%`). Callers should ensure
        the prefix format matches the stored path format they want to target.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT path, document_hash
                FROM ingest_manifest
                WHERE path LIKE %s
                """,
                (f"{path_prefix}%",),
            )
            rows = cur.fetchall()
            return {row[0]: row[1] for row in rows}


def delete_manifest_under_prefix(path_prefix: str) -> int:
    """Delete manifest rows whose path starts with the given prefix.

    Returns the number of deleted rows.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ingest_manifest WHERE path LIKE %s",
                (f"{path_prefix}%",),
            )
            return cur.rowcount


def delete_chunk_cache_for_document_hashes(document_hashes: Sequence[str]) -> int:
    """Delete chunk_cache rows for the provided document hashes.

    This removes cache entries for *all* strategies for those documents.
    Returns the number of deleted rows.
    """
    hashes = list(dict.fromkeys(document_hashes))
    if not hashes:
        return 0

    # Use `= ANY(%s)` to avoid string-building an `IN (...)` clause.
    # psycopg will adapt the Python list into a Postgres array.
    sql = "DELETE FROM chunk_cache WHERE document_hash = ANY(%s)"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (hashes,))
            return cur.rowcount
