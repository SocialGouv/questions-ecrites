from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import psycopg
from psycopg.types.json import Json


def _connection_kwargs() -> dict:
    url = os.getenv("DATABASE_DSN")
    if url:
        return {"conninfo": url}
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
