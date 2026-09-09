"""pgvector-backed VectorStore implementation.

Uses the existing PostgreSQL connection managed by qe.db — no additional
connection config required.

Collection name → SQL table mapping
------------------------------------
questions_opendata       →  vec_questions_opendata
answers_opendata         →  vec_answers_opendata

Point dict shapes
-----------------
Upsert input:  {"id": str, "vector": list[float], "payload": dict}
Search output: {"id": str, "score": float, "payload": dict}
               score is cosine similarity in [0, 1]
Scroll/get:    {"id": str, "vector": list[float], "payload": dict}
               (vector key absent when with_vectors=False)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence, cast

import sqlalchemy as sa
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult

from qe import db
from qe.models import (
    AnswersOpendataVec,
    QuestionsOpendataVec,
)

logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    "questions_opendata": QuestionsOpendataVec,
    "answers_opendata": AnswersOpendataVec,
}

# pgvector reserves the `hnsw` GUC prefix, so `SET LOCAL hnsw.iterative_scan`
# raises "unrecognized configuration parameter" (aborting the transaction)
# on an extension older than 0.8, where the GUC doesn't exist yet. Some
# environments (e.g. Atlas Sandbox before migration 087d1c73ddbc lands)
# are still on 0.6.0 -- probe rather than catching the error, since a
# failed statement can't be recovered from within the same transaction
# without a savepoint. Cached with a TTL (not for the process lifetime):
# a long-running server started before an extension upgrade migration
# lands would otherwise cache the pre-upgrade answer forever and never
# pick up the fix without a restart.
_ITERATIVE_SCAN_MIN_VERSION = (0, 8)
_ITERATIVE_SCAN_PROBE_TTL_SECONDS = float(
    os.environ.get("PGVECTOR_VERSION_PROBE_TTL_SECONDS", "300")
)
_supports_iterative_scan: bool | None = None
_supports_iterative_scan_probed_at: float = float("-inf")


def _pgvector_supports_iterative_scan(session: Any) -> bool:
    """`session` only needs `.execute(...).scalar()` — typed loosely (not
    `Session`) so a lightweight fake can stand in for tests without a
    database, since SQLAlchemy's real `Session.execute` overloads don't
    structurally match a narrower Protocol."""
    global _supports_iterative_scan, _supports_iterative_scan_probed_at
    now = time.monotonic()
    stale = (
        now - _supports_iterative_scan_probed_at
    ) >= _ITERATIVE_SCAN_PROBE_TTL_SECONDS
    if _supports_iterative_scan is None or stale:
        version = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        parts = tuple(int(p) for p in (version or "0.0").split(".")[:2])
        _supports_iterative_scan = parts >= _ITERATIVE_SCAN_MIN_VERSION
        _supports_iterative_scan_probed_at = now
        if not _supports_iterative_scan:
            logger.warning(
                "pgvector extension %s does not support hnsw.iterative_scan "
                "(needs >= %s) -- search() falls back to the ef_search-capped "
                "legacy behaviour (see migration 087d1c73ddbc).",
                version,
                ".".join(str(p) for p in _ITERATIVE_SCAN_MIN_VERSION),
            )
    return _supports_iterative_scan


def _resolve(name: str):
    model = _COLLECTION_MAP.get(name)
    if model is None:
        known = ", ".join(f"'{k}'" for k in _COLLECTION_MAP)
        raise ValueError(f"Unknown collection '{name}'. Known collections: {known}.")
    return model


def _build_where_clause(model, filter_payload: dict) -> sa.ColumnElement[bool] | None:
    """Translate a filter dict into a SQLAlchemy WHERE clause.

    Supported patterns (the only ones used in this codebase):
      must  / key+match  →  payload ->> key = value
      must  / has_id     →  id IN (...)
      must_not / has_id  →  id NOT IN (...)
    """
    clauses: list[sa.ColumnElement[bool]] = []

    for clause in filter_payload.get("must", []):
        if "key" in clause and "match" in clause:
            key = clause["key"]
            match = clause["match"]
            if "value" in match:
                clauses.append(model.payload[key].astext == str(match["value"]))
            elif "any" in match:
                values = [str(v) for v in match["any"]]
                clauses.append(model.payload[key].astext.in_(values))
        elif "has_id" in clause:
            clauses.append(model.id.in_(clause["has_id"]))

    for clause in filter_payload.get("must_not", []):
        if "has_id" in clause:
            clauses.append(model.id.not_in(clause["has_id"]))

    if not clauses:
        return None
    return sa.and_(*clauses)


class PgvectorClient:
    """VectorStore implementation backed by PostgreSQL + pgvector.

    Tables are created by the Alembic migration
    ``62ff467c436e_init_schema``. Run ``alembic upgrade head`` before
    using this client.
    """

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self, name: str) -> bool:
        """Return True if the backing SQL table exists."""
        model = _COLLECTION_MAP.get(name)
        if model is None:
            return False
        with db.get_session() as session:
            row = session.execute(
                text("SELECT 1 FROM pg_tables WHERE tablename = :t"),
                {"t": model.__tablename__},
            ).scalar()
        return row is not None

    def get_vector_size(self, name: str) -> int | None:
        """Return the vector dimension from the column definition, or None."""
        model = _COLLECTION_MAP.get(name)
        if model is None:
            return None
        with db.get_session() as session:
            row = session.execute(
                text(
                    """
                    SELECT (regexp_match(
                        format_type(atttypid, atttypmod),
                        'vector\\((\\d+)\\)'
                    ))[1]::int
                    FROM   pg_attribute
                    JOIN   pg_class ON pg_class.oid = pg_attribute.attrelid
                    WHERE  pg_class.relname = :table
                      AND  pg_attribute.attname = 'vector'
                    """
                ),
                {"table": model.__tablename__},
            ).scalar()
        return int(row) if row is not None else None

    def create_collection(self, name: str, vector_size: int) -> None:
        """No-op if the table exists with the correct dimension.

        Raises RuntimeError if the table is absent (run ``alembic upgrade head``).
        Raises ValueError if the existing dimension does not match vector_size.
        """
        existing_dim = self.get_vector_size(name)
        if existing_dim is None:
            raise RuntimeError(
                f"Collection '{name}' does not exist. "
                "Run 'poetry run alembic upgrade head' to create the vector tables."
            )
        if existing_dim != vector_size:
            raise ValueError(
                f"Collection '{name}' exists with dimension {existing_dim} "
                f"but the embedding model produces dimension {vector_size}. "
                "Update the _VECTOR_DIM constant in qe/models.py and create "
                "a new Alembic migration, or switch to the correct embedding model."
            )

    def delete_collection(self, name: str) -> bool:
        """Delete all rows from the collection table (truncate). Returns True if any rows were deleted."""
        model = _COLLECTION_MAP.get(name)
        if model is None:
            return False
        with db.get_session() as session:
            result = session.execute(delete(model))
            return (cast(CursorResult, result).rowcount or 0) > 0

    # ------------------------------------------------------------------
    # Point reads
    # ------------------------------------------------------------------

    def get_point(
        self, name: str, point_id: str, *, with_vectors: bool = False
    ) -> dict | None:
        model = _resolve(name)
        cols = [model.id, model.payload]
        if with_vectors:
            cols.append(model.vector)
        with db.get_session() as session:
            row = session.execute(
                select(*cols).where(model.id == point_id)
            ).one_or_none()
        if row is None:
            return None
        result: dict = {"id": row.id, "payload": row.payload}
        if with_vectors:
            result["vector"] = list(row.vector)
        return result

    def get_points_by_ids(
        self, name: str, ids: list[str], *, with_vectors: bool = True
    ) -> list[dict]:
        model = _resolve(name)
        cols = [model.id, model.payload]
        if with_vectors:
            cols.append(model.vector)
        with db.get_session() as session:
            rows = session.execute(select(*cols).where(model.id.in_(ids))).all()
        results = []
        for row in rows:
            pt: dict = {"id": row.id, "payload": row.payload}
            if with_vectors:
                pt["vector"] = list(row.vector)
            results.append(pt)
        return results

    def scroll_all(
        self,
        collection: str,
        *,
        filter: dict | None = None,
        with_vectors: bool = True,
        batch_size: int = 100,
    ) -> list[dict]:
        """Return all points from the table, optionally filtered.

        batch_size is accepted for interface compatibility but ignored — a
        single SQL query is more efficient than paginated scrolling.
        """
        model = _resolve(collection)
        cols = [model.id, model.payload]
        if with_vectors:
            cols.append(model.vector)
        stmt = select(*cols)
        if filter is not None:
            where = _build_where_clause(model, filter)
            if where is not None:
                stmt = stmt.where(where)
        with db.get_session() as session:
            rows = session.execute(stmt).all()
        results = []
        for row in rows:
            pt: dict = {"id": row.id, "payload": row.payload}
            if with_vectors:
                pt["vector"] = list(row.vector)
            results.append(pt)
        return results

    # ------------------------------------------------------------------
    # Point writes
    # ------------------------------------------------------------------

    def upsert_points(self, name: str, points: list[dict]) -> None:
        """Insert or update points. On conflict on id, overwrites vector and payload."""
        if not points:
            return
        model = _resolve(name)
        stmt = pg_insert(model).values(
            [
                {"id": p["id"], "vector": p["vector"], "payload": p["payload"]}
                for p in points
            ]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"vector": stmt.excluded.vector, "payload": stmt.excluded.payload},
        )
        with db.get_session() as session:
            session.execute(stmt)

    def delete_points_by_filter(self, name: str, filter_payload: dict) -> None:
        """Delete all points matching the filter."""
        model = _resolve(name)
        where = _build_where_clause(model, filter_payload)
        stmt = delete(model)
        if where is not None:
            stmt = stmt.where(where)
        with db.get_session() as session:
            session.execute(stmt)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        top_k: int,
        *,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Return the top_k nearest neighbours by cosine similarity.

        Score mapping: pgvector's <=> operator returns cosine *distance* in
        [0, 2]. BAAI/bge-m3 outputs L2-normalised vectors, so:
            similarity = 1 - distance
        The returned score is cosine similarity ∈ [0, 1].
        """
        model = _resolve(collection)
        query_vec = list(vector)
        dist_expr = model.vector.cosine_distance(query_vec)

        stmt = (
            select(model.id, model.payload, dist_expr.label("dist"))
            .order_by(dist_expr)
            .limit(top_k)
        )

        if score_threshold is not None:
            # similarity >= threshold  ⟺  distance <= 1 - threshold
            stmt = stmt.where(dist_expr <= (1.0 - score_threshold))

        if filter is not None:
            where = _build_where_clause(model, filter)
            if where is not None:
                stmt = stmt.where(where)

        with db.get_session() as session:
            # pgvector >= 0.8 caps an HNSW scan at hnsw.ef_search rows (40 by
            # default) unless iterative scanning is on, so a LIMIT above it
            # silently returns fewer rows. strict_order keeps exact distance
            # order while the walker continues until top_k rows are found.
            if _pgvector_supports_iterative_scan(session):
                session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
                session.execute(
                    text(f"SET LOCAL hnsw.ef_search = {min(1000, max(40, int(top_k)))}")
                )
            rows = session.execute(stmt).all()

        return [
            {"id": row.id, "score": float(1.0 - row.dist), "payload": row.payload}
            for row in rows
        ]
