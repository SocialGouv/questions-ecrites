"""Add pgvector extension and three vector store tables.

Replaces the external Qdrant collections:
  office_responsibilities  →  vec_office_responsibilities
  questions_opendata       →  vec_questions_opendata
  answers_opendata         →  vec_answers_opendata

Revision ID: c0d1e2f3a4b5
Revises: b3c4d5e6f7a8
Create Date: 2026-05-19
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "c0d1e2f3a4b5"
down_revision = "b3c4d5e6f7a8"
branch_labels = None
depends_on = None

# Must match the _VECTOR_DIM constant in qe/models.py and the embedding model
# in use (BAAI/bge-m3 → 1024 dimensions).
VECTOR_DIM = 1024

_TABLES = [
    "vec_office_responsibilities",
    "vec_questions_opendata",
    "vec_answers_opendata",
]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for table_name in _TABLES:
        op.create_table(
            table_name,
            sa.Column("id", sa.Text(), primary_key=True, nullable=False),
            sa.Column("vector", Vector(VECTOR_DIM), nullable=False),
            sa.Column(
                "payload",
                JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )
        # HNSW index for approximate nearest-neighbour cosine search.
        # m=16, ef_construction=64 are pgvector defaults; increase
        # ef_construction to 128 for higher recall if needed.
        op.execute(
            f"CREATE INDEX {table_name}_hnsw_idx "
            f"ON {table_name} "
            f"USING hnsw (vector vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )


def downgrade() -> None:
    for table_name in reversed(_TABLES):
        op.drop_index(f"{table_name}_hnsw_idx", table_name=table_name)
        op.drop_table(table_name)
    # The vector extension is intentionally NOT dropped — other objects may
    # depend on it and recreation requires superuser privileges.
