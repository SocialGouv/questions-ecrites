"""Add vec_questions_experiments for A/B embedding experiments.

Separate table from `vec_questions_opendata` so tests don't corrupt live
similarity results. Variants coexist here and are distinguished by
`payload ->> 'variant_tag'`. Same schema and HNSW index as the
production table for compatibility.

Revision ID: b1c2d3e4f501
Revises: a9b0c1d2e3f4
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "b1c2d3e4f501"
down_revision = "a9b0c1d2e3f4"
branch_labels = None
depends_on = None

# Must match _VECTOR_DIM in qe/models.py and the embedding model in use.
VECTOR_DIM = 1024
TABLE_NAME = "vec_questions_experiments"


def upgrade() -> None:
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("vector", Vector(VECTOR_DIM), nullable=False),
        sa.Column(
            "payload",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.execute(
        f"CREATE INDEX {TABLE_NAME}_hnsw_idx "
        f"ON {TABLE_NAME} "
        f"USING hnsw (vector vector_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64)"
    )
    # Frequently filtered on at eval time to isolate a specific variant.
    op.execute(
        f"CREATE INDEX {TABLE_NAME}_variant_tag_idx "
        f"ON {TABLE_NAME} ((payload ->> 'variant_tag'))"
    )


def downgrade() -> None:
    op.drop_index(f"{TABLE_NAME}_variant_tag_idx", table_name=TABLE_NAME)
    op.drop_index(f"{TABLE_NAME}_hnsw_idx", table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)
