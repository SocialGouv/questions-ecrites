"""drop vec_office_responsibilities and question_attribution_suggestions

The office-attribution feature (ingest office-responsibility XLSX -> embed
-> match questions to offices) never had a real data source: no XLSX ever
shipped in the repo, so vec_office_responsibilities stayed empty since it
was created and the ingestion script was already removed as dead code (see
"chore: remove unused ingest_office_responsibilities.py script"). Dropping
the table and its two consumer API endpoints (/attributions,
/direction-attributions) retires the feature entirely rather than leaving
a wired-but-empty path.

question_attribution_suggestions is qe-front's cache of that same backend
service's suggestions (used only by its now-removed /admin/performance
attribution-bureau/attribution-direction pages), so it's retired alongside
it -- no matching SQLAlchemy model here, same as other qe-front-owned
tables (see 62ff467c436e_init_schema.py).

Revision ID: f972851671ad
Revises: b2c3d4e5f6a7
Create Date: 2026-08-21 16:08:25.859874

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f972851671ad"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_VECTOR_DIM = 1024


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(
        "vec_office_responsibilities_hnsw_idx",
        table_name="vec_office_responsibilities",
        if_exists=True,
    )
    op.drop_table("vec_office_responsibilities", if_exists=True)
    op.drop_table("question_attribution_suggestions", if_exists=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "vec_office_responsibilities",
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("vector", Vector(_VECTOR_DIM), nullable=False),
        sa.Column(
            "payload",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.execute(
        "CREATE INDEX vec_office_responsibilities_hnsw_idx "
        "ON vec_office_responsibilities "
        "USING hnsw (vector vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.create_table(
        "question_attribution_suggestions",
        sa.Column("question_id", sa.String(100), primary_key=True, nullable=False),
        sa.Column("top1_office_id", sa.String(50), nullable=True),
        sa.Column("top2_office_id", sa.String(50), nullable=True),
        sa.Column("top3_office_id", sa.String(50), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
