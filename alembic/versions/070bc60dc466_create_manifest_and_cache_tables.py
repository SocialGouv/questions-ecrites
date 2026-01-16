"""create manifest and cache tables

Revision ID: 070bc60dc466
Revises:
Create Date: 2026-01-26 21:17:54.657492

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "070bc60dc466"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingest_manifest",
        sa.Column("path", sa.Text(), primary_key=True, nullable=False),
        sa.Column("document_hash", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "chunk_cache",
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("document_hash", sa.Text(), nullable=False),
        sa.Column("chunks", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("strategy", "document_hash"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("chunk_cache")
    op.drop_table("ingest_manifest")
