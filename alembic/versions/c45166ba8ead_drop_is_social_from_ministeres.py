"""drop is_social from ministeres

Revision ID: c45166ba8ead
Revises: 3f8a2c1d9b47
Create Date: 2026-03-20

The is_social boolean was a manually-set flag to scope ingestion to social
ministries.  It is replaced by --ministry substring filtering at the CLI
level, which is more flexible and does not require manual DB tagging.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c45166ba8ead"
down_revision: Union[str, Sequence[str], None] = "3f8a2c1d9b47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("ministeres", "is_social")


def downgrade() -> None:
    op.add_column(
        "ministeres",
        sa.Column("is_social", sa.Boolean(), server_default="false", nullable=False),
    )
