"""Add expression index on direction field in vec_office_responsibilities.

Revision ID: a8b9c0d1e2f3
Revises: f2a3b4c5d6e7
Create Date: 2026-05-27
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, Sequence[str], None] = "c0d1e2f3a4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_vec_office_responsibilities_direction
        ON vec_office_responsibilities ((payload ->> 'direction'))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_vec_office_responsibilities_direction")
