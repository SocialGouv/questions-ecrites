"""add objet column to questions

Revision ID: a1b2c3d4e5f6
Revises: f2a3b4c5d6e7
Create Date: 2026-03-25

Adds the <Objet> short title field from DILA opendata XML to the questions table.
"""

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "f2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("questions", sa.Column("objet", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("questions", "objet")
