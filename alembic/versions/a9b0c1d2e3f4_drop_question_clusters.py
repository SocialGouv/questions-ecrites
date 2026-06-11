"""drop question_clusters table

Revision ID: a9b0c1d2e3f4
Revises: f2a3b4c5d6e7
Create Date: 2026-06-11

The question_clusters table was introduced for a clustering feature that was
never completed. save_clusters() is never called by any active script, and no
API endpoint reads from this table. Dropping it to remove dead schema.
"""

import sqlalchemy as sa

from alembic import op

revision = "a9b0c1d2e3f4"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("idx_question_clusters_cluster_id", table_name="question_clusters")
    op.drop_table("question_clusters")


def downgrade() -> None:
    op.create_table(
        "question_clusters",
        sa.Column(
            "question_id",
            sa.Text,
            sa.ForeignKey("questions.id"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("cluster_id", sa.Integer, nullable=False),
        sa.Column("similarity_to_centroid", sa.Float, nullable=False),
    )
    op.create_index(
        "idx_question_clusters_cluster_id",
        "question_clusters",
        ["cluster_id"],
    )
