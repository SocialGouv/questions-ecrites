"""add question_cluster_runs and question_cluster_members tables

Revision ID: e1f2a3b4c5d6
Revises: d7e8f9a0b1c2
Create Date: 2026-03-25

Stores the output of cluster_questions.py so that a UI can display
questions that are semantically similar to any given question.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "e1f2a3b4c5d6"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "question_cluster_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("mode", sa.Text, nullable=False),
        sa.Column("threshold", sa.Float, nullable=True),
        sa.Column("total_clusters", sa.Integer, nullable=False),
        sa.Column("total_questions", sa.Integer, nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "question_cluster_members",
        sa.Column(
            "run_id",
            sa.Integer,
            sa.ForeignKey("question_cluster_runs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
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

    # fast "get all members of a cluster"
    op.create_index(
        "idx_cluster_members_run_cluster",
        "question_cluster_members",
        ["run_id", "cluster_id"],
    )
    # fast "find all runs containing a question"
    op.create_index(
        "idx_cluster_members_question",
        "question_cluster_members",
        ["question_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_cluster_members_question", table_name="question_cluster_members")
    op.drop_index(
        "idx_cluster_members_run_cluster", table_name="question_cluster_members"
    )
    op.drop_table("question_cluster_members")
    op.drop_table("question_cluster_runs")
