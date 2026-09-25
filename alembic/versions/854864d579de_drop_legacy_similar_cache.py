"""drop legacy similar cache

`question_similar_cache` and `question_similar_precompute_empty` are
superseded by the per-stage caches of c821895d1c5c; qe-front no longer
reads or writes them. Apply only once that qe-front version is deployed.

Revision ID: 854864d579de
Revises: c821895d1c5c
Create Date: 2026-09-25
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "854864d579de"
down_revision: Union[str, Sequence[str], None] = "c821895d1c5c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("question_similar_precompute_empty")
    op.drop_table("question_similar_cache")


def downgrade() -> None:
    op.create_table(
        "question_similar_cache",
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("candidate_question_id", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("rerank_model", sa.String(100), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("verdict", sa.String(10), nullable=False),
        sa.Column("origin", sa.String(10), server_default="direct", nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source_question_id", "candidate_question_id", "model"),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["candidate_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "verdict IN ('keep', 'drop')", name="question_similar_cache_verdict_check"
        ),
        sa.CheckConstraint(
            "origin IN ('direct', 'reciprocal')",
            name="question_similar_cache_origin_check",
        ),
    )
    op.create_index(
        "question_similar_cache_fulllist_idx",
        "question_similar_cache",
        ["source_question_id", "model", "rerank_model", "verdict", "score"],
    )
    op.create_index(
        "question_similar_cache_candidate_idx",
        "question_similar_cache",
        ["candidate_question_id"],
    )
    op.create_table(
        "question_similar_precompute_empty",
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("rerank_model", sa.String(100), nullable=False),
        sa.Column(
            "attempted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source_question_id", "model", "rerank_model"),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
    )
