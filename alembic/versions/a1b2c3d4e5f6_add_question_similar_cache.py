"""add question_similar_cache

Caches the combined rerank+judge output of the qe-front `/similar`
endpoint pipeline, keyed by (source_question_id, candidate_question_id,
model). Verdicts are a pure function of question text + model, and
question text is immutable once published, so entries never need
invalidation — see docs/llm-judge-caching-plan.md in qe-front for the
full design (Phase 1: runtime cache-or-compute; Phase 2: precompute
during ingestion).

Read two ways:
  - full-list read: WHERE source_question_id = ? AND model = ? AND
    verdict = 'keep' ORDER BY score DESC — the fast path, no Albert
    calls at all when it returns enough rows.
  - per-pair lookup: WHERE source_question_id = ? AND
    candidate_question_id = ? AND model = ? — skips redundant
    rerank/judge calls for pairs already known from a previous run or
    a reciprocal write (Phase 2).

Schema ownership rule (root CLAUDE.md): this table is defined here;
qe-front mirrors it as a `pgTable` in src/db/schema.ts for querying
only.

Revision ID: a1b2c3d4e5f6
Revises: bc275e498860
Create Date: 2026-09-04
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "bc275e498860"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_similar_cache",
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("candidate_question_id", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("verdict", sa.String(10), nullable=False),
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
            "verdict IN ('keep', 'drop')",
            name="question_similar_cache_verdict_check",
        ),
    )
    # Backs the full-list read: all keep-verdicts for a source+model,
    # ordered by score.
    op.create_index(
        "question_similar_cache_fulllist_idx",
        "question_similar_cache",
        ["source_question_id", "model", "verdict", "score"],
    )


def downgrade() -> None:
    op.drop_index(
        "question_similar_cache_fulllist_idx", table_name="question_similar_cache"
    )
    op.drop_table("question_similar_cache")
