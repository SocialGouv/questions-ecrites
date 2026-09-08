"""add question_similar_precompute_empty

Fixes a permanent-backlog bug in the precompute route (qe-front
app/api/internal/similar-cache/precompute/route.ts): the route's target
selection is a NOT EXISTS anti-join on `question_similar_cache` rows with
`origin = 'direct'` — a question only drops out of the "still missing"
set once a genuine pipeline run writes at least one such row.

But `precomputeOne()` legitimately returns zero cache-worthy rows in
several cases that have nothing to do with failure (no similar
candidates found at all, the source question's text is missing, or no
candidate cleared the display/score threshold after reranking) — and an
empty result writes NOTHING to `question_similar_cache`, so those
questions were re-selected, and their pipeline re-run from scratch, by
every single future batch forever: a permanent, silently-growing tax on
Albert API calls and qe-front load with zero possible progress.

This table records that a (question, model, rerank_model) triple was
genuinely attempted and found nothing to cache, distinct from "never
attempted". The precompute route's target selection now also excludes a
question with a *fresh* row here (see EMPTY_RETRY_TTL_DAYS in route.ts)
— fresh, not permanent, because the candidate pool changes as new
questions get embedded, so an empty result today may not stay empty
forever. A stale row (past the TTL) makes the question eligible again.

Deliberately a separate table rather than a sentinel row in
`question_similar_cache` itself: that table's primary key includes
`candidate_question_id` NOT NULL with a FK to `questions.id`, so an
"empty" marker has no candidate to point to and would need either a
nullable PK column (not supported) or a magic non-question sentinel id
— both messier than a small dedicated table keyed on exactly the
(source_question_id, model, rerank_model) triple the anti-join already
filters on.

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-08
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.drop_table("question_similar_precompute_empty")
