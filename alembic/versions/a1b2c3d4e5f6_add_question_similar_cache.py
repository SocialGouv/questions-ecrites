"""add question_similar_cache

Caches the combined rerank+judge output of the qe-front `/similar`
endpoint pipeline, keyed by (source_question_id, candidate_question_id,
model). Verdicts are a pure function of question text + model, and
question text is immutable once published, so entries never need
invalidation — see docs/llm-judge-caching-plan.md in qe-front for the
full design (Phase 1: runtime cache-or-compute; Phase 2: precompute
during ingestion).

`score` also depends on the rerank model, which is not part of the
primary key — `rerank_model` records it and every read filters on it,
so a reranker change starts a cold key instead of mixing score scales.

Read two ways:
  - full-list read: WHERE source_question_id = ? AND model = ? AND
    rerank_model = ? AND verdict = 'keep' ORDER BY score DESC — the
    fast path, no Albert calls at all when it returns enough rows.
  - per-pair lookup: WHERE source_question_id = ? AND
    candidate_question_id = ? AND model = ? AND rerank_model = ? —
    skips redundant rerank/judge calls for pairs already known from a
    previous run or a reciprocal write (Phase 2).

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
        # Rerank model that produced `score`. Not part of the primary key —
        # a row is overwritten in place when the reranker changes — but
        # every read filters on it, so a reranker change starts a cold key
        # instead of silently mixing score scales.
        sa.Column("rerank_model", sa.String(100), nullable=False),
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
    # Backs the full-list read: all keep-verdicts for a source+model+rerank
    # model, ordered by score.
    op.create_index(
        "question_similar_cache_fulllist_idx",
        "question_similar_cache",
        ["source_question_id", "model", "rerank_model", "verdict", "score"],
    )
    # Backs the ON DELETE CASCADE from questions.id on the candidate side —
    # the PK index and the fulllist index above both lead with
    # source_question_id, so without this a cascading delete on `questions`
    # would sequentially scan this table once per deleted row.
    op.create_index(
        "question_similar_cache_candidate_idx",
        "question_similar_cache",
        ["candidate_question_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "question_similar_cache_candidate_idx", table_name="question_similar_cache"
    )
    op.drop_index(
        "question_similar_cache_fulllist_idx", table_name="question_similar_cache"
    )
    op.drop_table("question_similar_cache")
