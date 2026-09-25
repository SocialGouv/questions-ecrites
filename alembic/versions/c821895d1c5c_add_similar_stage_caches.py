"""add similar stage caches

One cache per stage of qe-front's `/similar` pipeline, filled by the
precompute job and read by the live route:

  - question_similar_neighbors: retrieve output (top-K candidates and their
    cosine) per (source, pool); pool is the status searched.
    `precomputed_for` is the rerank/judge key the list was completed for,
    reset whenever its candidates change.
  - question_pair_rerank: rerank score per pair and rerank model.
  - question_pair_verdict: judge verdict per pair and judge key
    (model + prompt hash).
  - question_similar_precompute_queue: sources a live request found
    incomplete, drained first by the job.

Schema ownership rule (root CLAUDE.md): defined here, mirrored as
`pgTable`s in qe-front src/db/schema.ts.

Revision ID: c821895d1c5c
Revises: c99252387f8c
Create Date: 2026-09-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c821895d1c5c"
down_revision: Union[str, Sequence[str], None] = "c99252387f8c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

POOLS = "pool IN ('EN_COURS', 'REPONDU')"


def _computed_at() -> sa.Column:
    return sa.Column(
        "computed_at",
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )


def _pair_table(name: str, key: str, value: sa.Column) -> None:
    op.create_table(
        name,
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("candidate_question_id", sa.String(100), nullable=False),
        sa.Column(key, sa.String(150), nullable=False),
        value,
        _computed_at(),
        sa.PrimaryKeyConstraint("source_question_id", "candidate_question_id", key),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["candidate_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
    )
    # The PK leads with source_question_id: without this, a cascading
    # delete on the candidate side scans the table once per deleted row.
    op.create_index(f"{name}_candidate_idx", name, ["candidate_question_id"])


def upgrade() -> None:
    op.create_table(
        "question_similar_neighbors",
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("pool", sa.String(10), nullable=False),
        sa.Column("candidate_ids", postgresql.ARRAY(sa.String(100)), nullable=False),
        sa.Column("cosines", postgresql.ARRAY(sa.REAL), nullable=False),
        sa.Column("retrieve_key", sa.String(200), nullable=False),
        _computed_at(),
        sa.Column("last_viewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("precomputed_for", sa.String(300), nullable=True),
        sa.PrimaryKeyConstraint("source_question_id", "pool"),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint(POOLS, name="question_similar_neighbors_pool_check"),
        sa.CheckConstraint(
            "cardinality(candidate_ids) = cardinality(cosines)",
            name="question_similar_neighbors_arrays_check",
        ),
    )

    _pair_table(
        "question_pair_rerank",
        "rerank_model",
        sa.Column("score", sa.REAL, nullable=False),
    )

    _pair_table(
        "question_pair_verdict",
        "judge_key",
        sa.Column("verdict", sa.String(10), nullable=False),
    )
    op.create_check_constraint(
        "question_pair_verdict_verdict_check",
        "question_pair_verdict",
        "verdict IN ('keep', 'drop')",
    )

    op.create_table(
        "question_similar_precompute_queue",
        sa.Column("source_question_id", sa.String(100), nullable=False),
        sa.Column("pool", sa.String(10), nullable=False),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source_question_id", "pool"),
        sa.ForeignKeyConstraint(
            ["source_question_id"], ["questions.id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint(POOLS, name="question_similar_precompute_queue_pool_check"),
    )


def downgrade() -> None:
    op.drop_table("question_similar_precompute_queue")
    op.drop_table("question_pair_verdict")
    op.drop_table("question_pair_rerank")
    op.drop_table("question_similar_neighbors")
