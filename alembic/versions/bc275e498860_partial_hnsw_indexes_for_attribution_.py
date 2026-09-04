"""Partial HNSW indexes so direction/bureau kNN voting can search only attributed questions.

Both votes join the HNSW candidate stream against a sparse attribution
source (~13-15% of questions). Postgres has no way to restrict an HNSW
walk to only the attributed subset, so at this corpus size it either
walks far past the LIMIT into unattributed candidates (slow, and gets
slower as attributions get rarer relative to the corpus) or abandons the
index for a full-table scan (slow, and gets slower as the corpus grows) —
measured at 83-98ms per vote, both plans converging once neither uses the
index effectively.

A partial HNSW index built only over the attributed rows removes the
sparsity problem entirely: every row in the index already matches, so the
walk needs exactly `LIMIT` steps. Measured after this fix: 3.8-5.7ms per
vote (15-22x), verified against exact brute-force search with zero
mismatches on 60 sample questions (this required raising `hnsw.ef_search`
to pgvector's max of 1000 — a smaller graph needs a larger relative search
effort to reach the same recall as the full index).

`has_direction_attribution`/`has_bureau_attribution` are denormalized flags,
kept in sync on write (see `qe/attributions.py`'s `sync_attribution_flags_for_points`
and `resync_bureau_attribution_flags`/`resync_direction_attribution_flags`, and
`src/lib/attributions/refresh-view.ts`'s `refreshAttributionFlags`) rather than
computed by the index predicate itself, because a partial index's predicate
must be immutable and can't reference other tables.

Both flags read base tables (`question_real_attributions`,
`question_bureau_extract`), not the `question_attributions_all` view —
computing from the view would make the flag depend on the view refresh
succeeding first, and a failed or late refresh would silently write a
permanently-wrong `false`. See `qe/attributions.py`'s module docstring for
the full reasoning.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "bc275e498860"
down_revision: Union[str, Sequence[str], None] = "26480459d027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "vec_questions_opendata",
        sa.Column(
            "has_direction_attribution",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "vec_questions_opendata",
        sa.Column(
            "has_bureau_attribution",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Guarded on the same EXISTS checks: both columns already hold the
    # correct `false` from server_default for every row that isn't
    # attributed, so only the ~13%/~6% of rows that need to flip to `true`
    # get a new tuple version (and an HNSW insert) instead of all ~59k.
    # Reads base tables, not question_attributions_all — see the module
    # docstring for why.
    op.execute("""
        UPDATE vec_questions_opendata v
        SET has_direction_attribution = EXISTS (
              SELECT 1 FROM question_real_attributions qa
              WHERE qa.question_id = v.payload ->> 'question_id'
                AND qa.direction_reelle_id IS NOT NULL
            ),
            has_bureau_attribution = EXISTS (
              SELECT 1 FROM question_real_attributions qa
              WHERE qa.question_id = v.payload ->> 'question_id'
                AND qa.bureau_reel_id IS NOT NULL
            ) OR EXISTS (
              SELECT 1 FROM question_bureau_extract qbe
              WHERE qbe.question_id = v.payload ->> 'question_id'
            )
        WHERE EXISTS (
              SELECT 1 FROM question_real_attributions qa
              WHERE qa.question_id = v.payload ->> 'question_id'
                AND (qa.direction_reelle_id IS NOT NULL OR qa.bureau_reel_id IS NOT NULL)
            )
           OR EXISTS (
              SELECT 1 FROM question_bureau_extract qbe
              WHERE qbe.question_id = v.payload ->> 'question_id'
            )
    """)

    op.execute("""
        CREATE INDEX vec_q_hnsw_direction_idx ON vec_questions_opendata
        USING hnsw (vector vector_cosine_ops)
        WHERE (has_direction_attribution)
    """)
    op.execute("""
        CREATE INDEX vec_q_hnsw_bureau_idx ON vec_questions_opendata
        USING hnsw (vector vector_cosine_ops)
        WHERE (has_bureau_attribution)
    """)

    # Without this, the planner has no statistics on the new columns until
    # autovacuum's own schedule catches up — and empirically, with stale
    # stats it sometimes picks a full-table scan over the partial index it
    # should use (confirmed by re-running EXPLAIN ANALYZE before and after
    # an explicit ANALYZE on this exact table).
    op.execute("ANALYZE vec_questions_opendata")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS vec_q_hnsw_bureau_idx")
    op.execute("DROP INDEX IF EXISTS vec_q_hnsw_direction_idx")
    op.drop_column("vec_questions_opendata", "has_bureau_attribution")
    op.drop_column("vec_questions_opendata", "has_direction_attribution")
