"""add direction_algo_id cache to questions

Persists the last direction top-1 returned by the kNN attribution algo
for each question. Priority read order is:

    COALESCE(question_attributions.direction_reelle_id,
             questions.direction_algo_id)

The cache is (re)written by:
  * a one-shot backfill script over the whole corpus (~260k QE)
  * the attribution-direction API route each time it runs (write-behind)
  * a small backfill script for new questions (cron / post-ingest hook)

Rationale: the "similar questions" pipeline benefits massively from a
direction filter on candidates (+22 pts on hit@3 in leave-one-out eval,
see docs/rapport_performance_v2.md, Expé 9). That filter needs a
direction per question, and running the kNN algo synchronously per
request would be prohibitively slow — hence a cache. This migration
only adds the columns; scripts and wiring live in the qe-front repo.

Revision ID: d5a1c2f3e4b6
Revises: e5f6c7d8a9b1
Create Date: 2026-07-18

Made idempotent after the fact: the `fix/migration-chain-order` rechain
moved this migration from before to after `b2c3d4e5f6a7` in the DAG.
Atlas Sandbox had already run it (under the old order) by the time that
rechain landed, so its stamp (`b2c3d4e5f6a7`) now sits *before* this
migration instead of after it — a plain `alembic upgrade head` would try
to add `direction_algo_id` again and fail with `DuplicateColumn`. Every
statement below is guarded so this is a no-op there and unchanged
everywhere else.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "d5a1c2f3e4b6"
down_revision: Union[str, Sequence[str], None] = "e5f6c7d8a9b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable — most questions will be NULL until the backfill runs.
    op.execute(
        """
        ALTER TABLE questions
          ADD COLUMN IF NOT EXISTS direction_algo_id INTEGER,
          ADD COLUMN IF NOT EXISTS direction_algo_computed_at TIMESTAMPTZ
        """
    )
    # FK constraint explicitly named so re-runs don't create silent
    # duplicates and downgrade() knows what to drop.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'fk_questions_direction_algo_id'
            ) THEN
                ALTER TABLE questions
                    ADD CONSTRAINT fk_questions_direction_algo_id
                    FOREIGN KEY (direction_algo_id)
                    REFERENCES directions(id) ON DELETE SET NULL;
            END IF;
        END $$;
        """
    )
    # Partial index — skip the ~260k NULL rows we'll have between the
    # migration and the first backfill run. The filter predicate we
    # care about is always `direction_algo_id = X` (or IN (…)), so NULL
    # rows in the index are pure overhead.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_questions_direction_algo_id
            ON questions (direction_algo_id)
            WHERE direction_algo_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index("ix_questions_direction_algo_id", table_name="questions")
    op.drop_column("questions", "direction_algo_computed_at")
    op.drop_column("questions", "direction_algo_id")
