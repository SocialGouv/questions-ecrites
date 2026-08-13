"""Catch-up for pre-squash databases: give qe-front's attribution table its new name.

Two unrelated tables were both called `question_attributions`, one per
project: this repo's append-only log of inter-ministry reattributions, and
qe-front's direction/bureau ground truth. Only ever one of them existed in
any given database, so the collision stayed invisible.

`62ff467c436e` settles it by creating both under distinct names. That
covers databases created from now on. A database provisioned *before* it
still holds qe-front's ground truth under the old name, while qe-front's
code (`src/db/schema.ts`, the attribution routes) queries
`question_real_attributions` — every attribution route fails at runtime
with `42P01`.

This migration is the catch-up path for those databases, and a no-op
everywhere else:

  - post-squash database: `question_real_attributions` already exists, the
    guard short-circuits. The `question_attributions` sitting next to it is
    the reattribution log, and is left alone — which is why the guard tests
    for the *destination*, not just the source.
  - pre-squash database: the only `question_attributions` present is
    qe-front's ground truth, and it gets renamed.

Nothing is dropped or rewritten either way, so it is reversible in one
statement.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # `to_regclass` resolves through `search_path` and returns NULL when the
    # relation is absent. `pg_tables` would not do: it spans every schema, so
    # an unrelated table of the same name elsewhere makes the guard read the
    # wrong answer, while the bare ALTER below still targets whatever
    # `search_path` resolves to.
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('question_attributions') IS NOT NULL
               AND to_regclass('question_real_attributions') IS NULL
            THEN
                ALTER TABLE question_attributions RENAME TO question_real_attributions;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('question_real_attributions') IS NOT NULL
               AND to_regclass('question_attributions') IS NULL
            THEN
                ALTER TABLE question_real_attributions RENAME TO question_attributions;
            END IF;
        END $$;
        """
    )
