"""Add columns for the JO question parser output.

Stores the result of running `qe.analysis.question_parser` on each row
of `questions` (see the companion PR adding the analyser). Ported from
the Drizzle migration originally opened as SocialGouv/qe-front#102:
qe-front has since dropped its Drizzle setup ("let backend handle it"),
so schema changes belong here.

None of these columns are required for the app to work — they are
additive structured metadata. NULL means "not yet analysed" or "the
pattern did not match on this row"; consumers must degrade gracefully
to the full `texte_question` when a column is NULL.

`est_rappel` is NOT NULL DEFAULT FALSE on a ~260k-row table, which would
be alarming on an older PostgreSQL. It is not here: since 11, a DEFAULT
that is not volatile is recorded in the catalog and materialised lazily
on read, so ADD COLUMN stays metadata-only — ACCESS EXCLUSIVE is held for
that catalog update alone, with no table rewrite and no scan. We run 16.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "62ff467c436e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE questions
          ADD COLUMN IF NOT EXISTS contexte_extrait  TEXT,
          ADD COLUMN IF NOT EXISTS question_extraite TEXT,
          ADD COLUMN IF NOT EXISTS est_rappel        BOOLEAN NOT NULL DEFAULT FALSE,
          ADD COLUMN IF NOT EXISTS analyzed_at       TIMESTAMPTZ
        """
    )
    # Cheap partial index on the flag: lets us skip rappels from the
    # embedding batch and from stats queries.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_questions_est_rappel
          ON questions (est_rappel)
          WHERE est_rappel = TRUE
        """
    )
    # Partial index used by the backfill mode of the analyser to find rows
    # still needing parsing. We index `id`, not `analyzed_at`: every row
    # matching the predicate has analyzed_at = NULL, so indexing the column
    # itself would carry zero information, whereas indexing the PK lets the
    # planner do an index-only scan on
    # `SELECT id FROM questions WHERE analyzed_at IS NULL`.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_questions_analyzed_at_null
          ON questions (id)
          WHERE analyzed_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_questions_analyzed_at_null")
    op.execute("DROP INDEX IF EXISTS idx_questions_est_rappel")
    op.execute(
        """
        ALTER TABLE questions
          DROP COLUMN IF EXISTS analyzed_at,
          DROP COLUMN IF EXISTS est_rappel,
          DROP COLUMN IF EXISTS question_extraite,
          DROP COLUMN IF EXISTS contexte_extrait
        """
    )
