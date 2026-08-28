"""add comment column to correction_feedback

The correcteur orthotypographique in qe-front (`app/correction`) lets
agents accept or reject each suggested typo/orthographic replacement.
Their decisions land in `correction_feedback` — schema
`(original, replacement, context_before, context_after, decision,
created_at)` — but there is no free-text field.

Agents kept asking for a way to add a short note ("bon dans ce
contexte-là, faux ici", "l'usage varie selon le ministère"…), which
would sharply cut the reverse-engineering time when the model
disagrees with expected behaviour. We add a nullable `comment TEXT`
column so the client can attach optional context.

Nullable + no default → strictly additive, safe on the ~1 800 rows
already in prod. The qe-front POST /api/correction/feedback and the
CorrectionClient will populate it in the follow-up qe-front PR; this
migration ships the column standalone so the front change can land
independently without a dependency dance.

Revision ID: a2b3c4d5e6f8
Revises: 087d1c73ddbc
Create Date: 2026-08-27 09:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a2b3c4d5e6f8"
down_revision: Union[str, Sequence[str], None] = "087d1c73ddbc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # `op.add_column` (dialect-abstracted, type-checked) instead of raw
    # SQL. Idempotency handled via an inspector check rather than
    # `IF NOT EXISTS`, which SQLAlchemy's DDL compiler doesn't emit —
    # the precedent set by earlier Atlas Sandbox catch-up migrations
    # is « an additive change should be safe to re-run », and we keep
    # that here without leaving raw SQL in the migration.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("correction_feedback")}
    if "comment" not in columns:
        op.add_column(
            "correction_feedback",
            sa.Column("comment", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Mirror the upgrade's idempotency: `drop_column` raises if the
    # column is already gone, which would trip anyone running
    # `alembic downgrade -1` twice or against an env where the column
    # was never applied.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("correction_feedback")}
    if "comment" in columns:
        op.drop_column("correction_feedback", "comment")
