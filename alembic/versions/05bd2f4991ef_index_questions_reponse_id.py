"""index questions.reponse_id

`questions.reponse_id` was only a foreign key, with no index. Every lookup
of an answer's questions (`Reponse.questions` selectin loading in
qe/answer_embedding.py, the `allotissements_jo` view join) was a sequential
scan of `questions`.

Built CONCURRENTLY so the deploy never blocks writes to `questions`. That
cannot run inside a transaction, hence the autocommit block. A CONCURRENTLY
build that is interrupted leaves an INVALID index behind, which
`IF NOT EXISTS` would then keep forever, so a leftover invalid index is
dropped first.

Revision ID: 05bd2f4991ef
Revises: 854864d579de
Create Date: 2026-09-25
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "05bd2f4991ef"
down_revision: Union[str, Sequence[str], None] = "854864d579de"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        invalid = op.get_bind().scalar(
            sa.text(
                "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE c.relname = 'ix_questions_reponse_id' AND NOT i.indisvalid"
            )
        )
        if invalid:
            op.execute("DROP INDEX CONCURRENTLY ix_questions_reponse_id")
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_questions_reponse_id "
            "ON questions (reponse_id)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_questions_reponse_id")
