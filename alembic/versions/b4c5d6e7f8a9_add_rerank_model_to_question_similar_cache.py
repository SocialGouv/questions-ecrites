"""add rerank_model to question_similar_cache

Fixes a merge mishap: the `rerank_model` column (score depends on the
rerank model, not just the judge model) was added to
`a1b2c3d4e5f6_add_question_similar_cache.py` on the feature branch, but
that fix was committed after SocialGouv/questions-ecrites#69 had already
merged, so it never made it into the migration actually shipped — the
column silently never existed. qe-front's `pgTable` mirror
(src/db/schema.ts) has always had `rerank_model`, so every read/write
against this table (both the live `/similar` route and the Phase 2
precompute route) has been failing with "column rerank_model does not
exist" against any DB actually migrated to `a1b2c3d4e5f6`.

Added with a temporary empty-string server_default so the column can be
NOT NULL from the start without failing on any pre-existing row — in
practice there should be none, since every prior insert into this table
already required `rerank_model` and would have errored out before this
fix. The default is dropped right after backfilling, matching the
original migration's intent (every future insert must supply it
explicitly).

Revision ID: b4c5d6e7f8a9
Revises: a1b2c3d4e5f6
Create Date: 2026-09-08
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "b4c5d6e7f8a9"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_similar_cache",
        sa.Column(
            "rerank_model",
            sa.String(100),
            nullable=False,
            server_default="",
        ),
    )
    op.alter_column("question_similar_cache", "rerank_model", server_default=None)

    # Recreate the full-list index with rerank_model folded in, matching
    # what the column's docstring always claimed: a reranker change must
    # start a cold key, not mix score scales under the same index entry.
    op.drop_index(
        "question_similar_cache_fulllist_idx", table_name="question_similar_cache"
    )
    op.create_index(
        "question_similar_cache_fulllist_idx",
        "question_similar_cache",
        ["source_question_id", "model", "rerank_model", "verdict", "score"],
    )


def downgrade() -> None:
    op.drop_index(
        "question_similar_cache_fulllist_idx", table_name="question_similar_cache"
    )
    op.create_index(
        "question_similar_cache_fulllist_idx",
        "question_similar_cache",
        ["source_question_id", "model", "verdict", "score"],
    )
    op.drop_column("question_similar_cache", "rerank_model")
