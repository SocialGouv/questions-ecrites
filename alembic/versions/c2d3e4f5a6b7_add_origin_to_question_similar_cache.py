"""add origin to question_similar_cache

Addresses a Phase 2 review finding (SocialGouv/qe-front#127): the
reciprocal backward-fill heuristic (docs/llm-judge-caching-plan.md)
writes borrowed `keep` rows for a candidate question without ever
running that question's own retrieve->rerank->judge pipeline. Nothing
distinguished those rows from genuinely-judged ones, so a question that
only ever received reciprocal donations (and, per the precompute
route's target selection, could never become its own precompute target
again once it had ANY cached row) could be served indefinitely from the
full-list fast path with `llm_filter.applied: true` despite never having
been judged from its own perspective.

`origin` records provenance: `'direct'` for a row written by a genuine
pipeline run (live route or precompute route), `'reciprocal'` for a
borrowed row. qe-front's fast path now additionally requires at least
one `'direct'` row before trusting a source's cached list, and the
precompute route's target-selection anti-join now checks for a
`'direct'` row specifically (not "any row"), so a reciprocal-only
question is re-selected for a genuine pipeline run instead of being
permanently skipped.

Defaults to `'direct'`: every pre-existing row (Phase 1's live route
never writes reciprocal entries — that's Phase 2/precompute-only) is a
genuine pipeline result.

Revision ID: c2d3e4f5a6b7
Revises: b4c5d6e7f8a9
Create Date: 2026-09-08
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c2d3e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "b4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_similar_cache",
        sa.Column(
            "origin",
            sa.String(10),
            nullable=False,
            server_default="direct",
        ),
    )
    op.create_check_constraint(
        "question_similar_cache_origin_check",
        "question_similar_cache",
        "origin IN ('direct', 'reciprocal')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "question_similar_cache_origin_check",
        "question_similar_cache",
        type_="check",
    )
    op.drop_column("question_similar_cache", "origin")
