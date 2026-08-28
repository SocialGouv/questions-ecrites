"""add statut + min15_key columns to bureaux

The `bureaux` referential currently holds only the 36 DGCS/DSS bureaus
seeded from the org-chart documents ("qui fait quoi") provided by those
two directions. Meanwhile the MIN15 extracts surface ~127 additional
bureau keys (all of DGOS/DGS coverage) that live as free text in the
`question_attributions_all` view — invisible in the admin Organisation
page and impossible to rename or curate.

`scripts/sync_bureaux_from_min15.py` (added alongside this migration)
promotes the legitimate MIN15-discovered bureaus into `bureaux` so the
referential becomes the single home of every bureau the application can
display. This migration adds the two columns that sync needs:

- `statut` — 'organigramme' for the existing curated rows (backfilled),
  'min15' for rows auto-created from the extracts. Lets the admin UI
  badge the provenance and lets future imports treat curated rows as
  read-only ground truth.
- `min15_key` — the normalized key ('SDSP/SP3', 'SDRH1'…) that the
  attribution view derives from `poste_etape`. Unique, nullable: only
  rows reachable from MIN15 data carry one. It is the idempotency
  anchor for the sync (re-running never duplicates) and the future
  join point if `question_attributions_all` moves from text keys to
  referential ids.

Revision ID: b7c8d9e0f1a2
Revises: a2b3c4d5e6f8
Create Date: 2026-08-28 12:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, Sequence[str], None] = "a2b3c4d5e6f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _bureaux_columns(bind: sa.engine.Connection) -> set[str]:
    inspector = sa.inspect(bind)
    return {c["name"] for c in inspector.get_columns("bureaux")}


def upgrade() -> None:
    """Upgrade schema."""
    columns = _bureaux_columns(op.get_bind())
    if "statut" not in columns:
        # server_default so the 36 existing rows are backfilled to
        # 'organigramme' in the same statement — they all predate MIN15
        # sync by construction.
        op.add_column(
            "bureaux",
            sa.Column(
                "statut",
                sa.Text(),
                nullable=False,
                server_default="organigramme",
            ),
        )
    if "min15_key" not in columns:
        op.add_column("bureaux", sa.Column("min15_key", sa.Text(), nullable=True))
        op.create_unique_constraint(
            "bureaux_min15_key_unique", "bureaux", ["min15_key"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    columns = _bureaux_columns(op.get_bind())
    if "min15_key" in columns:
        op.drop_constraint("bureaux_min15_key_unique", "bureaux", type_="unique")
        op.drop_column("bureaux", "min15_key")
    if "statut" in columns:
        op.drop_column("bureaux", "statut")
