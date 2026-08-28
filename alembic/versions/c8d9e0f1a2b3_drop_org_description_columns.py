"""drop description columns from directions and bureaux

The `description` texts ("Attribuer ici les questions sur…") were
written for the retired responsibilities-similarity attribution
feature. Since its removal, no code reads them — the only surface that
touched them was the admin Organisation page's display/edit form, whose
help text still (wrongly) claimed they fed the attribution algorithms.

Product call: clean the schema so what remains is what the system
actually uses. The curated texts are NOT lost — they are archived,
verbatim, in `docs/perimetres_directions_bureaux.md` (committed
alongside this migration), available for future reuse such as tooltips
on the attribution pages.

Also drops `directions_desc_backup`, a one-off safety copy of the
direction descriptions taken before a past rewrite — superseded by the
markdown archive.

Deploy order: qe-front must deploy its side FIRST (stop selecting the
`description` columns in the organisation routes / Drizzle schema),
otherwise those routes fail with UndefinedColumn once this migration
runs.

Revision ID: c8d9e0f1a2b3
Revises: b7c8d9e0f1a2
Create Date: 2026-08-28 15:10:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d9e0f1a2b3"
down_revision: Union[str, Sequence[str], None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(bind: sa.engine.Connection, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    if "description" in _columns(bind, "directions"):
        op.drop_column("directions", "description")
    if "description" in _columns(bind, "bureaux"):
        op.drop_column("bureaux", "description")
    op.execute("DROP TABLE IF EXISTS directions_desc_backup")


def downgrade() -> None:
    """Downgrade schema."""
    # Columns come back empty — the content lives in
    # docs/perimetres_directions_bureaux.md and would need a manual
    # re-import. directions_desc_backup is not recreated (it was itself
    # a one-off snapshot).
    bind = op.get_bind()
    if "description" not in _columns(bind, "directions"):
        op.add_column("directions", sa.Column("description", sa.Text(), nullable=True))
    if "description" not in _columns(bind, "bureaux"):
        op.add_column("bureaux", sa.Column("description", sa.Text(), nullable=True))
