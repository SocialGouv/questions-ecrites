"""merge heads: JO parser columns + office-attribution removal

`main` had two divergent heads (`alembic heads` failing with "Multiple head
revisions are present") since the JO-parser-columns branch (`f1a2b3c4d5e6`)
and the office-attribution-removal branch (`f972851671ad`) both merged
independently without reconciling the migration DAG — both branch off
`b2c3d4e5f6a7` and neither touches an object the other cares about, so a
plain no-op merge revision is sufficient; there's no DDL to reconcile.

Revision ID: 4bcdeece2262
Revises: f1a2b3c4d5e6, f972851671ad
Create Date: 2026-08-21 18:37:08.619499

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "4bcdeece2262"
down_revision: Union[str, Sequence[str], None] = ("f1a2b3c4d5e6", "f972851671ad")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
