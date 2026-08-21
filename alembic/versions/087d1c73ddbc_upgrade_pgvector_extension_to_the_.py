"""upgrade pgvector extension to the version bundled in the postgres image

qe-front's attribution routes (see `src/lib/similarity/knn-vote.ts`,
`withHnswSearch`) run `SET LOCAL hnsw.iterative_scan = strict_order`, a GUC
pgvector only exposes from 0.8.0 onward. On Atlas Sandbox the `vector`
extension was created (at bootstrap time, `postInitApplicationSQL`) when
the postgres image only bundled pgvector 0.6.0 -- `CREATE EXTENSION`
installed 0.6.0 and it was never revisited, so every attribution query
fails with "unrecognized configuration parameter" even once the
`question_attributions_all` view exists.

`ALTER EXTENSION vector UPDATE` (no target version) upgrades to whatever
version is bundled in the *currently running* postgres image, alongside
the `imageName` bump in `data-platform-argocd/questions-ecrites/postgres.yaml`
(confirmed locally: `ghcr.io/cloudnative-pg/postgresql:16.8` bundles
pgvector 0.8.0, vs. 0.6.0 in the `16.1` this cluster runs today).

This statement is a safe no-op in two cases that matter here:
  - already at the latest bundled version (confirmed locally: emits a
    NOTICE, no error, `extversion` unchanged);
  - the postgres image hasn't been bumped yet, i.e. `pg_available_extension_versions`
    still tops out at 0.6.0 -- `UPDATE` then targets 0.6.0, which is also a
    no-op. This migration is safe to ship and deploy independently of the
    image bump landing; whichever deploy (this migration's image, or the
    postgres image bump) lands second is the one that actually performs
    the upgrade.

Revision ID: 087d1c73ddbc
Revises: 6cee3d0e7f23
Create Date: 2026-08-21 18:41:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "087d1c73ddbc"
down_revision: Union[str, Sequence[str], None] = "6cee3d0e7f23"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER EXTENSION vector UPDATE")


def downgrade() -> None:
    """Downgrade schema."""
    # Downgrading a shared, cluster-wide extension version on request of a
    # single app's migration history is not something we want to automate:
    # other tables' hnsw indexes may already depend on behaviour from the
    # newer version. No-op, same reasoning as 6cee3d0e7f23's downgrade.
    pass
