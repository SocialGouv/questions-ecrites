"""add query indices on questions table

Revision ID: d7e8f9a0b1c2
Revises: c45166ba8ead
Create Date: 2026-03-25

Adds composite and GIN trigram indices covering the main query patterns:
list filtering (etat + ministry + date), ministry dropdown, and ILIKE
searches on texte_question, auteur_nom, auteur_prenom.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "d7e8f9a0b1c2"
down_revision: Union[str, Sequence[str], None] = "c45166ba8ead"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_questions_etat_min_date
        ON questions (etat_question, ministre_attributaire_libelle, date_publication_jo DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_questions_min_attributaire
        ON questions (ministre_attributaire_libelle)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_questions_texte_trgm
        ON questions USING GIN (texte_question gin_trgm_ops)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_questions_auteur_nom_trgm
        ON questions USING GIN (auteur_nom gin_trgm_ops)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_questions_auteur_prenom_trgm
        ON questions USING GIN (auteur_prenom gin_trgm_ops)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_questions_auteur_prenom_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_auteur_nom_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_texte_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_min_attributaire")
    op.execute("DROP INDEX IF EXISTS idx_questions_etat_min_date")
    # pg_trgm extension is intentionally NOT dropped — it may be used elsewhere
