"""add question_bureau_extract — bureau réel extrait de MIN15

Isolée dans une table dédiée (pas de colonne sur `question_real_attributions`)
pour bien séparer la source. Une ligne par (question_id, direction) pour
supporter les QE réattribuées entre directions — la ligne "dernière
étape rédaction" est celle qui compte pour le bureau final.

Règle d'extraction (implémentée dans scripts/extract_bureau_from_min15.py) :
- type_etape ∈ {'Pour rédaction', 'Pour rédaction interfacée'}
- poste_etape a >= 3 segments splittés sur ' - '
- premier segment ∈ {DGCS, DGOS, DSS, DGS, DGEFP, DGT, DFAS, DGE, ...}
- pour chaque (QE, direction) : garder la ligne la plus récente

BDC (correspondants), CABINET, SGG, DDC sont ignorés — ce sont des étapes
administratives/politiques, pas des rédactions terrain.

Le rapprochement avec `bureaux.id` est différé : le référentiel `bureaux`
est DGCS-first et n'a pas les sous-directions DGOS/DGS/DSS. On stocke
d'abord les libellés bruts + une provenance, on créera le lien FK plus
tard une fois qu'on aura étendu le référentiel.

Revision ID: c1d2e3f4a5b6
Revises: d5a1c2f3e4b6
Create Date: 2026-07-28
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "d5a1c2f3e4b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_bureau_extract",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("question_id", sa.String(100), nullable=False),
        sa.Column(
            "direction_txt", sa.Text(), nullable=False,
            comment="Direction extraite du 1er segment de poste_etape "
                    "(ex 'DGOS', 'DSS'). Non lié à directions.id pour l'instant."
        ),
        sa.Column(
            "sous_direction", sa.Text(), nullable=True,
            comment="2e segment (ex 'SDRH1', 'SD2 A')"
        ),
        sa.Column(
            "bureau", sa.Text(), nullable=True,
            comment="3e segment (ex 'Pharmacie', 'MCGRM', 'Bureau SP5')"
        ),
        sa.Column(
            "bureau_full", sa.Text(), nullable=True,
            comment="Concaténation seg 3+ (ex '3e - 4e - 5e' pour les postes longs)"
        ),
        sa.Column(
            "source_etape_id", sa.Integer(), nullable=True,
            comment="FK vers l'étape MIN15 d'origine (pour provenance/debug)"
        ),
        sa.Column("date_debut_etape", sa.Date(), nullable=True),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_etape_id"], ["reponses_extract_etapes.id"], ondelete="SET NULL"
        ),
        # Une ligne par (QE, direction) : si la QE a été passée à plusieurs
        # directions, on garde la trace de chacune.
        sa.UniqueConstraint("question_id", "direction_txt",
                            name="uq_question_bureau_extract_qid_dir"),
    )
    op.create_index(
        "ix_question_bureau_extract_qid",
        "question_bureau_extract",
        ["question_id"],
    )
    op.create_index(
        "ix_question_bureau_extract_direction",
        "question_bureau_extract",
        ["direction_txt"],
    )
    op.execute(
        "COMMENT ON TABLE question_bureau_extract IS "
        "'Bureau réel extrait des workflows MIN15 (outil Réponses). "
        "Source complémentaire à question_real_attributions.bureau_reel_id "
        "qui n''est peuplé que pour DGCS. Voir scripts/extract_bureau_from_min15.py "
        "pour la règle d''extraction.'"
    )


def downgrade() -> None:
    op.drop_index("ix_question_bureau_extract_direction",
                  table_name="question_bureau_extract")
    op.drop_index("ix_question_bureau_extract_qid",
                  table_name="question_bureau_extract")
    op.drop_table("question_bureau_extract")
