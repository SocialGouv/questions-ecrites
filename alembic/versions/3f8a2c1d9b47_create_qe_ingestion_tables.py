"""create QE ingestion tables

Revision ID: 3f8a2c1d9b47
Revises: 070bc60dc466
Create Date: 2026-03-12

Tables created:
  - ministeres          : référentiel des ministères (id Réponse, intitulé, flag social)
  - questions           : questions écrites AN + Sénat avec réponse éventuelle
  - question_state_changes : historique des changements d'état (EN_COURS → REPONDU, RETIRE…)
  - question_attributions  : historique des ré-attributions / ré-affectations inter-ministères
  - ingest_cursors      : checkpoints de polling (jeton WS, dernière date open data)
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f8a2c1d9b47"
down_revision: Union[str, Sequence[str], None] = "070bc60dc466"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # ------------------------------------------------------------------
    # ministeres
    # Référentiel des ministères tel que connu de l'application Réponse.
    # Peuplé au fil de l'ingestion open data (ministre_depot /
    # ministre_attributaire présents dans chaque question XML), puis
    # enrichi via chercherMembresGouvernement quand le WS sera accessible.
    # is_social est taggé manuellement pour restreindre le périmètre
    # aux ministères santé / solidarités / travail et leurs équivalents
    # historiques.
    # ------------------------------------------------------------------
    op.create_table(
        "ministeres",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("titre_jo", sa.Text(), nullable=False),
        sa.Column("intitule_min", sa.Text(), nullable=False),
        sa.Column("en_fonction", sa.Boolean(), nullable=True),
        sa.Column("date_debut", sa.Date(), nullable=True),
        sa.Column("date_fin", sa.Date(), nullable=True),
        sa.Column("is_social", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # questions
    # Clé primaire : "{SOURCE}-{LEGISLATURE}-{TYPE}-{NUMERO}"
    # ex. "AN-17-QE-12345"
    #
    # Indexation : les deux assemblées ont des schémas différents.
    #   AN    → rubrique, rubrique_ta, analyses (array)
    #   Sénat → titre_senat, themes (array), rubriques_senat (array)
    # Les champs de l'autre chambre sont NULL selon la source.
    #
    # texte_reponse / ministre_reponse_* / date_reponse_jo sont NULL
    # tant que la question est EN_COURS.
    # ------------------------------------------------------------------
    op.create_table(
        "questions",
        # --- identifiant composite sérialisé ---
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        # --- composantes de l'identifiant ---
        sa.Column("numero_question", sa.Integer(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),  # QE, QOSD, QOAD…
        sa.Column("source", sa.Text(), nullable=False),  # AN | SENAT
        sa.Column("legislature", sa.Integer(), nullable=False),
        # --- état ---
        sa.Column("etat_question", sa.Text(), nullable=False),
        # EN_COURS | REPONDU | RETIRE | SIGNALE | CADUQUE | RENOUVELE | CLOTURE_AUTRE
        # --- publication JO ---
        sa.Column("date_publication_jo", sa.Date(), nullable=True),
        sa.Column("page_jo", sa.Integer(), nullable=True),
        # --- ministère dépôt ---
        sa.Column(
            "ministre_depot_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_depot_libelle", sa.Text(), nullable=True),
        # --- ministère attributaire (peut changer via ré-attribution) ---
        sa.Column(
            "ministre_attributaire_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_attributaire_libelle", sa.Text(), nullable=True),
        # --- auteur ---
        sa.Column("auteur_id_mandat", sa.Text(), nullable=True),
        sa.Column("auteur_nom", sa.Text(), nullable=True),
        sa.Column("auteur_prenom", sa.Text(), nullable=True),
        sa.Column("auteur_grp_pol", sa.Text(), nullable=True),
        sa.Column("auteur_circonscription", sa.Text(), nullable=True),
        # --- indexation AN (NULL si source=SENAT) ---
        sa.Column("rubrique", sa.Text(), nullable=True),
        sa.Column("rubrique_ta", sa.Text(), nullable=True),
        sa.Column("analyses", sa.ARRAY(sa.Text()), nullable=True),
        # --- indexation Sénat (NULL si source=AN) ---
        sa.Column("titre_senat", sa.Text(), nullable=True),
        sa.Column("themes", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("rubriques_senat", sa.ARRAY(sa.Text()), nullable=True),
        # --- textes ---
        sa.Column("texte_question", sa.Text(), nullable=False),
        sa.Column("texte_reponse", sa.Text(), nullable=True),
        # --- réponse (NULL si pas encore répondu) ---
        sa.Column(
            "ministre_reponse_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_reponse_libelle", sa.Text(), nullable=True),
        sa.Column("date_reponse_jo", sa.Date(), nullable=True),
        sa.Column("page_reponse_jo", sa.Integer(), nullable=True),
        # --- liens ---
        sa.Column("rappel_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=True),
        # question qui en rappelle une autre (RENOUVELE)
        sa.Column("date_retrait", sa.Date(), nullable=True),
        # --- méta ingestion ---
        sa.Column("ingest_source", sa.Text(), nullable=False),
        # opendata | ws_polling
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_questions_source_legislature_type",
        "questions",
        ["source", "legislature", "type"],
    )
    op.create_index(
        "ix_questions_ministre_attributaire_id",
        "questions",
        ["ministre_attributaire_id"],
    )
    op.create_index(
        "ix_questions_date_publication_jo", "questions", ["date_publication_jo"]
    )
    op.create_index("ix_questions_etat_question", "questions", ["etat_question"])

    # ------------------------------------------------------------------
    # question_state_changes
    # Log append-only des transitions d'état (émises par changerEtatQuestions
    # côté WS, ou détectées par diff lors du polling open data).
    # ------------------------------------------------------------------
    op.create_table(
        "question_state_changes",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column(
            "question_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=False
        ),
        sa.Column("etat", sa.Text(), nullable=False),
        sa.Column("date_modif", sa.Date(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_question_state_changes_question_id",
        "question_state_changes",
        ["question_id"],
    )

    # ------------------------------------------------------------------
    # question_attributions
    # Log append-only des ré-attributions / ré-affectations inter-ministères
    # (type REATTRIBUTION ou REAFFECTATION dans le WS Réponse).
    # Garde la trace des changements de ministère attributaire au fil du temps.
    # ------------------------------------------------------------------
    op.create_table(
        "question_attributions",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column(
            "question_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=False
        ),
        sa.Column(
            "type_attribution", sa.Text(), nullable=False
        ),  # REATTRIBUTION | REAFFECTATION
        sa.Column(
            "attributaire_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("attributaire_libelle", sa.Text(), nullable=True),
        sa.Column("date_attribution", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_question_attributions_question_id", "question_attributions", ["question_id"]
    )

    # ------------------------------------------------------------------
    # ingest_cursors
    # Checkpoints persistés pour les deux pipelines :
    #   - "opendata_an" / "opendata_senat" : dernière date de JO ingérée
    #   - "ws_questions_an" / "ws_questions_senat" : dernier jeton consommé
    # ------------------------------------------------------------------
    op.create_table(
        "ingest_cursors",
        sa.Column("cursor_name", sa.Text(), primary_key=True, nullable=False),
        sa.Column("jeton", sa.Text(), nullable=True),
        sa.Column("last_date", sa.Date(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("ingest_cursors")
    op.drop_index(
        "ix_question_attributions_question_id", table_name="question_attributions"
    )
    op.drop_table("question_attributions")
    op.drop_index(
        "ix_question_state_changes_question_id", table_name="question_state_changes"
    )
    op.drop_table("question_state_changes")
    op.drop_index("ix_questions_etat_question", table_name="questions")
    op.drop_index("ix_questions_date_publication_jo", table_name="questions")
    op.drop_index("ix_questions_ministre_attributaire_id", table_name="questions")
    op.drop_index("ix_questions_source_legislature_type", table_name="questions")
    op.drop_table("questions")
    op.drop_table("ministeres")
