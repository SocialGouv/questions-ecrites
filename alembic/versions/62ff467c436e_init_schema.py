"""init schema

Revision ID: 62ff467c436e
Revises:
Create Date: 2026-08-11 23:30:59.098076

Single starting point for this project's Alembic history, replacing the
previous 12-migration chain (2026-01-16 -> 2026-08-11). That history had
accumulated several dead ends (a run-based question_clusters design added
then fully dropped, an is_social flag added then dropped) and mixed two
projects' schema ownership; squashing it into one migration reflecting the
schema as it stands today gives a clean base to build on going forward.

Tables created, in three groups:

  - questions-ecrites domain tables (backed by SQLAlchemy models in
    qe/models.py): ingest_manifest, chunk_cache, ministeres, reponses,
    questions, question_state_changes, question_attributions, ingest_cursors.
  - pgvector store (backed by models in qe/models.py, replacing the former
    Qdrant collections): vec_office_responsibilities, vec_questions_opendata,
    vec_answers_opendata — each with an HNSW cosine index.
  - qe-front's application schema (no SQLAlchemy model here — see env.py's
    include_object filter): directions, sous_directions, bureaux,
    question_real_attributions, question_attribution_suggestions,
    question_similar_suggestions, suggestion_feedback, correction_feedback,
    app_settings, users, accounts, sessions, verificationTokens. Schema
    ownership for the whole shared database lives here; qe-front keeps
    drizzle-orm only as its runtime query builder (see that project's
    src/db/schema.ts and schema-auth.ts, which must match these tables).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "62ff467c436e"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match the _VECTOR_DIM constant in qe/models.py and the embedding model
# in use (BAAI/bge-m3 -> 1024 dimensions).
VECTOR_DIM = 1024

_VEC_TABLES = [
    "vec_office_responsibilities",
    "vec_questions_opendata",
    "vec_answers_opendata",
]


def upgrade() -> None:
    """Upgrade schema."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ------------------------------------------------------------------
    # ingest_manifest / chunk_cache
    # Ingest pipeline pour les fiches de poste (job descriptions) :
    # suivi des fichiers ingérés et cache des découpages en chunks.
    # ------------------------------------------------------------------
    op.create_table(
        "ingest_manifest",
        sa.Column("path", sa.Text(), primary_key=True, nullable=False),
        sa.Column("document_hash", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "chunk_cache",
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("document_hash", sa.Text(), nullable=False),
        sa.Column("chunks", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("strategy", "document_hash"),
    )

    # ------------------------------------------------------------------
    # ministeres
    # Référentiel des ministères tel que connu de l'application Réponse.
    # ------------------------------------------------------------------
    op.create_table(
        "ministeres",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column("titre_jo", sa.Text(), nullable=False),
        sa.Column("intitule_min", sa.Text(), nullable=False),
        sa.Column("en_fonction", sa.Boolean(), nullable=True),
        sa.Column("date_debut", sa.Date(), nullable=True),
        sa.Column("date_fin", sa.Date(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # reponses
    # Réponses aux questions écrites, extraites dans leur propre table
    # pour permettre une réponse partagée par plusieurs questions.
    # ------------------------------------------------------------------
    op.create_table(
        "reponses",
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("source", sa.Text(), nullable=False),  # AN | SENAT
        sa.Column("no_publication", sa.Text(), nullable=False),
        sa.Column("texte_reponse", sa.Text(), nullable=False),
        sa.Column(
            "ministre_reponse_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_reponse_libelle", sa.Text(), nullable=True),
        sa.Column("date_reponse_jo", sa.Date(), nullable=True),
        sa.Column("page_reponse_jo", sa.Integer(), nullable=True),
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

    # ------------------------------------------------------------------
    # questions
    # Clé primaire : "{SOURCE}-{LEGISLATURE}-{TYPE}-{NUMERO}", ex.
    # "AN-17-QE-12345". Les deux assemblées ont des schémas différents :
    #   AN    -> rubrique, rubrique_ta, analyses (array)
    #   Sénat -> titre_senat, themes (array), rubriques_senat (array)
    # Les champs de l'autre chambre sont NULL selon la source.
    # ------------------------------------------------------------------
    op.create_table(
        "questions",
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("numero_question", sa.Integer(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),  # QE, QOSD, QOAD…
        sa.Column("source", sa.Text(), nullable=False),  # AN | SENAT
        sa.Column("legislature", sa.Integer(), nullable=False),
        sa.Column("etat_question", sa.Text(), nullable=False),
        # EN_COURS | REPONDU | RETIRE | SIGNALE | CADUQUE | RENOUVELE | CLOTURE_AUTRE
        sa.Column("date_publication_jo", sa.Date(), nullable=True),
        sa.Column("page_jo", sa.Integer(), nullable=True),
        sa.Column(
            "ministre_depot_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_depot_libelle", sa.Text(), nullable=True),
        sa.Column(
            "ministre_attributaire_id",
            sa.Integer(),
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_attributaire_libelle", sa.Text(), nullable=True),
        sa.Column("auteur_id_mandat", sa.Text(), nullable=True),
        sa.Column("auteur_nom", sa.Text(), nullable=True),
        sa.Column("auteur_prenom", sa.Text(), nullable=True),
        sa.Column("auteur_grp_pol", sa.Text(), nullable=True),
        sa.Column("auteur_circonscription", sa.Text(), nullable=True),
        sa.Column("objet", sa.Text(), nullable=True),  # DILA <Objet>, AN uniquement
        sa.Column("rubrique", sa.Text(), nullable=True),
        sa.Column("rubrique_ta", sa.Text(), nullable=True),
        sa.Column("analyses", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("titre_senat", sa.Text(), nullable=True),
        sa.Column("themes", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("rubriques_senat", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("texte_question", sa.Text(), nullable=False),
        sa.Column("reponse_id", sa.Text(), sa.ForeignKey("reponses.id"), nullable=True),
        sa.Column("rappel_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=True),
        sa.Column("date_retrait", sa.Date(), nullable=True),
        sa.Column("ingest_source", sa.Text(), nullable=False),  # opendata | ws_polling
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

    # Composite/trigram indices covering the main query patterns: list
    # filtering (etat + ministry + date), ministry dropdown, and ILIKE
    # searches on texte_question, auteur_nom, auteur_prenom.
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
    # (type REATTRIBUTION ou REAFFECTATION dans le WS Réponse). Distincte de
    # question_real_attributions (qe-front, vérité terrain direction/bureau) —
    # collision de nom historique entre les deux projets, résolue en gardant
    # les deux tables sous des noms différents.
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
    # Checkpoints persistés pour les pipelines d'ingestion (jeton WS,
    # dernière date open data ingérée).
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

    # ------------------------------------------------------------------
    # vec_office_responsibilities / vec_questions_opendata / vec_answers_opendata
    # Store pgvector remplaçant les anciennes collections Qdrant.
    # ------------------------------------------------------------------
    for table_name in _VEC_TABLES:
        op.create_table(
            table_name,
            sa.Column("id", sa.Text(), primary_key=True, nullable=False),
            sa.Column("vector", Vector(VECTOR_DIM), nullable=False),
            sa.Column(
                "payload",
                JSONB(),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )
        # HNSW index for approximate nearest-neighbour cosine search.
        # m=16, ef_construction=64 are pgvector defaults.
        op.execute(
            f"CREATE INDEX {table_name}_hnsw_idx "
            f"ON {table_name} "
            f"USING hnsw (vector vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_vec_office_responsibilities_direction
        ON vec_office_responsibilities ((payload ->> 'direction'))
    """)

    # ==================================================================
    # qe-front tables — no SQLAlchemy model here, see env.py's
    # include_object filter (_QE_FRONT_TABLES). Schema ownership for the
    # whole shared database lives in this project's Alembic history;
    # qe-front keeps drizzle-orm only as its runtime query builder — see
    # qe-front's src/db/schema.ts and src/db/schema-auth.ts, which must
    # match these tables.
    # ==================================================================

    # ------------------------------------------------------------------
    # directions / sous_directions / bureaux
    # Référentiel organisationnel utilisé pour l'attribution des questions
    # (direction rédactrice, sous-direction, bureau). Pas de FK entre ces
    # trois tables ni vers `directions` sur `sous_directions.direction_id` /
    # `bureaux.direction_id` — confirmé sur la base de dev existante, aucune
    # contrainte de ce type n'y a jamais été créée.
    # ------------------------------------------------------------------
    op.create_table(
        "directions",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column("nom", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
    )

    op.create_table(
        "sous_directions",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column("nom", sa.String(200), nullable=False),
        sa.Column("direction_id", sa.Integer(), nullable=False),
    )

    op.create_table(
        "bureaux",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column("nom", sa.String(200), nullable=False),
        sa.Column("direction_id", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
    )

    # ------------------------------------------------------------------
    # question_real_attributions
    # Vérité terrain (direction + bureau rédacteur) par question, tenue par
    # qe-front. `source` / `imported_at` tracent la provenance de la valeur
    # courante (nom de fichier d'import en masse, ou "manual" pour une
    # édition humaine via l'admin UI).
    # ------------------------------------------------------------------
    op.create_table(
        "question_real_attributions",
        sa.Column("question_id", sa.String(100), primary_key=True, nullable=False),
        sa.Column("direction_reelle_id", sa.Integer(), nullable=True),
        sa.Column("bureau_reel_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # question_attribution_suggestions
    # Cache des suggestions du service ML (top-3 office_id), pour calculer
    # les métriques de performance sans rappeler le service à chaque vue.
    # ------------------------------------------------------------------
    op.create_table(
        "question_attribution_suggestions",
        sa.Column("question_id", sa.String(100), primary_key=True, nullable=False),
        sa.Column("top1_office_id", sa.String(50), nullable=True),
        sa.Column("top2_office_id", sa.String(50), nullable=True),
        sa.Column("top3_office_id", sa.String(50), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # question_similar_suggestions
    # Cache des questions similaires (kNN sur les embeddings).
    # ------------------------------------------------------------------
    op.create_table(
        "question_similar_suggestions",
        sa.Column("question_id", sa.String(100), primary_key=True, nullable=False),
        sa.Column("similar_question_ids", sa.ARRAY(sa.String(100)), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # suggestion_feedback
    # Pouces haut/bas sur les suggestions (attribution direction/bureau,
    # question similaire). FK vers `questions` (table de ce projet).
    # ------------------------------------------------------------------
    op.create_table(
        "suggestion_feedback",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column(
            "question_id",
            sa.String(100),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("suggestion_kind", sa.String(50), nullable=False),
        sa.Column("suggestion_target", sa.String(200), nullable=False),
        sa.Column("thumb", sa.String(10), nullable=False),  # 'up' | 'down'
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "suggestion_kind IN ('attribution_direction', 'attribution_bureau', 'similar')",
            name="suggestion_feedback_kind_check",
        ),
        sa.CheckConstraint(
            "thumb IN ('up', 'down')",
            name="suggestion_feedback_thumb_check",
        ),
    )

    op.create_index(
        "suggestion_feedback_question_id_idx", "suggestion_feedback", ["question_id"]
    )
    op.create_index(
        "suggestion_feedback_kind_idx", "suggestion_feedback", ["suggestion_kind"]
    )

    # ------------------------------------------------------------------
    # correction_feedback
    # Accepté/rejeté/reset sur les corrections orthotypographiques
    # suggérées par PLIAGE.
    # ------------------------------------------------------------------
    op.create_table(
        "correction_feedback",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False
        ),
        sa.Column("original", sa.Text(), nullable=False),
        sa.Column("replacement", sa.Text(), nullable=False),
        sa.Column("context_before", sa.Text(), nullable=True),
        sa.Column("context_after", sa.Text(), nullable=True),
        sa.Column(
            "decision", sa.String(20), nullable=False
        ),  # 'accepted' | 'rejected' | 'reset'
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision IN ('accepted', 'rejected', 'reset')",
            name="correction_feedback_decision_check",
        ),
    )

    op.create_index(
        "correction_feedback_created_at_idx", "correction_feedback", ["created_at"]
    )

    # ------------------------------------------------------------------
    # app_settings
    # Clé/valeur générique pour les réglages applicatifs de qe-front.
    # ------------------------------------------------------------------
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(100), primary_key=True, nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # users / accounts / sessions / verificationTokens
    # Schéma imposé par @auth/drizzle-adapter (Auth.js v5) côté qe-front.
    # Les colonnes en casse mixte ("userId", "providerAccountId", …) sont
    # requises telles quelles par l'adapter — sa.Column("...") avec
    # majuscule force Postgres à les quoter à l'identique.
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("emailVerified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image", sa.Text(), nullable=True),
        sa.Column("role", sa.Text(), server_default="user", nullable=False),
        sa.CheckConstraint("role IN ('user', 'admin')", name="users_role_check"),
    )

    op.create_table(
        "accounts",
        sa.Column(
            "userId",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("providerAccountId", sa.Text(), nullable=False),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.Integer(), nullable=True),
        sa.Column("token_type", sa.Text(), nullable=True),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("id_token", sa.Text(), nullable=True),
        sa.Column("session_state", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("provider", "providerAccountId"),
    )

    op.create_table(
        "sessions",
        sa.Column("sessionToken", sa.Text(), primary_key=True, nullable=False),
        sa.Column(
            "userId",
            sa.Text(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "verificationTokens",
        sa.Column("identifier", sa.Text(), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("expires", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("identifier", "token"),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_table("verificationTokens")
    op.drop_table("sessions")
    op.drop_table("accounts")
    op.drop_table("users")
    op.drop_table("app_settings")
    op.drop_index(
        "correction_feedback_created_at_idx", table_name="correction_feedback"
    )
    op.drop_table("correction_feedback")
    op.drop_index("suggestion_feedback_kind_idx", table_name="suggestion_feedback")
    op.drop_index(
        "suggestion_feedback_question_id_idx", table_name="suggestion_feedback"
    )
    op.drop_table("suggestion_feedback")
    op.drop_table("question_similar_suggestions")
    op.drop_table("question_attribution_suggestions")
    op.drop_table("question_real_attributions")
    op.drop_table("bureaux")
    op.drop_table("sous_directions")
    op.drop_table("directions")

    op.execute("DROP INDEX IF EXISTS idx_vec_office_responsibilities_direction")
    for table_name in reversed(_VEC_TABLES):
        op.drop_index(f"{table_name}_hnsw_idx", table_name=table_name)
        op.drop_table(table_name)

    op.drop_table("ingest_cursors")

    op.drop_index(
        "ix_question_attributions_question_id", table_name="question_attributions"
    )
    op.drop_table("question_attributions")

    op.drop_index(
        "ix_question_state_changes_question_id", table_name="question_state_changes"
    )
    op.drop_table("question_state_changes")

    op.execute("DROP INDEX IF EXISTS idx_questions_auteur_prenom_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_auteur_nom_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_texte_trgm")
    op.execute("DROP INDEX IF EXISTS idx_questions_min_attributaire")
    op.execute("DROP INDEX IF EXISTS idx_questions_etat_min_date")
    op.drop_index("ix_questions_etat_question", table_name="questions")
    op.drop_index("ix_questions_date_publication_jo", table_name="questions")
    op.drop_index("ix_questions_ministre_attributaire_id", table_name="questions")
    op.drop_index("ix_questions_source_legislature_type", table_name="questions")
    op.drop_table("questions")

    op.drop_table("reponses")
    op.drop_table("ministeres")
    op.drop_table("chunk_cache")
    op.drop_table("ingest_manifest")

    # vector / pg_trgm extensions are intentionally NOT dropped — other
    # objects may depend on them and recreation requires superuser privileges.
