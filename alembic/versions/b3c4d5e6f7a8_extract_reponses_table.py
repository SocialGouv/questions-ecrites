"""extract reponses table from questions

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-03-26

Moves response data out of the questions table into a dedicated reponses table.
A response is identified by "{source}-{no_publication}-{page_reponse_jo}" and
can be linked to multiple questions (joint parliamentary answers).

No data backfill — re-ingest from data/opendata/ after upgrading.
"""

from alembic import op
import sqlalchemy as sa

revision = "b3c4d5e6f7a8"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create the reponses table
    op.create_table(
        "reponses",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("no_publication", sa.Text, nullable=False),
        sa.Column("texte_reponse", sa.Text, nullable=False),
        sa.Column(
            "ministre_reponse_id",
            sa.Integer,
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
        sa.Column("ministre_reponse_libelle", sa.Text, nullable=True),
        sa.Column("date_reponse_jo", sa.Date, nullable=True),
        sa.Column("page_reponse_jo", sa.Integer, nullable=True),
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

    # 2. Add reponse_id FK to questions
    op.add_column(
        "questions",
        sa.Column(
            "reponse_id",
            sa.Text,
            sa.ForeignKey("reponses.id"),
            nullable=True,
        ),
    )

    # 3. Drop old response columns from questions (no backfill)
    op.drop_column("questions", "texte_reponse")
    op.drop_column("questions", "ministre_reponse_id")
    op.drop_column("questions", "ministre_reponse_libelle")
    op.drop_column("questions", "date_reponse_jo")
    op.drop_column("questions", "page_reponse_jo")


def downgrade() -> None:
    # Restore response columns on questions (data will be lost)
    op.add_column("questions", sa.Column("texte_reponse", sa.Text, nullable=True))
    op.add_column(
        "questions",
        sa.Column(
            "ministre_reponse_id",
            sa.Integer,
            sa.ForeignKey("ministeres.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "questions", sa.Column("ministre_reponse_libelle", sa.Text, nullable=True)
    )
    op.add_column("questions", sa.Column("date_reponse_jo", sa.Date, nullable=True))
    op.add_column("questions", sa.Column("page_reponse_jo", sa.Integer, nullable=True))

    op.drop_column("questions", "reponse_id")
    op.drop_table("reponses")
