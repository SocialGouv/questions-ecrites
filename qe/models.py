from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ministere(Base):
    __tablename__ = "ministeres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    titre_jo: Mapped[str] = mapped_column(Text, nullable=False)
    intitule_min: Mapped[str] = mapped_column(Text, nullable=False)
    en_fonction: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    date_debut: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_fin: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_social: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # back-references (convenient but not eagerly loaded)
    questions_deposees: Mapped[list[Question]] = relationship(
        "Question",
        foreign_keys="Question.ministre_depot_id",
        back_populates="ministre_depot",
    )
    questions_attribuees: Mapped[list[Question]] = relationship(
        "Question",
        foreign_keys="Question.ministre_attributaire_id",
        back_populates="ministre_attributaire",
    )
    questions_repondues: Mapped[list[Question]] = relationship(
        "Question",
        foreign_keys="Question.ministre_reponse_id",
        back_populates="ministre_reponse",
    )


class Question(Base):
    __tablename__ = "questions"

    # --- identifiant composite sérialisé : "{SOURCE}-{LEGISLATURE}-{TYPE}-{NUMERO}" ---
    # ex. "AN-17-QE-12345"
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    # --- composantes de l'identifiant ---
    numero_question: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)  # QE, QOSD, QOAD…
    source: Mapped[str] = mapped_column(Text, nullable=False)  # AN | SENAT
    legislature: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- état ---
    # EN_COURS | REPONDU | RETIRE | SIGNALE | CADUQUE | RENOUVELE | CLOTURE_AUTRE
    etat_question: Mapped[str] = mapped_column(Text, nullable=False)

    # --- publication JO ---
    date_publication_jo: Mapped[date | None] = mapped_column(Date, nullable=True)
    page_jo: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- ministère dépôt ---
    ministre_depot_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ministeres.id"), nullable=True
    )
    ministre_depot_libelle: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- ministère attributaire (peut changer via ré-attribution) ---
    ministre_attributaire_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ministeres.id"), nullable=True
    )
    ministre_attributaire_libelle: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # --- auteur ---
    auteur_id_mandat: Mapped[str | None] = mapped_column(Text, nullable=True)
    auteur_nom: Mapped[str | None] = mapped_column(Text, nullable=True)
    auteur_prenom: Mapped[str | None] = mapped_column(Text, nullable=True)
    auteur_grp_pol: Mapped[str | None] = mapped_column(Text, nullable=True)
    auteur_circonscription: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- indexation AN (None si source=SENAT) ---
    rubrique: Mapped[str | None] = mapped_column(Text, nullable=True)
    rubrique_ta: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyses: Mapped[list[str] | None] = mapped_column(ARRAY(Text()), nullable=True)

    # --- indexation Sénat (None si source=AN) ---
    titre_senat: Mapped[str | None] = mapped_column(Text, nullable=True)
    themes: Mapped[list[str] | None] = mapped_column(ARRAY(Text()), nullable=True)
    rubriques_senat: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text()), nullable=True
    )

    # --- textes ---
    texte_question: Mapped[str] = mapped_column(Text, nullable=False)
    texte_reponse: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- réponse (None tant que EN_COURS) ---
    ministre_reponse_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ministeres.id"), nullable=True
    )
    ministre_reponse_libelle: Mapped[str | None] = mapped_column(Text, nullable=True)
    date_reponse_jo: Mapped[date | None] = mapped_column(Date, nullable=True)
    page_reponse_jo: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- liens ---
    # question dont celle-ci est le renouvellement
    rappel_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("questions.id"), nullable=True
    )
    date_retrait: Mapped[date | None] = mapped_column(Date, nullable=True)

    # --- méta ingestion ---
    ingest_source: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # opendata | ws_polling
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # --- relationships ---
    ministre_depot: Mapped[Ministere | None] = relationship(
        "Ministere",
        foreign_keys=[ministre_depot_id],
        back_populates="questions_deposees",
    )
    ministre_attributaire: Mapped[Ministere | None] = relationship(
        "Ministere",
        foreign_keys=[ministre_attributaire_id],
        back_populates="questions_attribuees",
    )
    ministre_reponse: Mapped[Ministere | None] = relationship(
        "Ministere",
        foreign_keys=[ministre_reponse_id],
        back_populates="questions_repondues",
    )
    rappel: Mapped[Question | None] = relationship(
        "Question", remote_side="Question.id", foreign_keys=[rappel_id]
    )
    state_changes: Mapped[list[QuestionStateChange]] = relationship(
        "QuestionStateChange", back_populates="question", cascade="all, delete-orphan"
    )
    attributions: Mapped[list[QuestionAttribution]] = relationship(
        "QuestionAttribution", back_populates="question", cascade="all, delete-orphan"
    )


class QuestionStateChange(Base):
    __tablename__ = "question_state_changes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(
        Text, ForeignKey("questions.id"), nullable=False
    )
    etat: Mapped[str] = mapped_column(Text, nullable=False)
    date_modif: Mapped[date] = mapped_column(Date, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    question: Mapped[Question] = relationship(
        "Question", back_populates="state_changes"
    )


class QuestionAttribution(Base):
    __tablename__ = "question_attributions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(
        Text, ForeignKey("questions.id"), nullable=False
    )
    type_attribution: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # REATTRIBUTION | REAFFECTATION
    attributaire_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ministeres.id"), nullable=True
    )
    attributaire_libelle: Mapped[str | None] = mapped_column(Text, nullable=True)
    date_attribution: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    question: Mapped[Question] = relationship("Question", back_populates="attributions")
    attributaire: Mapped[Ministere | None] = relationship(
        "Ministere", foreign_keys=[attributaire_id]
    )


class IngestCursor(Base):
    __tablename__ = "ingest_cursors"

    cursor_name: Mapped[str] = mapped_column(Text, primary_key=True)
    jeton: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
