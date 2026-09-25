"""The questions/reponses upsert of `qe.ingestion_an.upsert_questions`.

Re-ingesting an unchanged row must not write it (the nightly job re-reads
every question), while a real change must still land — including the status
copy in vec_questions_by_status. Runs against PostgreSQL inside a rolled-back
transaction; skipped without one.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from qe.ingestion_an import IngestStats, ParsedQuestion, upsert_questions


@pytest.fixture
def pg():
    from qe import db

    try:
        connection = db.get_engine().connect()
    except OperationalError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no PostgreSQL available: {exc}")
    with connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def _question(
    etat: str, reponse_id: str | None = None, qid: str | None = None
) -> ParsedQuestion:
    return ParsedQuestion(
        id=qid or f"TEST-QE-{uuid.uuid4()}",
        numero_question=1,
        type="QE",
        source="AN",
        legislature=99,
        etat_question=etat,
        date_publication_jo=None,
        page_jo=None,
        ministre_libelle=None,
        auteur_nom="Dupont",
        objet="Objet",
        texte_question="Texte",
        reponse_id=reponse_id,
        texte_reponse="Réponse" if reponse_id else None,
        no_publication="20260101" if reponse_id else None,
    )


def _upsert(conn: Connection, *questions: ParsedQuestion) -> None:
    upsert_questions(Session(bind=conn), list(questions), "test", IngestStats())


def _ctid(conn: Connection, table: str, row_id: str) -> str:
    # A new ctid means a new tuple version, i.e. the row was rewritten.
    return conn.execute(
        text(f"SELECT ctid::text FROM {table} WHERE id = :id"),  # noqa: S608
        {"id": row_id},
    ).scalar_one()


@pytest.mark.integration
def test_reingesting_unchanged_rows_writes_nothing(pg):
    pq = _question("REPONDU", reponse_id=f"TEST-REP-{uuid.uuid4()}")
    _upsert(pg, pq)
    before = (_ctid(pg, "questions", pq.id), _ctid(pg, "reponses", pq.reponse_id))

    _upsert(pg, pq)
    # A downgrade attempt resolves to the stored REPONDU: no change, no write.
    _upsert(pg, replace(pq, etat_question="EN_COURS", reponse_id=None))

    assert (_ctid(pg, "questions", pq.id), _ctid(pg, "reponses", pq.reponse_id)) == (
        before
    )


@pytest.mark.integration
def test_status_change_still_updates_and_reaches_status_index(pg):
    has_status_index = (
        pg.execute(text("SELECT to_regclass('vec_questions_by_status')")).scalar()
        is not None
    )
    pq = _question("EN_COURS")
    _upsert(pg, pq)
    point_id = str(uuid.uuid4())
    pg.execute(
        text("""
            INSERT INTO vec_questions_opendata (id, vector, payload)
            VALUES (:id, CAST(:v AS vector), jsonb_build_object('question_id', CAST(:qid AS text)))
        """),
        {"id": point_id, "v": "[" + ",".join(["0.5"] * 1024) + "]", "qid": pq.id},
    )

    answered = _question("REPONDU", reponse_id=f"TEST-REP-{uuid.uuid4()}", qid=pq.id)
    _upsert(pg, answered)

    row = pg.execute(
        text("SELECT etat_question, reponse_id FROM questions WHERE id = :id"),
        {"id": pq.id},
    ).one()
    assert tuple(row) == ("REPONDU", answered.reponse_id)
    if has_status_index:
        assert (
            pg.execute(
                text(
                    "SELECT etat_question FROM vec_questions_by_status WHERE id = :id"
                ),
                {"id": point_id},
            ).scalar_one()
            == "REPONDU"
        )


@pytest.mark.integration
def test_duplicate_id_in_one_batch_is_applied_in_order(pg):
    pq = _question("EN_COURS")
    answered = _question("REPONDU", reponse_id=f"TEST-REP-{uuid.uuid4()}", qid=pq.id)

    # One multi-VALUES statement would fail here ("ON CONFLICT DO UPDATE
    # command cannot affect row a second time"); per-row statements don't.
    _upsert(pg, pq, answered)

    assert (
        pg.execute(
            text("SELECT etat_question FROM questions WHERE id = :id"), {"id": pq.id}
        ).scalar_one()
        == "REPONDU"
    )
