"""The triggers keeping vec_questions_by_status in sync (migration c99252387f8c).

Runs against PostgreSQL inside a rolled-back transaction; skipped without one,
or when the migration isn't applied.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import OperationalError


def _vector(value: float) -> str:
    return "[" + ",".join([str(value)] * 1024) + "]"


@dataclass
class Db:
    conn: Connection
    # Fresh per test: xdist workers share the database, and concurrent
    # uncommitted writes on the same keys deadlock.
    qid: str
    point_id: str

    def insert_question(self, etat: str) -> None:
        self.conn.execute(
            text("""
                INSERT INTO questions (id, numero_question, type, source, legislature,
                                       etat_question, texte_question, ingest_source)
                VALUES (:id, 1, 'QE', 'AN', 99, :etat, 'texte', 'test')
            """),
            {"id": self.qid, "etat": etat},
        )

    def upsert_vec(self, value: float) -> None:
        self.conn.execute(
            text("""
                INSERT INTO vec_questions_opendata (id, vector, payload)
                VALUES (:id, CAST(:v AS vector),
                        jsonb_build_object('question_id', CAST(:qid AS text)))
                ON CONFLICT (id) DO UPDATE
                SET vector = EXCLUDED.vector, payload = EXCLUDED.payload
            """),
            {"id": self.point_id, "v": _vector(value), "qid": self.qid},
        )

    def copy(self):
        return self.conn.execute(
            text("""
                SELECT question_id, etat_question,
                       vector::vector = CAST(:v AS vector) AS same_vector,
                       ctid::text AS ctid
                FROM vec_questions_by_status WHERE id = :id
            """),
            {"id": self.point_id, "v": _vector(0.5)},
        ).one_or_none()


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
            if (
                connection.execute(
                    text("SELECT to_regclass('vec_questions_by_status')")
                ).scalar_one()
                is None
            ):
                pytest.skip("vec_questions_by_status not migrated")
            yield Db(connection, f"TEST-QE-{uuid.uuid4()}", str(uuid.uuid4()))
        finally:
            transaction.rollback()


@pytest.mark.integration
def test_vector_insert_copies_live_status(pg):
    pg.insert_question("EN_COURS")
    pg.upsert_vec(0.5)

    row = pg.copy()
    assert (row.question_id, row.etat_question, row.same_vector) == (
        pg.qid,
        "EN_COURS",
        True,
    )


@pytest.mark.integration
def test_status_change_follows_questions(pg):
    pg.insert_question("EN_COURS")
    pg.upsert_vec(0.5)

    pg.conn.execute(
        text("UPDATE questions SET etat_question = 'REPONDU' WHERE id = :id"),
        {"id": pg.qid},
    )

    assert pg.copy().etat_question == "REPONDU"


@pytest.mark.integration
def test_question_inserted_after_its_vector(pg):
    pg.upsert_vec(0.5)
    assert pg.copy().etat_question is None

    pg.insert_question("EN_COURS")

    assert pg.copy().etat_question == "EN_COURS"


@pytest.mark.integration
def test_reembedding_updates_the_copy(pg):
    pg.insert_question("EN_COURS")
    pg.upsert_vec(0.25)
    assert pg.copy().same_vector is False

    pg.upsert_vec(0.5)

    assert pg.copy().same_vector is True


@pytest.mark.integration
def test_noop_writes_leave_the_copy_untouched(pg):
    pg.insert_question("EN_COURS")
    pg.upsert_vec(0.5)
    # Any rewrite of the row moves it to a new ctid.
    before = pg.copy().ctid

    pg.upsert_vec(0.5)
    pg.conn.execute(
        text("UPDATE questions SET texte_question = 'autre' WHERE id = :id"),
        {"id": pg.qid},
    )

    assert pg.copy().ctid == before


@pytest.mark.integration
def test_vector_delete_cascades(pg):
    pg.insert_question("EN_COURS")
    pg.upsert_vec(0.5)

    pg.conn.execute(
        text("DELETE FROM vec_questions_opendata WHERE id = :id"),
        {"id": pg.point_id},
    )

    assert pg.copy() is None
