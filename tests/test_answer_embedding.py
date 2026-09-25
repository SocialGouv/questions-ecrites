"""The answer loader of qe/answer_embedding.py.

Runs against PostgreSQL inside a rolled-back transaction, and is skipped
without one.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from qe.answer_embedding import _answers_stmt


@pytest.fixture
def conn():
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


@pytest.mark.integration
def test_linked_questions_are_loaded_with_their_id_only(conn):
    # Every linked question used to be loaded in full (texte_question,
    # analyses…) for its id alone, which dominated the nightly job's peak RSS.
    rid = f"TEST-R-{uuid.uuid4()}"
    qids = sorted(f"TEST-QE-{uuid.uuid4()}" for _ in range(2))
    conn.execute(
        text("""
            INSERT INTO reponses (id, source, no_publication, texte_reponse)
            VALUES (:rid, 'AN', '1', 'réponse')
        """),
        {"rid": rid},
    )
    for qid in qids:
        conn.execute(
            text("""
                INSERT INTO questions (id, numero_question, type, source, legislature,
                                       etat_question, texte_question, ingest_source,
                                       reponse_id)
                VALUES (:qid, 1, 'QE', 'AN', 99, 'REPONDU', 'texte', 'test', :rid)
            """),
            {"qid": qid, "rid": rid},
        )

    with Session(bind=conn, join_transaction_mode="create_savepoint") as session:
        [answer] = session.execute(_answers_stmt("AN", 99)).scalars().all()
        assert answer.id == rid
        assert sorted(q.id for q in answer.questions) == qids
        for question in answer.questions:
            assert "texte_question" in inspect(question).unloaded
