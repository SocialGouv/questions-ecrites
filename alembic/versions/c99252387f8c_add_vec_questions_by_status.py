"""Add vec_questions_by_status: halfvec copy of question vectors, indexed per status.

A partial HNSW index per etat_question searched by qe-front's `/similar`
(REPONDU for now), over a halfvec copy of `vec_questions_opendata.vector`.
Kept in sync by triggers on `vec_questions_opendata` (vector/payload writes)
and `questions` (etat_question writes), so no writer has to know about it.

The upgrade locks both tables against writes (reads are unaffected) until it
commits, so that no status change lands between the backfill and the triggers.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "c99252387f8c"
down_revision: Union[str, Sequence[str], None] = "5b1c9e2d7a4f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEXED_STATUSES = ("REPONDU",)


def _index_name(status: str) -> str:
    return f"vec_q_status_hnsw_{status.lower()}_idx"


def upgrade() -> None:
    op.execute(
        "LOCK TABLE questions, vec_questions_opendata IN SHARE ROW EXCLUSIVE MODE"
    )
    # The REPONDU graph (~2.6 KB/row) must fit in memory to build in one pass.
    # Serial build: a parallel one puts the graph in /dev/shm, 64 MB by
    # default under Docker.
    op.execute("SET LOCAL maintenance_work_mem = '768MB'")
    op.execute("SET LOCAL max_parallel_maintenance_workers = 0")

    op.execute("""
        CREATE TABLE vec_questions_by_status (
            id text PRIMARY KEY
                REFERENCES vec_questions_opendata (id) ON DELETE CASCADE,
            question_id text,
            etat_question text,
            vector halfvec(1024) NOT NULL
        )
    """)
    # Inline storage: a halfvec(1024) row is just over the TOAST threshold,
    # and every search hit reads the vector back from the heap.
    op.execute(
        "ALTER TABLE vec_questions_by_status ALTER COLUMN vector SET STORAGE PLAIN"
    )

    op.execute("""
        INSERT INTO vec_questions_by_status (id, question_id, etat_question, vector)
        SELECT v.id, v.payload ->> 'question_id', q.etat_question, v.vector::halfvec
        FROM vec_questions_opendata v
        LEFT JOIN questions q ON q.id = v.payload ->> 'question_id'
    """)

    op.execute(
        "CREATE INDEX vec_q_status_question_id_idx "
        "ON vec_questions_by_status (question_id)"
    )
    for status in INDEXED_STATUSES:
        op.execute(f"""
            CREATE INDEX {_index_name(status)} ON vec_questions_by_status
            USING hnsw (vector halfvec_cosine_ops)
            WHERE etat_question = '{status}'
        """)

    op.execute("""
        CREATE FUNCTION vec_questions_by_status_sync_vector() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            etat text;
        BEGIN
            -- FOR SHARE: waits for a pending status update on the question,
            -- and makes a later one wait until this copy is committed —
            -- otherwise its UPDATE below can't see the copy and is lost.
            SELECT q.etat_question INTO etat FROM questions q
            WHERE q.id = NEW.payload ->> 'question_id'
            FOR SHARE;
            INSERT INTO vec_questions_by_status (id, question_id, etat_question, vector)
            VALUES (NEW.id, NEW.payload ->> 'question_id', etat, NEW.vector::halfvec)
            ON CONFLICT (id) DO UPDATE
            SET question_id = EXCLUDED.question_id,
                etat_question = EXCLUDED.etat_question,
                vector = EXCLUDED.vector
            -- Skip no-op rewrites: each one is a new HNSW entry.
            WHERE (vec_questions_by_status.question_id,
                   vec_questions_by_status.etat_question,
                   vec_questions_by_status.vector)
                  IS DISTINCT FROM
                  (EXCLUDED.question_id, EXCLUDED.etat_question, EXCLUDED.vector);
            RETURN NULL;
        END
        $$
    """)
    op.execute("""
        CREATE TRIGGER vec_questions_by_status_sync_vector
        AFTER INSERT OR UPDATE OF vector, payload ON vec_questions_opendata
        FOR EACH ROW EXECUTE FUNCTION vec_questions_by_status_sync_vector()
    """)

    op.execute("""
        CREATE FUNCTION vec_questions_by_status_sync_etat() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            UPDATE vec_questions_by_status
            SET etat_question = NEW.etat_question
            WHERE question_id = NEW.id
              AND etat_question IS DISTINCT FROM NEW.etat_question;
            RETURN NULL;
        END
        $$
    """)
    op.execute("""
        CREATE TRIGGER vec_questions_by_status_sync_etat_insert
        AFTER INSERT ON questions
        FOR EACH ROW EXECUTE FUNCTION vec_questions_by_status_sync_etat()
    """)
    op.execute("""
        CREATE TRIGGER vec_questions_by_status_sync_etat_update
        AFTER UPDATE OF etat_question ON questions
        FOR EACH ROW
        WHEN (OLD.etat_question IS DISTINCT FROM NEW.etat_question)
        EXECUTE FUNCTION vec_questions_by_status_sync_etat()
    """)

    op.execute("ANALYZE vec_questions_by_status")


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS vec_questions_by_status_sync_etat_update ON questions"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS vec_questions_by_status_sync_etat_insert ON questions"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS vec_questions_by_status_sync_vector "
        "ON vec_questions_opendata"
    )
    op.execute("DROP FUNCTION IF EXISTS vec_questions_by_status_sync_etat()")
    op.execute("DROP FUNCTION IF EXISTS vec_questions_by_status_sync_vector()")
    op.execute("DROP TABLE IF EXISTS vec_questions_by_status")
