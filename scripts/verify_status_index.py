#!/usr/bin/env python3
"""Verify vec_questions_by_status against its sources and exact search.

Read-only. Exits non-zero when any check fails:

* every `vec_questions_opendata` row has its copy, with the live status
  of its question (the triggers keep both in sync);
* for a random sample of EN_COURS sources, the per-status partial index
  query (same shape and settings as qe-front's pgvector-searcher) returns
  the exact brute-force top-K at or above `--min-recall`, and the plan
  really scans the partial index.

Usage:
    poetry run python scripts/verify_status_index.py [--sample-size 30] [--min-recall 0.95]
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text as sqltext

from qe import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Mirrors qe-front's RERANK_POOL / RERANK_EF_SEARCH (src/lib/similar-pipeline.ts).
LIMIT = 100
EF_SEARCH = 500
STATUSES = ("REPONDU",)

MISSING_SQL = sqltext("""
    SELECT count(*) FROM vec_questions_opendata v
    WHERE NOT EXISTS (SELECT 1 FROM vec_questions_by_status s WHERE s.id = v.id)
""")
DRIFT_SQL = sqltext("""
    SELECT count(*) FROM vec_questions_by_status s
    JOIN questions q ON q.id = s.question_id
    WHERE s.etat_question IS DISTINCT FROM q.etat_question
""")
SAMPLE_SQL = sqltext("""
    SELECT id FROM vec_questions_by_status
    WHERE etat_question = 'EN_COURS' ORDER BY random() LIMIT :n
""")
EXACT_SQL = sqltext("""
    SELECT q.id FROM vec_questions_opendata v
    JOIN questions q ON q.id = v.payload ->> 'question_id'
    WHERE v.id <> :id AND q.etat_question = :status
    ORDER BY v.vector <=> (SELECT vector FROM vec_questions_opendata WHERE id = :id)
    LIMIT :limit
""")


def _indexed_sql(status: str) -> str:
    # Literal status, as qe-front inlines it: a bound one can't match the
    # partial index predicate.
    return f"""
        SELECT q.id FROM vec_questions_by_status v
        JOIN questions q ON q.id = v.question_id
        WHERE v.id <> :id
          AND v.etat_question = '{status}' AND q.etat_question = '{status}'
        ORDER BY v.vector <=> (SELECT vector::halfvec FROM vec_questions_opendata WHERE id = :id)
        LIMIT :limit
    """  # noqa: S608 -- status is one of STATUSES, not input


def _recall(conn, point_id: str, status: str) -> float:
    params = {"id": point_id, "status": status, "limit": LIMIT}
    conn.execute(sqltext("SET LOCAL enable_indexscan = off"))
    conn.execute(sqltext("SET LOCAL enable_bitmapscan = off"))
    # The source-vector lookup stays a PK scan: only the HNSW walk is disabled.
    exact = set(conn.execute(EXACT_SQL, params).scalars().all())

    conn.execute(sqltext("SET LOCAL enable_indexscan = on"))
    conn.execute(sqltext("SET LOCAL enable_bitmapscan = on"))
    conn.execute(sqltext("SET LOCAL enable_seqscan = off"))
    conn.execute(sqltext(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}"))
    conn.execute(sqltext("SET LOCAL hnsw.iterative_scan = strict_order"))
    indexed_sql = _indexed_sql(status)
    plan = str(
        conn.execute(
            sqltext(f"EXPLAIN (FORMAT JSON) {indexed_sql}"), params
        ).scalar_one()
    )
    index_name = f"vec_q_status_hnsw_{status.lower()}_idx"
    if index_name not in plan:
        raise AssertionError(f"{point_id}/{status}: plan does not scan {index_name}")
    if not exact:
        raise AssertionError(f"{point_id}/{status}: exact search returned nothing")
    got = set(conn.execute(sqltext(indexed_sql), params).scalars().all())
    return len(got & exact) / len(exact)


def run_checks(sample_size: int, min_recall: float) -> int:
    engine = db.get_engine()
    failures = 0
    with engine.connect() as conn:
        missing = conn.execute(MISSING_SQL).scalar_one()
        drift = conn.execute(DRIFT_SQL).scalar_one()
        sample = conn.execute(SAMPLE_SQL, {"n": sample_size}).scalars().all()
    logger.info("copy: %d missing row(s), %d status drift(s)", missing, drift)
    failures += int(missing > 0) + int(drift > 0)

    for status in STATUSES:
        recalls = []
        for point_id in sample:
            with engine.begin() as conn:
                conn.execute(sqltext("SET TRANSACTION READ ONLY"))
                recalls.append(_recall(conn, point_id, status))
        if not recalls:
            logger.error("%s: empty sample — recall not measured.", status)
            failures += 1
            continue
        mean = sum(recalls) / len(recalls)
        worst = min(recalls)
        logger.info(
            "%s: recall@%d mean %.1f%%, worst %.1f%% (n=%d, ef_search=%d)",
            status,
            LIMIT,
            100 * mean,
            100 * worst,
            len(recalls),
            EF_SEARCH,
        )
        failures += int(mean < min_recall)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample-size", type=int, default=30)
    ap.add_argument("--min-recall", type=float, default=0.95)
    args = ap.parse_args()

    failures = run_checks(args.sample_size, args.min_recall)
    if failures:
        logger.error("%d check(s) failed.", failures)
        sys.exit(1)
    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
