#!/usr/bin/env python3
"""Verify partial HNSW index recall for direction/bureau kNN voting.

`vec_q_hnsw_direction_idx` / `vec_q_hnsw_bureau_idx` (see the Alembic
migration that adds them) are approximate — a smaller/sparser graph can
need more relative search effort (`hnsw.ef_search`) than the full index to
reach the same recall as exact brute-force search. This was measured
directly: at the attributed pool sizes when the partial indexes were
introduced (~3.4k bureau / ~7.7k direction rows), `ef_search=500` gave
wrong votes on 11/15 test questions (including changed #1-ranked bureaus);
`ef_search=1000` (pgvector's max) gave zero mismatches on 60 questions.

Re-run this after any large attribution import, or periodically as the
attributed pool grows, to catch a recall regression before it silently
changes real votes. It compares the partial-index query (current
`ef_search`, plus `hnsw.iterative_scan = strict_order` — NOT currently set
by production's `withHnswSearch`; see docs/direction-bureau-attribution.md's
"open risk" note, this canary deliberately checks the stricter setting)
against exact brute-force search (`enable_seqscan` forced on,
`enable_indexscan`/`enable_bitmapscan` forced off) on a random sample, and
reports any mismatch. Also asserts via `EXPLAIN` that the partial query
actually scanned the expected partial index — `enable_seqscan = off` is
only a cost penalty, so without this a missing/invalid/unpicked index
would silently degrade the "partial" query to the same plan as "exact"
and report a vacuous pass.

Historical trend (larger attributed pools need *less* relative ef_search,
not more — a denser graph is easier for HNSW to navigate correctly):
    ~3.4k rows: ef_search=500 -> 11/15 wrong,  ef_search=1000 -> 0/60 wrong
    ~20k rows:  ef_search=100 -> 1/30 wrong,   ef_search=250  -> 0/30 wrong
    ~59k rows (full table): ef_search=500 -> 0/15 wrong

Usage:
    poetry run python scripts/verify_partial_index_recall.py [--sample-size 30]
"""

from __future__ import annotations

import argparse
import logging
import sys

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam
from sqlalchemy import text as sqltext

from qe import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Must match the production values in qe-front's attributionAlgo.ts /
# attributions/route.ts — this script only reads, so it can't desync data,
# but the ef_search value below IS a separate hardcoded copy that will
# silently stop reflecting reality if production's is ever tuned without
# updating this file too.
DIRECTION_EF_SEARCH = 1000
BUREAU_EF_SEARCH = 1000
DIRECTION_KNN = 15
BUREAU_KNN = 25

# Source vector for `:qid`, fetched once per check while index scans are
# still on (see `_check_one`) — the EXACT_SQL/PARTIAL_SQL queries below take
# it as a `:vector` bind param instead of re-looking it up inline, so that
# forcing `enable_indexscan = off` for the exact half doesn't also force a
# full-table scan for the self-lookup.
SRC_VECTOR_SQL = sqltext(
    "SELECT vector FROM vec_questions_opendata WHERE id = :qid"
).columns(vector=Vector())

DIRECTION_EXACT_SQL = sqltext("""
    SELECT qa.direction_reelle_id AS key, 1 - (v.vector <=> :vector) AS similarity
    FROM vec_questions_opendata v
    JOIN question_real_attributions qa ON qa.question_id = v.payload ->> 'question_id'
    WHERE v.id <> :qid AND qa.direction_reelle_id IS NOT NULL
    ORDER BY v.vector <=> :vector
    LIMIT :knn
""").bindparams(bindparam("vector", type_=Vector()))

DIRECTION_PARTIAL_SQL = sqltext("""
    SELECT qa.direction_reelle_id AS key, 1 - (v.vector <=> :vector) AS similarity
    FROM vec_questions_opendata v
    JOIN question_real_attributions qa ON qa.question_id = v.payload ->> 'question_id'
    WHERE v.id <> :qid AND v.has_direction_attribution AND qa.direction_reelle_id IS NOT NULL
    ORDER BY v.vector <=> :vector
    LIMIT :knn
""").bindparams(bindparam("vector", type_=Vector()))

BUREAU_EXACT_SQL = sqltext("""
    SELECT va.bureau_key AS key, 1 - (v.vector <=> :vector) AS similarity
    FROM vec_questions_opendata v
    JOIN question_attributions_all va ON va.question_id = v.payload ->> 'question_id'
    WHERE v.id <> :qid
    ORDER BY v.vector <=> :vector
    LIMIT :knn
""").bindparams(bindparam("vector", type_=Vector()))

BUREAU_PARTIAL_SQL = sqltext("""
    SELECT va.bureau_key AS key, 1 - (v.vector <=> :vector) AS similarity
    FROM vec_questions_opendata v
    JOIN question_attributions_all va ON va.question_id = v.payload ->> 'question_id'
    WHERE v.id <> :qid AND v.has_bureau_attribution
    ORDER BY v.vector <=> :vector
    LIMIT :knn
""").bindparams(bindparam("vector", type_=Vector()))


def _vote(rows) -> dict[object, float]:
    votes: dict[object, float] = {}
    for r in rows:
        votes[r.key] = votes.get(r.key, 0.0) + r.similarity
    return votes


def _plan_uses_index(conn, sql, params: dict, index_name: str) -> bool:
    """Return True if EXPLAIN shows `index_name` in the plan for `sql`.

    `enable_seqscan = off` is a cost penalty, not a prohibition — if the
    partial index is missing, invalid, or simply not chosen by the planner,
    the "partial" query silently falls back to the same plan as "exact",
    and the two votes then match by construction. This runs under the same
    GUCs as the real query (planning only, no `ANALYZE`, so it doesn't
    execute the scan).
    """
    explain_sql = sqltext(f"EXPLAIN (FORMAT JSON) {sql.text}").bindparams(
        bindparam("vector", type_=Vector())
    )
    plan = conn.execute(explain_sql, params).scalar_one()
    return index_name in str(plan)


def _check_one(
    conn, qid: str, exact_sql, partial_sql, knn: int, ef_search: int, index_name: str
) -> bool:
    """Return True if the partial-index vote matches exact search."""
    # Resolved before touching enable_indexscan/enable_bitmapscan below, so
    # this PK lookup stays an index scan rather than a full table scan.
    vector = conn.execute(SRC_VECTOR_SQL, {"qid": qid}).scalar_one()

    conn.execute(sqltext("SET LOCAL enable_seqscan = on"))
    conn.execute(sqltext("SET LOCAL enable_indexscan = off"))
    conn.execute(sqltext("SET LOCAL enable_bitmapscan = off"))
    exact = _vote(
        conn.execute(exact_sql, {"qid": qid, "knn": knn, "vector": vector}).fetchall()
    )

    conn.execute(sqltext("SET LOCAL enable_seqscan = off"))
    conn.execute(sqltext("SET LOCAL enable_indexscan = on"))
    conn.execute(sqltext("SET LOCAL enable_bitmapscan = on"))
    conn.execute(sqltext(f"SET LOCAL hnsw.ef_search = {ef_search}"))
    # NOT currently set by production (qe-front's withHnswSearch only sets
    # ef_search — see docs/direction-bureau-attribution.md's "open risk"
    # note). Set here deliberately anyway: without it this arm runs
    # pgvector's default non-iterative scan, which can under-fill the vote
    # after the join drops candidates has_bureau_attribution's superset
    # doesn't cover — the stricter setting this canary should be checking
    # against, even though it's not (yet) what production runs.
    conn.execute(sqltext("SET LOCAL hnsw.iterative_scan = strict_order"))
    params = {"qid": qid, "knn": knn, "vector": vector}
    if not _plan_uses_index(conn, partial_sql, params, index_name):
        raise AssertionError(
            f"partial query for qid={qid} did not use {index_name} — "
            "recall check would pass vacuously; is the index missing, "
            "invalid, or not being picked by the planner?"
        )
    partial = _vote(conn.execute(partial_sql, params).fetchall())

    def round4(votes: dict[object, float]) -> dict[object, float]:
        return {k: round(v, 4) for k, v in votes.items()}

    return round4(exact) == round4(partial)


def run_checks(sample_size: int = 30) -> int:
    """Run the direction+bureau recall check, return the total mismatch count."""
    engine = db.get_engine()
    failures = 0

    sample_sql = {
        "direction": sqltext(
            "SELECT id FROM vec_questions_opendata WHERE has_direction_attribution ORDER BY random() LIMIT :n"
        ),
        "bureau": sqltext(
            "SELECT id FROM vec_questions_opendata WHERE has_bureau_attribution ORDER BY random() LIMIT :n"
        ),
    }
    pool_size_sql = {
        "direction": sqltext(
            "SELECT count(*) FROM vec_questions_opendata WHERE has_direction_attribution"
        ),
        "bureau": sqltext(
            "SELECT count(*) FROM vec_questions_opendata WHERE has_bureau_attribution"
        ),
    }

    for label, exact_sql, partial_sql, knn, ef_search, index_name in [
        (
            "direction",
            DIRECTION_EXACT_SQL,
            DIRECTION_PARTIAL_SQL,
            DIRECTION_KNN,
            DIRECTION_EF_SEARCH,
            "vec_q_hnsw_direction_idx",
        ),
        (
            "bureau",
            BUREAU_EXACT_SQL,
            BUREAU_PARTIAL_SQL,
            BUREAU_KNN,
            BUREAU_EF_SEARCH,
            "vec_q_hnsw_bureau_idx",
        ),
    ]:
        with engine.connect() as conn:
            sample = (
                conn.execute(sample_sql[label], {"n": sample_size}).scalars().all()
            )
            pool_size = conn.execute(pool_size_sql[label]).scalar_one()

        mismatches = 0
        for qid in sample:
            with engine.begin() as conn:
                if not _check_one(
                    conn, qid, exact_sql, partial_sql, knn, ef_search, index_name
                ):
                    mismatches += 1
                    logger.warning("MISMATCH (%s): %s", label, qid)

        logger.info(
            "%s: %d/%d mismatches (ef_search=%d, attributed pool size=%d)",
            label,
            mismatches,
            len(sample),
            ef_search,
            pool_size,
        )
        failures += mismatches

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample-size", type=int, default=30)
    args = ap.parse_args()

    failures = run_checks(args.sample_size)
    if failures:
        logger.error(
            "%d total mismatch(es) — partial index recall has regressed.", failures
        )
        sys.exit(1)
    logger.info("All checks passed — partial index votes match exact search.")


if __name__ == "__main__":
    main()
