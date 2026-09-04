#!/usr/bin/env python3
"""A/B eval : algo d'attribution BUREAU avec vs sans MIN15.

Contrairement à `eval_direction_with_min15.py` qui compare des
direction_id (FK), on compare ici des LIBELLÉS bureau normalisés
("SD2/2B", "SDRH1/Pharmacie") : le référentiel `bureaux` est
DGCS-first et n'a pas encore d'entrées DGOS/DGS/DSS, donc il n'y a
pas de FK propre à utiliser.

Sources de voters (bureau) :
- **baseline** : bureaux issus de `question_real_attributions.bureau_reel_id`
  (5 925 DGCS + 1 011 DSS + rares autres)
- **enriched** : baseline + `question_bureau_extract` (1 275 DSS +
  1 143 DGOS + 378 DGS + 251 DGE + 153 DGCS + etc.)

Attendu : gain massif sur DGOS/DGS où l'algo baseline ne peut PAS
prédire un bureau correct (aucun exemple humain de leurs bureaux).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from sqlalchemy import text as sqltext
from tqdm import tqdm

from qe import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KNN = 15
HNSW_EF_SEARCH = 1000  # matches production (attributions/route.ts) as of the partial-index fix

# Extract SD-code and bureau-code from a "bureaux.nom" like:
#   "[SD2/2B] Protection de l'enfance et de l'adolescence"
_DGCS_NOM_RE = re.compile(r"^\s*\[(SD\d+)/(\w+)\]", re.IGNORECASE)


def canonical_from_dgcs_nom(nom: str) -> str | None:
    """Return 'SD2/2B' from '[SD2/2B] Protection…'."""
    m = _DGCS_NOM_RE.match(nom)
    if not m:
        return None
    return f"{m.group(1).upper()}/{m.group(2).upper()}"


def canonical_from_extract(sous_direction: str | None, bureau: str | None) -> str | None:
    """Return canonical key from question_bureau_extract fields.

    Formats seen in the wild:
    - MIN15 DGCS  : sous_direction='SD2', bureau='Bureau 2B'  → 'SD2/2B'
    - MIN15 DGOS  : sous_direction='SDRH1', bureau='Pharmacie' → 'SDRH1/Pharmacie'
    - MIN15 DGS   : sous_direction='SD SP', bureau='Bureau SP5 - Maladies…' → 'SDSP/SP5'
    """
    if not sous_direction:
        return None
    sd = sous_direction.strip().upper().replace(" ", "")
    if not bureau:
        return sd
    b = bureau.strip()
    # Try to extract "2B" from "Bureau 2B"
    m = re.match(r"^\s*Bureau\s+(\w+)", b, re.IGNORECASE)
    if m:
        return f"{sd}/{m.group(1).upper()}"
    # Try "SP5" from "Bureau SP5" or full name
    m = re.match(r"^\s*Bureau\s+([A-Z]+\d+)", b, re.IGNORECASE)
    if m:
        return f"{sd}/{m.group(1).upper()}"
    # Otherwise use the first token normalised
    first = re.split(r"[\s/,-]+", b)[0].strip().upper()
    return f"{sd}/{first}" if first else sd


def _point_id(qid: str) -> str:
    return str(UUID(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:32]))


def knn_top1_bureau(session, src_pt_id: str, src_qid: str, voters_sql: str) -> str | None:
    """kNN vote returning the top-1 canonical bureau key.

    Adds production's `v.has_bureau_attribution` predicate so the planner
    can pick `vec_q_hnsw_bureau_idx` instead of the full index. Safe for
    both arms here (unlike the direction eval): the flag is defined as
    EXISTS(question_real_attributions.bureau_reel_id) OR
    EXISTS(question_bureau_extract) — exactly the union both the baseline
    and enriched voter tables are built from, so every voter already
    satisfies it.
    """
    # SET (not SET LOCAL) is transactional — a rollback after a failed kNN
    # reverts it, so it's re-issued on every call rather than once, to avoid
    # silently falling back to the server default ef_search.
    session.execute(sqltext(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"))
    # NOT currently set by production (qe-front's withHnswSearch only sets
    # ef_search — see docs/direction-bureau-attribution.md's "open risk" note)
    # but set here and in verify_partial_index_recall because without it,
    # pgvector's non-iterative scan caps candidates at ef_search *before* the
    # `JOIN voters` post-filter, which can under-fill the KNN vote in the
    # baseline arm — its voter set is a strict subset of
    # vec_q_hnsw_bureau_idx's membership.
    session.execute(sqltext("SET hnsw.iterative_scan = strict_order"))
    sql = sqltext(f"""
        WITH src AS (
            SELECT vector FROM vec_questions_opendata WHERE id = :src_pt_id
        ),
        voters AS ({voters_sql}),
        neighbours AS (
            SELECT
                voters.bureau_key AS bk,
                1 - (v.vector <=> (SELECT vector FROM src)) AS similarity
            FROM vec_questions_opendata v
            JOIN voters ON voters.question_id = v.payload ->> 'question_id'
            WHERE v.id <> :src_pt_id AND v.has_bureau_attribution
            ORDER BY v.vector <=> (SELECT vector FROM src)
            LIMIT {KNN}
        )
        SELECT bk, SUM(similarity) AS vote
        FROM neighbours
        WHERE bk IS NOT NULL
        GROUP BY bk
        ORDER BY vote DESC
        LIMIT 1
    """)
    row = session.execute(sql, {"src_pt_id": src_pt_id, "src_qid": src_qid}).first()
    return row.bk if row else None


# --- Voter tables (materialized once as temp tables for speed) ---------------

DDL_BASELINE_VOTERS = """
    DROP TABLE IF EXISTS tmp_voters_baseline;
    CREATE TEMP TABLE tmp_voters_baseline AS
    SELECT qa.question_id, b.nom AS bureau_nom
    FROM question_real_attributions qa
    JOIN bureaux b ON b.id = qa.bureau_reel_id
    WHERE qa.bureau_reel_id IS NOT NULL;
    CREATE INDEX ON tmp_voters_baseline(question_id);
"""

DDL_ENRICHED_VOTERS_SUFFIX = """
    DROP TABLE IF EXISTS tmp_voters_min15;
    CREATE TEMP TABLE tmp_voters_min15 AS
    SELECT question_id, sous_direction, bureau
    FROM question_bureau_extract;
    CREATE INDEX ON tmp_voters_min15(question_id);
"""

VOTERS_SQL_BASELINE = """
    SELECT question_id,
           NULL AS __placeholder,
           bureau_nom AS __raw_dgcs_nom,
           NULL::text AS __raw_sd,
           NULL::text AS __raw_bur
    FROM tmp_voters_baseline
    WHERE question_id <> :src_qid
"""

# For SQL simplicity, we'll do the canonicalisation Python-side and
# recreate a `voters` table with (question_id, bureau_key) each time.


def make_voters_table(session, table: str, enriched: bool) -> None:
    """Materialize a voter table (question_id, bureau_key) in-session."""
    session.execute(sqltext(f"DROP TABLE IF EXISTS {table}"))
    session.execute(sqltext(f"""
        CREATE TEMP TABLE {table} (
            question_id text PRIMARY KEY,
            bureau_key  text NOT NULL
        )
    """))
    # 1) DGCS/DSS from question_real_attributions
    rows_dgcs = session.execute(sqltext("""
        SELECT qa.question_id, b.nom
        FROM question_real_attributions qa
        JOIN bureaux b ON b.id = qa.bureau_reel_id
        WHERE qa.bureau_reel_id IS NOT NULL
    """)).all()
    kv: dict[str, str] = {}
    for r in rows_dgcs:
        k = canonical_from_dgcs_nom(r.nom)
        if k:
            kv.setdefault(r.question_id, k)

    if enriched:
        # 2) MIN15 extract
        rows_min15 = session.execute(sqltext("""
            SELECT question_id, sous_direction, bureau FROM question_bureau_extract
        """)).all()
        for r in rows_min15:
            k = canonical_from_extract(r.sous_direction, r.bureau)
            if k and r.question_id not in kv:  # ne pas écraser DGCS existant
                kv[r.question_id] = k

    # Bulk insert
    if kv:
        session.execute(sqltext(f"""
            INSERT INTO {table} (question_id, bureau_key) VALUES (:qid, :bk)
            ON CONFLICT (question_id) DO NOTHING
        """), [{"qid": q, "bk": k} for q, k in kv.items()])
    session.execute(sqltext(f"CREATE INDEX ON {table}(question_id)"))
    logger.info("Materialized %s : %d voters (enriched=%s)", table, len(kv), enriched)


VOTERS_SQL_FMT = """
    SELECT question_id, bureau_key
    FROM {table}
    WHERE question_id <> :src_qid
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="data/eval_bureau_min15_ab.json")
    args = ap.parse_args()

    logger.info("Loading test items from question_bureau_extract …")
    with db.get_session() as session:
        rows = session.execute(sqltext("""
            SELECT question_id, direction_txt, sous_direction, bureau
              FROM question_bureau_extract
             ORDER BY question_id
        """)).all()

    # Group rows by qid — a QE may have multiple (direction, bureau)
    by_qid: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for r in rows:
        k = canonical_from_extract(r.sous_direction, r.bureau)
        if k:
            by_qid[r.question_id].append((r.direction_txt, k))

    test_items = []
    for qid, entries in by_qid.items():
        gt_keys = {e[1] for e in entries}
        gt_dirs = [e[0] for e in entries]
        test_items.append((qid, _point_id(qid), gt_keys, gt_dirs))
    logger.info("Test items with resolvable canonical bureau: %d", len(test_items))
    if args.limit:
        test_items = test_items[: args.limit]
        logger.info("Capped to %d", len(test_items))

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hit_baseline": 0, "hit_enriched": 0})

    with db.get_session() as session:
        # Materialize both voter tables
        make_voters_table(session, "tmp_voters_baseline", enriched=False)
        make_voters_table(session, "tmp_voters_enriched", enriched=True)
        voters_baseline_sql = VOTERS_SQL_FMT.format(table="tmp_voters_baseline")
        voters_enriched_sql = VOTERS_SQL_FMT.format(table="tmp_voters_enriched")

        for qid, pt_id, gt_keys, gt_dirs in tqdm(test_items, unit="qe"):
            try:
                pred_b = knn_top1_bureau(session, pt_id, qid, voters_baseline_sql)
                pred_e = knn_top1_bureau(session, pt_id, qid, voters_enriched_sql)
            except Exception as exc:
                logger.warning("kNN failed for %s: %s", qid, exc)
                session.rollback()
                # re-materialise voters (rollback dropped temp tables)
                make_voters_table(session, "tmp_voters_baseline", enriched=False)
                make_voters_table(session, "tmp_voters_enriched", enriched=True)
                continue
            for d in gt_dirs:
                s = stats[d]
                s["n"] += 1
                if pred_b in gt_keys:
                    s["hit_baseline"] += 1
                if pred_e in gt_keys:
                    s["hit_enriched"] += 1

    logger.info("")
    logger.info("%-8s %-8s %-14s %-14s %-8s", "dir", "n", "baseline", "enriched", "delta")
    total_n = total_b = total_e = 0
    for dname in sorted(stats.keys()):
        s = stats[dname]
        pct_b = 100.0 * s["hit_baseline"] / s["n"] if s["n"] else 0
        pct_e = 100.0 * s["hit_enriched"] / s["n"] if s["n"] else 0
        logger.info("%-8s %-8d %-14s %-14s %+.1f",
                    dname, s["n"],
                    f"{s['hit_baseline']} ({pct_b:.1f}%)",
                    f"{s['hit_enriched']} ({pct_e:.1f}%)",
                    pct_e - pct_b)
        total_n += s["n"]; total_b += s["hit_baseline"]; total_e += s["hit_enriched"]
    pct_b = 100.0 * total_b / total_n if total_n else 0
    pct_e = 100.0 * total_e / total_n if total_n else 0
    logger.info("%-8s %-8d %-14s %-14s %+.1f", "TOTAL", total_n,
                f"{total_b} ({pct_b:.1f}%)",
                f"{total_e} ({pct_e:.1f}%)",
                pct_e - pct_b)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "n_test_items": len(test_items),
        "per_direction": dict(stats),
        "total": {"n": total_n, "hit_baseline": total_b, "hit_enriched": total_e,
                  "pct_baseline": round(pct_b, 2), "pct_enriched": round(pct_e, 2),
                  "delta_pts": round(pct_e - pct_b, 2)},
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
