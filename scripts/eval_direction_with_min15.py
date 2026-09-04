#!/usr/bin/env python3
"""A/B eval : algo d'attribution direction AVEC vs SANS MIN15 dans le
training set du kNN.

Mesure le gain effectif de nourrir `question_bureau_extract` au kNN,
sur le set de test constitué de `question_bureau_extract` lui-même
(source-of-truth extraite des workflows MIN15, indépendante du training
actuel qui n'utilise que `question_real_attributions`).

Pipeline pour chaque QE test :
1. On récupère son vecteur.
2. On fait un cosine top-K sur le corpus, en JOIN avec la source des
   voters (voir plus bas), en EXCLUANT la QE elle-même du pool.
3. On vote pondéré par similarité → top-1/3 direction.
4. On compare avec la direction MIN15 de référence.

Deux sources de voters à tester :
- **baseline** : `question_real_attributions.direction_reelle_id` seul (le training
  actuel de la prod)
- **enriched** : UNION(direction_reelle_id, question_bureau_extract.direction_txt→id)

Usage :
    poetry run python scripts/eval_direction_with_min15.py [--limit 500]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from sqlalchemy import text as sqltext

from qe import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KNN = 15  # matches production
HNSW_EF_SEARCH = 1000  # matches production (attributionAlgo.ts) as of the partial-index fix
TOP_K = 3

# Directions we test on — those where the algo has ANY training signal.
# DGE / DGCCRF are excluded because they have no attributions at all,
# so the algo can't fairly be measured on them.
TESTABLE_DIRECTIONS = ("DGCS", "DSS", "DGOS", "DGS", "DGEFP", "DGT")


# --- Voter SQL fragments -----------------------------------------------------
#
# Both are wrapped in a subquery producing (question_id, direction_id).
# `EXCLUDE_QID` placeholder is replaced at runtime.

VOTERS_BASELINE = """
    SELECT qa.question_id, qa.direction_reelle_id AS direction_id
    FROM question_real_attributions qa
    WHERE qa.direction_reelle_id IS NOT NULL
      AND qa.question_id <> :src_qid
"""

# MIN15 direction_txt is a text; map to directions.id via directions.nom.
# Only directions we can map (nom exact match) contribute.
VOTERS_ENRICHED = """
    SELECT qa.question_id, qa.direction_reelle_id AS direction_id
    FROM question_real_attributions qa
    WHERE qa.direction_reelle_id IS NOT NULL
      AND qa.question_id <> :src_qid
    UNION
    SELECT qbe.question_id, d.id AS direction_id
    FROM question_bureau_extract qbe
    JOIN directions d ON d.nom = qbe.direction_txt
    WHERE qbe.question_id <> :src_qid
"""


_EF_SEARCH_SET = False


def knn_top1(
    session, src_pt_id: str, src_qid: str, voters_sql: str, use_partial_index: bool = False
) -> int | None:
    """Run the kNN vote for `src_qid`, return the winning direction_id.

    `use_partial_index` adds production's `v.has_direction_attribution`
    predicate so the planner can pick `vec_q_hnsw_direction_idx` instead of
    the full index — only valid for the baseline arm, whose voters are
    exactly the rows that flag covers. The enriched arm's MIN15-only voters
    aren't covered by the flag, so adding it there would silently drop them.
    """
    global _EF_SEARCH_SET
    if not _EF_SEARCH_SET:
        session.execute(sqltext(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"))
        _EF_SEARCH_SET = True
    partial_predicate = " AND v.has_direction_attribution" if use_partial_index else ""
    sql = sqltext(f"""
        WITH src AS (
            SELECT vector FROM vec_questions_opendata WHERE id = :src_pt_id
        ),
        voters AS ({voters_sql}),
        neighbours AS (
            SELECT
                voters.direction_id AS direction_id,
                1 - (v.vector <=> (SELECT vector FROM src)) AS similarity
            FROM vec_questions_opendata v
            JOIN voters ON voters.question_id = v.payload ->> 'question_id'
            WHERE v.id <> :src_pt_id{partial_predicate}
            ORDER BY v.vector <=> (SELECT vector FROM src)
            LIMIT {KNN}
        )
        SELECT direction_id, SUM(similarity) AS vote
        FROM neighbours
        GROUP BY direction_id
        ORDER BY vote DESC
        LIMIT 1
    """)
    row = session.execute(sql, {"src_pt_id": src_pt_id, "src_qid": src_qid}).first()
    return int(row.direction_id) if row else None


# Point-id derivation (same as embed_questions.py).
def _point_id(qid: str) -> str:
    import hashlib
    from uuid import UUID
    return str(UUID(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:32]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap total test QE (0 = all).")
    ap.add_argument("--output", default="data/eval_direction_min15_ab.json")
    args = ap.parse_args()

    logger.info("Loading test set from question_bureau_extract …")
    with db.get_session() as session:
        rows = session.execute(sqltext("""
            SELECT question_id, array_agg(direction_txt) AS gt_dirs
              FROM question_bureau_extract
             WHERE direction_txt = ANY(:testable)
             GROUP BY question_id
             ORDER BY question_id
        """), {"testable": list(TESTABLE_DIRECTIONS)}).all()
        dir_id_by_name = {
            r.nom: r.id for r in session.execute(sqltext(
                "SELECT id, nom FROM directions"
            )).all()
        }
    logger.info("Test set: %d QE", len(rows))

    test_items = []
    for r in rows:
        pt_id = _point_id(r.question_id)
        gt_ids = {dir_id_by_name[n] for n in r.gt_dirs if n in dir_id_by_name}
        if gt_ids:
            test_items.append((r.question_id, pt_id, gt_ids, r.gt_dirs))
    logger.info("Testable (with resolvable gt direction_id): %d", len(test_items))
    if args.limit:
        test_items = test_items[: args.limit]
        logger.info("Capped to %d", len(test_items))

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hit_baseline": 0, "hit_enriched": 0})

    from tqdm import tqdm
    with db.get_session() as session:
        for src_qid, pt_id, gt_ids, gt_names in tqdm(test_items, unit="qe"):
            try:
                pred_baseline = knn_top1(
                    session, pt_id, src_qid, VOTERS_BASELINE, use_partial_index=True
                )
                pred_enriched = knn_top1(session, pt_id, src_qid, VOTERS_ENRICHED)
            except Exception as exc:
                logger.warning("kNN failed for %s: %s", src_qid, exc)
                session.rollback()
                continue
            for dname in gt_names:
                s = stats[dname]
                s["n"] += 1
                if pred_baseline in gt_ids:
                    s["hit_baseline"] += 1
                if pred_enriched in gt_ids:
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
        total_n += s["n"]
        total_b += s["hit_baseline"]
        total_e += s["hit_enriched"]
    pct_b = 100.0 * total_b / total_n if total_n else 0
    pct_e = 100.0 * total_e / total_n if total_n else 0
    logger.info("%-8s %-8d %-14s %-14s %+.1f", "TOTAL", total_n,
                f"{total_b} ({pct_b:.1f}%)",
                f"{total_e} ({pct_e:.1f}%)",
                pct_e - pct_b)

    import json
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_test_items": len(test_items),
        "per_direction": dict(stats),
        "total": {
            "n": total_n,
            "hit_baseline": total_b,
            "hit_enriched": total_e,
            "pct_baseline": round(pct_b, 2),
            "pct_enriched": round(pct_e, 2),
            "delta_pts": round(pct_e - pct_b, 2),
        },
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
