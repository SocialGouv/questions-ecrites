#!/usr/bin/env python3
"""Evaluate allotissement retrieval with the *real* operating constraint:
at the instant an agent sees a newly-arrived source QE (time t =
`date_publication_jo(source)`), the candidate pool is restricted to
questions that are still `EN_COURS` at t.

That means, for a candidate q :
    date_publication_jo(q)              <= t   (already existed)
    (q.reponse_id IS NULL) OR
    (date_reponse_jo(q.reponse_id)      >  t)  (not yet answered at t)

Ground-truth mates of the source are also filtered the same way :
only mates that were already published at t are counted (mates sharing
the same reponse_id are, by construction, not answered at t since the
source itself will be answered on the shared date, which is > t).

The prior `eval_allotissement.py` script anchored candidates to
"published on or before the response date" — this includes questions
that were *already answered* at t, and therefore never in the pool the
agent actually queries. The number here is the honest one.

Usage:
    poetry run python scripts/eval_realistic_encours.py \\
        --collection questions_opendata \\
        --sample 500 \\
        --output data/eval_encours_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from tqdm import tqdm

from qe import db
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.models import Question, Reponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_POOL = 500       # candidates fed to the reranker
DEFAULT_OVERFETCH = 4    # over-fetch multiplier from pgvector before filter
TOP_KS = (1, 3, 5, 10, 20)
MAX_K = max(TOP_KS)


def _point_id(question_id: str, variant_tag: str | None = None) -> str:
    key = f"{question_id}::{variant_tag}" if variant_tag else question_id
    return str(UUID(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]))


def _load_allotment_groups() -> list[tuple[str, list[str], str]]:
    """Return groups of size >= 2 : (reponse_id, [qids], date_reponse_iso)."""
    with db.get_session() as session:
        stmt = (
            select(
                Question.reponse_id,
                func.array_agg(Question.id),
                Reponse.date_reponse_jo,
            )
            .join(Reponse, Reponse.id == Question.reponse_id)
            .where(Question.reponse_id.is_not(None))
            .where(Reponse.date_reponse_jo.is_not(None))
            .group_by(Question.reponse_id, Reponse.date_reponse_jo)
            .having(func.count() >= 2)
        )
        return [
            (row[0], row[1], row[2].isoformat())
            for row in session.execute(stmt).all()
        ]


def _load_dgcs_groups(csv_path: Path) -> list[tuple[str, list[str], str]]:
    """Load DGCS groups from a CSV. Auto-detects two schemas:
    - lot_id, question_id, …             → real DGCS allotments extracted
      from the "Commentaires" column of Salomé's Excel (`Lot AN 15650`)
    - dgcs_group_id, question_id, …      → legacy: groups by exact objet
      (mostly thematic, NOT real allotments)"""
    import csv
    from collections import defaultdict
    buckets: dict[str, list[str]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key = "lot_id" if "lot_id" in reader.fieldnames else "dgcs_group_id"
        for row in reader:
            buckets[row[key]].append(row["question_id"])
    return [(gid, qids, "") for gid, qids in buckets.items() if len(qids) >= 2]


def _load_allotment_groups_dgcs_scoped(csv_path: Path):
    """Ministerial hash+date allotments restricted to the DGCS perimeter.

    Each group = a reponse_id shared by >= 2 QE that are ALSO in
    Salomé's DGCS CSV. This gives us a fiable GT (hash+date validated
    by LLM at ~98 %) on the scope that matters for the eval (agents'
    real queue). Returns (reponse_id, [qids in DGCS], date_reponse_iso)."""
    import csv
    from collections import defaultdict
    from sqlalchemy import text as sqltext
    dgcs_qids: set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dgcs_qids.add(row["question_id"])
    with db.get_session() as session:
        rows = session.execute(sqltext(
            """
            SELECT q.reponse_id, q.id, r.date_reponse_jo
              FROM questions q
              JOIN reponses r ON r.id = q.reponse_id
             WHERE q.reponse_id IS NOT NULL
               AND r.date_reponse_jo IS NOT NULL
               AND q.id = ANY(:ids)
            """
        ), {"ids": list(dgcs_qids)}).all()
    buckets: dict[str, tuple[list[str], str]] = defaultdict(lambda: ([], ""))
    for rid, qid, date_rep in rows:
        arr, _ = buckets[rid]
        arr.append(qid)
        buckets[rid] = (arr, date_rep.isoformat())
    return [(rid, arr, drep) for rid, (arr, drep) in buckets.items()
            if len(arr) >= 2]


def _load_meta() -> tuple[dict[str, str], dict[str, str | None]]:
    """Return ({qid: date_publication_jo}, {qid: date_reponse_jo|None})."""
    with db.get_session() as session:
        rows = session.execute(
            select(
                Question.id,
                Question.date_publication_jo,
                Reponse.date_reponse_jo,
            ).select_from(Question).join(
                Reponse, Reponse.id == Question.reponse_id, isouter=True
            )
        ).all()
    pub: dict[str, str] = {}
    rep: dict[str, str | None] = {}
    for qid, pub_d, rep_d in rows:
        if pub_d is None:
            continue
        pub[qid] = pub_d.isoformat()
        rep[qid] = rep_d.isoformat() if rep_d else None
    return pub, rep


def _load_effective_direction() -> dict[str, int]:
    """Return {qid: direction_id} using COALESCE(reelle, algo) — matches
    what qe-front's `getEffectiveDirection` returns. Only entries with
    a resolvable direction appear in the map. Raw SQL to avoid coupling
    on the branch-local ORM (direction_algo_id was added out-of-band)."""
    from sqlalchemy import text as sqltext
    with db.get_session() as session:
        rows = session.execute(
            sqltext(
                """
                SELECT q.id,
                       COALESCE(qa.direction_reelle_id, q.direction_algo_id) AS eff
                  FROM questions q
                  LEFT JOIN question_attributions qa ON qa.question_id = q.id
                 WHERE COALESCE(qa.direction_reelle_id, q.direction_algo_id) IS NOT NULL
                """
            )
        ).all()
    return {qid: int(eff) for qid, eff in rows}


def _load_texts(ids: list[str], field: str = "texte_question") -> dict[str, str]:
    """Return {qid: text}. `field` is 'texte_question' (default) or
    'question_extraite' — for the latter, fall back to texte_question
    when the extract is empty/null (2 % of the corpus)."""
    from sqlalchemy import text as sqltext
    if field == "question_extraite":
        stmt = sqltext(
            "SELECT id, COALESCE(NULLIF(question_extraite, ''), texte_question) "
            "FROM questions WHERE id = ANY(:ids)"
        )
    else:
        stmt = sqltext(
            "SELECT id, texte_question FROM questions WHERE id = ANY(:ids)"
        )
    with db.get_session() as session:
        rows = session.execute(stmt, {"ids": ids}).all()
    return {r[0]: (r[1] or "") for r in rows}


def _hits_at_k(top: list[str], mates: set[str]) -> dict[int, int]:
    return {k: int(bool(mates & set(top[:k]))) for k in TOP_KS}


def _recall_at_k(top: list[str], mates: set[str]) -> dict[int, float]:
    n = len(mates)
    if n == 0:
        return {k: 0.0 for k in TOP_KS}
    return {k: len(mates & set(top[:k])) / n for k in TOP_KS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--variant-tag", default=None)
    parser.add_argument("--sample", type=int, default=500,
                        help="Number of groups to sample (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL)
    parser.add_argument("--overfetch", type=int, default=DEFAULT_OVERFETCH)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--dgcs-csv",
        type=Path,
        default=None,
        help="Use Salomé's DGCS-annotated groups instead of the ministerial "
             "hash+date allotments. Path to dgcs_groups.csv. WARNING: most "
             "DGCS groups are thematic, not real allotments — prefer "
             "--dgcs-scoped-allots for a proper allotment eval on DGCS scope.",
    )
    parser.add_argument(
        "--dgcs-scoped-allots",
        type=Path,
        default=None,
        help="Path to dgcs_groups.csv — same time-anchor + rerank pipeline, "
             "but ground truth = hash+date allotments RESTRICTED to QE that "
             "appear in the DGCS CSV. Fiable GT (LLM-validated at 98 %) on "
             "the perimeter that matters for the eval.",
    )
    parser.add_argument(
        "--rerank-field",
        choices=["texte_question", "question_extraite"],
        default="texte_question",
        help="Which column to feed to the reranker. `question_extraite` "
             "uses the LLM-extracted actual question (with texte_question "
             "fallback when empty). Retrieval always uses the embedding "
             "collection unchanged.",
    )
    parser.add_argument(
        "--filter-direction",
        action="store_true",
        help="Apply direction filter (auto mode: source's effective direction "
             "→ keep only candidates with the same effective direction). "
             "Matches the UI's Auto option; sources without a resolvable "
             "direction fall back to no filter (also matches the UI).",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("data/eval_encours_baseline.json"))
    args = parser.parse_args()

    settings = get_settings()
    store = PgvectorClient()
    reranker = None
    if not args.no_rerank:
        reranker = RerankClient(
            base_url=settings.albert_base_url,
            model=settings.albert_rerank_model,
            api_key=settings.albert_api_key,
        )

    logger.info("Loading allotment groups + full question meta …")
    if args.dgcs_scoped_allots:
        groups = _load_allotment_groups_dgcs_scoped(args.dgcs_scoped_allots)
        logger.info("Using ministerial allotments RESTRICTED to DGCS scope "
                    "(%s) — %d groups", args.dgcs_scoped_allots, len(groups))
    elif args.dgcs_csv:
        groups = _load_dgcs_groups(args.dgcs_csv)
        logger.info("Using RAW DGCS GT from %s (thematic groups incl.)",
                    args.dgcs_csv)
    else:
        groups = _load_allotment_groups()
    pub_date, rep_date = _load_meta()
    dir_map: dict[str, int] = {}
    if args.filter_direction:
        dir_map = _load_effective_direction()
        logger.info("Direction map loaded: %d qids with effective direction",
                    len(dir_map))
    logger.info("Groups: %d ; questions with pub_date: %d",
                len(groups), len(pub_date))

    if args.sample and args.sample < len(groups):
        random.Random(args.seed).shuffle(groups)
        groups = groups[: args.sample]
        logger.info("Sampled %d groups", len(groups))

    hit_counts = {k: 0 for k in TOP_KS}
    recall_sums = {k: 0.0 for k in TOP_KS}
    queries = 0
    skipped_no_source_pub = 0
    skipped_no_mates_at_t = 0
    skipped_no_vector = 0
    source_no_direction = 0  # counted when filter is on but source has no dir

    for _, qids, date_reponse in tqdm(groups, unit="grp"):
        for src in qids:
            t = pub_date.get(src)
            if t is None:
                skipped_no_source_pub += 1
                continue
            mates_all = set(qids) - {src}
            # Mates must already be published at t
            mates = {m for m in mates_all if pub_date.get(m) and pub_date[m] <= t}
            if not mates:
                skipped_no_mates_at_t += 1
                continue

            src_pt = store.get_point(
                args.collection, _point_id(src, args.variant_tag),
                with_vectors=True,
            )
            if src_pt is None:
                skipped_no_vector += 1
                continue

            # Over-fetch from pgvector; we'll drop candidates that were
            # already answered at t (that check requires the DB, so we
            # can't push it server-side without more schema plumbing).
            raw = store.search(
                args.collection,
                src_pt["vector"],
                args.pool * args.overfetch,
                filter=None,
                score_threshold=0.0,
            )

            # Direction filter (auto mode) — matches UI's Auto option.
            # If source has no resolvable direction, fall back to no
            # filter (like the UI does). Candidates without a direction
            # are dropped when a filter is applied (matches
            # COALESCE(reelle, algo) IN (…) semantics of the SQL filter).
            required_dir: int | None = None
            if args.filter_direction:
                required_dir = dir_map.get(src)
                if required_dir is None:
                    source_no_direction += 1

            candidates: list[str] = []
            for h in raw:
                cid = h["payload"].get("question_id")
                if not cid or cid == src:
                    continue
                cpub = pub_date.get(cid)
                if cpub is None or cpub > t:
                    continue
                crep = rep_date.get(cid)
                # Answered before or at t → not in pool at t
                if crep is not None and crep <= t:
                    continue
                if required_dir is not None:
                    cd = dir_map.get(cid)
                    if cd is None or cd != required_dir:
                        continue
                candidates.append(cid)
                if len(candidates) >= args.pool:
                    break

            top: list[str] = candidates[:MAX_K]
            if reranker and candidates:
                texts = _load_texts([src, *candidates], field=args.rerank_field)
                q_text = texts.get(src, "")
                docs = [texts.get(c, "") for c in candidates]
                try:
                    rr = reranker.rerank(q_text, docs, top_n=MAX_K)
                    top = [candidates[r["index"]] for r in rr]
                except Exception as exc:
                    logger.warning("Rerank failed on %s: %s", src, exc)

            h = _hits_at_k(top, mates)
            r = _recall_at_k(top, mates)
            for k in TOP_KS:
                hit_counts[k] += h[k]
                recall_sums[k] += r[k]
            queries += 1

    results = {
        "n_queries": queries,
        "skipped_no_source_pub": skipped_no_source_pub,
        "skipped_no_mates_at_t": skipped_no_mates_at_t,
        "skipped_no_vector": skipped_no_vector,
        "source_no_direction_fallback": source_no_direction,
        **{f"hit_at_{k}": round(hit_counts[k] / queries, 4) if queries else 0.0
           for k in TOP_KS},
        **{f"recall_at_{k}": round(recall_sums[k] / queries, 4) if queries else 0.0
           for k in TOP_KS},
    }
    payload = {
        "conditions": {
            "gt": ("Real allotments hash(texte_reponse)+date, mates filtered "
                   "to those already published at source date"),
            "corpus": "full ~260k",
            "time_anchor": ("STRICT EN_COURS at date_publication_jo(source): "
                            "candidates must be pub_date<=t AND (no reponse "
                            "OR reponse_date>t)"),
            "rerank": None if args.no_rerank else "Albert",
            "pool": args.pool,
            "overfetch_x": args.overfetch,
            "sample_groups": args.sample or "all",
            "collection": args.collection,
            "variant_tag": args.variant_tag,
            "filter_direction": args.filter_direction,
            "rerank_field": args.rerank_field,
            "gt_source": ("dgcs_csv:" + str(args.dgcs_csv)) if args.dgcs_csv
                          else "ministerial hash+date",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s", args.output)
    logger.info("Hit@3=%.3f  Hit@20=%.3f  (n=%d)",
                results["hit_at_3"], results["hit_at_20"], queries)


if __name__ == "__main__":
    main()
