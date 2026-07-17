#!/usr/bin/env python3
"""Evaluate the allotissement pipeline the way the perf report does :
retrieve top-N candidates from a pgvector collection, rerank them with
Albert, and measure "hit@20" — at least one true sibling in the top-20.

The perf report headline number (74.4% hit@20 on 13 444 groups) is
computed by /api/admin/performance/allotissements in qe-front. This
script reproduces that logic in Python so we can point it at any
pgvector collection (`--collection`) and A/B different embeddings on
the same ground truth.

Usage:
    poetry run python scripts/eval_allotissement.py \\
        --collection questions_experiments \\
        --legislature 17 \\
        --output data/eval_alloti_q_only.json

Metrics:
    hit@20 : fraction of query questions with >= 1 sibling in top-20
    recall@20 : mean per-question (# siblings found / # siblings total)

For each query we do:
    1. fetch its own vector from `--collection` (skip if missing)
    2. top-500 cosine neighbours from the same collection
    3. Albert rerank on the pool → top-20
    4. compare to the sibling set derived from reponse_id
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from tqdm import tqdm

from qe import db
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.models import Question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_POOL = 500  # matches the production retrieval pool
DEFAULT_TOP_K = 20  # matches the display count / perf metric definition


def _question_point_id(question_id: str, variant_tag: str | None = None) -> str:
    """Same deterministic UUID as embed_questions.py.

    When variant_tag is set, it's mixed into the hash so multiple
    variants can coexist in the same collection.
    """
    key = f"{question_id}::{variant_tag}" if variant_tag else question_id
    return str(UUID(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]))


def _load_sibling_groups(
    legislature: int | None,
) -> list[tuple[str, list[str], str | None]]:
    """[(reponse_id, [question_id, ...], date_reponse_jo), ...].

    We include the response date so callers can time-anchor the eval :
    at the moment the historical allotment was decided, the algo could
    only see questions posted on or before that date.
    """
    from qe.models import Reponse
    with db.get_session() as session:
        stmt = (
            select(
                Question.reponse_id,
                func.array_agg(Question.id),
                Reponse.date_reponse_jo,
            )
            .join(Reponse, Reponse.id == Question.reponse_id)
            .where(Question.reponse_id.is_not(None))
        )
        if legislature is not None:
            stmt = stmt.where(Question.legislature == legislature)
        stmt = stmt.group_by(Question.reponse_id, Reponse.date_reponse_jo).having(func.count() >= 2)
        return [
            (row[0], row[1], row[2].isoformat() if row[2] else None)
            for row in session.execute(stmt).all()
        ]


def _load_texts(ids: list[str]) -> dict[str, str]:
    """Return {question_id: texte_question} for reranking documents."""
    with db.get_session() as session:
        rows = session.execute(
            select(Question.id, Question.texte_question).where(Question.id.in_(ids))
        ).all()
    return {r[0]: (r[1] or "") for r in rows}


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the allotissement pipeline (retrieve + rerank) against "
            "reponse_id-shared sibling groups."
        )
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--variant-tag", default=None,
        help="Variant tag used at embedding time — mixed into the point_id hash.",
    )
    parser.add_argument(
        "--legislature",
        type=int,
        default=None,
        help="Restrict ground-truth groups to one legislature (for pilot batches).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the Albert rerank step (measure raw cosine hit@K).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_allotissement.json"),
    )
    args = parser.parse_args()

    settings = get_settings()
    vector_store = PgvectorClient()
    reranker = None
    if not args.no_rerank:
        reranker = RerankClient(
            base_url=settings.albert_base_url,
            model=settings.albert_rerank_model,
            api_key=settings.albert_api_key,
        )

    groups = _load_sibling_groups(args.legislature)
    logger.info("Loaded %d sibling groups (legislature=%s)", len(groups), args.legislature)

    hit_count = 0
    recall_sum = 0.0
    query_count = 0
    missing = 0
    skipped_no_date = 0

    for reponse_id, qids, date_reponse in tqdm(groups, unit="group"):
        if not date_reponse:
            skipped_no_date += 1
            continue
        for query_id in qids:
            siblings = set(qids) - {query_id}
            if not siblings:
                continue

            # Fetch source vector
            src = vector_store.get_point(
                args.collection, _question_point_id(query_id, args.variant_tag), with_vectors=True
            )
            if src is None:
                missing += 1
                continue

            # Retrieve a bigger raw pool (2×) since we'll drop
            # anything posted AFTER the historical allotment date —
            # future questions would leak information the algo could
            # never have seen at decision time.
            hits = vector_store.search(
                args.collection,
                src["vector"],
                args.pool * 2,
                filter=None,
                score_threshold=0.0,
            )
            # Time-anchored filter : keep only questions dated on or
            # before the response date, drop the query itself.
            candidate_ids: list[str] = []
            for h in hits:
                payload = h.get("payload", {})
                cid = payload.get("question_id")
                cdate = payload.get("date_publication_jo")
                if not cid or cid == query_id:
                    continue
                if cdate and cdate > date_reponse:
                    continue
                candidate_ids.append(cid)
                if len(candidate_ids) >= args.pool:
                    break

            # Optional rerank
            top_ids: list[str] = candidate_ids[: args.top_k]
            if reranker and candidate_ids:
                texts_map = _load_texts([query_id, *candidate_ids])
                query_text = texts_map.get(query_id, "")
                docs = [texts_map.get(c, "") for c in candidate_ids]
                try:
                    reranked = reranker.rerank(query_text, docs, top_n=args.top_k)
                    # `reranked` is a list of {"index": i, "relevance_score": s}
                    top_ids = [candidate_ids[r["index"]] for r in reranked]
                except Exception as exc:
                    logger.warning("Rerank failed for %s: %s — falling back to cosine top-K", query_id, exc)

            found = siblings & set(top_ids)
            hit_count += 1 if found else 0
            recall_sum += len(found) / len(siblings)
            query_count += 1

    hit_at_k = hit_count / query_count if query_count else 0.0
    recall_at_k = recall_sum / query_count if query_count else 0.0

    report = {
        "summary": {
            "collection": args.collection,
            "pool": args.pool,
            "top_k": args.top_k,
            "legislature": args.legislature,
            "reranked": not args.no_rerank,
            "total_query_questions": query_count,
            "missing_from_collection": missing,
            "groups_skipped_no_date": skipped_no_date,
            f"hit_at_{args.top_k}": round(hit_at_k, 6),
            f"recall_at_{args.top_k}": round(recall_at_k, 6),
        }
    }
    logger.info(
        "Hit@%d = %.4f    Recall@%d = %.4f    (n=%d queries, %d missing)",
        args.top_k, hit_at_k, args.top_k, recall_at_k, query_count, missing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
