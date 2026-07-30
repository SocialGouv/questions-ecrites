#!/usr/bin/env python3
"""Compute rank distribution of the FIRST GT-mate in the reranked top-K.

Complements hit@K by answering "when the mate is found, where is it?".
An algo can have hit@20 = 92 % while placing 60 % of mates beyond rank
5 — which explains why agents feel results are "nul" even though our
coverage metric looks great."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from tqdm import tqdm

from qe import db
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings
from qe.models import Question, Reponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _point_id(qid: str) -> str:
    return str(UUID(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:32]))


def _groups() -> list[tuple[str, list[str], str]]:
    with db.get_session() as s:
        return [
            (r[0], r[1], r[2].isoformat())
            for r in s.execute(
                select(Question.reponse_id, func.array_agg(Question.id),
                       Reponse.date_reponse_jo)
                .join(Reponse, Reponse.id == Question.reponse_id)
                .where(Question.reponse_id.is_not(None))
                .where(Reponse.date_reponse_jo.is_not(None))
                .group_by(Question.reponse_id, Reponse.date_reponse_jo)
                .having(func.count() >= 2)
            ).all()
        ]


def _meta():
    with db.get_session() as s:
        rows = s.execute(
            select(Question.id, Question.date_publication_jo,
                   Reponse.date_reponse_jo).select_from(Question)
            .join(Reponse, Reponse.id == Question.reponse_id, isouter=True)
        ).all()
    pub, rep = {}, {}
    for qid, p, r in rows:
        if p is None:
            continue
        pub[qid] = p.isoformat()
        rep[qid] = r.isoformat() if r else None
    return pub, rep


def _texts(ids):
    with db.get_session() as s:
        return {r[0]: (r[1] or "") for r in s.execute(
            select(Question.id, Question.texte_question)
            .where(Question.id.in_(ids))
        ).all()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output", type=Path,
                    default=Path("data/eval_encours_rank_distribution.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    settings = get_settings()
    store = PgvectorClient()
    rr = RerankClient(base_url=settings.albert_base_url,
                      model=settings.albert_rerank_model,
                      api_key=settings.albert_api_key)

    logger.info("Loading …")
    groups = _groups()
    pub, rep = _meta()
    random.Random(args.seed).shuffle(groups)
    groups = groups[: args.sample]

    ranks: list[int | None] = []      # rank of first mate, or None if miss
    for _, qids, _drep in tqdm(groups, unit="grp"):
        for src in qids:
            t = pub.get(src)
            if t is None:
                continue
            mates = {m for m in qids
                     if m != src and pub.get(m) and pub[m] <= t}
            if not mates:
                continue
            pt = store.get_point("questions_opendata", _point_id(src),
                                 with_vectors=True)
            if pt is None:
                continue
            raw = store.search("questions_opendata", pt["vector"],
                               args.pool * 4, filter=None, score_threshold=0.0)
            cand = []
            for h in raw:
                cid = h["payload"].get("question_id")
                if not cid or cid == src:
                    continue
                cp = pub.get(cid)
                if cp is None or cp > t:
                    continue
                cr = rep.get(cid)
                if cr is not None and cr <= t:
                    continue
                cand.append(cid)
                if len(cand) >= args.pool:
                    break
            if not cand:
                ranks.append(None)
                continue
            tx = _texts([src, *cand])
            try:
                out = rr.rerank(tx.get(src, ""),
                                [tx.get(c, "") for c in cand],
                                top_n=args.top_k)
                top = [cand[o["index"]] for o in out]
            except Exception:
                top = cand[: args.top_k]
            first = None
            for i, cid in enumerate(top, start=1):
                if cid in mates:
                    first = i
                    break
            ranks.append(first)

    n = len(ranks)
    found = [r for r in ranks if r is not None]
    dist = Counter(found)
    buckets = {
        "@1": sum(1 for r in found if r == 1),
        "@2-3": sum(1 for r in found if 2 <= r <= 3),
        "@4-5": sum(1 for r in found if 4 <= r <= 5),
        "@6-10": sum(1 for r in found if 6 <= r <= 10),
        "@11-20": sum(1 for r in found if 11 <= r <= 20),
        ">20 (miss)": n - len(found),
    }
    result = {
        "n_queries": n,
        "n_found": len(found),
        "hit_at_20": round(len(found) / n, 4) if n else 0.0,
        "mean_first_rank": round(sum(found) / len(found), 2) if found else None,
        "median_first_rank": sorted(found)[len(found) // 2] if found else None,
        "buckets": buckets,
        "buckets_pct": {k: round(v / n * 100, 1) for k, v in buckets.items()},
        "raw_dist": dict(sorted(dist.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    logger.info("Wrote %s", args.output)
    for k, v in result["buckets_pct"].items():
        logger.info("  %-12s %5.1f %%", k, v)
    logger.info("mean rank first-mate = %s", result["mean_first_rank"])


if __name__ == "__main__":
    main()
