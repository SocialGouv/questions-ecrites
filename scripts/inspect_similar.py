#!/usr/bin/env python3
"""Inspect the top-K similar suggestions for a given QE, exactly as the
qe-front /similar endpoint would return them : same time-anchor
(candidates still EN_COURS at date_publication_jo(source)), same
pool=100, same Albert rerank. Prints objets + rerank score + whether
each result is a real allotment mate.

    poetry run python scripts/inspect_similar.py AN-17-QE-13335
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from uuid import UUID

from sqlalchemy import text as sqltext

from qe import db
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings

POOL = 100
TOP_K = 20


def _point_id(qid: str) -> str:
    return str(UUID(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:32]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qid")
    args = parser.parse_args()

    settings = get_settings()
    store = PgvectorClient()
    reranker = RerankClient(
        base_url=settings.albert_base_url,
        model=settings.albert_rerank_model,
        api_key=settings.albert_api_key,
    )

    with db.get_session() as session:
        row = session.execute(sqltext(
            """
            SELECT q.objet, q.texte_question, q.date_publication_jo,
                   q.reponse_id, r.date_reponse_jo
              FROM questions q
              LEFT JOIN reponses r ON r.id = q.reponse_id
             WHERE q.id = :qid
            """
        ), {"qid": args.qid}).one_or_none()
    if row is None:
        print(f"Not found: {args.qid}", file=sys.stderr)
        sys.exit(1)
    src_objet, src_text, src_pub, src_reponse_id, src_rep_date = row
    t_iso = src_pub.isoformat()
    print(f"\n=== SOURCE {args.qid} ===")
    print(f"date_publication_jo : {t_iso}")
    print(f"reponse_id          : {src_reponse_id}")
    print(f"objet               : {src_objet}\n")

    mates: set[str] = set()
    if src_reponse_id:
        with db.get_session() as session:
            mates = {r[0] for r in session.execute(sqltext(
                "SELECT id FROM questions WHERE reponse_id = :rid AND id != :qid"
            ), {"rid": src_reponse_id, "qid": args.qid}).all()}
    print(f"GT-mates (partagent reponse_id) : {sorted(mates) if mates else 'aucun'}\n")

    src_vec = store.get_point("questions_opendata", _point_id(args.qid),
                              with_vectors=True)
    if not src_vec:
        print("No embedding for source.", file=sys.stderr)
        sys.exit(1)

    # Over-fetch, then filter EN_COURS at t
    raw = store.search("questions_opendata", src_vec["vector"], POOL * 5,
                       filter=None, score_threshold=0.0)

    with db.get_session() as session:
        rep_map = dict(session.execute(sqltext(
            """
            SELECT q.id, r.date_reponse_jo
              FROM questions q
              LEFT JOIN reponses r ON r.id = q.reponse_id
            """
        )).all())

    candidates: list[tuple[str, float, str]] = []
    for h in raw:
        cid = h["payload"].get("question_id")
        cpub = h["payload"].get("date_publication_jo")
        if not cid or cid == args.qid or not cpub:
            continue
        if cpub > t_iso:
            continue
        crep = rep_map.get(cid)
        if crep is not None and crep.isoformat() <= t_iso:
            continue
        candidates.append((cid, h["score"], cpub))
        if len(candidates) >= POOL:
            break

    ids = [c[0] for c in candidates]
    with db.get_session() as session:
        obj_map = dict(session.execute(sqltext(
            "SELECT id, COALESCE(objet, '') FROM questions WHERE id = ANY(:ids)"
        ), {"ids": ids}).all())
        txt_map = dict(session.execute(sqltext(
            "SELECT id, texte_question FROM questions WHERE id = ANY(:ids)"
        ), {"ids": ids}).all())

    docs = [txt_map.get(c, "") for c in ids]
    try:
        rr = reranker.rerank(src_text, docs, top_n=TOP_K)
    except Exception as exc:
        print(f"Rerank failed: {exc}", file=sys.stderr)
        sys.exit(1)

    cos_map = {c[0]: c[1] for c in candidates}

    print(f"=== TOP-{TOP_K} après rerank Albert ===\n")
    print(f"{'rk':>3} {'M':>1} {'cos':>5} {'rer':>5}  {'id':<18} {'pub':<10}  objet")
    print("-" * 120)
    for rk, r in enumerate(rr, start=1):
        cid = ids[r["index"]]
        score = r.get("score", r.get("relevance_score", 0.0))
        cos = cos_map.get(cid, 0.0)
        mate = "*" if cid in mates else " "
        pub = next((c[2] for c in candidates if c[0] == cid), "?")
        objet = (obj_map.get(cid) or "")[:80]
        print(f"{rk:>3} {mate:>1} {cos:5.2f} {score:5.2f}  {cid:<18} {pub:<10}  {objet}")

    print(f"\nMates trouvés dans top-{TOP_K} : "
          f"{sum(1 for r in rr if ids[r['index']] in mates)} / {len(mates)}")


if __name__ == "__main__":
    main()
