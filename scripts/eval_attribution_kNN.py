#!/usr/bin/env python3
"""Leave-one-out kNN eval for direction / bureau attribution.

Reproduces the production algorithm described in the perf report:

  For each question that has a real (manually-labelled) direction (resp.
  bureau) in `question_attributions`, temporarily remove it from the
  reference pool, find its k nearest neighbours among the OTHER labelled
  questions in the vector collection, aggregate their labels weighted by
  cosine similarity, propose top-1 and top-3 predictions, and compare
  to the withheld truth.

Baseline (report): direction top-1 = 90.4 %, top-3 = 98.5 % ; bureau
top-1 = 83.6 %, top-3 = 95.6 %. Reproduced by running with the default
`--collection questions_opendata`. Swap the collection to A/B the
embedding text source (e.g. `question_extraite`).

Usage:
    poetry run python scripts/eval_attribution_kNN.py \\
        --collection questions_opendata --target direction \\
        --output data/attribution_baseline_direction.json

    poetry run python scripts/eval_attribution_kNN.py \\
        --collection questions_experiments --target bureau \\
        --output data/attribution_qonly_bureau.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy import text as sa_text
from tqdm import tqdm

from qe import db
from qe.clients.pgvector_client import PgvectorClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Report defaults — reproduce the same k values as production.
DEFAULT_K_DIRECTION = 15
DEFAULT_K_BUREAU = 25


def _question_point_id(question_id: str, variant_tag: str | None = None) -> str:
    key = f"{question_id}::{variant_tag}" if variant_tag else question_id
    return str(UUID(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]))


def _load_labels(target: str) -> dict[str, int]:
    """{question_id: label_id} for questions that have a real attribution."""
    col = "direction_reelle_id" if target == "direction" else "bureau_reel_id"
    with db.get_session() as session:
        rows = session.execute(
            sa_text(
                f"SELECT question_id, {col} FROM question_attributions "
                f"WHERE {col} IS NOT NULL"
            )
        ).all()
    return {r[0]: r[1] for r in rows}


def _fetch_vectors(
    collection: str,
    qids: list[str],
    variant_tag: str | None = None,
    batch_size: int = 1000,
) -> dict[str, np.ndarray]:
    """Batch-fetch vectors for the given question_ids. Missing → absent."""
    vs = PgvectorClient()
    pid_to_qid = {_question_point_id(q, variant_tag): q for q in qids}
    pids = list(pid_to_qid.keys())
    result: dict[str, np.ndarray] = {}
    for i in tqdm(range(0, len(pids), batch_size), desc="Fetching vectors", unit="batch"):
        batch = pids[i : i + batch_size]
        points = vs.get_points_by_ids(collection, batch, with_vectors=True)
        for p in points:
            qid = pid_to_qid.get(p.get("id"))
            vec = p.get("vector")
            if qid and vec:
                result[qid] = np.asarray(vec, dtype=np.float32)
    return result


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--collection", default="questions_opendata")
    parser.add_argument(
        "--variant-tag", default=None,
        help="Variant tag used at embedding time — mixed into the point_id hash.",
    )
    parser.add_argument(
        "--target", choices=("direction", "bureau"), required=True,
        help="Which attribution label to predict.",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help=f"Number of neighbours (default: {DEFAULT_K_DIRECTION} for direction, "
             f"{DEFAULT_K_BUREAU} for bureau — matches production).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_attribution.json"),
    )
    args = parser.parse_args()

    k = args.k or (DEFAULT_K_DIRECTION if args.target == "direction" else DEFAULT_K_BUREAU)

    # ------------------------------------------------------------------
    # 1. Ground truth
    # ------------------------------------------------------------------
    labels = _load_labels(args.target)
    logger.info(
        "Loaded %d labelled questions (%s) from question_attributions.",
        len(labels), args.target,
    )

    # ------------------------------------------------------------------
    # 2. Fetch vectors for all labelled questions
    # ------------------------------------------------------------------
    qids = list(labels.keys())
    vecs = _fetch_vectors(args.collection, qids, args.variant_tag)
    logger.info("  Got vectors for %d / %d labelled questions.", len(vecs), len(qids))

    covered = [q for q in qids if q in vecs]
    y_true = np.array([labels[q] for q in covered], dtype=np.int64)
    M = np.stack([vecs[q] for q in covered]).astype(np.float32)  # (n, d)
    # Assume unit-norm BGE-M3 vectors → dot product = cosine similarity.
    n = len(covered)
    logger.info("Running leave-one-out kNN over %d questions, k=%d...", n, k)

    # ------------------------------------------------------------------
    # 3. Leave-one-out kNN — compute full cosine similarity matrix once
    # ------------------------------------------------------------------
    # For 11k questions, M @ M.T is 11k×11k = 500 MB float32. Doable in
    # RAM. For larger corpora we'd batch by query rows.
    sims = M @ M.T  # (n, n) cosine similarities
    np.fill_diagonal(sims, -np.inf)  # exclude self (leave-one-out)

    top1_hits = 0
    top3_hits = 0
    top10_hits = 0

    for i in tqdm(range(n), desc="Predicting", unit="q"):
        # Top-k neighbours (largest similarity)
        neigh_idx = np.argpartition(-sims[i], k)[:k]
        # Order those k by descending similarity so weights are meaningful
        neigh_idx = neigh_idx[np.argsort(-sims[i][neigh_idx])]
        neigh_labels = y_true[neigh_idx]
        neigh_scores = sims[i][neigh_idx]

        # Weighted vote per label: sum of cosine similarities per label.
        votes: dict[int, float] = defaultdict(float)
        for lbl, sc in zip(neigh_labels.tolist(), neigh_scores.tolist(), strict=True):
            votes[lbl] += sc

        # Ranked prediction: descending vote score.
        ranked = sorted(votes.items(), key=lambda x: -x[1])
        truth = int(y_true[i])
        rank = next(
            (r for r, (lbl, _) in enumerate(ranked, 1) if lbl == truth),
            None,
        )
        if rank is not None:
            if rank <= 1:
                top1_hits += 1
            if rank <= 3:
                top3_hits += 1
            if rank <= 10:
                top10_hits += 1

    top1 = top1_hits / n
    top3 = top3_hits / n
    top10 = top10_hits / n

    report = {
        "summary": {
            "collection": args.collection,
            "target": args.target,
            "k": k,
            "total_labelled": len(labels),
            "total_evaluated": n,
            "top_1_accuracy": round(top1, 6),
            "top_3_accuracy": round(top3, 6),
            "top_10_accuracy": round(top10, 6),
        }
    }
    logger.info(
        "%s / collection=%s / k=%d — top-1 = %.4f    top-3 = %.4f    top-10 = %.4f    (n=%d)",
        args.target, args.collection, k, top1, top3, top10, n,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
