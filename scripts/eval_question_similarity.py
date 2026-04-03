#!/usr/bin/env python3
"""Evaluate question similarity retrieval using exact cosine similarity.

Ground truth: questions that share the same reponse_id in PostgreSQL received a
joint parliamentary answer — they are "similar" by definition. For each such
question we compute the exact cosine similarity to each of its siblings using
their stored Qdrant vectors and measure what fraction of siblings score >=
score_threshold.

Metrics (rank-free — only the threshold matters):
  - Recall@threshold: fraction of siblings found above the threshold, averaged
                      over all query questions.
  - Hit@threshold:    1 if at least one sibling was found above the threshold.

Implementation note: instead of issuing one ANN search per question (~11 000
HTTP calls), this script batch-fetches all relevant vectors from Qdrant upfront
(~11 HTTP calls) and computes cosine similarity locally with numpy. BGE-M3
vectors are unit-norm so cosine similarity equals the dot product.

Output: JSON report with summary stats and ~20 failure cases that include the
score each missed sibling actually received.

Usage:
    poetry run python scripts/eval_question_similarity.py
    poetry run python scripts/eval_question_similarity.py \\
        --score-threshold 0.8 \\
        --num-failures 20 \\
        --output data/eval_question_similarity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from uuid import UUID

import numpy as np
import requests
from sqlalchemy import func, select
from tqdm import tqdm

from qe import db
from qe.clients.qdrant import QdrantClient
from qe.models import Question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_NUM_FAILURES = 20
DEFAULT_OUTPUT = Path("data/eval_question_similarity.json")
# For failure-case diagnostics: how many raw (no-threshold) results to fetch
FAILURE_TOP_K = 50
# Batch size for fetching vectors from Qdrant
VECTOR_FETCH_BATCH_SIZE = 1_000


# ---------------------------------------------------------------------------
# Point ID helper (mirrors embed_questions.py)
# ---------------------------------------------------------------------------


def _question_point_id(question_id: str) -> str:
    """Deterministic Qdrant point UUID derived from the question's string ID."""
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


# ---------------------------------------------------------------------------
# Qdrant client extended with recommend support
# ---------------------------------------------------------------------------


class _RecommendQdrantClient(QdrantClient):
    def recommend(
        self,
        collection: str,
        point_id: str,
        top_k: int,
        *,
        score_threshold: float | None = None,
        filter: dict | None = None,
    ) -> list[dict] | None:
        """Find neighbors of an existing point by its ID.

        Returns None if the point does not exist in the collection (not yet
        embedded), so callers can handle missing questions gracefully.
        """
        payload: dict = {
            "positive": [point_id],
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if filter is not None:
            payload["filter"] = filter

        response = requests.post(
            f"{self.base_url}/collections/{collection}/points/recommend",
            json=payload,
            timeout=60,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("result", [])


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _fetch_all_vectors(
    qdrant: _RecommendQdrantClient,
    collection: str,
    question_ids: list[str],
    batch_size: int = VECTOR_FETCH_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Batch-fetch vectors for all given question IDs from Qdrant.

    Returns a dict mapping question_id → unit-norm vector. Questions not found
    in Qdrant (not yet embedded) are simply absent from the result.
    """
    point_id_to_question_id = {_question_point_id(qid): qid for qid in question_ids}
    point_ids = list(point_id_to_question_id.keys())

    vectors: dict[str, np.ndarray] = {}
    for i in tqdm(
        range(0, len(point_ids), batch_size),
        desc="Fetching vectors",
        unit="batch",
    ):
        batch = point_ids[i : i + batch_size]
        points = qdrant.get_points_by_ids(collection, batch, with_vectors=True)
        for point in points:
            pid = point.get("id")
            vec = point.get("vector")
            qid = point_id_to_question_id.get(pid)
            if qid and vec:
                vectors[qid] = np.array(vec, dtype=np.float32)

    return vectors


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------


def _load_sibling_groups() -> list[tuple[str, list[str]]]:
    """Return [(reponse_id, [question_id, ...]), ...] for groups with >= 2 questions."""
    with db.get_session() as session:
        stmt = (
            select(Question.reponse_id, func.array_agg(Question.id))
            .where(Question.reponse_id.is_not(None))
            .group_by(Question.reponse_id)
            .having(func.count() >= 2)
        )
        rows = session.execute(stmt).all()
    return [(row[0], row[1]) for row in rows]


def _load_question_metadata(question_ids: list[str]) -> dict[str, dict]:
    """Fetch display metadata for a list of question IDs from PostgreSQL."""
    with db.get_session() as session:
        rows = session.execute(
            select(
                Question.id,
                Question.texte_question,
                Question.ministre_attributaire_libelle,
                Question.legislature,
            ).where(Question.id.in_(question_ids))
        ).all()
    return {
        row[0]: {
            "text": row[1] or "",
            "ministry": row[2],
            "legislature": row[3],
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Evaluate question similarity retrieval via exact cosine similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--num-failures",
        type=int,
        default=DEFAULT_NUM_FAILURES,
        help="Number of failure cases to include in the report.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    qdrant = _RecommendQdrantClient(args.qdrant_url)

    # ------------------------------------------------------------------
    # 1. Load ground truth from PostgreSQL
    # ------------------------------------------------------------------
    logger.info("Loading sibling groups from PostgreSQL...")
    all_groups = _load_sibling_groups()
    logger.info("  Found %d groups with >= 2 questions.", len(all_groups))

    # ------------------------------------------------------------------
    # 2. Batch-fetch all relevant vectors from Qdrant (once)
    # ------------------------------------------------------------------
    all_question_ids = list({qid for _, qids in all_groups for qid in qids})
    logger.info("  Fetching vectors for %d questions...", len(all_question_ids))
    vectors = _fetch_all_vectors(qdrant, args.collection, all_question_ids)
    logger.info(
        "  Got vectors for %d / %d questions.", len(vectors), len(all_question_ids)
    )

    # ------------------------------------------------------------------
    # 3. Evaluate each query question (pure numpy, zero HTTP calls)
    # ------------------------------------------------------------------
    per_question_results: list[dict] = []
    not_embedded: list[str] = []

    for reponse_id, question_ids in all_groups:
        siblings_set = set(question_ids)

        for query_id in question_ids:
            if query_id not in vectors:
                not_embedded.append(query_id)
                continue

            q_vec = vectors[query_id]
            siblings = siblings_set - {query_id}

            above_threshold: dict[str, float] = {}
            for sib_id in siblings:
                if sib_id in vectors:
                    score = float(np.dot(q_vec, vectors[sib_id]))
                    if score >= args.score_threshold:
                        above_threshold[sib_id] = score

            found = siblings & above_threshold.keys()
            recall = len(found) / len(siblings)
            hit = 1 if found else 0

            per_question_results.append(
                {
                    "reponse_id": reponse_id,
                    "original_group_size": len(question_ids),
                    "query_id": query_id,
                    "siblings": list(siblings),
                    "recall": recall,
                    "hit": hit,
                    "above_threshold": above_threshold,
                }
            )

    if not_embedded:
        logger.warning(
            "%d question(s) were not found in Qdrant and were skipped.",
            len(not_embedded),
        )

    # ------------------------------------------------------------------
    # 4. Aggregate metrics
    # ------------------------------------------------------------------
    n = len(per_question_results)
    mean_recall = sum(r["recall"] for r in per_question_results) / n if n else 0.0
    mean_hit = sum(r["hit"] for r in per_question_results) / n if n else 0.0

    logger.info(
        "Recall@%.2f = %.4f   Hit@%.2f = %.4f   (n=%d query questions)",
        args.score_threshold,
        mean_recall,
        args.score_threshold,
        mean_hit,
        n,
    )

    # ------------------------------------------------------------------
    # 5. Failure cases: Recall = 0, one per group for diversity
    # ------------------------------------------------------------------
    zero_recall = [r for r in per_question_results if r["recall"] == 0]

    seen_reponse_ids: set[str] = set()
    failures_diverse: list[dict] = []
    for r in zero_recall:
        if r["reponse_id"] not in seen_reponse_ids:
            failures_diverse.append(r)
            seen_reponse_ids.add(r["reponse_id"])
        if len(failures_diverse) >= args.num_failures:
            break
    if len(failures_diverse) < args.num_failures:
        for r in zero_recall:
            if r not in failures_diverse:
                failures_diverse.append(r)
            if len(failures_diverse) >= args.num_failures:
                break

    all_failure_qids = list(
        {r["query_id"] for r in failures_diverse}
        | {sid for r in failures_diverse for sid in r["siblings"]}
    )
    metadata = _load_question_metadata(all_failure_qids)

    failures_output = []
    for r in failures_diverse:
        query_id = r["query_id"]
        point_id = _question_point_id(query_id)

        missed_sib_ids = [
            sib_id for sib_id in r["siblings"] if sib_id not in r["above_threshold"]
        ]

        # Compute actual scores from cached vectors (no extra HTTP call needed)
        q_vec = vectors[query_id]
        missed_siblings = [
            {
                "question_id": sib_id,
                "text": metadata.get(sib_id, {}).get("text", ""),
                "actual_score": (
                    float(np.dot(q_vec, vectors[sib_id])) if sib_id in vectors else None
                ),
            }
            for sib_id in missed_sib_ids
        ]

        # Top-5 results (no threshold) for context on what the model did find
        raw_hits = (
            qdrant.recommend(args.collection, point_id, top_k=FAILURE_TOP_K) or []
        )

        query_meta = metadata.get(query_id, {})
        failures_output.append(
            {
                "reponse_id": r["reponse_id"],
                "query_question_id": query_id,
                "query_text": query_meta.get("text", ""),
                "ministre_attributaire_libelle": query_meta.get("ministry"),
                "legislature": query_meta.get("legislature"),
                "missed_siblings": missed_siblings,
                "top5_results": [
                    {
                        "question_id": h["payload"]["question_id"],
                        "score": h["score"],
                        "text_preview": h["payload"].get("texte_question", "")[:200],
                    }
                    for h in raw_hits[:5]
                ],
            }
        )

    # ------------------------------------------------------------------
    # 6. Write report
    # ------------------------------------------------------------------
    report = {
        "summary": {
            "score_threshold": args.score_threshold,
            "total_groups_in_db": len(all_groups),
            "total_query_questions": n,
            "not_embedded_questions": not_embedded,
            "recall_at_threshold": round(mean_recall, 6),
            "hit_at_threshold": round(mean_hit, 6),
        },
        "failures": failures_output,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
