#!/usr/bin/env python3
"""Cluster parliamentary questions by semantic similarity.

Fetches question embeddings from Qdrant (populated by embed_questions.py) and
groups them into clusters using one of two algorithms:

  allotissement (default)
    Agglomerative clustering with average linkage and cosine distance.
    Tunable via --threshold (cosine *distance*, so 1 − similarity).
    Use this for grouping near-identical unanswered questions so that a single
    response can address all of them at once.

  loose
    HDBSCAN with cosine metric.  No threshold required — the algorithm finds
    natural density peaks automatically.  Use this for broad thematic grouping
    in the UI.  Questions that don't fit any cluster (noise) are excluded.

Only EN_COURS questions are fetched by default (use --filter-status to change).
Singleton clusters and noise points are excluded from the output.

Usage:
    # Strict grouping for allotissement (default)
    poetry run python scripts/cluster_questions.py

    # Adjust strictness
    poetry run python scripts/cluster_questions.py --threshold 0.85

    # Thematic grouping
    poetry run python scripts/cluster_questions.py --mode loose --min-cluster-size 3

    # Custom collection / output
    poetry run python scripts/cluster_questions.py \\
        --collection questions_opendata \\
        --output data/my_clusters.json

Requires:
    - A populated Qdrant collection (run embed_questions.py first)
    - scikit-learn >= 1.3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from qe.clients.qdrant import QdrantClient
from qe.db import save_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_THRESHOLD = 0.90  # cosine *similarity* threshold for allotissement
DEFAULT_MIN_CLUSTER_SIZE = 2


@dataclass(frozen=True)
class ClusterConfig:
    collection: str
    qdrant_url: str
    mode: str  # "allotissement" | "loose"
    threshold: float  # similarity threshold (allotissement only)
    min_cluster_size: int
    filter_status: str
    output: Path


def _parse_args() -> ClusterConfig:
    parser = argparse.ArgumentParser(
        description="Cluster parliamentary questions by semantic similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Base URL for Qdrant (default: http://localhost:6333).",
    )
    parser.add_argument(
        "--mode",
        choices=["allotissement", "loose"],
        default="allotissement",
        help="Clustering mode: 'allotissement' (strict) or 'loose' (thematic). Default: allotissement.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Minimum cosine *similarity* for allotissement mode (default: 0.90). "
            "Ignored for loose mode."
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            f"Minimum number of questions per cluster (default: {DEFAULT_MIN_CLUSTER_SIZE}). "
            "In allotissement mode, singletons are always excluded regardless of this value."
        ),
    )
    parser.add_argument(
        "--filter-status",
        default="EN_COURS",
        metavar="STATUS",
        help="Fetch only questions with this etat_question (default: EN_COURS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: data/clusters_{mode}.json).",
    )
    args = parser.parse_args()

    output = args.output or Path(f"data/clusters_{args.mode}.json")

    return ClusterConfig(
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        mode=args.mode,
        threshold=args.threshold,
        min_cluster_size=max(2, args.min_cluster_size),
        filter_status=args.filter_status,
        output=output,
    )


def _fetch_points(
    qdrant: QdrantClient, collection: str, filter_status: str
) -> list[dict]:
    """Scroll all points for the given etat_question from the collection."""
    qdrant_filter = {
        "must": [{"key": "etat_question", "match": {"value": filter_status}}]
    }
    points = qdrant.scroll_all(collection, filter=qdrant_filter, with_vectors=True)
    logger.info(
        "Fetched %d point(s) from '%s' (status=%s).",
        len(points),
        collection,
        filter_status,
    )
    return points


def _run_allotissement(
    matrix: np.ndarray,
    threshold: float,
    min_cluster_size: int,
) -> np.ndarray:
    """Agglomerative clustering with cosine distance, tunable threshold.

    Returns an array of integer cluster labels (same length as matrix).
    Singletons are relabelled to −1 (noise) when min_cluster_size > 1.
    """

    # distance_threshold = 1 − similarity_threshold (cosine distance)
    distance_threshold = 1.0 - threshold

    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels: np.ndarray = model.fit_predict(matrix)

    if min_cluster_size > 1:
        # Count cluster sizes and relabel small clusters as noise (−1).
        unique, counts = np.unique(labels, return_counts=True)
        small = {
            label
            for label, count in zip(unique, counts, strict=False)
            if count < min_cluster_size
        }
        labels = np.where(np.isin(labels, list(small)), -1, labels)

    return labels


def _run_loose(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN with cosine metric.

    Returns an array of integer cluster labels; −1 = noise / unclustered.
    """
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
    )
    return model.fit_predict(matrix)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_output(
    points: list[dict],
    vectors: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """Convert cluster labels into the output JSON structure."""
    from collections import defaultdict

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # noise / singleton
        clusters[label].append(idx)

    result = []
    cluster_id = 1

    # Sort by size descending.
    for label in sorted(clusters, key=lambda idx: len(clusters[idx]), reverse=True):
        indices = clusters[label]
        member_vectors = vectors[indices]
        centroid = member_vectors.mean(axis=0)

        questions = []
        for idx in indices:
            payload = points[idx].get("payload", {})
            sim = _cosine_similarity(vectors[idx], centroid)
            preview = (
                payload.get("texte_preview")
                or (payload.get("texte_question", "")[:240])
            )
            questions.append(
                {
                    "question_id": payload.get("question_id", ""),
                    "etat_question": payload.get("etat_question", ""),
                    "source": payload.get("source", ""),
                    "legislature": payload.get("legislature"),
                    "auteur_nom": payload.get("auteur_nom"),
                    "ministre_attributaire_libelle": payload.get(
                        "ministre_attributaire_libelle"
                    ),
                    "date_publication_jo": payload.get("date_publication_jo"),
                    "texte_preview": preview,
                    "similarity_to_centroid": round(sim, 6),
                }
            )

        # Sort members by similarity to centroid descending.
        questions.sort(key=lambda q: q["similarity_to_centroid"], reverse=True)

        result.append(
            {
                "cluster_id": cluster_id,
                "size": len(questions),
                "questions": questions,
            }
        )
        cluster_id += 1

    return result


def cluster_questions(config: ClusterConfig) -> list[dict]:
    """Main clustering logic.  Returns the list of cluster dicts."""
    qdrant = QdrantClient(config.qdrant_url)

    points = _fetch_points(qdrant, config.collection, config.filter_status)
    if len(points) < 2:
        logger.warning("Need at least 2 points to cluster; found %d.", len(points))
        return []

    # Build numpy matrix; skip points without a vector.
    valid_points = [p for p in points if p.get("vector")]
    if len(valid_points) < len(points):
        logger.warning(
            "%d point(s) had no vector and will be skipped.",
            len(points) - len(valid_points),
        )

    vectors = np.array([p["vector"] for p in valid_points], dtype=np.float32)
    logger.info("Matrix shape: %s", vectors.shape)

    logger.info("Running %s clustering...", config.mode)
    if config.mode == "allotissement":
        labels = _run_allotissement(vectors, config.threshold, config.min_cluster_size)
    else:
        labels = _run_loose(vectors, config.min_cluster_size)

    noise_count = int((labels == -1).sum())
    cluster_count = len(set(labels) - {-1})
    logger.info(
        "Found %d cluster(s); %d point(s) unclustered (noise/singletons).",
        cluster_count,
        noise_count,
    )

    return _build_output(valid_points, vectors, labels)


def main() -> None:
    config = _parse_args()

    clusters = cluster_questions(config)

    output_json = json.dumps(clusters, ensure_ascii=False, indent=2)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(output_json, encoding="utf-8")
    logger.info("Wrote %d cluster(s) to %s.", len(clusters), config.output)

    threshold = config.threshold if config.mode == "allotissement" else None
    run_id = save_clusters(config.mode, threshold, clusters)
    logger.info("Saved cluster run to database (run_id=%d).", run_id)


if __name__ == "__main__":
    main()
