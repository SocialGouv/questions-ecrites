#!/usr/bin/env python3
"""Cluster parliamentary questions by semantic similarity.

Fetches question embeddings from Qdrant (populated by embed_questions.py) and
groups them using agglomerative clustering with average linkage and cosine distance.
Questions whose cosine similarity exceeds --threshold are placed in the same cluster.
Singleton clusters and noise points are excluded from the output.

Usage:
    # Default threshold (0.90)
    poetry run python scripts/cluster_questions.py

    # Adjust strictness
    poetry run python scripts/cluster_questions.py --threshold 0.85

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
DEFAULT_THRESHOLD = 0.90  # cosine *similarity* threshold
DEFAULT_MIN_CLUSTER_SIZE = 2


@dataclass(frozen=True)
class ClusterConfig:
    collection: str
    qdrant_url: str
    threshold: float  # cosine similarity threshold
    min_cluster_size: int
    embedding_model: str  # only cluster questions embedded with this model
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
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine *similarity* to place questions in the same cluster (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=f"Minimum number of questions per cluster (default: {DEFAULT_MIN_CLUSTER_SIZE}).",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        metavar="MODEL",
        help=("Only cluster questions embedded with this model (e.g. 'BAAI/bge-m3')"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: data/clusters.json).",
    )
    args = parser.parse_args()

    output = args.output or Path("data/clusters.json")

    return ClusterConfig(
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        threshold=args.threshold,
        min_cluster_size=max(2, args.min_cluster_size),
        embedding_model=args.embedding_model,
        output=output,
    )


def _fetch_points(
    qdrant: QdrantClient,
    collection: str,
    embedding_model: str,
) -> list[dict]:
    """Scroll all points matching the given filters from the collection."""
    qdrant_filter = {
        "must": [{"key": "embedding_model", "match": {"value": embedding_model}}]
    }
    points = qdrant.scroll_all(collection, filter=qdrant_filter, with_vectors=True)
    logger.info(
        "Fetched %d point(s) from '%s' (model=%s).",
        len(points),
        collection,
        embedding_model,
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

    points = _fetch_points(qdrant, config.collection, config.embedding_model)
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

    labels = _run_allotissement(vectors, config.threshold, config.min_cluster_size)

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

    total_questions = sum(c["size"] for c in clusters)
    save_clusters(clusters)
    logger.info(
        "Saved %d question(s) in %d cluster(s) to database.",
        total_questions,
        len(clusters),
    )


if __name__ == "__main__":
    main()
