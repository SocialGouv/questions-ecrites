#!/usr/bin/env python3
"""Embed parliamentary answers from PostgreSQL into a Qdrant collection.

Reads answers from the `reponses` table (populated by ingest_an_legacy.py or
ingest_senat.py), generates embeddings using ``texte_reponse``, and upserts the
result into Qdrant.

Incremental: answers are skipped if they are already in Qdrant with the same
embedding model and the same texte_reponse content (tracked via a SHA-256 hash
stored in the point payload alongside ``embedding_model``). If the model
changes, all answers are re-embedded.

Answers deleted from PostgreSQL are cleaned up from Qdrant automatically.

Usage:
    # All answers
    poetry run python scripts/embed_answers.py

    # Assemblée Nationale only, rate-limited
    poetry run python scripts/embed_answers.py --source AN --rate-limit 60

Requires:
    - SOCLE_IA_API_KEY environment variable set
    - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
    - A running PostgreSQL with ingested answers (run ingest_an_legacy.py / ingest_senat.py first)
    - A running Qdrant instance
"""

from __future__ import annotations

import argparse
import logging

from qe.answer_embedding import DEFAULT_COLLECTION, embed_answers
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings
from qe.rate_limiter import TokenBucketRateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed answers from PostgreSQL into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Base URL for Qdrant (default: {DEFAULT_QDRANT_URL}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Socle IA embedding model name (overrides settings).",
    )
    parser.add_argument(
        "--source",
        choices=["AN", "SENAT"],
        default=None,
        help="Restrict to answers from one chamber: AN or SENAT.",
    )
    parser.add_argument(
        "--legislature",
        type=int,
        default=17,
        metavar="N",
        help="Only embed answers linked to questions from this legislature (default: 17). Pass 0 for all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Number of answers to embed per API call (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum embedding API calls per minute. Omit for no rate limiting.",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedding_model = args.embedding_model or settings.embedding_model

    embedder = EmbeddingClient(
        url=settings.embeddings_url,
        model=embedding_model,
        api_key=settings.socle_api_key,
    )
    qdrant = QdrantClient(args.qdrant_url)
    rate_limiter = (
        TokenBucketRateLimiter(rate_per_minute=args.rate_limit)
        if args.rate_limit
        else None
    )

    legislature = args.legislature or None  # treat 0 as "all"

    logger.info(
        "Starting — source: %s, legislature: %s, collection: %s, model: %s, batch_size: %d%s",
        args.source or "all",
        legislature or "all",
        args.collection,
        embedding_model,
        args.batch_size,
        f", rate_limit: {args.rate_limit} calls/min" if args.rate_limit else "",
    )

    stats = embed_answers(
        embedder=embedder,
        qdrant=qdrant,
        embedding_model=embedding_model,
        collection=args.collection,
        source=args.source,
        legislature=legislature,
        batch_size=args.batch_size,
        rate_limiter=rate_limiter,
    )

    logger.info(
        "Summary — total: %d, embedded: %d, skipped: %d, stale removed: %d",
        stats.total,
        stats.embedded,
        stats.skipped,
        stats.deleted,
    )


if __name__ == "__main__":
    main()
