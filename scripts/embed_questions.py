#!/usr/bin/env python3
"""Embed parliamentary questions from PostgreSQL into a Qdrant collection.

Reads questions from the `questions` table (populated by ingest_an_legacy.py or ingest_senat.py),
generates embeddings using ``texte_question``, and upserts the result into
Qdrant.

Incremental: questions are skipped if they are already in Qdrant with the same
embedding model and the same texte_question content (tracked via a SHA-256 hash
stored in the point payload alongside ``embedding_model``).  If the model
changes, all questions are re-embedded.

Questions deleted from PostgreSQL are cleaned up from Qdrant automatically.

Filters (all combinable):
    --filter-status   EN_COURS | REPONDU | …   question status
    --ministry TEXT                             substring match on ministry label
    --source          AN | SENAT                parliamentary chamber
    --legislature N                             legislature number (e.g. 17)
    --date-from       YYYY-MM-DD                published on or after this date (JO)
    --date-to         YYYY-MM-DD                published on or before this date (JO)

Performance:
    --batch-size N      embed N questions per API call (default: 32)
    --rate-limit N      max API calls per minute; omit for no limit

Usage:
    # Questions for social ministries (cohésion sociale), unanswered only
    poetry run python scripts/embed_questions.py --ministry "cohésion sociale" --filter-status EN_COURS

    # Current legislature, Assemblée Nationale only, rate-limited
    poetry run python scripts/embed_questions.py --source AN --legislature 17 --rate-limit 60

Requires:
    - SOCLE_IA_API_KEY environment variable set
    - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
    - A running PostgreSQL with ingested questions (run ingest_an_legacy.py / ingest_senat.py first)
    - A running Qdrant instance
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from itertools import islice
from uuid import UUID

from sqlalchemy import select
from tqdm import tqdm

from qe import db
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings, require_api_key
from qe.hashing import make_preview
from qe.models import Question
from qe.rate_limiter import TokenBucketRateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class EmbedConfig:
    collection: str
    qdrant_url: str
    embedding_model: str
    embeddings_url: str
    api_key: str
    filter_status: str | None  # None = all statuses
    ministry: str | None  # substring match on ministre_attributaire_libelle
    source: str | None  # "AN" | "SENAT"
    legislature: int | None  # e.g. 17
    date_from: date | None  # date_publication_jo >= this date
    date_to: date | None  # date_publication_jo <= this date
    batch_size: int  # questions per embedding API call
    rate_limit: int | None  # max API calls per minute; None = unlimited


def _parse_args() -> EmbedConfig:
    parser = argparse.ArgumentParser(
        description="Embed questions from PostgreSQL into Qdrant.",
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
        help="Base URL for Qdrant (default: http://localhost:6333).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Socle IA embedding model name (overrides settings).",
    )
    parser.add_argument(
        "--filter-status",
        default=None,
        metavar="STATUS",
        help="Embed only questions with this etat_question (e.g. EN_COURS, REPONDU).",
    )
    parser.add_argument(
        "--ministry",
        default=None,
        metavar="TEXT",
        help=(
            "Case-insensitive substring filter on the ministry label "
            "(ministre_attributaire_libelle). E.g. 'cohésion sociale'."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["AN", "SENAT"],
        default=None,
        help="Restrict to questions from one chamber: AN or SENAT.",
    )
    parser.add_argument(
        "--legislature",
        type=int,
        default=None,
        metavar="N",
        help="Restrict to a specific legislature number (e.g. 17).",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        metavar="YYYY-MM-DD",
        help="Embed only questions published in the JO on or after this date.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        metavar="YYYY-MM-DD",
        help="Embed only questions published in the JO on or before this date.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Number of questions to embed per API call (default: {DEFAULT_BATCH_SIZE}).",
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
    api_key = require_api_key("SOCLE_IA_API_KEY")

    def _parse_date(val: str | None, flag: str) -> date | None:
        if val is None:
            return None
        try:
            return date.fromisoformat(val)
        except ValueError:
            parser.error(f"{flag}: invalid date '{val}', expected YYYY-MM-DD")

    return EmbedConfig(
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model or settings.embedding_model,
        embeddings_url=settings.embeddings_url,
        api_key=api_key,
        filter_status=args.filter_status,
        ministry=args.ministry,
        source=args.source,
        legislature=args.legislature,
        date_from=_parse_date(args.date_from, "--date-from"),
        date_to=_parse_date(args.date_to, "--date-to"),
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
    )


def _question_point_id(question_id: str) -> str:
    """Deterministic UUID derived from the question's string ID."""
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_questions(
    filter_status: str | None,
    ministry: str | None,
    source: str | None,
    legislature: int | None,
    date_from: date | None,
    date_to: date | None,
) -> list[Question]:
    """Fetch questions from PostgreSQL, applying all active filters."""
    stmt = select(Question)

    if filter_status:
        stmt = stmt.where(Question.etat_question == filter_status)

    if ministry:
        stmt = stmt.where(Question.ministre_attributaire_libelle.ilike(f"%{ministry}%"))

    if source:
        stmt = stmt.where(Question.source == source)

    if legislature is not None:
        stmt = stmt.where(Question.legislature == legislature)

    if date_from is not None:
        stmt = stmt.where(Question.date_publication_jo >= date_from)

    if date_to is not None:
        stmt = stmt.where(Question.date_publication_jo <= date_to)

    with db.get_session() as session:
        return list(session.execute(stmt).scalars().all())


def _load_all_question_ids() -> set[str]:
    """Return all question IDs in PostgreSQL, unfiltered, for stale detection."""
    with db.get_session() as session:
        return set(session.execute(select(Question.id)).scalars().all())


def _load_existing_points(
    qdrant: QdrantClient, collection: str
) -> dict[str, tuple[str, str]]:
    """Scroll all existing points (no vectors) and return {question_id: (model, content_hash)}."""
    if not qdrant.collection_exists(collection):
        return {}
    logger.info("Loading existing points from Qdrant collection '%s'...", collection)
    points = qdrant.scroll_all(collection, with_vectors=False)
    result: dict[str, tuple[str, str]] = {}
    for point in points:
        payload = point.get("payload", {})
        qid = payload.get("question_id")
        model = payload.get("embedding_model")
        chash = payload.get("content_hash")
        if qid and model and chash:
            result[qid] = (model, chash)
    logger.info("  Found %d existing point(s) with tracking metadata.", len(result))
    return result


def _batched(iterable, n):
    """Split an iterable into chunks of at most n items."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def embed_questions(  # noqa: C901
    *,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    embedding_model: str,
    filter_status: str | None,
    ministry: str | None = None,
    source: str | None = None,
    legislature: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limiter: TokenBucketRateLimiter | None = None,
) -> None:
    """Embed all matching questions from PostgreSQL into Qdrant.

    Skips questions already in Qdrant with the same embedding model and content
    hash.  Re-embeds if the model or the question text has changed.  Cleans up
    Qdrant points for questions no longer in the database.

    Args:
        collection: Qdrant collection name.
        embedder: Embedding API client.
        qdrant: Qdrant REST client.
        embedding_model: Model name stored in each point payload.
        filter_status: If set, only embed questions with this etat_question.
        ministry: Substring filter on ministre_attributaire_libelle.
        source: If set, restrict to "AN" or "SENAT".
        legislature: If set, restrict to this legislature number.
        date_from: If set, only embed questions published on or after this date.
        date_to: If set, only embed questions published on or before this date.
        batch_size: Number of questions per embedding API call.
        rate_limiter: Optional global rate limiter (API calls/min).
    """
    questions = _load_questions(
        filter_status, ministry, source, legislature, date_from, date_to
    )
    if not questions:
        logger.warning(
            "No questions found (status=%s, ministry=%r, source=%s, legislature=%s).",
            filter_status,
            ministry,
            source,
            legislature,
        )
        return

    logger.info("Loaded %d question(s) from PostgreSQL.", len(questions))

    # Load existing Qdrant points (no vectors) to determine what to skip/clean.
    existing = _load_existing_points(qdrant, collection)

    # --- Stale point cleanup ---
    # Use all DB IDs (unfiltered) so questions outside the current filter scope
    # are never incorrectly treated as stale and deleted.
    all_db_ids = _load_all_question_ids()
    stale_ids = [qid for qid in existing if qid not in all_db_ids]
    if stale_ids:
        logger.info("Removing %d stale point(s) (deleted from DB)...", len(stale_ids))
        for qid in stale_ids:
            qdrant.delete_points_by_filter(
                collection,
                {"must": [{"key": "question_id", "match": {"value": qid}}]},
            )
            logger.debug("  Removed stale point for question %s.", qid)
        logger.info("  Done removing stale points.")

    # --- Determine which questions need (re-)embedding ---
    to_embed: list[Question] = []
    skipped = 0
    empty = 0

    for q in questions:
        text = q.texte_question
        if not text or not text.strip():
            empty += 1
            logger.debug("Skipping empty question %s.", q.id)
            continue
        cached = existing.get(q.id)
        if cached is not None:
            cached_model, cached_hash = cached
            if cached_model == embedding_model and cached_hash == _content_hash(text):
                skipped += 1
                continue
        to_embed.append(q)

    logger.info(
        "%d to embed, %d already up-to-date (skipped), %d empty.",
        len(to_embed),
        skipped,
        empty,
    )

    if not to_embed:
        logger.info("Nothing to do.")
        return

    # --- Probe first batch to get vector dimension, create collection if needed ---
    first_batch_texts = [q.texte_question for q in to_embed[:batch_size]]
    if rate_limiter:
        rate_limiter.acquire(1)
    logger.info(
        "Probing embedding dimension with first batch (%d question(s))...",
        len(first_batch_texts),
    )
    first_embeddings = embedder.embed_batch(first_batch_texts)
    vector_size = len(first_embeddings[0])

    if not qdrant.collection_exists(collection):
        qdrant.create_collection(collection, vector_size=vector_size)
        logger.info("Created collection '%s' (dim=%d).", collection, vector_size)

    # --- Batch embed + upsert with progress bar ---
    upserted = 0
    batches = list(_batched(to_embed, batch_size))

    with tqdm(total=len(to_embed), unit="q", desc="Embedding") as progress:
        for batch_idx, batch in enumerate(batches):
            texts = [q.texte_question for q in batch]

            # Use pre-computed embeddings for the first batch.
            if batch_idx == 0:
                embeddings = first_embeddings
            else:
                if rate_limiter:
                    rate_limiter.acquire(1)
                embeddings = embedder.embed_batch(texts)

            points = []
            for question, embedding in zip(batch, embeddings, strict=True):
                date_str = (
                    question.date_publication_jo.isoformat()
                    if question.date_publication_jo
                    else None
                )
                text = question.texte_question
                points.append(
                    {
                        "id": _question_point_id(question.id),
                        "vector": embedding,
                        "payload": {
                            "kind": "question",
                            "question_id": question.id,
                            "embedding_model": embedding_model,
                            "content_hash": _content_hash(text),
                            "etat_question": question.etat_question,
                            "source": question.source,
                            "legislature": question.legislature,
                            "texte_question": text[:2000],
                            "texte_preview": make_preview(text),
                            "auteur_nom": question.auteur_nom,
                            "ministre_attributaire_libelle": question.ministre_attributaire_libelle,
                            "date_publication_jo": date_str,
                        },
                    }
                )

            qdrant.upsert_points(collection, points)
            upserted += len(batch)
            progress.update(len(batch))

    logger.info(
        "Done — %d upserted, %d skipped (up-to-date), %d stale removed.",
        upserted,
        skipped,
        len(stale_ids),
    )


def main() -> None:
    config = _parse_args()

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=config.api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)
    rate_limiter = (
        TokenBucketRateLimiter(rate_per_minute=config.rate_limit)
        if config.rate_limit
        else None
    )

    if rate_limiter:
        logger.info("Rate limiting enabled: %d API calls/min.", config.rate_limit)

    embed_questions(
        collection=config.collection,
        embedder=embedder,
        qdrant=qdrant,
        embedding_model=config.embedding_model,
        filter_status=config.filter_status,
        ministry=config.ministry,
        source=config.source,
        legislature=config.legislature,
        date_from=config.date_from,
        date_to=config.date_to,
        batch_size=config.batch_size,
        rate_limiter=rate_limiter,
    )


if __name__ == "__main__":
    main()
