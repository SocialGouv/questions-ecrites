#!/usr/bin/env python3
"""Embed parliamentary questions from PostgreSQL into a Qdrant collection.

Reads questions from the `questions` table (populated by ingest_opendata.py),
generates one embedding per question using ``texte_question``, and upserts the
result into Qdrant.  Incremental: unchanged questions (same texte_question hash
in the manifest) are skipped.  Questions deleted from PostgreSQL are cleaned up
from Qdrant automatically.

Filters (all combinable):
    --filter-status   EN_COURS | REPONDU | …   question status
    --ministry TEXT                             substring match on ministry label
    --source          AN | SENAT                parliamentary chamber
    --legislature N                             legislature number (e.g. 17)
    --date-from       YYYY-MM-DD                published on or after this date (JO)
    --date-to         YYYY-MM-DD                published on or before this date (JO)

Usage:
    # Questions for social ministries (cohésion sociale), unanswered only
    poetry run python scripts/embed_questions.py --ministry "cohésion sociale" --filter-status EN_COURS

    # Questions for the DGCS specifically, last two years
    poetry run python scripts/embed_questions.py --ministry "cohésion sociale" --date-from 2024-01-01

    # Current legislature, Assemblée Nationale only
    poetry run python scripts/embed_questions.py --source AN --legislature 17

Requires:
    - SOCLE_IA_API_KEY environment variable set
    - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
    - A running PostgreSQL with ingested questions (run ingest_opendata.py first)
    - A running Qdrant instance
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy import select

from qe import db
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings, require_api_key
from qe.hashing import make_preview
from qe.models import Question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_QDRANT_URL = "http://localhost:6333"


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
    )


def _question_point_id(question_id: str) -> str:
    """Deterministic UUID derived from the question's string ID."""
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return str(UUID(digest[:32]))


def _manifest_key(collection: str, question_id: str) -> str:
    return f"{collection}:{question_id}"


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


def embed_questions(
    *,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    filter_status: str | None,
    ministry: str | None = None,
    source: str | None = None,
    legislature: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> None:
    """Embed all matching questions from PostgreSQL into Qdrant.

    Skips unchanged questions (same texte_question hash in the manifest).
    Cleans up Qdrant points for questions no longer in the database.

    Args:
        collection: Qdrant collection name.
        embedder: Embedding API client.
        qdrant: Qdrant REST client.
        filter_status: If set, only embed questions with this etat_question.
        ministry: Substring filter on ministre_attributaire_libelle.
        source: If set, restrict to "AN" or "SENAT".
        legislature: If set, restrict to this legislature number.
        date_from: If set, only embed questions published on or after this date.
        date_to: If set, only embed questions published on or before this date.
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

    # Probe the first question to determine vector dimension.
    probe_text = questions[0].texte_question
    probe_embedding = embedder.embed(probe_text)
    vector_size = len(probe_embedding)

    if not qdrant.collection_exists(collection):
        qdrant.create_collection(collection, vector_size=vector_size)
        logger.info("Created collection '%s' (dim=%d).", collection, vector_size)

    # Load manifest entries scoped to this collection.
    prefix = f"{collection}:"
    manifest = db.get_manifest_entries_under_prefix(prefix)

    # Identify and remove stale points (questions deleted from DB).
    db_question_ids = {q.id for q in questions}
    for manifest_key in list(manifest):
        question_id = manifest_key[len(prefix) :]
        if question_id not in db_question_ids:
            point_id = _question_point_id(question_id)
            qdrant.delete_points_by_filter(
                collection,
                {"must": [{"key": "question_id", "match": {"value": question_id}}]},
            )
            db.delete_manifest(manifest_key)
            logger.info("Removed stale point for question %s.", question_id)

    # Embed and upsert new / changed questions.
    skipped = 0
    upserted = 0

    for i, question in enumerate(questions):
        text = question.texte_question
        if not text or not text.strip():
            logger.debug("Skipping empty question %s.", question.id)
            continue

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        mkey = _manifest_key(collection, question.id)

        if manifest.get(mkey) == content_hash:
            skipped += 1
            continue

        point_id = _question_point_id(question.id)

        # Use pre-computed probe embedding for the first question.
        if i == 0:
            embedding = probe_embedding
        else:
            embedding = embedder.embed(text)

        date_str = (
            question.date_publication_jo.isoformat()
            if question.date_publication_jo
            else None
        )

        point = {
            "id": point_id,
            "vector": embedding,
            "payload": {
                "kind": "question",
                "question_id": question.id,
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

        qdrant.upsert_points(collection, [point])
        db.upsert_manifest(mkey, content_hash)
        upserted += 1

        if upserted % 50 == 0:
            logger.info("  %d upserted so far...", upserted)

    logger.info(
        "Done — %d upserted, %d skipped (unchanged).",
        upserted,
        skipped,
    )


def main() -> None:
    config = _parse_args()

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=config.api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)

    embed_questions(
        collection=config.collection,
        embedder=embedder,
        qdrant=qdrant,
        filter_status=config.filter_status,
        ministry=config.ministry,
        source=config.source,
        legislature=config.legislature,
        date_from=config.date_from,
        date_to=config.date_to,
    )


if __name__ == "__main__":
    main()
