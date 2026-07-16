#!/usr/bin/env python3
"""Embed parliamentary questions from PostgreSQL into pgvector.

Reads questions from the `questions` table (populated by ingest_an_legacy.py or ingest_senat.py),
generates embeddings using ``texte_question``, and upserts the result into
the `vec_questions_opendata` pgvector table.

Incremental: questions are skipped if they are already embedded with the same
embedding model and the same texte_question content (tracked via a SHA-256 hash
stored in the point payload alongside ``embedding_model``).  If the model
changes, all questions are re-embedded.

Questions deleted from PostgreSQL are cleaned up from the vector table automatically.

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
    # Baseline (existing behaviour)
    poetry run python scripts/embed_questions.py

    # A/B: embed only the question_extraite in a separate collection
    poetry run python scripts/embed_questions.py \\
        --text-source question_extraite \\
        --skip-rappels \\
        --collection questions_opendata_q_only

    # A/B: contexte-only for attribution eval
    poetry run python scripts/embed_questions.py \\
        --text-source contexte_extrait \\
        --skip-rappels \\
        --collection questions_opendata_ctx_only

    # Filters: current legislature, Assemblée Nationale only, rate-limited
    poetry run python scripts/embed_questions.py --source AN --legislature 17 --rate-limit 60

Requires:
    - PLIAGE_API_KEY environment variable set
    - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
    - A running PostgreSQL with ingested questions (run ingest_an_legacy.py / ingest_senat.py first)
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
from qe.clients.pgvector_client import PgvectorClient
from qe.clients.vector_store import VectorStore
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
DEFAULT_BATCH_SIZE = 32

# Text source options for --text-source. Every value falls back to
# `texte_question` when the primary field is NULL (row not yet analysed
# by the regex parser, or parser failed on this row). This guarantees
# every row still gets a vector — no silent gaps in the index.
#
#   texte_question       : baseline — the raw JO text (current behaviour)
#   question_extraite    : only the closing demand ("Il lui demande…")
#   contexte_extrait     : only the context body ("attire l'attention…")
#   contexte_and_question: contexte + " " + question (drops opener boilerplate)
TEXT_SOURCES = (
    "texte_question",
    "question_extraite",
    "contexte_extrait",
    "contexte_and_question",
)


def _select_text(question: Question, text_source: str) -> str | None:
    """Pick the text to embed according to `text_source`.

    Always returns `texte_question` when the primary field is NULL so
    the row still gets a vector — the alternative is a corpus with
    silent gaps that would fail every recall test on those rows.
    """
    if text_source == "texte_question":
        return question.texte_question
    if text_source == "question_extraite":
        return question.question_extraite or question.texte_question
    if text_source == "contexte_extrait":
        return question.contexte_extrait or question.texte_question
    if text_source == "contexte_and_question":
        parts = [p for p in (question.contexte_extrait, question.question_extraite) if p]
        return " ".join(parts) if parts else question.texte_question
    raise ValueError(f"unknown text_source: {text_source!r}")


@dataclass(frozen=True)
class EmbedConfig:
    collection: str
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
    text_source: str  # one of TEXT_SOURCES — picks which field to embed
    skip_rappels: bool  # drop rows where est_rappel = TRUE (bruit dans l'index)
    variant_tag: str | None  # label recorded in payload for A/B filtering


def _parse_args() -> EmbedConfig:
    parser = argparse.ArgumentParser(
        description="Embed questions from PostgreSQL into pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION}).",
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
    parser.add_argument(
        "--text-source",
        choices=TEXT_SOURCES,
        default="texte_question",
        help=(
            "Which text to embed: texte_question (raw JO, baseline), "
            "question_extraite (closing demand only), contexte_extrait "
            "(context body only), or contexte_and_question (both, drops "
            "the opener boilerplate). Falls back to texte_question when "
            "the primary field is NULL."
        ),
    )
    parser.add_argument(
        "--skip-rappels",
        action="store_true",
        help=(
            "Skip rows where est_rappel = TRUE — their generic wording "
            "is noise in the similarity index."
        ),
    )
    parser.add_argument(
        "--variant-tag",
        default=None,
        metavar="TAG",
        help=(
            "Label stored in each point's payload as `variant_tag`. Only "
            "useful when writing to the shared experiments collection "
            "(`--collection questions_experiments`) — lets multiple A/B "
            "variants coexist in the same table and be filtered at eval "
            "time. Ignored otherwise but harmless."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("pliage", "albert"),
        default="pliage",
        help=(
            "Which service to call for embeddings. `pliage` (default) uses "
            "the internal gateway at LLM_BASE_URL (`ia.social.gouv.fr`) — "
            "IP-whitelisted, only reachable from the office wifi. `albert` "
            "uses Albert Etalab's public API "
            "(https://albert.api.etalab.gouv.fr) — reachable from anywhere. "
            "Same BGE-M3 model behind both, so vectors are compatible; "
            "picks the matching API key automatically."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    # Resolve the embedding endpoint + credential according to the provider.
    # `--embedding-provider albert` forces the Etalab public URL regardless
    # of settings.embeddings_url (which defaults to PLIAGE via LLM_BASE_URL).
    if args.embedding_provider == "albert":
        api_key = require_api_key("ALBERT_API_KEY")
        embeddings_url = f"{settings.albert_base_url.rstrip('/')}/v1/embeddings"
    else:
        api_key = require_api_key("PLIAGE_API_KEY")
        embeddings_url = settings.embeddings_url

    def _parse_date(val: str | None, flag: str) -> date | None:
        if val is None:
            return None
        try:
            return date.fromisoformat(val)
        except ValueError:
            parser.error(f"{flag}: invalid date '{val}', expected YYYY-MM-DD")

    return EmbedConfig(
        collection=args.collection,
        embedding_model=args.embedding_model or settings.embedding_model,
        embeddings_url=embeddings_url,
        api_key=api_key,
        filter_status=args.filter_status,
        ministry=args.ministry,
        source=args.source,
        legislature=args.legislature,
        date_from=_parse_date(args.date_from, "--date-from"),
        date_to=_parse_date(args.date_to, "--date-to"),
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        text_source=args.text_source,
        skip_rappels=args.skip_rappels,
        variant_tag=args.variant_tag,
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
    skip_rappels: bool = False,
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

    if skip_rappels:
        stmt = stmt.where(Question.est_rappel.is_(False))

    with db.get_session() as session:
        return list(session.execute(stmt).scalars().all())


def _load_all_question_ids() -> set[str]:
    """Return all question IDs in PostgreSQL, unfiltered, for stale detection."""
    with db.get_session() as session:
        return set(session.execute(select(Question.id)).scalars().all())


def _load_existing_points(
    vector_store: VectorStore, collection: str
) -> dict[str, tuple[str, str]]:
    """Scroll all existing points (no vectors) and return {question_id: (model, content_hash)}."""
    if not vector_store.collection_exists(collection):
        return {}
    logger.info("Loading existing points from collection '%s'...", collection)
    points = vector_store.scroll_all(collection, with_vectors=False)
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
    vector_store: VectorStore,
    embedding_model: str,
    filter_status: str | None,
    ministry: str | None = None,
    source: str | None = None,
    legislature: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limiter: TokenBucketRateLimiter | None = None,
    text_source: str = "texte_question",
    skip_rappels: bool = False,
    variant_tag: str | None = None,
) -> None:
    """Embed all matching questions from PostgreSQL into pgvector.

    Skips questions already embedded with the same embedding model and content
    hash.  Re-embeds if the model or the question text has changed.  Cleans up
    rows for questions no longer in the database.

    Args:
        collection: Vector table collection name.
        embedder: Embedding API client.
        vector_store: Vector store client.
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
        filter_status, ministry, source, legislature, date_from, date_to,
        skip_rappels=skip_rappels,
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

    existing = _load_existing_points(vector_store, collection)

    # --- Stale point cleanup ---
    # Use all DB IDs (unfiltered) so questions outside the current filter scope
    # are never incorrectly treated as stale and deleted.
    all_db_ids = _load_all_question_ids()
    stale_ids = [qid for qid in existing if qid not in all_db_ids]
    if stale_ids:
        logger.info("Removing %d stale point(s) (deleted from DB)...", len(stale_ids))
        for qid in stale_ids:
            vector_store.delete_points_by_filter(
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
        text = _select_text(q, text_source)
        if not text or not text.strip():
            empty += 1
            logger.debug("Skipping empty question %s (text_source=%s).", q.id, text_source)
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
    first_batch_texts = [_select_text(q, text_source) for q in to_embed[:batch_size]]
    if rate_limiter:
        rate_limiter.acquire(1)
    logger.info(
        "Probing embedding dimension with first batch (%d question(s))...",
        len(first_batch_texts),
    )
    first_embeddings = embedder.embed_batch(first_batch_texts)
    vector_size = len(first_embeddings[0])

    if not vector_store.collection_exists(collection):
        vector_store.create_collection(collection, vector_size=vector_size)
        logger.info("Created collection '%s' (dim=%d).", collection, vector_size)

    # --- Batch embed + upsert with progress bar ---
    upserted = 0
    batches = list(_batched(to_embed, batch_size))

    with tqdm(total=len(to_embed), unit="q", desc="Embedding") as progress:
        for batch_idx, batch in enumerate(batches):
            texts = [_select_text(q, text_source) for q in batch]

            # Use pre-computed embeddings for the first batch.
            if batch_idx == 0:
                embeddings = first_embeddings
            else:
                if rate_limiter:
                    rate_limiter.acquire(1)
                embeddings = embedder.embed_batch(texts)

            points = []
            for question, embedding, text in zip(batch, embeddings, texts, strict=True):
                date_str = (
                    question.date_publication_jo.isoformat()
                    if question.date_publication_jo
                    else None
                )
                points.append(
                    {
                        "id": _question_point_id(question.id),
                        "vector": embedding,
                        "payload": {
                            "kind": "question",
                            "question_id": question.id,
                            "embedding_model": embedding_model,
                            # `text_source` and `content_hash` together
                            # form the cache key: switching the source
                            # forces a re-embedding of every row.
                            "text_source": text_source,
                            "content_hash": _content_hash(text),
                            # Present only when the caller passes --variant-tag,
                            # so we don't pollute the payload of non-experiment
                            # collections.
                            **({"variant_tag": variant_tag} if variant_tag else {}),
                            "etat_question": question.etat_question,
                            "source": question.source,
                            "legislature": question.legislature,
                            "texte_question": text[:2000],
                            "texte_preview": make_preview(text),
                            "auteur_nom": question.auteur_nom,
                            # Snapshot at embedding time — can go stale if a reattribution happens later.
                            "ministre_attributaire_libelle": question.ministre_attributaire_libelle,
                            "date_publication_jo": date_str,
                        },
                    }
                )

            vector_store.upsert_points(collection, points)
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
    vector_store = PgvectorClient()
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
        vector_store=vector_store,
        embedding_model=config.embedding_model,
        filter_status=config.filter_status,
        ministry=config.ministry,
        source=config.source,
        legislature=config.legislature,
        date_from=config.date_from,
        date_to=config.date_to,
        batch_size=config.batch_size,
        rate_limiter=rate_limiter,
        text_source=config.text_source,
        skip_rappels=config.skip_rappels,
        variant_tag=config.variant_tag,
    )


if __name__ == "__main__":
    main()
