"""Embed parliamentary answers (Reponse) from PostgreSQL into pgvector.

Incremental: answers are skipped if they are already embedded with the same
embedding model and the same texte_reponse content (tracked via a SHA-256 hash
stored in the point payload alongside ``embedding_model``).

Answers deleted from PostgreSQL are cleaned up from the vector table automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import batched

from sqlalchemy import exists, select
from sqlalchemy.orm import selectinload
from tqdm import tqdm

from qe import db
from qe.clients.embedding import EmbeddingClient
from qe.clients.vector_store import VectorStore
from qe.hashing import compute_content_hash, make_preview, stable_answer_point_id
from qe.models import Question, Reponse
from qe.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "answers_opendata"
DEFAULT_BATCH_SIZE = 32
TEXT_MAX_CHARS = 4000


@dataclass(frozen=True)
class EmbedStats:
    total: int
    embedded: int
    skipped: int
    deleted: int


def _load_answers(source: str | None, legislature: int | None) -> list[Reponse]:
    stmt = select(Reponse).options(selectinload(Reponse.questions))
    if source:
        stmt = stmt.where(Reponse.source == source)
    if legislature is not None:
        stmt = stmt.where(
            exists().where(
                Question.reponse_id == Reponse.id,
                Question.legislature == legislature,
            )
        )
    with db.get_session() as session:
        return list(session.execute(stmt).scalars().all())


def _load_all_answer_ids(source: str | None) -> set[str]:
    stmt = select(Reponse.id)
    if source:
        stmt = stmt.where(Reponse.source == source)
    with db.get_session() as session:
        return set(session.execute(stmt).scalars().all())


def _load_existing_points(
    vector_store: VectorStore, collection: str, source: str | None
) -> dict[str, tuple[str, str]]:
    """Scroll existing points matching the active source filter; return {reponse_id: (model, hash)}."""
    if not vector_store.collection_exists(collection):
        return {}
    logger.info("Loading existing points from collection '%s'...", collection)
    filter_ = (
        {"must": [{"key": "source", "match": {"value": source}}]} if source else None
    )
    points = vector_store.scroll_all(collection, filter=filter_, with_vectors=False)
    result: dict[str, tuple[str, str]] = {}
    for point in points:
        payload = point.get("payload", {})
        rid = payload.get("reponse_id")
        model = payload.get("embedding_model")
        chash = payload.get("content_hash")
        if rid and model and chash:
            result[rid] = (model, chash)
    logger.info("  Found %d existing point(s) with tracking metadata.", len(result))
    return result


def embed_answers(  # noqa: C901
    *,
    embedder: EmbeddingClient,
    vector_store: VectorStore,
    embedding_model: str,
    collection: str = DEFAULT_COLLECTION,
    source: str | None = None,
    legislature: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limiter: TokenBucketRateLimiter | None = None,
) -> EmbedStats:
    """Embed all answers from PostgreSQL into pgvector.

    Skips answers already embedded with the same embedding model and content
    hash. Cleans up rows for answers no longer in the database.

    Args:
        embedder: Embedding API client.
        vector_store: Vector store client.
        embedding_model: Model name stored in each point payload.
        collection: Vector table collection name.
        source: If set, restrict to "AN" or "SENAT".
        batch_size: Number of answers per embedding API call.
        rate_limiter: Optional global rate limiter (API calls/min).
    """
    answers = _load_answers(source, legislature)
    if not answers:
        logger.warning(
            "No answers found (source=%s, legislature=%s).", source, legislature
        )
        return EmbedStats(total=0, embedded=0, skipped=0, deleted=0)

    logger.info("Loaded %d answer(s) from PostgreSQL.", len(answers))

    existing = _load_existing_points(vector_store, collection, source)

    # Stale cleanup: scope to the same source filter so a scoped run never
    # touches points outside its remit.
    # When legislature is set, answers is a subset of all source answers, so we
    # still need a separate query to avoid marking legislature-filtered answers
    # as stale.
    all_db_ids = (
        {a.id for a in answers} if legislature is None else _load_all_answer_ids(source)
    )
    stale_ids = [rid for rid in existing if rid not in all_db_ids]
    if stale_ids:
        logger.info("Removing %d stale point(s) (deleted from DB)...", len(stale_ids))
        stale_point_ids = [str(stable_answer_point_id(rid)) for rid in stale_ids]
        vector_store.delete_points_by_filter(
            collection,
            {"must": [{"has_id": stale_point_ids}]},
        )
        logger.info("  Done removing stale points.")

    to_embed: list[Reponse] = []
    skipped = 0

    for answer in answers:
        text = answer.texte_reponse
        if not text or not text.strip():
            logger.debug("Skipping empty answer %s.", answer.id)
            continue
        cached = existing.get(answer.id)
        if cached is not None:
            cached_model, cached_hash = cached
            if cached_model == embedding_model and cached_hash == compute_content_hash(
                text
            ):
                skipped += 1
                continue
        to_embed.append(answer)

    logger.info(
        "%d to embed, %d already up-to-date (skipped).",
        len(to_embed),
        skipped,
    )

    if not to_embed:
        logger.info("Nothing to do.")
        return EmbedStats(
            total=len(answers), embedded=0, skipped=skipped, deleted=len(stale_ids)
        )

    # Probe first batch for vector dimension, create collection if needed.
    first_batch_texts = [
        a.texte_reponse[:TEXT_MAX_CHARS] for a in to_embed[:batch_size]
    ]
    if rate_limiter:
        rate_limiter.acquire(1)
    logger.info(
        "Probing embedding dimension with first batch (%d answer(s))...",
        len(first_batch_texts),
    )
    first_embeddings = embedder.embed_batch_partial(first_batch_texts)
    probe = next((e for e in first_embeddings if e is not None), None)
    if probe is None:
        raise ValueError(
            "Every text in the probe batch was rejected by the content "
            "guardrail — cannot determine the embedding dimension."
        )
    vector_size = len(probe)

    existing_dim = vector_store.get_vector_size(collection)
    if existing_dim is None:
        vector_store.create_collection(collection, vector_size=vector_size)
        logger.info("Created collection '%s' (dim=%d).", collection, vector_size)
    elif existing_dim != vector_size:
        raise ValueError(
            f"Collection '{collection}' has dimension {existing_dim} but "
            f"model '{embedding_model}' produces dimension {vector_size}. "
            "Delete the collection or switch to the correct model."
        )

    upserted = 0
    blocked = 0
    batches = list(batched(to_embed, batch_size))

    with tqdm(total=len(to_embed), unit="a", desc="Embedding answers") as progress:
        for batch_idx, batch in enumerate(batches):
            if batch_idx == 0:
                embeddings = first_embeddings
            else:
                if rate_limiter:
                    rate_limiter.acquire(1)
                embeddings = embedder.embed_batch_partial(
                    [a.texte_reponse[:TEXT_MAX_CHARS] for a in batch]
                )

            points = []
            for answer, embedding in zip(batch, embeddings, strict=True):
                if embedding is None:
                    # Rejected by the content guardrail — skip this answer
                    # but keep embedding the rest of the corpus.
                    blocked += 1
                    logger.warning(
                        "Answer %s skipped (blocked by content guardrail).",
                        answer.id,
                    )
                    continue
                date_str = (
                    answer.date_reponse_jo.isoformat()
                    if answer.date_reponse_jo
                    else None
                )
                text = answer.texte_reponse
                question_ids = [q.id for q in answer.questions]
                points.append(
                    {
                        "id": stable_answer_point_id(answer.id),
                        "vector": embedding,
                        "payload": {
                            "kind": "answer",
                            "reponse_id": answer.id,
                            "source": answer.source,
                            "embedding_model": embedding_model,
                            "content_hash": compute_content_hash(text),
                            "texte_reponse": text[:TEXT_MAX_CHARS],
                            "texte_preview": make_preview(text),
                            "ministre_reponse_libelle": answer.ministre_reponse_libelle,
                            "date_reponse_jo": date_str,
                            "question_ids": question_ids,
                        },
                    }
                )

            if points:
                vector_store.upsert_points(collection, points)
            upserted += len(points)
            progress.update(len(batch))

    logger.info(
        "Done — %d upserted, %d skipped (up-to-date), %d blocked by "
        "guardrail, %d stale removed.",
        upserted,
        skipped,
        blocked,
        len(stale_ids),
    )
    return EmbedStats(
        total=len(answers), embedded=upserted, skipped=skipped, deleted=len(stale_ids)
    )


def try_embed_answers_from_env(source: str) -> EmbedStats | None:
    """Embed answers into pgvector using env-configured clients.

    Intended for use at the end of ingest scripts. Logs a warning and returns
    without raising if the required env vars are not set or if embedding fails.
    """
    from qe.clients.embedding import EmbeddingClient
    from qe.clients.pgvector_client import PgvectorClient
    from qe.config import get_settings

    try:
        settings = get_settings()
    except ValueError as exc:
        logger.warning(
            "Skipping answer embedding: %s. "
            "Set the required env vars and run scripts/embed_answers.py manually.",
            exc,
        )
        return None

    embedder = EmbeddingClient(
        url=settings.albert_embeddings_url,
        model=settings.albert_embedding_model,
        api_key=settings.albert_api_key,
    )
    vector_store = PgvectorClient()

    try:
        stats = embed_answers(
            embedder=embedder,
            vector_store=vector_store,
            embedding_model=settings.albert_embedding_model,
            source=source,
        )
        logger.info(
            "Answers embedded: %d upserted, %d skipped, %d stale removed.",
            stats.embedded,
            stats.skipped,
            stats.deleted,
        )
        return stats
    except Exception as exc:
        logger.warning(
            "Answer embedding failed: %s. Run scripts/embed_answers.py manually.", exc
        )
        return None
