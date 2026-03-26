"""WS polling pipeline — ingest questions and state changes from the Réponse WS.

This module orchestrates three distinct polling loops:

1. **rechercherDossier** (sliding date window):
   Fetches question+answer dossiers published in the last ``lookback_days`` up
   to today + 7 days (to catch upcoming AN publications, which are announced
   on Tuesdays).  Results are upserted into the ``questions`` table with the
   same conflict strategy as the open data ingestion (never downgrade state,
   preserve original publication date with COALESCE).

2. **chercherChangementDEtatQuestions** (jeton queue):
   Drains the state-change event queue, updating ``etat_question`` in the
   ``questions`` table and inserting rows into ``question_state_changes``.
   The jeton cursor is persisted in ``ingest_cursors`` so successive runs
   resume from where they left off.

3. **chercherAttributionsDate** (jeton queue):
   Drains the attribution event queue, updating ``ministre_attributaire_*`` in
   the ``questions`` table and inserting rows into ``question_attributions``.
   The jeton cursor is persisted as well.

For **dossier** ingestion, ``ministry_filter`` (optional substring) is applied
client-side: only questions whose ``ministre_attributaire.titre_jo`` contains
the filter string (case-insensitive) are stored.  For state-change and
attribution queues, questions not already in the DB are silently ignored when a
filter is active.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import case, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement
from typing import Any

from qe.clients.reponse_ws import (
    ReponseWSClient,
    WSAttributionDate,
    WSChangementEtat,
    WSQuestion,
)
from qe.db import get_session
from qe.models import (
    IngestCursor,
    Ministere,
    Question,
    QuestionAttribution,
    QuestionStateChange,
    Reponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURSOR_ETAT = "ws_changements_etat"
_CURSOR_ATTRIBUTION = "ws_attributions_date"

# ---------------------------------------------------------------------------
# Polling statistics
# ---------------------------------------------------------------------------


@dataclass
class PollingStats:
    """Aggregate counters for one polling run."""

    questions_upserted: int = 0
    questions_skipped: int = 0
    state_changes_processed: int = 0
    attributions_processed: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Ministry cache (shared across functions in a single run)
# ---------------------------------------------------------------------------


def _load_ministere_cache(session: Session) -> dict[str, int]:
    """Load all known ministeries from DB into a titre_jo -> id mapping."""
    rows = session.execute(select(Ministere.id, Ministere.titre_jo)).all()
    return {row.titre_jo: row.id for row in rows}


def _get_or_create_ministere(
    session: Session,
    ws_id: int,
    titre_jo: str,
    intitule_min: str,
    cache_by_titre: dict[str, int],
    cache_by_id: dict[int, str],
) -> int:
    """Upsert a ministry from WS data into the ministeres table.

    The WS provides a numeric @id that open data lacks.  We upsert using the
    WS id as primary key to avoid duplicating rows already created from open
    data (which used an auto-incremented id).

    Strategy:
    - If the WS id already exists in DB: update titre_jo / intitule_min if
      they changed, return the id.
    - If titre_jo is already in the DB (inserted by open data with a different
      auto-id): update that row's id to the canonical WS id.  This is handled
      by the upsert on ``id`` PK.
    - If neither: insert a new row with the WS id.
    """
    if ws_id in cache_by_id:
        return ws_id  # already present; no write needed

    # Upsert by WS numeric id
    stmt = (
        pg_insert(Ministere)
        .values(
            id=ws_id,
            titre_jo=titre_jo,
            intitule_min=intitule_min,
        )
        .on_conflict_do_update(
            index_elements=["id"],
            set_={
                "titre_jo": titre_jo,
                "intitule_min": intitule_min,
                "updated_at": func.now(),
            },
        )
    )
    session.execute(stmt)
    cache_by_id[ws_id] = titre_jo
    cache_by_titre[titre_jo] = ws_id
    return ws_id


# ---------------------------------------------------------------------------
# Dossier upsert
# ---------------------------------------------------------------------------

# Fields set on INSERT only (immutable after first write)
_INSERT_ONLY = frozenset(
    {
        "id",
        "numero_question",
        "type",
        "source",
        "legislature",
        "date_publication_jo",
        "page_jo",
        "ministre_depot_id",
        "ministre_depot_libelle",
        "auteur_id_mandat",
        "auteur_nom",
        "auteur_prenom",
        "auteur_grp_pol",
        "auteur_circonscription",
        "texte_question",
        "rubrique",
        "rubrique_ta",
        "analyses",
        "titre_senat",
        "themes",
        "rubriques_senat",
        "ingest_source",
    }
)


def _upsert_ws_question(
    session: Session,
    wq: WSQuestion,
    min_cache_by_titre: dict[str, int],
    min_cache_by_id: dict[int, str],
) -> None:
    """Upsert a single WSQuestion into the questions table."""
    # Resolve ministry IDs using the WS numeric id (canonical)
    depot_id: int | None = None
    if wq.ministre_depot is not None:
        depot_id = _get_or_create_ministere(
            session,
            wq.ministre_depot.id,
            wq.ministre_depot.titre_jo,
            wq.ministre_depot.intitule_min,
            min_cache_by_titre,
            min_cache_by_id,
        )

    attr_id: int | None = None
    if wq.ministre_attributaire is not None:
        attr_id = _get_or_create_ministere(
            session,
            wq.ministre_attributaire.id,
            wq.ministre_attributaire.titre_jo,
            wq.ministre_attributaire.intitule_min,
            min_cache_by_titre,
            min_cache_by_id,
        )

    reponse_fk: str | None = None
    if wq.reponse is not None and wq.reponse.date_jo and wq.reponse.page_jo:
        # WS does not provide noPublication; use ISO date as stand-in
        no_pub = wq.reponse.date_jo.isoformat()
        reponse_fk = f"{wq.source}-{no_pub}-{wq.reponse.page_jo}"

        min_rep_id: int | None = None
        min_rep_libelle: str | None = None
        if wq.reponse.ministre_reponse is not None:
            min_rep_id = _get_or_create_ministere(
                session,
                wq.reponse.ministre_reponse.id,
                wq.reponse.ministre_reponse.titre_jo,
                wq.reponse.ministre_reponse.intitule_min,
                min_cache_by_titre,
                min_cache_by_id,
            )
            min_rep_libelle = wq.reponse.ministre_reponse.titre_jo

        rep_stmt = (
            pg_insert(Reponse)
            .values(
                id=reponse_fk,
                source=wq.source,
                no_publication=no_pub,
                texte_reponse=wq.reponse.texte_reponse or "",
                ministre_reponse_id=min_rep_id,
                ministre_reponse_libelle=min_rep_libelle,
                date_reponse_jo=wq.reponse.date_jo,
                page_reponse_jo=wq.reponse.page_jo,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"updated_at": func.now()},
            )
        )
        session.execute(rep_stmt)

    analyses: list[str] | None = None
    if wq.indexation_an is not None:
        analyses = wq.indexation_an.analyses or None

    values: dict[str, Any] = {
        "id": wq.id,
        "numero_question": wq.numero_question,
        "type": wq.type,
        "source": wq.source,
        "legislature": wq.legislature,
        "etat_question": wq.etat_question,
        "date_publication_jo": wq.date_publication_jo,
        "page_jo": wq.page_jo,
        "ministre_depot_id": depot_id,
        "ministre_depot_libelle": wq.ministre_depot.titre_jo
        if wq.ministre_depot
        else None,
        "ministre_attributaire_id": attr_id,
        "ministre_attributaire_libelle": (
            wq.ministre_attributaire.titre_jo if wq.ministre_attributaire else None
        ),
        "auteur_id_mandat": wq.auteur.id_mandat if wq.auteur else None,
        "auteur_nom": wq.auteur.nom if wq.auteur else None,
        "auteur_prenom": wq.auteur.prenom if wq.auteur else None,
        "auteur_grp_pol": wq.auteur.grp_pol if wq.auteur else None,
        "auteur_circonscription": wq.auteur.circonscription if wq.auteur else None,
        "texte_question": wq.texte_question,
        "reponse_id": reponse_fk,
        # Indexation — AN
        "rubrique": wq.indexation_an.rubrique if wq.indexation_an else None,
        "rubrique_ta": wq.indexation_an.rubrique_ta if wq.indexation_an else None,
        "analyses": analyses,
        # Indexation — Senate
        "titre_senat": wq.titre_senat,
        "themes": wq.indexation_senat.themes or None if wq.indexation_senat else None,
        "rubriques_senat": (
            wq.indexation_senat.rubriques or None if wq.indexation_senat else None
        ),
        # Links
        "rappel_id": wq.rappel_id,
        "date_retrait": wq.date_retrait,
        # Meta
        "ingest_source": "ws_polling",
    }

    insert_stmt = pg_insert(Question).values(**values)

    _existing: dict[str, ColumnElement[Any]] = {
        col: literal_column(f"questions.{col}")
        for col in ("etat_question", "date_publication_jo", "page_jo")
    }

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={
            # Never downgrade a REPONDU question back to EN_COURS
            "etat_question": case(
                (_existing["etat_question"] == "REPONDU", "REPONDU"),
                else_=insert_stmt.excluded.etat_question,
            ),
            # Preserve original publication date if already set
            "date_publication_jo": func.coalesce(
                _existing["date_publication_jo"],
                insert_stmt.excluded.date_publication_jo,
            ),
            "page_jo": func.coalesce(
                _existing["page_jo"],
                insert_stmt.excluded.page_jo,
            ),
            # Response FK: always overwrite (WS is authoritative)
            "reponse_id": insert_stmt.excluded.reponse_id,
            # Attributee ministry may change via re-attribution
            "ministre_attributaire_id": insert_stmt.excluded.ministre_attributaire_id,
            "ministre_attributaire_libelle": insert_stmt.excluded.ministre_attributaire_libelle,
            # Update rich fields that open data does not provide
            "auteur_id_mandat": insert_stmt.excluded.auteur_id_mandat,
            "auteur_prenom": insert_stmt.excluded.auteur_prenom,
            "auteur_grp_pol": insert_stmt.excluded.auteur_grp_pol,
            "auteur_circonscription": insert_stmt.excluded.auteur_circonscription,
            "rubrique": insert_stmt.excluded.rubrique,
            "rubrique_ta": insert_stmt.excluded.rubrique_ta,
            "analyses": insert_stmt.excluded.analyses,
            "titre_senat": insert_stmt.excluded.titre_senat,
            "themes": insert_stmt.excluded.themes,
            "rubriques_senat": insert_stmt.excluded.rubriques_senat,
            "date_retrait": insert_stmt.excluded.date_retrait,
            "rappel_id": insert_stmt.excluded.rappel_id,
            "updated_at": func.now(),
        },
    )
    session.execute(upsert_stmt)


# ---------------------------------------------------------------------------
# Loop 1 — rechercherDossier
# ---------------------------------------------------------------------------


def poll_new_questions(
    client: ReponseWSClient,
    *,
    lookback_days: int = 7,
    ministry_filter: str | None = None,
    sources: list[str] | None = None,
) -> PollingStats:
    """Fetch new/answered questions from the WS and upsert them into the DB.

    Uses ``rechercherDossier`` with a sliding date window:
      ``[today - lookback_days, today + 7d]``

    The forward window (+7 days) ensures upcoming publications (AN: Tuesdays,
    Senate: Thursdays) are not missed when the cron runs mid-week.

    Args:
        client:          Authenticated :class:`ReponseWSClient` instance.
        lookback_days:   Number of past days to include in the query window.
        ministry_filter: If set, only ingest questions whose
                         ``ministre_attributaire.titre_jo`` contains this
                         string (case-insensitive).  None = ingest all.
        sources:         List of sources to query ("AN", "SENAT").  Defaults
                         to both.
    """
    stats = PollingStats()
    today = date.today()
    date_debut = today - timedelta(days=lookback_days)
    date_fin = today + timedelta(days=7)

    logger.info(
        "poll_new_questions: window %s → %s, ministry_filter=%r",
        date_debut,
        date_fin,
        ministry_filter,
    )

    questions = client.rechercher_dossier(
        date_debut=date_debut,
        date_fin=date_fin,
        sources=sources,
    )
    logger.info("  %d dossiers returned by WS", len(questions))

    if not questions:
        return stats

    needle = ministry_filter.lower() if ministry_filter else None

    with get_session() as session:
        min_cache_by_titre = _load_ministere_cache(session)
        min_cache_by_id: dict[int, str] = {v: k for k, v in min_cache_by_titre.items()}

        for wq in questions:
            if needle is not None:
                attr = wq.ministre_attributaire
                label = attr.titre_jo if attr is not None else ""
                if needle not in label.lower():
                    stats.questions_skipped += 1
                    continue

            try:
                _upsert_ws_question(session, wq, min_cache_by_titre, min_cache_by_id)
                stats.questions_upserted += 1
            except Exception:
                logger.exception("Error upserting question %s", wq.id)
                stats.errors += 1

    logger.info(
        "poll_new_questions done: %d upserted, %d skipped (out of scope), %d errors",
        stats.questions_upserted,
        stats.questions_skipped,
        stats.errors,
    )
    return stats


# ---------------------------------------------------------------------------
# Loop 2 — chercherChangementDEtatQuestions
# ---------------------------------------------------------------------------


def poll_state_changes(
    client: ReponseWSClient,
    *,
    ministry_filter: str | None = None,
) -> PollingStats:
    """Drain the state-change event queue and apply updates to the DB.

    The jeton cursor is stored in ``ingest_cursors`` under the key
    ``ws_changements_etat``.  Each call advances the cursor until
    ``dernier_renvoi=True``.

    State changes are applied as follows:
    - ``etat_question`` on the question row is updated.
    - A row is inserted into ``question_state_changes`` for audit.

    Args:
        client:          Authenticated :class:`ReponseWSClient` instance.
        ministry_filter: If set, skip state changes for questions not already
                         in the DB (questions outside the filter were never
                         ingested).
    """
    stats = PollingStats()

    with get_session() as session:
        cursor_row = session.get(IngestCursor, _CURSOR_ETAT)
        jeton: str | None = cursor_row.jeton if cursor_row else None

    dernier_renvoi = False
    batch_count = 0

    while not dernier_renvoi:
        changes, new_jeton, dernier_renvoi = client.chercher_changements_etat(jeton)
        jeton = new_jeton
        batch_count += 1

        if changes:
            _apply_state_changes(changes, ministry_filter=ministry_filter)
            stats.state_changes_processed += len(changes)

        # Persist cursor after each successful batch
        _save_cursor(
            _CURSOR_ETAT,
            jeton=new_jeton,
            last_date=date.today(),
        )

        if dernier_renvoi:
            break

    logger.info(
        "poll_state_changes done: %d batches, %d changes applied",
        batch_count,
        stats.state_changes_processed,
    )
    return stats


def _apply_state_changes(
    changes: list[WSChangementEtat],
    *,
    ministry_filter: str | None,
) -> None:
    """Persist a batch of state changes to the DB."""
    with get_session() as session:
        for ce in changes:
            q = session.get(Question, ce.question_id)
            if q is None:
                if ministry_filter:
                    continue  # question was not ingested — outside the filter scope
                logger.debug(
                    "State change for unknown question %s — skipped", ce.question_id
                )
                continue

            old_etat = q.etat_question
            if old_etat == ce.nouvel_etat:
                continue  # no change

            q.etat_question = ce.nouvel_etat
            q.updated_at = func.now()  # type: ignore[assignment]

            session.add(
                QuestionStateChange(
                    question_id=ce.question_id,
                    etat=ce.nouvel_etat,
                    date_modif=ce.date_modif or date.today(),
                )
            )
            logger.debug(
                "State change %s: %s → %s", ce.question_id, old_etat, ce.nouvel_etat
            )


# ---------------------------------------------------------------------------
# Loop 3 — chercherAttributionsDate
# ---------------------------------------------------------------------------


def poll_attributions(
    client: ReponseWSClient,
    *,
    ministry_filter: str | None = None,
) -> PollingStats:
    """Drain the attribution event queue and apply updates to the DB.

    The jeton cursor is stored under ``ws_attributions_date``.

    Attribution events are applied as follows:
    - ``ministre_attributaire_id`` / ``ministre_attributaire_libelle`` on the
      question row are updated.
    - A row is inserted into ``question_attributions`` for audit.

    Args:
        client:          Authenticated :class:`ReponseWSClient` instance.
        ministry_filter: If set, skip attributions for questions not in the DB.
    """
    stats = PollingStats()

    with get_session() as session:
        cursor_row = session.get(IngestCursor, _CURSOR_ATTRIBUTION)
        jeton: str | None = cursor_row.jeton if cursor_row else None

    dernier_renvoi = False
    batch_count = 0

    while not dernier_renvoi:
        attributions, new_jeton, dernier_renvoi = client.chercher_attributions_date(
            jeton
        )
        jeton = new_jeton
        batch_count += 1

        if attributions:
            _apply_attributions(attributions, ministry_filter=ministry_filter)
            stats.attributions_processed += len(attributions)

        _save_cursor(
            _CURSOR_ATTRIBUTION,
            jeton=new_jeton,
            last_date=date.today(),
        )

        if dernier_renvoi:
            break

    logger.info(
        "poll_attributions done: %d batches, %d attributions applied",
        batch_count,
        stats.attributions_processed,
    )
    return stats


def _apply_attributions(
    attributions: list[WSAttributionDate],
    *,
    ministry_filter: str | None,
) -> None:
    """Persist a batch of attribution events to the DB."""
    with get_session() as session:
        for attr in attributions:
            q = session.get(Question, attr.question_id)
            if q is None:
                if ministry_filter:
                    continue
                logger.debug(
                    "Attribution for unknown question %s — skipped", attr.question_id
                )
                continue

            # Update the attributee ministry on the question
            new_id: int | None = None
            new_libelle: str | None = None
            if attr.attributaire is not None:
                new_id = attr.attributaire.id
                new_libelle = attr.attributaire.titre_jo
                # Ensure the ministry exists
                stmt = (
                    pg_insert(Ministere)
                    .values(
                        id=attr.attributaire.id,
                        titre_jo=attr.attributaire.titre_jo,
                        intitule_min=attr.attributaire.intitule_min,
                    )
                    .on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "titre_jo": attr.attributaire.titre_jo,
                            "intitule_min": attr.attributaire.intitule_min,
                            "updated_at": func.now(),
                        },
                    )
                )
                session.execute(stmt)

            q.ministre_attributaire_id = new_id  # type: ignore[assignment]
            q.ministre_attributaire_libelle = new_libelle  # type: ignore[assignment]
            q.updated_at = func.now()  # type: ignore[assignment]

            session.add(
                QuestionAttribution(
                    question_id=attr.question_id,
                    type_attribution=attr.type_attribution,
                    attributaire_id=new_id,
                    attributaire_libelle=new_libelle,
                    date_attribution=attr.date_attribution,
                )
            )
            logger.debug(
                "Attribution %s: %s → %s (%s)",
                attr.question_id,
                attr.type_attribution,
                new_libelle,
                attr.date_attribution,
            )


# ---------------------------------------------------------------------------
# Cursor persistence helpers
# ---------------------------------------------------------------------------


def _save_cursor(
    cursor_name: str, *, jeton: str | None, last_date: date | None
) -> None:
    """Upsert a cursor row in ingest_cursors."""
    with get_session() as session:
        stmt = (
            pg_insert(IngestCursor)
            .values(
                cursor_name=cursor_name,
                jeton=jeton,
                last_date=last_date,
            )
            .on_conflict_do_update(
                index_elements=["cursor_name"],
                set_={
                    "jeton": jeton,
                    "last_date": last_date,
                    "updated_at": func.now(),
                },
            )
        )
        session.execute(stmt)


# ---------------------------------------------------------------------------
# High-level entry point: run all three loops
# ---------------------------------------------------------------------------


def run_full_poll(
    client: ReponseWSClient,
    *,
    lookback_days: int = 7,
    ministry_filter: str | None = None,
    sources: list[str] | None = None,
    skip_dossier: bool = False,
    skip_state_changes: bool = False,
    skip_attributions: bool = False,
) -> PollingStats:
    """Run all three polling loops in sequence.

    Returns aggregated :class:`PollingStats`.

    Args:
        client:            Authenticated :class:`ReponseWSClient` instance.
        lookback_days:     Lookback window for rechercherDossier.
        ministry_filter:   Case-insensitive substring filter on
                           ``ministre_attributaire.titre_jo``.  None = no filter.
        sources:           AN / SENAT filter for dossier loop.
        skip_dossier:      Skip the rechercherDossier loop.
        skip_state_changes: Skip the state-change queue.
        skip_attributions: Skip the attribution queue.
    """
    total = PollingStats()

    if not skip_dossier:
        s = poll_new_questions(
            client,
            lookback_days=lookback_days,
            ministry_filter=ministry_filter,
            sources=sources,
        )
        total.questions_upserted += s.questions_upserted
        total.questions_skipped += s.questions_skipped
        total.errors += s.errors

    if not skip_state_changes:
        s = poll_state_changes(client, ministry_filter=ministry_filter)
        total.state_changes_processed += s.state_changes_processed
        total.errors += s.errors

    if not skip_attributions:
        s = poll_attributions(client, ministry_filter=ministry_filter)
        total.attributions_processed += s.attributions_processed
        total.errors += s.errors

    return total
