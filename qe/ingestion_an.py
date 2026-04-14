"""AN legacy question ingestion — ZIP archive parser and DB upsert.

Handles AN legislatures XIV, XV, XVI (static archives) and XVII (live archive).
Shared helpers (ParsedQuestion, IngestStats, ingest_questions,
_get_or_create_ministere) are also used by qe/ingestion_senat.py.

Two distinct ZIP layouts exist:

    XIV  — single file  Questions_ecrites_XIV.xml  with root <questionsEcrites>
            containing many <question> children.  No XML namespace.
            Dates in ISO 8601 format (YYYY-MM-DD).

    XV/XVI/XVII — one file per question under  xml/QANR5L*.xml.
            Namespace: http://schemas.assemblee-nationale.fr/referentiel
            Dates in DD/MM/YYYY format.

Key conventions:
  - The question primary key is "{SOURCE}-{LEGISLATURE}-{TYPE}-{NUMERO}",
    e.g. "AN-17-QE-12345".
  - Ministries are inserted on-the-fly into the `ministeres` table using their
    label.  `intitule_min` is initialised to the same label and can be
    corrected manually.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import ParseError, fromstring
from sqlalchemy import case, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from qe.db import get_session
from qe.models import Ministere, Question, Reponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legislature lookup — same numbering for AN and SENAT.
# Sorted descending so the first matching entry wins.
# ---------------------------------------------------------------------------
_LEGISLATURE_START_DATES: list[tuple[date, int]] = [
    (date(2024, 7, 18), 17),  # 17th: snap-election result
    (date(2022, 6, 22), 16),  # 16th
    (date(2017, 6, 21), 15),  # 15th
    (date(2012, 6, 20), 14),  # 14th
]


def _legislature_from_date(d: date) -> int:
    """Return the legislature number for a question published on date *d*."""
    for start, leg in _LEGISLATURE_START_DATES:
        if d >= start:
            return leg
    return 13  # fallback for very old data


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuestion:
    """Question extracted from XML or SQL before being written to the database."""

    id: str  # "AN-17-QE-12345" or "SENAT-16-QE-9876"
    numero_question: int
    type: str  # "QE"
    source: str  # "AN" | "SENAT"
    legislature: int
    etat_question: str  # "EN_COURS" | "REPONDU" | …
    date_publication_jo: date | None
    page_jo: int | None
    ministre_libelle: str | None  # depot/attributaire label
    auteur_nom: str | None  # parlementaire surname (or full string for AN)
    objet: str | None  # short title (AN only)
    texte_question: str
    # Response fields — None when etat_question == "EN_COURS"
    reponse_id: str | None = None
    texte_reponse: str | None = None
    no_publication: str | None = None
    date_reponse_jo: date | None = None
    page_reponse_jo: int | None = None
    # SENAT-specific / WS-enrichment fields
    auteur_prenom: str | None = None
    auteur_grp_pol: str | None = None
    auteur_circonscription: str | None = None
    titre_senat: str | None = None
    themes: list[str] | None = None
    ministre_reponse_libelle: str | None = None  # response ministry (SENAT)


@dataclass
class IngestStats:
    """Counters returned by ingest_an_zip_file / ingest_questions."""

    source_file: str = ""
    questions_parsed: int = 0
    questions_inserted: int = 0
    questions_updated: int = 0
    ministeres_created: int = 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str | None) -> date | None:
    """Parse a YYYY-MM-DD string; return None if missing or invalid."""
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Ministry cache (in-process, keyed by titre_jo)
# ---------------------------------------------------------------------------


def _load_ministere_cache(session: Session) -> dict[str, int]:
    """Load all known ministries into a titre_jo -> id dict."""
    rows = session.execute(select(Ministere.id, Ministere.titre_jo)).all()
    return {row.titre_jo: row.id for row in rows}


def _get_or_create_ministere(
    session: Session,
    titre_jo: str,
    cache: dict[str, int],
    stats: IngestStats,
) -> int:
    """Return the id of the ministry with the given titre_jo, inserting if needed."""
    if titre_jo in cache:
        return cache[titre_jo]

    # Not in cache — insert a new row
    new_min = Ministere(titre_jo=titre_jo, intitule_min=titre_jo)
    session.add(new_min)
    session.flush()  # obtain the auto-generated id
    cache[titre_jo] = new_min.id
    stats.ministeres_created += 1
    logger.debug("Created new ministry: %r (id=%d)", titre_jo, new_min.id)
    return new_min.id


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------


def ingest_questions(
    questions: list[ParsedQuestion],
    *,
    ingest_source: str = "an_legacy",
) -> IngestStats:
    """Upsert a list of ParsedQuestion into the `questions` table.

    Conflict strategy on primary key `id`:
      - New questions         -> full INSERT
      - Already known         -> UPDATE response fields only
        (etat_question, reponse_id, updated_at).
        Most fields (author, deposit date…) are preserved on conflict.

    Additional upsert guards:
      - etat_question is never downgraded: REPONDU stays REPONDU even if the
        incoming row says EN_COURS.
      - date_publication_jo / page_jo use COALESCE(existing, incoming) so a
        correct value is never overwritten by NULL.
      - texte_question uses COALESCE(NULLIF(existing, ''), incoming) so a
        question first inserted with empty text gets its text filled later.
      - reponse_id uses COALESCE(incoming, existing) so a valid reponse_id
        already in the DB is never cleared by a NULL.
      - SENAT-specific fields use COALESCE(incoming, existing) so WS-polling
        enrichment is not overwritten by a subsequent dump re-ingest.
    """
    stats = IngestStats(questions_parsed=len(questions))
    if not questions:
        return stats

    with get_session() as session:
        ministere_cache = _load_ministere_cache(session)

        # --- upsert responses first (FK target must exist before questions) ---
        seen_reponse_ids: set[str] = set()
        for pq in questions:
            if pq.reponse_id is None or pq.reponse_id in seen_reponse_ids:
                continue
            seen_reponse_ids.add(pq.reponse_id)

            # Use the dedicated response ministry label when available (SENAT),
            # otherwise fall back to the depot/attributaire label.
            reponse_min_label = pq.ministre_reponse_libelle or pq.ministre_libelle
            min_id_rep: int | None = None
            if reponse_min_label:
                min_id_rep = _get_or_create_ministere(
                    session, reponse_min_label, ministere_cache, stats
                )

            rep_values = {
                "id": pq.reponse_id,
                "source": pq.source,
                "no_publication": pq.no_publication,
                "texte_reponse": pq.texte_reponse or "",
                "ministre_reponse_id": min_id_rep,
                "ministre_reponse_libelle": reponse_min_label,
                "date_reponse_jo": pq.date_reponse_jo,
                "page_reponse_jo": pq.page_reponse_jo,
            }
            rep_stmt = (
                pg_insert(Reponse)
                .values(**rep_values)
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={"updated_at": func.now()},
                )
            )
            session.execute(rep_stmt)

        # --- upsert questions ---
        for pq in questions:
            min_id: int | None = None
            if pq.ministre_libelle:
                min_id = _get_or_create_ministere(
                    session, pq.ministre_libelle, ministere_cache, stats
                )

            values: dict = {
                "id": pq.id,
                "numero_question": pq.numero_question,
                "type": pq.type,
                "source": pq.source,
                "legislature": pq.legislature,
                "etat_question": pq.etat_question,
                "date_publication_jo": pq.date_publication_jo,
                "page_jo": pq.page_jo,
                "ministre_depot_id": min_id,
                "ministre_depot_libelle": pq.ministre_libelle,
                "ministre_attributaire_id": min_id,
                "ministre_attributaire_libelle": pq.ministre_libelle,
                "auteur_nom": pq.auteur_nom,
                "auteur_prenom": pq.auteur_prenom,
                "auteur_grp_pol": pq.auteur_grp_pol,
                "auteur_circonscription": pq.auteur_circonscription,
                "objet": pq.objet,
                "titre_senat": pq.titre_senat,
                "themes": pq.themes,
                "texte_question": pq.texte_question,
                "reponse_id": pq.reponse_id,
                "ingest_source": ingest_source,
            }

            insert_stmt = pg_insert(Question).values(**values)

            # References to the existing row columns (the "target" side)
            _existing: dict[str, ColumnElement[Any]] = {
                col: literal_column(f"questions.{col}")
                for col in (
                    "etat_question",
                    "date_publication_jo",
                    "page_jo",
                    "texte_question",
                    "reponse_id",
                    "auteur_prenom",
                    "auteur_grp_pol",
                    "auteur_circonscription",
                    "titre_senat",
                    "themes",
                )
            }

            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    # Never downgrade REPONDU -> EN_COURS
                    "etat_question": case(
                        (_existing["etat_question"] == "REPONDU", "REPONDU"),
                        else_=insert_stmt.excluded.etat_question,
                    ),
                    # Preserve the original publication date if already set.
                    "date_publication_jo": func.coalesce(
                        _existing["date_publication_jo"],
                        insert_stmt.excluded.date_publication_jo,
                    ),
                    "page_jo": func.coalesce(
                        _existing["page_jo"],
                        insert_stmt.excluded.page_jo,
                    ),
                    # Fill in texte_question if the existing value is empty.
                    "texte_question": func.coalesce(
                        func.nullif(_existing["texte_question"], ""),
                        insert_stmt.excluded.texte_question,
                    ),
                    # Static fields: update if incoming is not NULL
                    "objet": func.coalesce(
                        insert_stmt.excluded.objet,
                        literal_column("questions.objet"),
                    ),
                    # Response field: take incoming if set, preserve existing otherwise.
                    "reponse_id": func.coalesce(
                        insert_stmt.excluded.reponse_id,
                        _existing["reponse_id"],
                    ),
                    # SENAT-specific / enrichment fields: prefer existing non-NULL
                    # value so that WS-polling enrichment is not lost on re-ingest.
                    "auteur_prenom": func.coalesce(
                        _existing["auteur_prenom"],
                        insert_stmt.excluded.auteur_prenom,
                    ),
                    "auteur_grp_pol": func.coalesce(
                        _existing["auteur_grp_pol"],
                        insert_stmt.excluded.auteur_grp_pol,
                    ),
                    "auteur_circonscription": func.coalesce(
                        _existing["auteur_circonscription"],
                        insert_stmt.excluded.auteur_circonscription,
                    ),
                    "titre_senat": func.coalesce(
                        _existing["titre_senat"],
                        insert_stmt.excluded.titre_senat,
                    ),
                    "themes": func.coalesce(
                        _existing["themes"],
                        insert_stmt.excluded.themes,
                    ),
                    "updated_at": func.now(),
                },
            )
            session.execute(upsert_stmt)
            stats.questions_inserted += 1  # INSERT or UPDATE

    return stats


# ---------------------------------------------------------------------------
# AN legacy ZIP archives (XIV, XV, XVI, and XVII legislatures)
#
# Two distinct ZIP layouts exist:
#
#   XIV  — single file  Questions_ecrites_XIV.xml  with root <questionsEcrites>
#           containing many <question> children.  No XML namespace.
#           Dates in ISO 8601 format (YYYY-MM-DD).
#
#   XV/XVI/XVII — one file per question under  xml/QANR5L*.xml.
#           Namespace: http://schemas.assemblee-nationale.fr/referentiel
#           Dates in DD/MM/YYYY format.
# ---------------------------------------------------------------------------

_AN_NS = "http://schemas.assemblee-nationale.fr/referentiel"
_AN_DATE_DMY = re.compile(r"(\d{2})/(\d{2})/(\d{4})")  # DD/MM/YYYY (XV+)

# Strip the "NUMERO. — DATE. — " prefix from texteQuestion text.
# Example: "13382. — 10 mars 2026. — Mme Corinne Vignon attire..."
_QE_HEADER_RE = re.compile(
    r"^\d+\.\s*[—\-–]\s*\d{1,2}\s+\w+\s+\d{4}\.\s*[—\-–]\s*",
    re.UNICODE,
)


def _parse_an_date(s: str | None) -> date | None:
    """Parse a date string from AN legacy XML — either ISO or DD/MM/YYYY."""
    if not s:
        return None
    s = s.strip()
    # ISO 8601 (XIV format)
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    # DD/MM/YYYY (XV+ format)
    m = _AN_DATE_DMY.fullmatch(s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


def _clean_texte(raw: str) -> str:
    """Strip the leading 'NUMERO. — DATE. — ' prefix from texteQuestion."""
    return _QE_HEADER_RE.sub("", raw.strip())


def _parse_an_question_element(  # noqa: C901
    elem: Element, tag: "Callable[[str], str]"
) -> ParsedQuestion | None:
    """Extract a ParsedQuestion from a single <question> Element.

    ``tag`` maps a plain field name to the qualified tag string, i.e.
    ``"{namespace}name"`` for XV/XVI/XVII (namespaced) or just ``"name"`` for XIV.
    """

    def _t(node: Element | None, *names: str) -> str | None:
        for name in names:
            if node is None:
                return None
            node = node.find(tag(name))
        return (node.text or "").strip() or None if node is not None else None

    identifiant = elem.find(tag("identifiant"))
    if identifiant is None:
        return None

    numero_str = _t(identifiant, "numero")
    legislature_str = _t(identifiant, "legislature")
    if not numero_str or not legislature_str:
        return None

    numero = int(numero_str)
    legislature = int(legislature_str)
    qid = f"AN-{legislature}-QE-{numero}"

    # Ministry: prefer last attributee entry, fall back to minInt
    ministre_libelle: str | None = None
    min_attribs = elem.find(tag("minAttribs"))
    if min_attribs is not None:
        attrib_list = min_attribs.findall(tag("minAttrib"))
        if attrib_list:
            ministre_libelle = _t(
                attrib_list[-1].find(tag("denomination")), "developpe"
            )
    if ministre_libelle is None:
        ministre_libelle = _t(elem.find(tag("minInt")), "developpe")

    # Publication date and question text (first texteQuestion entry)
    date_pub: date | None = None
    texte_question = ""
    textes_q = elem.find(tag("textesQuestion"))
    if textes_q is not None:
        first_tq = textes_q.find(tag("texteQuestion"))
        if first_tq is not None:
            date_pub = _parse_an_date(_t(first_tq.find(tag("infoJO")), "dateJO"))
            texte_question = _t(first_tq, "texte") or ""

    # Objet — teteAnalyse if set, else first ANA element
    objet: str | None = None
    idx = elem.find(tag("indexationAN"))
    if idx is not None:
        objet = _t(idx, "teteAnalyse")
        if not objet:
            analyse = idx.find(tag("ANALYSE"))
            if analyse is not None:
                objet = _t(analyse, "ANA")

    # Response
    etat = "EN_COURS"
    texte_reponse: str | None = None
    date_reponse: date | None = None
    textes_r = elem.find(tag("textesReponse"))
    if textes_r is not None:
        first_tr = textes_r.find(tag("texteReponse"))
        if first_tr is not None:
            etat = "REPONDU"
            texte_reponse = _t(first_tr, "texte")
            date_reponse = _parse_an_date(_t(first_tr.find(tag("infoJO")), "dateJO"))

    # Extract pageJO from cloture/infoJO (present for all REP_PUB answers).
    page_reponse_jo: int | None = None
    page_jo_str = _t(elem.find(tag("cloture")), "infoJO", "pageJO")
    if page_jo_str:
        try:
            page_reponse_jo = int(page_jo_str)
        except ValueError:
            pass

    # Use JO date + page as reponse_id so questions that received the same
    # joint response (same JO publication page) share a reponse_id.
    # Falls back to a per-question synthetic key only when page data is absent.
    reponse_id: str | None = None
    no_publication: str | None = None
    if texte_reponse:
        if date_reponse and page_reponse_jo is not None:
            no_publication = date_reponse.strftime("%Y%m%d")
            reponse_id = f"AN-{no_publication}-{page_reponse_jo}"
        else:
            reponse_id = f"AN-LEGACY-{qid}"
            no_publication = "LEGACY"

    return ParsedQuestion(
        id=qid,
        numero_question=numero,
        type="QE",
        source="AN",
        legislature=legislature,
        etat_question=etat,
        date_publication_jo=date_pub,
        page_jo=None,
        ministre_libelle=ministre_libelle,
        auteur_nom=None,  # only acteurRef available; full name requires separate lookup
        objet=objet,
        texte_question=texte_question,
        reponse_id=reponse_id,
        no_publication=no_publication,
        texte_reponse=texte_reponse,
        date_reponse_jo=date_reponse,
        page_reponse_jo=page_reponse_jo,
    )


def parse_an_archive_question_xml(xml_bytes: bytes) -> ParsedQuestion | None:
    """Parse a single namespaced question XML file (XV/XVI/XVII per-file format)."""
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        logger.debug("XML parse error (AN legacy): %s", exc)
        return None
    return _parse_an_question_element(root, lambda name: f"{{{_AN_NS}}}{name}")


def parse_an_bulk_xml(xml_bytes: bytes) -> list[ParsedQuestion]:
    """Parse the XIV bulk XML file (root <questionsEcrites>, no namespace)."""
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        logger.debug("XML parse error (AN legacy XIV): %s", exc)
        return []

    questions: list[ParsedQuestion] = []
    for elem in root.iter("question"):
        pq = _parse_an_question_element(elem, lambda name: name)
        if pq is not None:
            questions.append(pq)
    return questions


def ingest_an_zip_file(
    zip_path: Path, *, ingest_source: str = "an_legacy"
) -> IngestStats:
    """Parse and ingest an AN ZIP archive (XIV, XV, XVI, or XVII legislature).

    Detects the archive format automatically:
    - XIV: single ``Questions_ecrites_XIV.xml`` bulk file, no namespace, ISO dates.
    - XV/XVI/XVII: one file per question under ``xml/``, namespaced, DD/MM/YYYY dates.
    Questions are upserted in batches of 500 to limit memory usage.
    """
    logger.info("Processing %s", zip_path.name)

    BATCH = 500
    stats = IngestStats(source_file=zip_path.name)

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        logger.error("Failed to open %s: %s", zip_path.name, exc)
        return stats

    def _flush(batch: list[ParsedQuestion]) -> None:
        s = ingest_questions(batch, ingest_source=ingest_source)
        stats.questions_parsed += s.questions_parsed
        stats.questions_inserted += s.questions_inserted
        stats.ministeres_created += s.ministeres_created

    with zf:
        names = zf.namelist()
        per_file_entries = [
            n for n in names if n.startswith("xml/") and n.endswith(".xml")
        ]
        bulk_entries = [n for n in names if n.endswith(".xml") and "/" not in n]

        if per_file_entries:
            # XV/XVI/XVII format: one XML per question
            logger.info("  per-file format — %d XML entries", len(per_file_entries))
            batch: list[ParsedQuestion] = []
            for name in per_file_entries:
                pq = parse_an_archive_question_xml(zf.read(name))
                if pq is None:
                    continue
                batch.append(pq)
                if len(batch) >= BATCH:
                    _flush(batch)
                    batch = []
            if batch:
                _flush(batch)

        elif bulk_entries:
            # XIV format: single bulk XML file
            logger.info("  XIV format — bulk XML: %s", bulk_entries[0])
            questions = parse_an_bulk_xml(zf.read(bulk_entries[0]))
            logger.info("  %d questions parsed", len(questions))
            for i in range(0, len(questions), BATCH):
                _flush(questions[i : i + BATCH])

        else:
            logger.warning("No parseable XML found in %s", zip_path.name)

    logger.info(
        "  %s — %d questions upserted, %d new ministries",
        zip_path.name,
        stats.questions_inserted,
        stats.ministeres_created,
    )
    return stats
