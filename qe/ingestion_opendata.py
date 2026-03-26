"""DILA open data ingestion — parser and DB upsert for REDIF_ANQ*.xml.

Each .taz archive (e.g. ANQ20260010.taz) extracted from
  https://echanges.dila.gouv.fr/OPENDATA/Questions-Reponses/AN/Annee_en_cours/
contains:
  - REDIF_ANQ{NUM}.xml  — rich format, two sections:
        <Section part="QE">  -> new questions published in this JO
        <Section part="REP"> -> questions with answers published in this JO
  - XML1JO_AN{NUM}.xml  — lightweight index (numero + page_jo only), not used here
  - XML2JO_AN{NUM}.xml  — corrections/rectifications (usually empty)

Key conventions:
  - In the REP section, <idQuestion><type>REP</type> is NOT a real question type
    — it is a publication section marker.  It is normalised to "QE" since all
    ANQ files contain written questions (questions ecrites).
  - The question primary key is "{SOURCE}-{LEGISLATURE}-{TYPE}-{NUMERO}",
    e.g. "AN-17-QE-12345".
  - Ministries are inserted on-the-fly into the `ministeres` table using their
    `titre_jo` label.  `intitule_min` is initialised to the same label and can
    be corrected manually.
"""

from __future__ import annotations

import logging
import re
import tarfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
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
# Strip the "NUMERO. — DATE. — " prefix from texteQuestion text.
# Example: "13382. — 10 mars 2026. — Mme Corinne Vignon attire..."
# ---------------------------------------------------------------------------
_QE_HEADER_RE = re.compile(
    r"^\d+\.\s*[—\-–]\s*\d{1,2}\s+\w+\s+\d{4}\.\s*[—\-–]\s*",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuestion:
    """Question extracted from XML before being written to the database."""

    id: str  # "AN-17-QE-12345"
    numero_question: int
    type: str  # "QE"
    source: str  # "AN" | "SENAT"
    legislature: int
    etat_question: str  # "EN_COURS" | "REPONDU"
    date_publication_jo: date | None
    page_jo: int | None
    ministre_libelle: str | None  # ministreJO label
    auteur_nom: str | None  # parlementaire (full string)
    objet: str | None  # <Objet> short title
    texte_question: str
    # Response fields — None when etat_question == "EN_COURS"
    reponse_id: str | None = None
    texte_reponse: str | None = None
    no_publication: str | None = None
    date_reponse_jo: date | None = None
    page_reponse_jo: int | None = None


@dataclass
class IngestStats:
    """Counters returned by ingest_taz_file / ingest_questions."""

    taz_file: str = ""
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


def _clean_texte(raw: str) -> str:
    """Strip the leading 'NUMERO. — DATE. — ' prefix from texteQuestion."""
    return _QE_HEADER_RE.sub("", raw.strip())


def _parse_question_element(q_elem: Element) -> ParsedQuestion | None:
    """Parse a <question> XML element into a ParsedQuestion.

    Returns None if the element is malformed (required fields missing).
    """
    id_q = q_elem.find("idQuestion")
    if id_q is None:
        return None

    numero_str = (id_q.findtext("numeroQuestion") or "").strip()
    type_str = (id_q.findtext("type") or "").strip()
    legislature_str = (id_q.findtext("legislature") or "").strip()
    xml_source = (id_q.findtext("source") or "").strip()

    if not numero_str or not legislature_str or not xml_source:
        logger.debug("Skipping question with incomplete idQuestion fields")
        return None

    # "REP" in the REP section is not a real question type — normalise to "QE"
    canonical_type = "QE" if type_str in ("REP", "") else type_str

    legislature = int(legislature_str)
    numero = int(numero_str)
    qid = f"{xml_source}-{legislature}-{canonical_type}-{numero}"

    ref = q_elem.find("referencePublication")
    date_pub = _parse_date(ref.findtext("datePublication") if ref is not None else None)
    page_jo_str = (ref.findtext("pageJO") if ref is not None else None) or ""
    page_jo = int(page_jo_str.strip()) if page_jo_str.strip().isdigit() else None

    ministre_jo = (q_elem.findtext("ministreJO") or "").strip() or None
    parlementaire = (q_elem.findtext("parlementaire") or "").strip() or None
    objet = (q_elem.findtext("Objet") or "").strip() or None
    texte_raw = (q_elem.findtext("texteQuestion") or "").strip()
    texte = _clean_texte(texte_raw) if texte_raw else ""

    return ParsedQuestion(
        id=qid,
        numero_question=numero,
        type=canonical_type,
        source=xml_source,
        legislature=legislature,
        etat_question="EN_COURS",  # overridden by caller for REP section
        date_publication_jo=date_pub,
        page_jo=page_jo,
        ministre_libelle=ministre_jo,
        auteur_nom=parlementaire,
        objet=objet,
        texte_question=texte,
    )


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _parse_qe_section(section: Element) -> list[ParsedQuestion]:
    results: list[ParsedQuestion] = []
    for q_elem in section:
        pq = _parse_question_element(q_elem)
        if pq:
            results.append(pq)
    return results


def _parse_reponse_element(
    rep_elem: Element | None,
    source: str,
) -> tuple[str | None, str | None, str | None, date | None, int | None]:
    """Parse a <reponse> element.

    Returns (reponse_id, no_publication, texte_rep, date_rep, page_rep).
    reponse_id is None when the required JO reference fields are missing.
    """
    if rep_elem is None:
        return None, None, None, None, None

    texte_rep: str | None = (rep_elem.findtext("texteReponse") or "").strip() or None

    date_rep: date | None = None
    page_rep: int | None = None
    no_pub: str | None = None
    rep_ref = rep_elem.find("referencePublication")
    if rep_ref is not None:
        date_rep = _parse_date(rep_ref.findtext("datePublication"))
        page_str = (rep_ref.findtext("pageJO") or "").strip()
        page_rep = int(page_str) if page_str.isdigit() else None
        no_pub = (rep_ref.findtext("noPublication") or "").strip() or None

    reponse_id: str | None = None
    if no_pub and page_rep is not None:
        reponse_id = f"{source}-{no_pub}-{page_rep}"

    return reponse_id, no_pub, texte_rep, date_rep, page_rep


def _parse_rep_section(section: Element) -> list[ParsedQuestion]:
    results: list[ParsedQuestion] = []

    for entry in section:
        # Each entry: one or more <question> elements, then one <reponse>
        q_elems = [c for c in entry if c.tag == "question"]
        rep_elem = entry.find("reponse")

        # Derive source from the first question element in this entry
        source = ""
        if q_elems:
            id_q = q_elems[0].find("idQuestion")
            if id_q is not None:
                source = (id_q.findtext("source") or "").strip()

        reponse_id, no_pub, texte_rep, date_rep, page_rep = _parse_reponse_element(
            rep_elem, source
        )

        for q_elem in q_elems:
            pq = _parse_question_element(q_elem)
            if not pq:
                continue

            pq.etat_question = "REPONDU"
            pq.reponse_id = reponse_id
            pq.texte_reponse = texte_rep
            pq.no_publication = no_pub
            pq.date_reponse_jo = date_rep
            pq.page_reponse_jo = page_rep

            # In the REP section, referencePublication/datePublication reflects
            # the date of the *response* JO, not the original question
            # publication date. Clear these fields so we do not overwrite a
            # correct value already stored from a QE section entry.
            pq.date_publication_jo = None
            pq.page_jo = None

            results.append(pq)

    return results


def parse_redif_xml(xml_bytes: bytes) -> list[ParsedQuestion]:
    """Parse a REDIF_ANQ*.xml file and return a flat list of ParsedQuestion.

    The assembly source (AN / SENAT) is read directly from each
    <idQuestion><source> element in the XML, so no external parameter is needed.

    - QE section  -> etat_question = "EN_COURS"
    - REP section -> etat_question = "REPONDU" + response fields populated
    - Multiple <question> elements sharing one <reponse> (joint answer) each
      receive the same response data.
    """
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        logger.error("XML parse error: %s", exc)
        return []

    results: list[ParsedQuestion] = []

    for section in root.findall("Section"):
        part = section.attrib.get("part", "")
        if part == "QE":
            results.extend(_parse_qe_section(section))
        elif part == "REP":
            results.extend(_parse_rep_section(section))

    return results


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
    ingest_source: str = "opendata",
) -> IngestStats:
    """Upsert a list of ParsedQuestion into the `questions` table.

    Conflict strategy on primary key `id`:
      - New questions         -> full INSERT
      - Already known         -> UPDATE response fields only
        (etat_question, reponse_id, updated_at).
        Immutable fields (texte_question, author, deposit date…) are preserved.

    Additional upsert guards:
      - etat_question is never downgraded: REPONDU stays REPONDU even if the
        incoming row says EN_COURS.
      - date_publication_jo / page_jo use COALESCE(existing, incoming) so a
        correct value from a QE section is never overwritten by the NULL that
        REP section entries carry.
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

            min_id_rep: int | None = None
            if pq.ministre_libelle:
                min_id_rep = _get_or_create_ministere(
                    session, pq.ministre_libelle, ministere_cache, stats
                )

            rep_values = {
                "id": pq.reponse_id,
                "source": pq.source,
                "no_publication": pq.no_publication,
                "texte_reponse": pq.texte_reponse or "",
                # open data does not distinguish response ministry from depot ministry
                "ministre_reponse_id": min_id_rep,
                "ministre_reponse_libelle": pq.ministre_libelle,
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
                "objet": pq.objet,
                "texte_question": pq.texte_question,
                "reponse_id": pq.reponse_id,
                "ingest_source": ingest_source,
            }

            insert_stmt = pg_insert(Question).values(**values)

            # References to the existing row columns (the "target" side)
            _existing: dict[str, ColumnElement[Any]] = {
                col: literal_column(f"questions.{col}")
                for col in ("etat_question", "date_publication_jo", "page_jo")
            }

            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    # Never downgrade REPONDU -> EN_COURS: if the existing row
                    # is already REPONDU, keep it; otherwise take the incoming value.
                    "etat_question": case(
                        (_existing["etat_question"] == "REPONDU", "REPONDU"),
                        else_=insert_stmt.excluded.etat_question,
                    ),
                    # Preserve the original publication date if already set.
                    # REP section entries carry NULL here (date unknown); we must
                    # not overwrite a correct date from a QE section entry.
                    "date_publication_jo": func.coalesce(
                        _existing["date_publication_jo"],
                        insert_stmt.excluded.date_publication_jo,
                    ),
                    "page_jo": func.coalesce(
                        _existing["page_jo"],
                        insert_stmt.excluded.page_jo,
                    ),
                    # Static fields: update if incoming is not NULL
                    "objet": func.coalesce(
                        insert_stmt.excluded.objet,
                        literal_column("questions.objet"),
                    ),
                    # Response field: always update
                    "reponse_id": insert_stmt.excluded.reponse_id,
                    "updated_at": func.now(),
                },
            )
            session.execute(upsert_stmt)
            stats.questions_inserted += 1  # INSERT or UPDATE; can be refined later

    return stats


# ---------------------------------------------------------------------------
# Main entry point: process a single .taz archive
# ---------------------------------------------------------------------------


def ingest_taz_file(taz_path: Path, *, ingest_source: str = "opendata") -> IngestStats:
    """Extract and ingest a DILA open data .taz archive.

    Locates the REDIF_*.xml file inside the archive, parses it, then upserts
    the questions into the database.  The assembly source (AN / SENAT) is read
    from the XML itself.  Returns ingestion statistics.
    """
    logger.info("Processing %s", taz_path.name)

    try:
        with tarfile.open(taz_path) as tar:
            redif_member = next(
                (m for m in tar.getmembers() if m.name.startswith("REDIF_")),
                None,
            )
            if redif_member is None:
                logger.warning("No REDIF_*.xml found in %s", taz_path.name)
                return IngestStats(taz_file=taz_path.name)

            f = tar.extractfile(redif_member)
            if f is None:
                logger.warning(
                    "Could not read %s from %s",
                    redif_member.name,
                    taz_path.name,
                )
                return IngestStats(taz_file=taz_path.name)

            xml_bytes = f.read()

    except tarfile.TarError as exc:
        logger.error("Failed to open archive %s: %s", taz_path.name, exc)
        return IngestStats(taz_file=taz_path.name)

    questions = parse_redif_xml(xml_bytes)
    logger.info("  %d questions parsed", len(questions))

    stats = ingest_questions(questions, ingest_source=ingest_source)
    stats.taz_file = taz_path.name
    return stats
