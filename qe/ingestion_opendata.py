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
# Strip the "NUMERO. — DATE. — " prefix from texteQuestion text.
# Example: "13382. — 10 mars 2026. — Mme Corinne Vignon attire..."
# ---------------------------------------------------------------------------
_QE_HEADER_RE = re.compile(
    r"^\d+\.\s*[—\-–]\s*\d{1,2}\s+\w+\s+\d{4}\.\s*[—\-–]\s*",
    re.UNICODE,
)

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


# Strip HTML tags from legacy SENAT TexteQ / Texte_Reponse fields.
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)


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


def parse_senat_legacy_xml(xml_bytes: bytes) -> list[ParsedQuestion]:  # noqa: C901
    """Parse a legacy SENAT *SEQ*.xml file (pre-REDIF format).

    The root element is ``<QuestionsPubliees>`` (not ``<QuestionsPublieesAS>``).
    Elements use PascalCase (``<Question>``, ``<NumQ>``, ``<TexteQ>`` …) rather
    than the camelCase used in REDIF files.

    Sections handled:
    - ``<Section part="QE">``  → ``etat_question = "EN_COURS"``
    - ``<Section part="REP">`` → ``etat_question = "REPONDU"``; each
      ``<Question_Reponse>`` may contain multiple ``<Question>`` elements
      sharing one ``<Reponse>`` (joint answers).

    Legislature is derived from the question publication date using
    ``_legislature_from_date()``.
    """
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        logger.error("XML parse error (legacy SENAT): %s", exc)
        return []

    # Root attributes: DateJO="2024-03-07" NumJO="20240010"
    root_date = _parse_date(root.attrib.get("DateJO"))
    num_jo = (root.attrib.get("NumJO") or "").strip() or None

    def _parse_legacy_question(q_elem: Element) -> ParsedQuestion | None:
        gestion = q_elem.find("Gestion")
        if gestion is None:
            return None

        num_str = (gestion.findtext("NumQ") or "").strip()
        if not num_str:
            return None

        date_q = _parse_date(gestion.findtext("DateQ")) or root_date
        if date_q is None:
            logger.debug("Skipping legacy SENAT question with no date")
            return None

        legislature = _legislature_from_date(date_q)
        numero = int(num_str)
        qid = f"SENAT-{legislature}-QE-{numero}"

        page_str = (gestion.findtext("PageJO") or "").strip()
        page_jo = int(page_str) if page_str.isdigit() else None

        objet = (gestion.findtext("Objet") or "").strip() or None

        auteur = gestion.find("Auteur")
        if auteur is not None:
            nom = (auteur.findtext("Nom") or "").strip()
            prenom = (auteur.findtext("Prenom") or "").strip()
            civ = (auteur.findtext("Civilite") or "").strip()
            parts = [p for p in (civ, prenom, nom) if p]
            auteur_nom: str | None = " ".join(parts) or None
        else:
            auteur_nom = None

        dest = gestion.find("Destinataire")
        ministre_libelle: str | None = None
        if dest is not None:
            ministre_libelle = (dest.findtext("TitreMinistere") or "").strip() or None

        texte_raw = (q_elem.findtext("TexteQ") or "").strip()
        texte = _HTML_TAG_RE.sub("", texte_raw).strip() if texte_raw else ""

        return ParsedQuestion(
            id=qid,
            numero_question=numero,
            type="QE",
            source="SENAT",
            legislature=legislature,
            etat_question="EN_COURS",  # overridden by caller for REP section
            date_publication_jo=date_q,
            page_jo=page_jo,
            ministre_libelle=ministre_libelle,
            auteur_nom=auteur_nom,
            objet=objet,
            texte_question=texte,
        )

    results: list[ParsedQuestion] = []

    for section in root.findall("Section"):
        part = section.attrib.get("part", "")

        if part == "QE":
            for q_elem in section.findall("Question"):
                pq = _parse_legacy_question(q_elem)
                if pq:
                    results.append(pq)

        elif part == "REP":
            for entry in section.findall("Question_Reponse"):
                # Parse response metadata
                rep_elem = entry.find("Reponse")
                reponse_id: str | None = None
                texte_rep: str | None = None
                date_rep: date | None = None
                page_rep: int | None = None
                ministre_rep: str | None = None

                if rep_elem is not None:
                    rep_gestion = rep_elem.find("Gestion")
                    if rep_gestion is not None:
                        page_str = (rep_gestion.findtext("PageJO") or "").strip()
                        page_rep = int(page_str) if page_str.isdigit() else None
                        min_rep = rep_gestion.find("MinReponse")
                        if min_rep is not None:
                            ministre_rep = (
                                min_rep.findtext("TitreMinistereJO") or ""
                            ).strip() or None

                    texte_raw = (rep_elem.findtext("Texte_Reponse") or "").strip()
                    texte_rep = (
                        _HTML_TAG_RE.sub("", texte_raw).strip() if texte_raw else None
                    )

                    if num_jo and page_rep is not None:
                        reponse_id = f"SENAT-{num_jo}-{page_rep}"

                # Parse each question in this joint-answer group
                questions_elem = entry.find("Questions")
                q_elems = (
                    questions_elem.findall("Question")
                    if questions_elem is not None
                    else entry.findall("Question")
                )
                for q_elem in q_elems:
                    pq = _parse_legacy_question(q_elem)
                    if not pq:
                        continue

                    pq.etat_question = "REPONDU"
                    pq.reponse_id = reponse_id
                    pq.texte_reponse = texte_rep
                    pq.no_publication = num_jo
                    pq.date_reponse_jo = date_rep
                    pq.page_reponse_jo = page_rep
                    if ministre_rep:
                        pq.ministre_libelle = ministre_rep
                    # REP entries don't carry the original publication date
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
            members = tar.getmembers()

            redif_member = next(
                (m for m in members if m.name.startswith("REDIF_")), None
            )
            senat_legacy = next(
                (m for m in members if re.match(r"SEQ\d+\.xml$", m.name)), None
            )
            an_legacy = next(
                (m for m in members if m.name.startswith("XML1JO_AN")), None
            )

            if redif_member is not None:
                f = tar.extractfile(redif_member)
                if f is None:
                    logger.warning(
                        "Could not read %s from %s",
                        redif_member.name,
                        taz_path.name,
                    )
                    return IngestStats(taz_file=taz_path.name)
                xml_bytes = f.read()
                parser = parse_redif_xml

            elif senat_legacy is not None:
                f = tar.extractfile(senat_legacy)
                if f is None:
                    logger.warning(
                        "Could not read %s from %s",
                        senat_legacy.name,
                        taz_path.name,
                    )
                    return IngestStats(taz_file=taz_path.name)
                xml_bytes = f.read()
                parser = parse_senat_legacy_xml

            elif an_legacy is not None:
                logger.warning(
                    "Skipping %s: old AN format (XML1JO) contains no question text",
                    taz_path.name,
                )
                return IngestStats(taz_file=taz_path.name)

            else:
                logger.warning("No parseable XML found in %s", taz_path.name)
                return IngestStats(taz_file=taz_path.name)

    except tarfile.TarError as exc:
        logger.error("Failed to open archive %s: %s", taz_path.name, exc)
        return IngestStats(taz_file=taz_path.name)

    questions = parser(xml_bytes)
    logger.info("  %d questions parsed", len(questions))

    stats = ingest_questions(questions, ingest_source=ingest_source)
    stats.taz_file = taz_path.name
    return stats


# ---------------------------------------------------------------------------
# AN legacy ZIP archives (XIV and XV legislatures)
#
# Two distinct ZIP layouts exist:
#
#   XIV  — single file  Questions_ecrites_XIV.xml  with root <questionsEcrites>
#           containing many <question> children.  No XML namespace.
#           Dates in ISO 8601 format (YYYY-MM-DD).
#
#   XV   — one file per question under  xml/QANR5L15QE*.xml.
#           Namespace: http://schemas.assemblee-nationale.fr/referentiel
#           Dates in DD/MM/YYYY format.
# ---------------------------------------------------------------------------

_AN_NS = "http://schemas.assemblee-nationale.fr/referentiel"
_AN_DATE_DMY = re.compile(r"(\d{2})/(\d{2})/(\d{4})")  # DD/MM/YYYY (XV)


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
    # DD/MM/YYYY (XV format)
    m = _AN_DATE_DMY.fullmatch(s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


def _parse_an_question_element(  # noqa: C901
    elem: Element, tag: "Callable[[str], str]"
) -> ParsedQuestion | None:
    """Extract a ParsedQuestion from a single <question> Element.

    ``tag`` maps a plain field name to the qualified tag string, i.e.
    ``"{namespace}name"`` for XV (namespaced) or just ``"name"`` for XIV.
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

    # Synthetic reponse_id — the AN legacy format has no JO page number, so we
    # use the question id as the unique key to link to the reponses table.
    reponse_id: str | None = None
    no_publication: str | None = None
    if texte_reponse:
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
    )


def parse_an_archive_question_xml(xml_bytes: bytes) -> ParsedQuestion | None:
    """Parse a single namespaced question XML file (XV per-file format)."""
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        logger.debug("XML parse error (AN legacy XV): %s", exc)
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
    """Parse and ingest an AN legacy ZIP archive (XIV or XV legislature).

    Detects the archive format automatically:
    - XIV: single ``Questions_ecrites_XIV.xml`` bulk file, no namespace, ISO dates.
    - XV:  one file per question under ``xml/``, namespaced, DD/MM/YYYY dates.
    Questions are upserted in batches of 500 to limit memory usage.
    """
    logger.info("Processing %s", zip_path.name)

    BATCH = 500
    stats = IngestStats(taz_file=zip_path.name)

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
            # XV format: one XML per question
            logger.info("  XV format — %d XML entries", len(per_file_entries))
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
