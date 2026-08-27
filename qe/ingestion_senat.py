"""Sénat full-database SQL dump ingestion.

Parses the PostgreSQL pg_dump published at
    https://data.senat.fr/data/questions/questions.zip
(a ZIP containing a single .sql file) and upserts questions écrites from
legislature 14 onward into the database.

The dump uses pg_dump's ``COPY ... FROM stdin`` format with tab-delimited rows
and **internal database column names** (e.g. ``natquecod``, ``sorquecod``,
``txtque``), not the human-readable French labels shown in the CSV extracts.

Relevant tables in the dump (in dump order):
  - ``sortquestion``  — lookup: sorquecod → status label
  - ``tam_questions`` — main questions table (58 columns)
  - ``tam_reponses``  — response texts, linked by question id
  - ``the``           — theme lookup: thecle → thelib

Key conventions:
  - Only rows with ``natquecod = 'QE'`` are imported.
  - Legislature is read directly from the ``legislature`` column (no date
    derivation needed).  Only legislature 14 onward is kept.
  - The question primary key follows the shared convention:
    ``"SENAT-{legislature}-QE-{numero}"``.
  - Response text is in ``tam_reponses.txtrep`` (joined on question id).
  - ``reponse_id`` uses a synthetic key ``"SENAT-DUMP-{qid}"`` (no JO page
    number is available in this dump).
  - Theme codes (``#2#14#`` format) are resolved to labels via the ``the``
    lookup table, parsed in the same pass as ``tam_questions``.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import IO, Callable, Iterator

from qe.ingestion_an import (
    IngestStats,
    ParsedQuestion,
    ingest_questions,
)

logger = logging.getLogger(__name__)

_MIN_LEGISLATURE = 14

# pg_dump COPY header: COPY [schema.]table (col1, col2, ...) FROM stdin;
_COPY_RE = re.compile(
    r"^COPY\s+(\S+)\s*\(([^)]+)\)\s+FROM\s+stdin",
    re.IGNORECASE,
)

# pg_dump NULL marker
_NULL = r"\N"

# Columns we rely on from tam_questions — emit a warning if missing.
_EXPECTED_COLS = frozenset(
    {
        "natquecod",
        "legislature",
        "numero",
        "sorquecod",
        "titre",
        "nom",
        "prenom",
        "codequalite",
        "circonscription",
        "groupe",
        "datejodepot",
        "mindepotlib",
        "minreplib1",
        "datejorep1",
        "txtque",
        "themes",
        "id",
    }
)

# Keyword-based mapping from sorquelib → etat_question
# (handles any sorquelib label the Sénat might add in future dumps).
_SORQUELIB_TO_ETAT: list[tuple[str, str]] = [
    ("réponse", "REPONDU"),
    ("reçue", "REPONDU"),
    ("retirée", "RETIRE"),
    ("rappelée", "RETIRE"),
    ("caduque", "CADUQUE"),
    ("cours", "EN_COURS"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(fields: list[str], col_idx: dict[str, int], col: str) -> str | None:
    """Safe column lookup; returns None for missing columns or NULL markers."""
    idx = col_idx.get(col)
    if idx is None or idx >= len(fields):
        return None
    val = fields[idx]
    return None if (val == _NULL or not val) else val


def _parse_timestamp_as_date(s: str | None) -> date | None:
    """Parse a pg_dump timestamp 'YYYY-MM-DD HH:MM:SS' to a date."""
    if not s:
        return None
    s = s.strip()
    if len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _sorquelib_to_etat(lib: str | None) -> str:
    """Map a sortquestion label to an etat_question value."""
    if not lib:
        return "EN_COURS"
    lib_lower = lib.lower()
    for keyword, etat in _SORQUELIB_TO_ETAT:
        if keyword in lib_lower:
            return etat
    return "EN_COURS"


def _resolve_themes(raw: str | None, theme_map: dict[str, str]) -> list[str] | None:
    """Convert a '#2#14#' theme-code string to a list of theme labels."""
    if not raw:
        return None
    codes = [c.strip() for c in raw.split("#") if c.strip()]
    labels = [theme_map[c] for c in codes if c in theme_map]
    return labels if labels else None


# ---------------------------------------------------------------------------
# Intermediate structure for buffering a filtered question row
# ---------------------------------------------------------------------------


@dataclass
class _PartialQ:
    internal_id: str  # bigint id from tam_questions (for joining reponses)
    legislature: int
    numero: int
    sorquecod: str
    titre: str | None
    nom: str | None
    prenom: str | None
    codequalite: str | None
    circonscription: str | None
    groupe: str | None
    datejodepot: str | None  # raw timestamp — parsed later
    mindepotlib: str | None
    minreplib1: str | None  # response ministry from tam_questions
    datejorep1: str | None  # raw timestamp — parsed later
    txtque: str
    themes_raw: str | None  # "#2#14#" — resolved after the loop


# ---------------------------------------------------------------------------
# Two-pass parser
# ---------------------------------------------------------------------------


def _stream_copy_blocks(
    open_stream: Callable[[], IO[bytes]],
) -> Iterator[tuple[str, dict[str, int], list[str]]]:
    """Yield ``(table_name, col_idx, fields)`` for every data row in every COPY block.

    Opens a fresh stream each call so callers can iterate the dump more than
    once (e.g. one pass per table of interest) without holding it in memory.
    """
    with open_stream() as sql_file:
        reader = io.TextIOWrapper(sql_file, encoding="utf-8", errors="surrogateescape")
        current_table: str | None = None
        current_cols: dict[str, int] = {}

        for raw_line in reader:
            line = raw_line.rstrip("\n")

            if current_table is None:
                m = _COPY_RE.match(line)
                if m:
                    # Strip schema prefix ("questions.tam_questions" → "tam_questions")
                    full_name = m.group(1)
                    table_name = full_name.split(".")[-1]
                    cols = [c.strip().strip('"') for c in m.group(2).split(",")]
                    current_cols = {name: idx for idx, name in enumerate(cols)}
                    current_table = table_name

                    if table_name == "tam_questions":
                        missing = _EXPECTED_COLS - set(current_cols)
                        if missing:
                            logger.warning(
                                "Sénat dump: tam_questions is missing expected columns: %s",
                                ", ".join(sorted(missing)),
                            )
                        logger.debug("tam_questions COPY block: %d columns", len(cols))
                continue

            if line == "\\.":
                current_table = None
                current_cols = {}
                continue

            yield current_table, current_cols, line.split("\t")


def _parse_lookups_and_questions(  # noqa: C901
    open_stream: Callable[[], IO[bytes]],
) -> tuple[dict[str, str], dict[str, str], list[_PartialQ]]:
    """Pass 1: ``sortquestion``, ``the``, and filtered ``tam_questions`` rows.

    ``tam_reponses`` is skipped here — it's read separately in pass 2, once
    the set of question ids we actually care about is known, so that dict
    never has to hold response rows for the full 1978-present history.
    """
    sort_map: dict[str, str] = {}
    theme_map: dict[str, str] = {}
    partial: list[_PartialQ] = []

    for table, cols, fields in _stream_copy_blocks(open_stream):
        if table == "sortquestion":
            code = _get(fields, cols, "sorquecod")
            lib = _get(fields, cols, "sorquelib")
            if code and lib:
                sort_map[code] = lib

        elif table == "the":
            cle = _get(fields, cols, "thecle")
            lib = _get(fields, cols, "thelib")
            if cle and lib:
                theme_map[cle] = lib

        elif table == "tam_questions":
            # Filter: QE only
            if _get(fields, cols, "natquecod") != "QE":
                continue

            # Filter: legislature 14 onward
            leg_str = _get(fields, cols, "legislature")
            try:
                legislature = int(leg_str or "0")
            except ValueError:
                continue
            if legislature < _MIN_LEGISLATURE:
                continue

            # Question number
            numero_str = _get(fields, cols, "numero")
            if not numero_str:
                continue
            # Strip leading zeros; can be "01380"
            numero_str = numero_str.strip().lstrip("0") or "0"
            try:
                numero = int(numero_str)
            except ValueError:
                continue

            internal_id = _get(fields, cols, "id") or ""

            partial.append(
                _PartialQ(
                    internal_id=internal_id,
                    legislature=legislature,
                    numero=numero,
                    sorquecod=_get(fields, cols, "sorquecod") or "0",
                    titre=_get(fields, cols, "titre"),
                    nom=_get(fields, cols, "nom"),
                    prenom=_get(fields, cols, "prenom"),
                    codequalite=_get(fields, cols, "codequalite"),
                    circonscription=_get(fields, cols, "circonscription"),
                    groupe=_get(fields, cols, "groupe"),
                    datejodepot=_get(fields, cols, "datejodepot"),
                    mindepotlib=_get(fields, cols, "mindepotlib"),
                    minreplib1=_get(fields, cols, "minreplib1"),
                    datejorep1=_get(fields, cols, "datejorep1"),
                    txtque=_get(fields, cols, "txtque") or "",
                    themes_raw=_get(fields, cols, "themes"),
                )
            )

    return sort_map, theme_map, partial


def _parse_responses(
    open_stream: Callable[[], IO[bytes]], wanted_ids: set[str]
) -> dict[str, tuple]:
    """Pass 2: ``tam_reponses`` rows, kept only for ids in *wanted_ids*."""
    responses: dict[str, tuple] = {}
    for table, cols, fields in _stream_copy_blocks(open_stream):
        if table != "tam_reponses":
            continue
        idque = _get(fields, cols, "idque")
        if not idque or idque not in wanted_ids:
            continue
        txtrep = _get(fields, cols, "txtrep")
        datejorep = _get(fields, cols, "datejorep")
        minreplib = _get(fields, cols, "minreplib")
        responses[idque] = (txtrep, datejorep, minreplib)
    return responses


def parse_senat_sql_dump(
    open_stream: Callable[[], IO[bytes]],
) -> list[ParsedQuestion]:
    """Parse a Sénat pg_dump file, returning QE questions from leg 14 onward.

    Two passes over the dump, via *open_stream* (called once per pass — the
    caller must be able to reopen the same underlying file each time):
      1. ``sortquestion`` → sort_map, ``the`` → theme_map, ``tam_questions``
         → partial question rows (filtered to QE + leg >= 14).
      2. ``tam_reponses`` → responses dict, filtered to only the question ids
         kept in pass 1. Doing this as a second, filtered pass (rather than
         collecting every response row up front) keeps memory bounded to the
         legislatures we actually keep, instead of the full 1978-present
         history the dump covers.

    Question rows are then enriched with response text and resolved theme
    labels, and converted to ``ParsedQuestion`` objects.
    """
    sort_map, theme_map, partial = _parse_lookups_and_questions(open_stream)
    wanted_ids = {p.internal_id for p in partial}
    responses = _parse_responses(open_stream, wanted_ids)

    logger.debug(
        "Two-pass done: %d partial questions, %d responses, %d sort codes, %d themes",
        len(partial),
        len(responses),
        len(sort_map),
        len(theme_map),
    )

    # --- Enrich and convert ---
    questions: list[ParsedQuestion] = []
    for p in partial:
        qid = f"SENAT-{p.legislature}-QE-{p.numero}"

        etat = _sorquelib_to_etat(sort_map.get(p.sorquecod))

        date_pub = _parse_timestamp_as_date(p.datejodepot)
        date_rep = _parse_timestamp_as_date(p.datejorep1)

        themes = _resolve_themes(p.themes_raw, theme_map)

        # Response data: prefer tam_reponses (richer), fall back to tam_questions cols
        rep_row = responses.get(p.internal_id)
        if rep_row:
            txtrep, datejorep_raw, minreplib = rep_row
            date_rep = _parse_timestamp_as_date(datejorep_raw) or date_rep
            ministre_reponse = minreplib or p.minreplib1
        else:
            txtrep = None
            ministre_reponse = p.minreplib1 if etat == "REPONDU" else None

        reponse_id: str | None = None
        no_publication: str | None = None
        if etat == "REPONDU":
            reponse_id = f"SENAT-DUMP-{qid}"
            no_publication = "SENAT-DUMP"  # sentinel: no JO number in SQL dump

        questions.append(
            ParsedQuestion(
                id=qid,
                numero_question=p.numero,
                type="QE",
                source="SENAT",
                legislature=p.legislature,
                etat_question=etat,
                date_publication_jo=date_pub,
                page_jo=None,
                ministre_libelle=p.mindepotlib,
                auteur_nom=p.nom,
                auteur_prenom=p.prenom,
                auteur_grp_pol=p.groupe,
                auteur_circonscription=p.circonscription,
                titre_senat=p.titre,
                themes=themes,
                objet=None,
                texte_question=p.txtque,
                reponse_id=reponse_id,
                texte_reponse=txtrep,
                no_publication=no_publication,
                date_reponse_jo=date_rep,
                page_reponse_jo=None,
                ministre_reponse_libelle=ministre_reponse,
            )
        )

    return questions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest_senat_dump(
    zip_path: Path,
    *,
    ingest_source: str = "senat_portal",
    batch_size: int = 500,
) -> IngestStats:
    """Parse and ingest the Sénat full-database SQL dump ZIP.

    Opens the ZIP, extracts the .sql file, stream-parses COPY blocks for
    all relevant tables, filters to ``Nature=QE`` and legislature 14 onward,
    and upserts questions into the database in batches.
    """
    logger.info("Processing %s", zip_path.name)
    stats = IngestStats(source_file=zip_path.name)

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        logger.error("Failed to open %s: %s", zip_path.name, exc)
        return stats

    with zf:
        sql_names = [n for n in zf.namelist() if n.endswith(".sql")]
        if not sql_names:
            logger.error("No .sql file found in %s", zip_path.name)
            return stats
        if len(sql_names) > 1:
            logger.warning(
                "Multiple .sql files in %s, using first: %s",
                zip_path.name,
                sql_names[0],
            )
        sql_name = sql_names[0]

        questions = parse_senat_sql_dump(lambda: zf.open(sql_name))

    logger.info(
        "  %d questions parsed (legislature >= 14, natquecod=QE)", len(questions)
    )

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        s = ingest_questions(batch, ingest_source=ingest_source)
        stats.questions_parsed += s.questions_parsed
        stats.questions_inserted += s.questions_inserted
        stats.ministeres_created += s.ministeres_created

    logger.info(
        "  %s — %d questions upserted, %d new ministries",
        zip_path.name,
        stats.questions_inserted,
        stats.ministeres_created,
    )
    return stats
