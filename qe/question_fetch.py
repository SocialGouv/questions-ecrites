"""On-demand fetching of individual parliamentary questions from public sources.

**AN questions** are only published as a consolidated bulk ZIP archive — the AN
does not serve individual XML files over HTTP.  Use
``scripts/download_an_legacy.py --legislature 17 --ingest`` (run daily) to keep
the database current.

**Sénat questions** can be fetched individually by scraping the Sénat's public
question page:
    https://www.senat.fr/questions/base/{legislature}/q{legislature}{numero:05d}.html
"""

from __future__ import annotations

import logging
import re
from datetime import date
from html.parser import HTMLParser

import requests

from qe.ingestion_an import ParsedQuestion

logger = logging.getLogger(__name__)

_SENAT_BASE = "https://www.senat.fr"
_SENAT_ID_RE = re.compile(r"^SENAT-(\d+)-QE-(\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sénat — HTML scraping
# ---------------------------------------------------------------------------


def _senat_question_url(legislature: int, numero: int) -> str:
    return f"{_SENAT_BASE}/questions/base/{legislature}/q{legislature}{numero:05d}.html"


class _TextCollector(HTMLParser):
    """Strips HTML tags and collects visible text, inserting newlines at block boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag in {"p", "br", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    p = _TextCollector()
    p.feed(html)
    return p.get_text()


def _parse_date_fr(s: str | None) -> date | None:
    if not s:
        return None
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s.strip())
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


_RE_OBJET = re.compile(
    r"(?:Titre\s*:?|Question\s+(?:écrite|n[°º])\s*\d+\s*[-–]\s*)([^\n]+)",
    re.IGNORECASE,
)
_RE_AUTEUR = re.compile(
    r"(?:Auteur\s*:?\s*M(?:me|\.)?\.?\s*)([^\n,\(]+)",
    re.IGNORECASE,
)
_RE_PRENOM_NOM = re.compile(r"^([A-ZÀ-Ú][a-zà-ú\-]+)\s+([A-ZÀ-Ú\-]+)$")
_RE_MINISTERE = re.compile(
    r"(?:Minist[eè]re\s+(?:attributaire|int[eé]ress[eé]|de\s+d[eé]p[oô]t)\s*:?\s*)([^\n]+)",
    re.IGNORECASE,
)
_RE_DATE_PUB = re.compile(
    r"(?:Publi[eé]e?\s+(?:au\s+JO\s+)?(?:le\s+)?|Date\s+de\s+publication\s*:?\s*)(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
_RE_DATE_REP = re.compile(
    r"(?:R[ée]ponse\s+publi[eé]e?\s+(?:au\s+JO\s+)?(?:le\s+)?|Date\s+de\s+r[eé]ponse\s*:?\s*)(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
_RE_MIN_REP = re.compile(
    r"(?:Minist[eè]re\s+(?:charg[eé]\s+de\s+la\s+r[eé]ponse|ayant\s+r[eé]pondu)\s*:?\s*)([^\n]+)",
    re.IGNORECASE,
)
_RE_TEXTE_Q = re.compile(
    r"(?:Texte\s+de\s+la\s+question\s*:?\s*\n+)(.+?)(?=\n\s*(?:Texte\s+de\s+la\s+r[eé]ponse|R[eé]ponse|$))",
    re.IGNORECASE | re.DOTALL,
)
_RE_TEXTE_R = re.compile(
    r"(?:Texte\s+de\s+la\s+r[eé]ponse\s*:?\s*\n+)(.+?)(?=\n\s*(?:R[eé]f[eé]rence|Source|$))",
    re.IGNORECASE | re.DOTALL,
)
_STATUS_REPONDU = re.compile(r"r[eé]ponse\s+publi[eé]e", re.IGNORECASE)
_STATUS_RETIRE = re.compile(r"retir[eé]e|caduque", re.IGNORECASE)


def _extract_auteur(text: str) -> tuple[str | None, str | None]:
    """Return (auteur_nom, auteur_prenom) from the page plain text."""
    m = _RE_AUTEUR.search(text)
    if not m:
        return None, None
    raw = m.group(1).strip()
    pm = _RE_PRENOM_NOM.match(raw)
    if pm:
        return pm.group(2), pm.group(1)
    return raw, None


def _extract_reponse_fields(
    text: str, qid: str
) -> tuple[date | None, str | None, str | None, str | None, str | None]:
    """Return (date_rep, ministre_reponse, texte_reponse, reponse_id, no_publication)."""
    date_rep: date | None = None
    ministre_reponse: str | None = None
    texte_reponse: str | None = None

    m = _RE_DATE_REP.search(text)
    if m:
        date_rep = _parse_date_fr(m.group(1))
    m = _RE_MIN_REP.search(text)
    if m:
        ministre_reponse = m.group(1).strip()[:300] or None
    m = _RE_TEXTE_R.search(text)
    if m:
        texte_reponse = m.group(1).strip() or None

    return date_rep, ministre_reponse, texte_reponse, f"SENAT-WS-{qid}", "SENAT-WS"


def _extract_senat(html: str, legislature: int, numero: int) -> ParsedQuestion | None:
    text = _strip_html(html)
    text = re.sub(r"\n{3,}", "\n\n", text)

    qid = f"SENAT-{legislature}-QE-{numero}"

    if _STATUS_REPONDU.search(text):
        etat = "REPONDU"
    elif _STATUS_RETIRE.search(text):
        etat = "RETIRE"
    else:
        etat = "EN_COURS"

    objet: str | None = None
    m = _RE_OBJET.search(text)
    if m:
        objet = m.group(1).strip()[:500] or None

    auteur_nom, auteur_prenom = _extract_auteur(text)

    ministre_libelle: str | None = None
    m = _RE_MINISTERE.search(text)
    if m:
        ministre_libelle = m.group(1).strip()[:300] or None

    date_pub: date | None = None
    m = _RE_DATE_PUB.search(text)
    if m:
        date_pub = _parse_date_fr(m.group(1))

    date_rep: date | None = None
    ministre_reponse: str | None = None
    texte_reponse: str | None = None
    reponse_id: str | None = None
    no_publication: str | None = None
    if etat == "REPONDU":
        date_rep, ministre_reponse, texte_reponse, reponse_id, no_publication = (
            _extract_reponse_fields(text, qid)
        )

    texte_question = ""
    m = _RE_TEXTE_Q.search(text)
    if m:
        texte_question = m.group(1).strip()

    if not texte_question:
        logger.warning("%s — could not extract question text from page", qid)
        return None

    return ParsedQuestion(
        id=qid,
        numero_question=numero,
        type="QE",
        source="SENAT",
        legislature=legislature,
        etat_question=etat,
        date_publication_jo=date_pub,
        page_jo=None,
        ministre_libelle=ministre_libelle,
        auteur_nom=auteur_nom,
        auteur_prenom=auteur_prenom,
        titre_senat=objet,
        objet=None,
        texte_question=texte_question,
        reponse_id=reponse_id,
        texte_reponse=texte_reponse,
        no_publication=no_publication,
        date_reponse_jo=date_rep,
        page_reponse_jo=None,
        ministre_reponse_libelle=ministre_reponse,
    )


def fetch_senat_question(qid: str, http: requests.Session) -> ParsedQuestion | None:
    """Fetch a single Sénat question by ID (e.g. ``SENAT-17-QE-1234``)."""
    m = _SENAT_ID_RE.match(qid.strip())
    if not m:
        raise ValueError(
            f"Invalid Sénat question ID: {qid!r} (expected SENAT-NN-QE-NNNNN)"
        )
    legislature, numero = int(m.group(1)), int(m.group(2))

    url = _senat_question_url(legislature, numero)
    logger.info("Fetching %s from %s", qid, url)
    try:
        resp = http.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.error("%s — not found on senat.fr (404)", qid)
        else:
            logger.error("%s — HTTP error: %s", qid, exc)
        return None
    except requests.RequestException as exc:
        logger.error("%s — request failed: %s", qid, exc)
        return None

    parsed = _extract_senat(resp.text, legislature, numero)
    if parsed is None:
        logger.error(
            "%s — page fetched but could not be parsed; "
            "check URL pattern or page structure: %s",
            qid,
            url,
        )
    return parsed


def fetch_question(qid: str, http: requests.Session) -> ParsedQuestion | None:
    """Fetch a question from source based on the ID prefix.

    Only Sénat questions support individual on-demand fetch.  AN questions are
    only available through the bulk archive; call this for AN IDs will raise
    ``ValueError``.
    """
    upper = qid.upper()
    if upper.startswith("SENAT-"):
        return fetch_senat_question(qid, http)
    if upper.startswith("AN-"):
        raise ValueError(
            f"AN question '{qid}' cannot be fetched individually. "
            "Run scripts/download_an_legacy.py --legislature 17 --ingest to refresh the database."
        )
    raise ValueError(f"Unrecognised question ID format: {qid!r}")
