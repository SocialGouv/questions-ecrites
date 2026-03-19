"""DILA Réponse Web Service client (REST/XML v3.4).

Implements the ministry-facing read endpoints used to poll new questions,
state changes, attributions, and government members.

URL convention: POST {base_url}/{ServiceName}/{MethodName}
    e.g. https://ws.example.gouv.fr/WSquestion/rechercherDossier

Authentication: HTTP Basic Auth (username / password).

Content-Type for requests: text/xml;charset=UTF-8

Reference XSD schemas (provided by DILA):
    WSquestion.xsd   — rechercherDossier, chercherChangementDEtatQuestions
    WSattribution.xsd — chercherAttributionsDate, chercherMembresGouvernement
    questions.xsd    — Question, Attribution, Auteur, Indexation* types
    reponses-commons.xsd — Ministre, ChangementEtatQuestion, MembreGouvernement, …
    reponses.xsd     — ReponsePubliee
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests
from defusedxml.ElementTree import ParseError, fromstring

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML namespace constants
# ---------------------------------------------------------------------------

_NS_WS_QUESTION = "http://www.dila.premier-ministre.gouv.fr/solrep/reponses/WSquestion"
_NS_WS_ATTRIBUTION = (
    "http://www.dila.premier-ministre.gouv.fr/solrep/reponses/WSattribution"
)

# ---------------------------------------------------------------------------
# XML helpers (namespace-agnostic parsing)
# ---------------------------------------------------------------------------


def _strip_ns(tag: str) -> str:
    """Remove the XML namespace prefix from a tag string.

    '{http://...}localname' -> 'localname'
    """
    return tag.split("}")[-1] if "}" in tag else tag


def _find_child(elem: Any, local_name: str) -> Any | None:
    """Find the first direct child whose local name matches, ignoring namespace."""
    for child in elem:
        if _strip_ns(child.tag) == local_name:
            return child
    return None


def _find_all_children(elem: Any, local_name: str) -> list[Any]:
    """Return all direct children whose local name matches, ignoring namespace."""
    return [c for c in elem if _strip_ns(c.tag) == local_name]


def _text(elem: Any, *path: str) -> str | None:
    """Traverse nested child elements by local name and return the text content."""
    current = elem
    for local_name in path:
        current = _find_child(current, local_name)
        if current is None:
            return None
    return (current.text or "").strip() or None


def _attr(elem: Any, name: str) -> str | None:
    """Return an attribute value, searching with and without namespace prefix."""
    # Attributes are usually unqualified even in namespace-aware documents.
    val = elem.attrib.get(name)
    if val is not None:
        return val.strip() or None
    # Fallback: search all attribute keys stripping namespaces
    for key, v in elem.attrib.items():
        if _strip_ns(key) == name:
            return v.strip() or None
    return None


def _parse_date(s: str | None) -> date | None:
    """Parse a YYYY-MM-DD string; return None if missing or invalid."""
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


def _parse_datetime(s: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string (with or without timezone offset)."""
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Data-transfer objects (WS response types)
# ---------------------------------------------------------------------------


@dataclass
class WSMinistre:
    """A ministry as returned by the WS (includes the numeric DB id)."""

    id: int  # numeric ministry ID from the WS (@id attribute)
    titre_jo: str  # short JO label
    intitule_min: str  # long official name
    titre_min: str | None = None  # minister title (e.g. "M. le ministre de …")
    ordre_proto: int | None = None  # protocol order


@dataclass
class WSAuteur:
    """Author (MP) of a parliamentary question."""

    id_mandat: str | None
    civilite: str | None  # "M." | "Mme" | …
    prenom: str | None
    nom: str | None
    grp_pol: str | None  # political group
    circonscription: str | None  # constituency


@dataclass
class WSIndexationAN:
    """AN subject indexation."""

    rubrique: str | None
    rubrique_ta: str | None  # analysis heading
    analyses: list[str] = field(default_factory=list)


@dataclass
class WSIndexationSenat:
    """Senate subject indexation."""

    themes: list[str] = field(default_factory=list)
    rubriques: list[str] = field(default_factory=list)


@dataclass
class WSReponse:
    """Published response as returned inside a dossier."""

    ministre_reponse: WSMinistre | None
    texte_reponse: str | None
    date_jo: date | None = None
    page_jo: int | None = None


@dataclass
class WSQuestion:
    """A full question+answer dossier as returned by rechercherDossier."""

    # --- identity ---
    numero_question: int
    type: str  # "QE"
    source: str  # "AN" | "SENAT"
    legislature: int
    id: str  # composed key: "{source}-{legislature}-{type}-{numero}"

    # --- state ---
    etat_question: str

    # --- JO publication ---
    date_publication_jo: date | None
    page_jo: int | None

    # --- ministries ---
    ministre_depot: WSMinistre | None
    ministre_attributaire: WSMinistre | None

    # --- author ---
    auteur: WSAuteur | None

    # --- text ---
    texte_question: str

    # --- response (None when EN_COURS) ---
    reponse: WSReponse | None = None

    # --- links ---
    rappel_id: str | None = None
    date_retrait: date | None = None

    # --- indexation (AN or Senate) ---
    indexation_an: WSIndexationAN | None = None
    titre_senat: str | None = None
    indexation_senat: WSIndexationSenat | None = None


@dataclass
class WSChangementEtat:
    """A state change event as returned by chercherChangementDEtatQuestions."""

    question_id: str  # composed key
    numero_question: int
    type: str
    source: str
    legislature: int
    nouvel_etat: str  # EtatQuestion value
    date_modif: date | None


@dataclass
class WSAttributionDate:
    """An attribution event as returned by chercherAttributionsDate."""

    question_id: str  # composed key
    numero_question: int
    type: str
    source: str
    legislature: int
    type_attribution: str  # "REATTRIBUTION" | "REAFFECTATION"
    attributaire: WSMinistre | None
    date_attribution: datetime | None


@dataclass
class WSMembreGouvernement:
    """A government member as returned by chercherMembresGouvernement."""

    prenom: str | None
    nom: str | None
    civilite: str | None
    ministre: WSMinistre | None
    en_fonction: bool
    date_debut: date | None = None
    date_fin: date | None = None


# ---------------------------------------------------------------------------
# Parsing helpers for complex WS types
# ---------------------------------------------------------------------------


def _parse_ministre(elem: Any | None) -> WSMinistre | None:
    """Parse a <ministre> or <ministre_depot>/<ministre_attributaire> element."""
    if elem is None:
        return None
    id_str = _attr(elem, "id")
    if id_str is None or not id_str.isdigit():
        logger.debug("ministre element missing @id")
        return None
    titre_jo = _text(elem, "titre_jo") or ""
    intitule_min = _text(elem, "intitule_min") or titre_jo
    titre_min = _text(elem, "titre_min")
    ordre_proto_str = _text(elem, "ordre_proto")
    ordre_proto = (
        int(ordre_proto_str) if ordre_proto_str and ordre_proto_str.isdigit() else None
    )
    return WSMinistre(
        id=int(id_str),
        titre_jo=titre_jo,
        intitule_min=intitule_min,
        titre_min=titre_min,
        ordre_proto=ordre_proto,
    )


def _parse_auteur(elem: Any | None) -> WSAuteur | None:
    """Parse an <auteur> element."""
    if elem is None:
        return None
    return WSAuteur(
        id_mandat=_attr(elem, "id_mandat"),
        civilite=_text(elem, "civilite"),
        prenom=_text(elem, "prenom"),
        nom=_text(elem, "nom"),
        grp_pol=_text(elem, "grp_pol"),
        circonscription=_text(elem, "circonscription"),
    )


def _parse_indexation_an(elem: Any | None) -> WSIndexationAN | None:
    """Parse an <indexation_an> element."""
    if elem is None:
        return None
    rubrique = _text(elem, "rubrique")
    rubrique_ta = _text(elem, "rubrique_ta")
    analyses = [
        c.text.strip()
        for c in _find_all_children(elem, "analyse")
        if c.text and c.text.strip()
    ]
    return WSIndexationAN(rubrique=rubrique, rubrique_ta=rubrique_ta, analyses=analyses)


def _parse_indexation_senat(elem: Any | None) -> WSIndexationSenat | None:
    """Parse an <indexation_senat> element."""
    if elem is None:
        return None
    themes = [
        c.text.strip()
        for c in _find_all_children(elem, "theme")
        if c.text and c.text.strip()
    ]
    rubriques = [
        c.text.strip()
        for c in _find_all_children(elem, "rubrique")
        if c.text and c.text.strip()
    ]
    return WSIndexationSenat(themes=themes, rubriques=rubriques)


def _build_question_id(source: str, legislature: int, q_type: str, numero: int) -> str:
    return f"{source}-{legislature}-{q_type}-{numero}"


def _parse_question_id(id_elem: Any | None) -> tuple[int, str, str, int] | None:
    """Parse a <id_question> element.

    Returns (numero_question, type, source, legislature) or None if incomplete.
    """
    if id_elem is None:
        return None
    numero_str = _text(id_elem, "numero_question")
    q_type = _text(id_elem, "type")
    source = _text(id_elem, "source")
    legislature_str = _text(id_elem, "legislature")
    # NOTE: using `all([...])` does not narrow `str | None` to `str` for mypy.
    if (
        numero_str is None
        or q_type is None
        or source is None
        or legislature_str is None
    ):
        return None
    try:
        numero = int(numero_str)
        legislature = int(legislature_str)
    except ValueError:
        return None

    return numero, q_type, source, legislature


def _parse_reponse(elem: Any | None) -> WSReponse | None:
    """Parse a <reponse> element (ReponsePubliee type)."""
    if elem is None:
        return None
    ministre_elem = _find_child(elem, "ministre_reponse")
    texte_elem = _find_child(elem, "texte_reponse")
    texte = (
        texte_elem.text.strip() if texte_elem is not None and texte_elem.text else None
    )
    date_jo_str = _text(elem, "date_jo")
    page_jo_str = _text(elem, "page_jo")
    page_jo = int(page_jo_str) if page_jo_str and page_jo_str.isdigit() else None
    return WSReponse(
        ministre_reponse=_parse_ministre(ministre_elem),
        texte_reponse=texte,
        date_jo=_parse_date(date_jo_str),
        page_jo=page_jo,
    )


def _parse_question_element(q_elem: Any) -> WSQuestion | None:
    """Parse a <question> element (Question complex type)."""
    id_elem = _find_child(q_elem, "id_question")
    parsed_id = _parse_question_id(id_elem)
    if parsed_id is None:
        logger.debug("Skipping question with incomplete id_question")
        return None
    numero, q_type, source, legislature = parsed_id
    qid = _build_question_id(source, legislature, q_type, numero)

    date_pub_str = _text(q_elem, "date_publication_jo")
    page_jo_str = _text(q_elem, "page_jo")
    page_jo = int(page_jo_str) if page_jo_str and page_jo_str.isdigit() else None

    ministre_depot_elem = _find_child(q_elem, "ministre_depot")
    ministre_attr_elem = _find_child(q_elem, "ministre_attributaire")
    auteur_elem = _find_child(q_elem, "auteur")

    texte_elem = _find_child(q_elem, "texte")
    texte = (
        texte_elem.text.strip() if texte_elem is not None and texte_elem.text else ""
    )

    etat_str = _text(q_elem, "etat_question") or "EN_COURS"

    rappel_elem = _find_child(q_elem, "rappel")
    rappel_id: str | None = None
    if rappel_elem is not None:
        rappel_parsed = _parse_question_id(rappel_elem)
        if rappel_parsed:
            # Cleaner: unpack properly
            r_num, r_type, r_src, r_leg = rappel_parsed
            rappel_id = _build_question_id(r_src, r_leg, r_type, r_num)

    date_retrait_str = _text(q_elem, "date_retrait")

    # Indexation: either indexation_an or titre_senat + indexation_senat
    indexation_an_elem = _find_child(q_elem, "indexation_an")
    titre_senat = _text(q_elem, "titre_senat")
    indexation_senat_elem = _find_child(q_elem, "indexation_senat")

    return WSQuestion(
        id=qid,
        numero_question=numero,
        type=q_type,
        source=source,
        legislature=legislature,
        etat_question=etat_str,
        date_publication_jo=_parse_date(date_pub_str),
        page_jo=page_jo,
        ministre_depot=_parse_ministre(ministre_depot_elem),
        ministre_attributaire=_parse_ministre(ministre_attr_elem),
        auteur=_parse_auteur(auteur_elem),
        texte_question=texte,
        date_retrait=_parse_date(date_retrait_str),
        rappel_id=rappel_id,
        indexation_an=_parse_indexation_an(indexation_an_elem),
        titre_senat=titre_senat,
        indexation_senat=_parse_indexation_senat(indexation_senat_elem),
    )


def _parse_dossier(dossier_elem: Any) -> list[WSQuestion]:
    """Parse a <dossier> element (QuestionReponse type).

    A dossier may contain multiple <question> elements sharing one <reponse>.
    """
    results: list[WSQuestion] = []
    q_elems = _find_all_children(dossier_elem, "question")
    reponse_elem = _find_child(dossier_elem, "reponse")
    reponse = _parse_reponse(reponse_elem)

    for q_elem in q_elems:
        wq = _parse_question_element(q_elem)
        if wq is None:
            continue
        wq.reponse = reponse
        results.append(wq)

    return results


# ---------------------------------------------------------------------------
# HTTP transport + retry
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (1, 2, 4, 8, 16)  # seconds between retries (exponential back-off)


class WSError(Exception):
    """Raised when the WS returns statut=KO."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ReponseWSClient:
    """HTTP client for the DILA Réponse Web Service (REST/XML v3.4).

    Usage::

        client = ReponseWSClient(
            base_url="https://reponses.dila.gouv.fr/ws",
            username="ministry_login",
            password="secret", #pragma: allowlist secret
        )
        questions = client.rechercher_dossier(
            date_debut=date(2026, 3, 1),
            date_fin=date(2026, 3, 18),
        )

    All methods raise :exc:`WSError` if the service returns ``statut=KO``.
    Transient network errors are retried with exponential back-off.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: int = 30,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.auth = (username, password)
        self._session.headers.update({"Content-Type": "text/xml;charset=UTF-8"})
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _post(self, service: str, method: str, body: str) -> Any:
        """POST an XML body and return the parsed root Element.

        Retries on network / 5xx errors with exponential back-off.
        Raises :exc:`WSError` on statut=KO in the response body.
        """
        url = f"{self._base}/{service}/{method}"
        last_exc: Exception | None = None

        for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
            try:
                resp = self._session.post(
                    url, data=body.encode("utf-8"), timeout=self._timeout
                )
                resp.raise_for_status()
                root = fromstring(resp.content)
                return root
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "Network error calling %s (attempt %d): %s", url, attempt, exc
                )
            except requests.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "HTTP %d from %s (attempt %d)",
                    exc.response.status_code,
                    url,
                    attempt,
                )
            except ParseError as exc:
                raise WSError(
                    f"Could not parse XML response from {url}: {exc}"
                ) from exc

            if delay is None:
                break
            logger.info("Retrying in %ds…", delay)
            time.sleep(delay)

        raise WSError(f"All retries exhausted for {url}: {last_exc}")

    @staticmethod
    def _check_statut(root: Any, method: str) -> None:
        """Raise WSError if the response statut is KO."""
        statut_elem = _find_child(root, "statut")
        statut = (
            statut_elem.text.strip()
            if statut_elem is not None and statut_elem.text
            else ""
        )
        if statut != "OK":
            msg_elem = _find_child(root, "message_erreur")
            msg = (
                msg_elem.text.strip()
                if msg_elem is not None and msg_elem.text
                else "(no message)"
            )
            raise WSError(f"{method} returned statut={statut!r}: {msg}")

    # ------------------------------------------------------------------
    # rechercherDossier — sliding date window, no jeton
    # ------------------------------------------------------------------

    def rechercher_dossier(
        self,
        date_debut: date,
        date_fin: date,
        sources: list[str] | None = None,
        types: list[str] | None = None,
        legislature: int | None = None,
    ) -> list[WSQuestion]:
        """Search for question+answer dossiers matching the given filters.

        Returns all matching dossiers in a single call (no jeton pagination).

        Args:
            date_debut: Start of the publication date window (inclusive).
            date_fin:   End of the publication date window (inclusive).
            sources:    List of ``QuestionSource`` values ("AN", "SENAT").
                        Defaults to both when omitted.
            types:      List of ``QuestionType`` values ("QE", …).
                        Defaults to ["QE"] when omitted.
            legislature: Filter by legislature number.  Omit to include all.
        """
        sources = sources or ["AN", "SENAT"]
        types = types or ["QE"]

        lines: list[str] = [f'<rechercherDossierRequest xmlns="{_NS_WS_QUESTION}">']
        for t in types:
            lines.append(f"  <type>{t}</type>")
        for s in sources:
            lines.append(f"  <source>{s}</source>")
        if legislature is not None:
            lines.append(f"  <legislature>{legislature}</legislature>")
        lines.append(f"  <date_debut>{date_debut.isoformat()}</date_debut>")
        lines.append(f"  <date_fin>{date_fin.isoformat()}</date_fin>")
        lines.append("</rechercherDossierRequest>")
        body = "\n".join(lines)

        root = self._post("WSquestion", "rechercherDossier", body)
        self._check_statut(root, "rechercherDossier")

        questions: list[WSQuestion] = []
        for dossier_elem in _find_all_children(root, "dossier"):
            questions.extend(_parse_dossier(dossier_elem))

        logger.debug(
            "rechercherDossier [%s→%s]: %d questions",
            date_debut,
            date_fin,
            len(questions),
        )
        return questions

    # ------------------------------------------------------------------
    # chercherChangementDEtatQuestions — jeton-paginated
    # ------------------------------------------------------------------

    def chercher_changements_etat(
        self,
        jeton: str | None = None,
    ) -> tuple[list[WSChangementEtat], str, bool]:
        """Retrieve state changes since the last consumed jeton.

        Call repeatedly until ``dernier_renvoi=True`` to drain the queue.

        Args:
            jeton: The cursor from the previous call.  Pass ``None`` on the
                   very first call (the server will initialise a new cursor).

        Returns:
            A 3-tuple: (list of changes, new jeton string, dernier_renvoi flag).
        """
        if jeton:
            inner = f"  <jeton_changements_etat>{jeton}</jeton_changements_etat>"
        else:
            # No jeton → first call; XSD requires a choice so we still need a
            # jeton element.  Send an empty string to request initialisation.
            inner = "  <jeton_changements_etat></jeton_changements_etat>"

        body = (
            f'<chercherChangementDEtatQuestionsRequest xmlns="{_NS_WS_QUESTION}">\n'
            f"{inner}\n"
            "</chercherChangementDEtatQuestionsRequest>"
        )
        root = self._post("WSquestion", "chercherChangementDEtatQuestions", body)

        statut_elem = _find_child(root, "statut")
        statut = (
            statut_elem.text.strip()
            if statut_elem is not None and statut_elem.text
            else ""
        )
        if statut != "OK":
            msg_elem = _find_child(root, "message_erreur")
            msg = (
                msg_elem.text.strip()
                if msg_elem is not None and msg_elem.text
                else "(no message)"
            )
            raise WSError(
                f"chercherChangementDEtatQuestions returned statut={statut!r}: {msg}"
            )

        jeton_elem = _find_child(root, "jeton_changements_etat")
        new_jeton = (
            jeton_elem.text.strip()
            if jeton_elem is not None and jeton_elem.text
            else ""
        )
        dernier_elem = _find_child(root, "dernier_renvoi")
        dernier_renvoi = (
            dernier_elem is not None
            and dernier_elem.text is not None
            and dernier_elem.text.strip().lower() == "true"
        )

        changes: list[WSChangementEtat] = []
        for ce_elem in _find_all_children(root, "changements_etat"):
            id_elem = _find_child(ce_elem, "id_question")
            parsed = _parse_question_id(id_elem)
            if parsed is None:
                continue
            numero, q_type, source, legislature = parsed
            qid = _build_question_id(source, legislature, q_type, numero)
            etat = _text(ce_elem, "type_modif") or ""
            date_modif = _parse_date(_text(ce_elem, "date_modif"))
            changes.append(
                WSChangementEtat(
                    question_id=qid,
                    numero_question=numero,
                    type=q_type,
                    source=source,
                    legislature=legislature,
                    nouvel_etat=etat,
                    date_modif=date_modif,
                )
            )

        logger.debug(
            "chercherChangementDEtatQuestions: %d changes, dernier_renvoi=%s",
            len(changes),
            dernier_renvoi,
        )
        return changes, new_jeton, dernier_renvoi

    # ------------------------------------------------------------------
    # chercherAttributionsDate — jeton-paginated
    # ------------------------------------------------------------------

    def chercher_attributions_date(
        self,
        jeton: str | None = None,
    ) -> tuple[list[WSAttributionDate], str, bool]:
        """Retrieve attribution events since the last consumed jeton.

        Call repeatedly until ``dernier_renvoi=True`` to drain the queue.

        Args:
            jeton: The cursor from the previous call.  Pass ``None`` on the
                   very first call.

        Returns:
            A 3-tuple: (list of attributions, new jeton string, dernier_renvoi flag).
        """
        inner = f"  <jeton>{jeton}</jeton>" if jeton else "  <jeton></jeton>"
        body = (
            f'<chercherAttributionsDateRequest xmlns="{_NS_WS_ATTRIBUTION}">\n'
            f"{inner}\n"
            "</chercherAttributionsDateRequest>"
        )
        root = self._post("WSattribution", "chercherAttributionsDate", body)
        self._check_statut(root, "chercherAttributionsDate")

        jeton_elem = _find_child(root, "jeton_attributions")
        new_jeton = (
            jeton_elem.text.strip()
            if jeton_elem is not None and jeton_elem.text
            else ""
        )
        dernier_elem = _find_child(root, "dernier_renvoi")
        dernier_renvoi = (
            dernier_elem is not None
            and dernier_elem.text is not None
            and dernier_elem.text.strip().lower() == "true"
        )

        attributions: list[WSAttributionDate] = []
        for attr_elem in _find_all_children(root, "attributions"):
            id_elem = _find_child(attr_elem, "id_question")
            parsed = _parse_question_id(id_elem)
            if parsed is None:
                continue
            numero, q_type, source, legislature = parsed
            qid = _build_question_id(source, legislature, q_type, numero)

            attribution_type = _attr(attr_elem, "type") or ""
            date_attr_str = _text(attr_elem, "date_attribution")

            # attributaire -> ministre
            attributaire_elem = _find_child(attr_elem, "attributaire")
            ministre_elem = (
                _find_child(attributaire_elem, "ministre")
                if attributaire_elem is not None
                else None
            )

            attributions.append(
                WSAttributionDate(
                    question_id=qid,
                    numero_question=numero,
                    type=q_type,
                    source=source,
                    legislature=legislature,
                    type_attribution=attribution_type,
                    attributaire=_parse_ministre(ministre_elem),
                    date_attribution=_parse_datetime(date_attr_str),
                )
            )

        logger.debug(
            "chercherAttributionsDate: %d attributions, dernier_renvoi=%s",
            len(attributions),
            dernier_renvoi,
        )
        return attributions, new_jeton, dernier_renvoi

    # ------------------------------------------------------------------
    # chercherMembresGouvernement — full list, no jeton
    # ------------------------------------------------------------------

    def chercher_membres_gouvernement(
        self,
        en_fonction: str = "ALL",
    ) -> list[WSMembreGouvernement]:
        """Return the list of government members.

        Args:
            en_fonction: "ALL" | "TRUE" | "FALSE" — filter by current status.

        Returns:
            List of :class:`WSMembreGouvernement` objects.
        """
        body = (
            f'<chercherMembresGouvernementRequest xmlns="{_NS_WS_ATTRIBUTION}">\n'
            f"  <en_fonction>{en_fonction}</en_fonction>\n"
            "</chercherMembresGouvernementRequest>"
        )
        root = self._post("WSattribution", "chercherMembresGouvernement", body)
        self._check_statut(root, "chercherMembresGouvernement")

        membres: list[WSMembreGouvernement] = []
        for mg_elem in _find_all_children(root, "membre_gouvernement"):
            en_f_str = _attr(mg_elem, "en_fonction") or "false"
            en_f = en_f_str.lower() == "true"
            date_debut = _parse_date(_attr(mg_elem, "date_debut"))
            date_fin = _parse_date(_attr(mg_elem, "date_fin"))
            ministre_elem = _find_child(mg_elem, "ministre")
            membres.append(
                WSMembreGouvernement(
                    prenom=_text(mg_elem, "prenom"),
                    nom=_text(mg_elem, "nom"),
                    civilite=_text(mg_elem, "civilite"),
                    ministre=_parse_ministre(ministre_elem),
                    en_fonction=en_f,
                    date_debut=date_debut,
                    date_fin=date_fin,
                )
            )

        logger.debug("chercherMembresGouvernement: %d members", len(membres))
        return membres

    # ------------------------------------------------------------------
    # Connectivity test
    # ------------------------------------------------------------------

    def test_connectivity(self) -> dict[str, bool]:
        """Hit the /test endpoints of WSquestion and WSattribution.

        Returns a dict mapping service names to reachability booleans.
        """
        results: dict[str, bool] = {}
        for service in ("WSquestion", "WSattribution"):
            url = f"{self._base}/{service}/test"
            try:
                resp = self._session.get(url, timeout=self._timeout)
                results[service] = resp.ok
                logger.info("%s/test → HTTP %d", service, resp.status_code)
            except requests.RequestException as exc:
                results[service] = False
                logger.warning("%s/test failed: %s", service, exc)
        return results
