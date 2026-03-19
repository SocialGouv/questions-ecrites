"""Tests for qe/clients/reponse_ws.py.

Strategy: the most valuable code in this module is the XML parsing layer —
pure functions that transform raw XML into typed Python objects.  We test
those directly with hand-crafted XML strings, requiring zero mocking.

We deliberately skip:
- The `_post()` / HTTP transport wiring (thin wrapper around `requests`; the
  retry loop is covered by the retry-delay list constants, not worth faking a
  server for).
- SQLAlchemy upserts in ingestion_ws_polling (integration-level; needs a real
  DB — tested separately when the service is running).
"""

from __future__ import annotations

from datetime import date, datetime
from xml.etree.ElementTree import fromstring as stdlib_fromstring

import pytest

from qe.clients.reponse_ws import (
    ReponseWSClient,
    WSError,
    _attr,
    _build_question_id,
    _find_all_children,
    _find_child,
    _parse_auteur,
    _parse_date,
    _parse_datetime,
    _parse_dossier,
    _parse_indexation_an,
    _parse_indexation_senat,
    _parse_ministre,
    _parse_question_id,
    _strip_ns,
    _text,
)

# ---------------------------------------------------------------------------
# Helpers — we use stdlib ET to build test elements (trusted, hand-crafted XML)
# ---------------------------------------------------------------------------


def el(xml: str):
    """Parse a literal XML string into an Element (test helper)."""
    return stdlib_fromstring(xml)  # noqa: S314


# ===========================================================================
# _strip_ns
# ===========================================================================


def test_strip_ns_with_namespace():
    assert _strip_ns("{http://example.com/ns}localname") == "localname"


def test_strip_ns_without_namespace():
    assert _strip_ns("localname") == "localname"


def test_strip_ns_empty():
    assert _strip_ns("") == ""


# ===========================================================================
# _find_child / _find_all_children
# ===========================================================================


def test_find_child_present():
    root = el("<root><child>hello</child><other/></root>")
    child = _find_child(root, "child")
    assert child is not None
    assert child.text == "hello"


def test_find_child_absent_returns_none():
    root = el("<root><other/></root>")
    assert _find_child(root, "missing") is None


def test_find_child_ignores_namespace():
    """Elements declared with a namespace prefix are still found by local name."""
    root = el('<root xmlns:ns="http://example.com"><ns:target>hi</ns:target></root>')
    found = _find_child(root, "target")
    assert found is not None
    assert found.text == "hi"


def test_find_all_children_multiple():
    root = el("<root><item>a</item><item>b</item><other/></root>")
    items = _find_all_children(root, "item")
    assert [i.text for i in items] == ["a", "b"]


def test_find_all_children_none_match():
    root = el("<root><item>a</item></root>")
    assert _find_all_children(root, "ghost") == []


# ===========================================================================
# _text
# ===========================================================================


def test_text_direct_child():
    root = el("<root><value>42</value></root>")
    assert _text(root, "value") == "42"


def test_text_nested_path():
    root = el("<root><a><b>deep</b></a></root>")
    assert _text(root, "a", "b") == "deep"


def test_text_missing_intermediate_returns_none():
    root = el("<root><a><b>deep</b></a></root>")
    assert _text(root, "x", "b") is None


def test_text_empty_element_returns_none():
    root = el("<root><value></value></root>")
    assert _text(root, "value") is None


def test_text_whitespace_only_returns_none():
    root = el("<root><value>   </value></root>")
    assert _text(root, "value") is None


# ===========================================================================
# _attr
# ===========================================================================


def test_attr_present():
    root = el('<root id="99"/>')
    assert _attr(root, "id") == "99"


def test_attr_absent_returns_none():
    root = el("<root/>")
    assert _attr(root, "id") is None


def test_attr_namespaced_key():
    """Attributes set with a namespace prefix are found by local name."""
    root = el(
        '<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="foo"/>'
    )
    assert _attr(root, "type") == "foo"


# ===========================================================================
# _parse_date / _parse_datetime
# ===========================================================================


def test_parse_date_valid():
    assert _parse_date("2026-03-18") == date(2026, 3, 18)


def test_parse_date_none():
    assert _parse_date(None) is None


def test_parse_date_invalid_string():
    assert _parse_date("not-a-date") is None


def test_parse_date_strips_whitespace():
    assert _parse_date("  2026-01-01  ") == date(2026, 1, 1)


def test_parse_datetime_naive():
    result = _parse_datetime("2026-03-18T14:30:00")
    assert result == datetime(2026, 3, 18, 14, 30, 0)


def test_parse_datetime_with_offset():
    result = _parse_datetime("2026-03-18T14:30:00+01:00")
    assert result is not None
    assert result.year == 2026
    assert result.hour == 14


def test_parse_datetime_none():
    assert _parse_datetime(None) is None


def test_parse_datetime_invalid():
    assert _parse_datetime("not-a-datetime") is None


# ===========================================================================
# _build_question_id
# ===========================================================================


def test_build_question_id():
    assert _build_question_id("AN", 17, "QE", 12345) == "AN-17-QE-12345"
    assert _build_question_id("SENAT", 16, "QE", 999) == "SENAT-16-QE-999"


# ===========================================================================
# _parse_ministre
# ===========================================================================

_MINISTRE_XML = """
<ministre id="42">
  <titre_jo>affaires sociales</titre_jo>
  <intitule_min>Ministère des affaires sociales et de la santé</intitule_min>
  <titre_min>M. le ministre des affaires sociales</titre_min>
  <ordre_proto>5</ordre_proto>
</ministre>
"""


def test_parse_ministre_full():
    m = _parse_ministre(el(_MINISTRE_XML))
    assert m is not None
    assert m.id == 42
    assert m.titre_jo == "affaires sociales"
    assert m.intitule_min == "Ministère des affaires sociales et de la santé"
    assert m.titre_min == "M. le ministre des affaires sociales"
    assert m.ordre_proto == 5


def test_parse_ministre_minimal():
    """Only id + titre_jo + intitule_min are required; others default to None."""
    m = _parse_ministre(
        el(
            '<ministre id="7"><titre_jo>santé</titre_jo><intitule_min>Santé</intitule_min></ministre>'
        )
    )
    assert m is not None
    assert m.id == 7
    assert m.titre_min is None
    assert m.ordre_proto is None


def test_parse_ministre_missing_id_returns_none():
    """@id is mandatory; missing or non-numeric id must return None."""
    assert (
        _parse_ministre(
            el(
                "<ministre><titre_jo>x</titre_jo><intitule_min>y</intitule_min></ministre>"
            )
        )
        is None
    )
    assert (
        _parse_ministre(
            el(
                '<ministre id="abc"><titre_jo>x</titre_jo><intitule_min>y</intitule_min></ministre>'
            )
        )
        is None
    )


def test_parse_ministre_none_input():
    assert _parse_ministre(None) is None


def test_parse_ministre_fallback_intitule():
    """intitule_min falls back to titre_jo when the element is missing."""
    m = _parse_ministre(el('<ministre id="1"><titre_jo>santé</titre_jo></ministre>'))
    assert m is not None
    assert m.intitule_min == "santé"


# ===========================================================================
# _parse_auteur
# ===========================================================================

_AUTEUR_XML = """
<auteur id_mandat="PA720986">
  <civilite>M.</civilite>
  <prenom>Jean</prenom>
  <nom>Dupont</nom>
  <grp_pol>LFI-NFP</grp_pol>
  <circonscription>Paris (1ère)</circonscription>
</auteur>
"""


def test_parse_auteur_full():
    a = _parse_auteur(el(_AUTEUR_XML))
    assert a is not None
    assert a.id_mandat == "PA720986"
    assert a.prenom == "Jean"
    assert a.nom == "Dupont"
    assert a.grp_pol == "LFI-NFP"
    assert a.circonscription == "Paris (1ère)"
    assert a.civilite == "M."


def test_parse_auteur_none():
    assert _parse_auteur(None) is None


# ===========================================================================
# _parse_question_id
# ===========================================================================


def test_parse_question_id_complete():
    xml = """
    <id_question>
      <numero_question>12345</numero_question>
      <type>QE</type>
      <source>AN</source>
      <legislature>17</legislature>
    </id_question>
    """
    result = _parse_question_id(el(xml))
    assert result == (12345, "QE", "AN", 17)


def test_parse_question_id_senat():
    xml = """
    <id_question>
      <numero_question>456</numero_question>
      <type>QE</type>
      <source>SENAT</source>
      <legislature>16</legislature>
    </id_question>
    """
    result = _parse_question_id(el(xml))
    assert result == (456, "QE", "SENAT", 16)


def test_parse_question_id_missing_field_returns_none():
    # Missing <source>
    xml = """
    <id_question>
      <numero_question>1</numero_question>
      <type>QE</type>
      <legislature>17</legislature>
    </id_question>
    """
    assert _parse_question_id(el(xml)) is None


def test_parse_question_id_non_numeric_numero_returns_none():
    xml = """
    <id_question>
      <numero_question>abc</numero_question>
      <type>QE</type>
      <source>AN</source>
      <legislature>17</legislature>
    </id_question>
    """
    assert _parse_question_id(el(xml)) is None


def test_parse_question_id_none_input():
    assert _parse_question_id(None) is None


# ===========================================================================
# _parse_indexation_an / _parse_indexation_senat
# ===========================================================================


def test_parse_indexation_an_full():
    xml = """
    <indexation_an>
      <rubrique>santé</rubrique>
      <rubrique_ta>maladies chroniques</rubrique_ta>
      <analyse>diabète</analyse>
      <analyse>insuffisance rénale</analyse>
    </indexation_an>
    """
    idx = _parse_indexation_an(el(xml))
    assert idx is not None
    assert idx.rubrique == "santé"
    assert idx.rubrique_ta == "maladies chroniques"
    assert idx.analyses == ["diabète", "insuffisance rénale"]


def test_parse_indexation_an_no_analyses():
    xml = "<indexation_an><rubrique>travail</rubrique></indexation_an>"
    idx = _parse_indexation_an(el(xml))
    assert idx is not None
    assert idx.analyses == []


def test_parse_indexation_senat():
    xml = """
    <indexation_senat>
      <theme>politique sociale</theme>
      <theme>protection sociale</theme>
      <rubrique>sécurité sociale</rubrique>
    </indexation_senat>
    """
    idx = _parse_indexation_senat(el(xml))
    assert idx is not None
    assert idx.themes == ["politique sociale", "protection sociale"]
    assert idx.rubriques == ["sécurité sociale"]


# ===========================================================================
# _parse_dossier (the main integration point of all parsers)
# ===========================================================================

_DOSSIER_AN_XML = """
<dossier>
  <question>
    <id_question>
      <numero_question>12345</numero_question>
      <type>QE</type>
      <source>AN</source>
      <legislature>17</legislature>
    </id_question>
    <date_publication_jo>2026-03-10</date_publication_jo>
    <page_jo>3456</page_jo>
    <ministre_depot id="42">
      <titre_jo>affaires sociales</titre_jo>
      <intitule_min>Ministère des affaires sociales</intitule_min>
    </ministre_depot>
    <ministre_attributaire id="43">
      <titre_jo>travail</titre_jo>
      <intitule_min>Ministère du travail</intitule_min>
    </ministre_attributaire>
    <auteur id_mandat="PA720986">
      <civilite>M.</civilite>
      <prenom>Jean</prenom>
      <nom>Dupont</nom>
      <grp_pol>LFI-NFP</grp_pol>
      <circonscription>Paris (1ère)</circonscription>
    </auteur>
    <texte>Texte de la question parlementaire.</texte>
    <etat_question>EN_COURS</etat_question>
    <indexation_an>
      <rubrique>santé</rubrique>
      <analyse>diabète</analyse>
    </indexation_an>
  </question>
</dossier>
"""

_DOSSIER_WITH_REPONSE_XML = """
<dossier>
  <question>
    <id_question>
      <numero_question>99</numero_question>
      <type>QE</type>
      <source>AN</source>
      <legislature>17</legislature>
    </id_question>
    <date_publication_jo>2026-01-07</date_publication_jo>
    <ministre_depot id="10">
      <titre_jo>santé</titre_jo>
      <intitule_min>Santé</intitule_min>
    </ministre_depot>
    <auteur id_mandat="PA001">
      <civilite>Mme</civilite>
      <prenom>Marie</prenom>
      <nom>Martin</nom>
      <grp_pol>RN</grp_pol>
      <circonscription>Nord (1ère)</circonscription>
    </auteur>
    <texte>Question sur la santé publique.</texte>
    <etat_question>REPONDU</etat_question>
    <indexation_an>
      <rubrique>santé publique</rubrique>
    </indexation_an>
  </question>
  <reponse>
    <ministre_reponse id="10">
      <titre_jo>santé</titre_jo>
      <intitule_min>Santé</intitule_min>
    </ministre_reponse>
    <texte_reponse>Le gouvernement a répondu que…</texte_reponse>
    <date_jo>2026-02-18</date_jo>
    <page_jo>987</page_jo>
  </reponse>
</dossier>
"""

_DOSSIER_SENAT_XML = """
<dossier>
  <question>
    <id_question>
      <numero_question>777</numero_question>
      <type>QE</type>
      <source>SENAT</source>
      <legislature>16</legislature>
    </id_question>
    <date_publication_jo>2026-03-05</date_publication_jo>
    <ministre_depot id="20">
      <titre_jo>logement</titre_jo>
      <intitule_min>Logement</intitule_min>
    </ministre_depot>
    <auteur id_mandat="SEN42">
      <civilite>M.</civilite>
      <prenom>Pierre</prenom>
      <nom>Leroux</nom>
      <grp_pol>SER</grp_pol>
      <circonscription>Bouches-du-Rhône</circonscription>
    </auteur>
    <texte>Question sénatoriale.</texte>
    <etat_question>EN_COURS</etat_question>
    <titre_senat>Politique du logement social</titre_senat>
    <indexation_senat>
      <theme>logement</theme>
      <rubrique>HLM</rubrique>
    </indexation_senat>
  </question>
</dossier>
"""


def test_parse_dossier_an_en_cours():
    questions = _parse_dossier(el(_DOSSIER_AN_XML))
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "AN-17-QE-12345"
    assert q.numero_question == 12345
    assert q.source == "AN"
    assert q.legislature == 17
    assert q.etat_question == "EN_COURS"
    assert q.date_publication_jo == date(2026, 3, 10)
    assert q.page_jo == 3456
    assert q.ministre_depot is not None
    assert q.ministre_depot.id == 42
    assert q.ministre_attributaire is not None
    assert q.ministre_attributaire.id == 43
    assert q.auteur is not None
    assert q.auteur.nom == "Dupont"
    assert q.auteur.id_mandat == "PA720986"
    assert q.texte_question == "Texte de la question parlementaire."
    assert q.reponse is None
    assert q.indexation_an is not None
    assert q.indexation_an.rubrique == "santé"
    assert q.indexation_an.analyses == ["diabète"]


def test_parse_dossier_with_reponse():
    questions = _parse_dossier(el(_DOSSIER_WITH_REPONSE_XML))
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "AN-17-QE-99"
    assert q.etat_question == "REPONDU"
    assert q.reponse is not None
    assert q.reponse.texte_reponse == "Le gouvernement a répondu que…"
    assert q.reponse.date_jo == date(2026, 2, 18)
    assert q.reponse.page_jo == 987
    assert q.reponse.ministre_reponse is not None
    assert q.reponse.ministre_reponse.id == 10


def test_parse_dossier_senat():
    questions = _parse_dossier(el(_DOSSIER_SENAT_XML))
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "SENAT-16-QE-777"
    assert q.source == "SENAT"
    assert q.titre_senat == "Politique du logement social"
    assert q.indexation_an is None
    assert q.indexation_senat is not None
    assert q.indexation_senat.themes == ["logement"]
    assert q.indexation_senat.rubriques == ["HLM"]


def test_parse_dossier_empty_returns_empty_list():
    """A dossier with no <question> elements is silently skipped."""
    questions = _parse_dossier(el("<dossier/>"))
    assert questions == []


def test_parse_dossier_question_with_rappel():
    """A renewed question carries a rappel_id pointing to the original."""
    xml = """
    <dossier>
      <question>
        <id_question>
          <numero_question>200</numero_question>
          <type>QE</type>
          <source>AN</source>
          <legislature>17</legislature>
        </id_question>
        <date_publication_jo>2026-03-01</date_publication_jo>
        <ministre_depot id="5">
          <titre_jo>min</titre_jo>
          <intitule_min>Ministère</intitule_min>
        </ministre_depot>
        <auteur id_mandat="PA1">
          <civilite>M.</civilite><prenom>A</prenom><nom>B</nom>
          <grp_pol>X</grp_pol><circonscription>Y</circonscription>
        </auteur>
        <texte>Renouvellement.</texte>
        <etat_question>RENOUVELLE</etat_question>
        <rappel>
          <numero_question>100</numero_question>
          <type>QE</type>
          <source>AN</source>
          <legislature>17</legislature>
        </rappel>
        <indexation_an><rubrique>test</rubrique></indexation_an>
      </question>
    </dossier>
    """
    questions = _parse_dossier(el(xml))
    assert len(questions) == 1
    assert questions[0].rappel_id == "AN-17-QE-100"


def test_parse_dossier_question_with_retrait():
    xml = """
    <dossier>
      <question>
        <id_question>
          <numero_question>300</numero_question>
          <type>QE</type>
          <source>AN</source>
          <legislature>17</legislature>
        </id_question>
        <date_publication_jo>2026-02-01</date_publication_jo>
        <ministre_depot id="6">
          <titre_jo>m</titre_jo>
          <intitule_min>M</intitule_min>
        </ministre_depot>
        <auteur id_mandat="PA2">
          <civilite>Mme</civilite><prenom>C</prenom><nom>D</nom>
          <grp_pol>Z</grp_pol><circonscription>W</circonscription>
        </auteur>
        <texte>Question retirée.</texte>
        <etat_question>RETIRE</etat_question>
        <date_retrait>2026-02-28</date_retrait>
        <indexation_an><rubrique>test</rubrique></indexation_an>
      </question>
    </dossier>
    """
    questions = _parse_dossier(el(xml))
    assert len(questions) == 1
    assert questions[0].date_retrait == date(2026, 2, 28)
    assert questions[0].etat_question == "RETIRE"


def test_parse_dossier_skips_question_with_bad_id():
    """Questions whose id_question is incomplete are silently dropped."""
    xml = """
    <dossier>
      <question>
        <id_question>
          <numero_question>999</numero_question>
          <!-- missing type, source, legislature -->
        </id_question>
        <texte>Bad.</texte>
        <etat_question>EN_COURS</etat_question>
      </question>
    </dossier>
    """
    assert _parse_dossier(el(xml)) == []


# ===========================================================================
# WSError
# ===========================================================================


def test_ws_error_carries_message():
    err = WSError("something went wrong")
    assert err.message == "something went wrong"
    assert "something went wrong" in str(err)


# ===========================================================================
# ReponseWSClient._check_statut (static method — no HTTP involved)
# ===========================================================================


def test_check_statut_ok_does_not_raise():
    root = el(
        "<rechercherDossierResponse><statut>OK</statut></rechercherDossierResponse>"
    )
    # Should not raise
    ReponseWSClient._check_statut(root, "rechercherDossier")


def test_check_statut_ko_raises_ws_error():
    root = el(
        "<rechercherDossierResponse>"
        "<statut>KO</statut>"
        "<message_erreur>Credentials invalides</message_erreur>"
        "</rechercherDossierResponse>"
    )
    with pytest.raises(WSError, match="Credentials invalides"):
        ReponseWSClient._check_statut(root, "rechercherDossier")


def test_check_statut_ko_without_message():
    root = el("<response><statut>KO</statut></response>")
    with pytest.raises(WSError, match="no message"):
        ReponseWSClient._check_statut(root, "someMethod")


def test_check_statut_missing_statut_element_raises():
    """No <statut> element → treated as KO (empty string != 'OK')."""
    root = el("<response/>")
    with pytest.raises(WSError):
        ReponseWSClient._check_statut(root, "anyMethod")


# ===========================================================================
# Request body construction — verify XML shape without hitting HTTP
# ===========================================================================


def test_rechercher_dossier_request_body_contains_filters():
    """Check that the XML body built by rechercher_dossier is well-formed
    and contains the expected elements.  We subclass to intercept _post."""

    captured: dict = {}

    class _CaptureClient(ReponseWSClient):
        def _post(self, service, method, body):  # type: ignore[override]
            captured["service"] = service
            captured["method"] = method
            captured["body"] = body
            # Return a minimal OK response so the method can complete
            return stdlib_fromstring(  # noqa: S314
                "<rechercherDossierResponse><statut>OK</statut></rechercherDossierResponse>"
            )

    client = _CaptureClient(base_url="http://unused", username="u", password="p")  # noqa: S106
    client.rechercher_dossier(
        date_debut=date(2026, 3, 1),
        date_fin=date(2026, 3, 18),
        sources=["AN"],
        types=["QE"],
        legislature=17,
    )

    body = captured["body"]
    assert captured["service"] == "WSquestion"
    assert captured["method"] == "rechercherDossier"
    assert "<date_debut>2026-03-01</date_debut>" in body
    assert "<date_fin>2026-03-18</date_fin>" in body
    assert "<source>AN</source>" in body
    assert "<type>QE</type>" in body
    assert "<legislature>17</legislature>" in body
    # Well-formed XML
    stdlib_fromstring(body)  # raises if not parseable  # noqa: S314


def test_chercher_membres_gouvernement_request_body():
    """en_fonction parameter is serialised correctly."""

    captured: dict = {}

    class _CaptureClient(ReponseWSClient):
        def _post(self, service, method, body):  # type: ignore[override]
            captured["service"] = service
            captured["body"] = body
            return stdlib_fromstring(  # noqa: S314
                "<chercherMembresGouvernementResponse><statut>OK</statut></chercherMembresGouvernementResponse>"
            )

    client = _CaptureClient(base_url="http://unused", username="u", password="p")  # noqa: S106
    result = client.chercher_membres_gouvernement(en_fonction="TRUE")

    assert captured["service"] == "WSattribution"
    assert "<en_fonction>TRUE</en_fonction>" in captured["body"]
    assert result == []  # no membre_gouvernement elements in our stub response
