"""Tests for the AN question `objet` extraction cascade.

The AN reference schema uses different tag forms for the objet/analyse
field across legislatures. These tests pin the parser behaviour for
each known shape and the priority order documented in
`qe/ingestion_an.py`.
"""

from __future__ import annotations

import pytest

from qe.ingestion_an import parse_an_archive_question_xml

NS = "http://schemas.assemblee-nationale.fr/referentiel"


def _question_xml(indexation_inner: str) -> bytes:
    """Wrap an indexationAN fragment in a minimally-valid <question>."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<question xmlns="{NS}">'
        f"  <uid>QANR5L17QE1</uid>"
        f"  <identifiant>"
        f"    <numero>1</numero>"
        f"    <legislature>17</legislature>"
        f"  </identifiant>"
        f"  <type>QE</type>"
        f"  <indexationAN>{indexation_inner}</indexationAN>"
        f"</question>"
    ).encode("utf-8")


def test_modern_analyses_analyse_extracts_objet():
    """XVI/XVII format: <analyses><analyse>...</analyse></analyses>."""
    xml = _question_xml(
        "<teteAnalyse/>"
        "<analyses><analyse>PAC 2026 : mise à disposition des terres</analyse></analyses>"
    )
    pq = parse_an_archive_question_xml(xml)
    assert pq is not None
    assert pq.objet == "PAC 2026 : mise à disposition des terres"


def test_legacy_analyse_ana_extracts_objet():
    """XIV/XV legacy format: <ANALYSE><ANA>...</ANA></ANALYSE>."""
    xml = _question_xml(
        "<teteAnalyse/>"
        "<ANALYSE><ANA>Sujet legacy</ANA></ANALYSE>"
    )
    pq = parse_an_archive_question_xml(xml)
    assert pq is not None
    assert pq.objet == "Sujet legacy"


def test_tete_analyse_takes_precedence():
    """When <teteAnalyse> is non-empty, it wins over any analyse element."""
    xml = _question_xml(
        "<teteAnalyse>Title from teteAnalyse</teteAnalyse>"
        "<analyses><analyse>Should not be used</analyse></analyses>"
        "<ANALYSE><ANA>Should not be used either</ANA></ANALYSE>"
    )
    pq = parse_an_archive_question_xml(xml)
    assert pq is not None
    assert pq.objet == "Title from teteAnalyse"


def test_empty_indexation_returns_none_objet():
    """No usable tag → objet is None, but the question still parses."""
    xml = _question_xml("<teteAnalyse/>")
    pq = parse_an_archive_question_xml(xml)
    assert pq is not None
    assert pq.objet is None


@pytest.mark.parametrize(
    "inner,expected",
    [
        # whitespace-only teteAnalyse should NOT count as a value
        ("<teteAnalyse>   </teteAnalyse><analyses><analyse>fallback</analyse></analyses>", "fallback"),
        # empty <analyse> element falls through to the legacy form
        ("<teteAnalyse/><analyses><analyse/></analyses><ANALYSE><ANA>legacy</ANA></ANALYSE>", "legacy"),
    ],
)
def test_fallback_chain_robustness(inner: str, expected: str):
    pq = parse_an_archive_question_xml(_question_xml(inner))
    assert pq is not None
    assert pq.objet == expected
