"""Tests unitaires pour qe.analysis.question_parser.

Le module est pur : on injecte du texte, on vérifie la structure du résultat.
"""

from qe.analysis.question_parser import parse


def test_pattern_attire_attention_captures_contexte_and_question() -> None:
    text = (
        "M. Éric Woerth attire l'attention de M. le ministre d'État, ministre de "
        "l'intérieur, sur les situations d'usurpation d'identité lors des mariages. "
        "Du fait de la progressive numérisation des documents administratifs et "
        "de la digitalisation accélérée des entreprises, il apparaît nécessaire "
        "d'agir. Il lui demande donc de mettre en place des mécanismes permettant "
        "aux agents municipaux d'empêcher les usurpations."
    )
    p = parse(text)
    assert p.est_rappel is False
    assert p.contexte_extrait is not None
    assert "usurpation d'identité" in p.contexte_extrait
    assert p.question_extraite is not None
    assert p.question_extraite.startswith("Il lui demande")
    assert p.opener_label == "attire/appelle l'attention"
    assert p.closer_label == "il/elle demande"


def test_pattern_rappel_is_detected() -> None:
    text = (
        "Mme Nicole Bonnefoy rappelle à Mme la ministre de l'agriculture les "
        "termes de sa question n° 07181 sous le titre « Baisse des moyens "
        "alloués à l'enseignement agricole », qui n'a pas obtenu de réponse à "
        "ce jour."
    )
    p = parse(text)
    assert p.est_rappel is True


def test_pattern_interroge_and_souhaite_savoir() -> None:
    text = (
        "Mme Sandra Delannoy interroge Mme la ministre de la santé sur les "
        "difficultés d'accès aux soins. Le contexte local montre des tensions. "
        "Elle souhaite savoir quelles évolutions le Gouvernement envisage."
    )
    p = parse(text)
    assert p.contexte_extrait is not None
    assert "accès aux soins" in p.contexte_extrait
    assert p.question_extraite is not None
    assert p.question_extraite.startswith("Elle souhaite savoir")
    assert p.opener_label == "interroge"
    assert p.closer_label == "il/elle souhaite"


def test_inverted_form_souhaite_t_elle_is_detected() -> None:
    text = (
        "M. Machin attire l'attention de la ministre sur un sujet. "
        "Aussi souhaite-t-elle connaître l'avis du Gouvernement."
    )
    p = parse(text)
    assert p.question_extraite is not None
    assert (
        "souhaite-t-elle" in p.question_extraite
        or "Aussi" in p.question_extraite
    )


def test_empty_text_returns_nulls() -> None:
    p = parse("")
    assert p.est_rappel is False
    assert p.contexte_extrait is None
    assert p.question_extraite is None


def test_expose_a_as_opener() -> None:
    text = (
        "M. Roland Courteau expose à M. le ministre d'État que le bilan "
        "environnemental de la filière éolienne doit être positif. Le contexte "
        "sur le terrain montre des difficultés. Il lui demande donc quelles "
        "mesures seront prises."
    )
    p = parse(text)
    assert p.contexte_extrait is not None
    assert p.question_extraite is not None
