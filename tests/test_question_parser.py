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


def test_pattern_alerte_as_opener() -> None:
    text = (
        "Mme Sophie Errante alerte M. le ministre de l'intérieur sur la "
        "situation préoccupante des services d'urgence. Depuis plusieurs mois, "
        "les délais s'allongent. Elle souhaite connaître les mesures prévues."
    )
    p = parse(text)
    assert p.opener_label == "alerte"
    assert p.contexte_extrait is not None
    assert "services d'urgence" in p.contexte_extrait


def test_pattern_questionne_as_opener() -> None:
    text = (
        "M. Jean Dupont questionne Mme la ministre au sujet de la réforme "
        "des retraites. Il lui demande quels arbitrages seront rendus."
    )
    p = parse(text)
    assert p.opener_label == "questionne"
    assert p.contexte_extrait is not None
    assert "réforme des retraites" in p.contexte_extrait


def test_pattern_aux_fins_de_connaitre_as_opener() -> None:
    text = (
        "Mme Anne Martin interpelle le ministre aux fins de connaître les "
        "évolutions envisagées sur la fiscalité locale. Il lui demande "
        "des précisions."
    )
    p = parse(text)
    assert p.opener_label == "aux fins de connaître"
    assert p.contexte_extrait is not None


def test_pattern_souhaite_attirer_attention_as_opener() -> None:
    text = (
        "M. Paul Bernard souhaite attirer l'attention de Mme la ministre "
        "sur la pénurie de médicaments. Aussi souhaiterait-il connaître "
        "les mesures envisagées."
    )
    p = parse(text)
    assert p.opener_label == "souhaite attirer/appeler l'attention"
    assert p.contexte_extrait is not None
    assert "pénurie de médicaments" in p.contexte_extrait


def test_pattern_que_compte_as_closer() -> None:
    text = (
        "M. Machin attire l'attention de la ministre sur un sujet difficile. "
        "Que compte-t-elle mettre en place pour y remédier ?"
    )
    p = parse(text)
    assert p.closer_label == "que compte/comptent"
    assert p.question_extraite is not None
    assert "compte-t-elle" in p.question_extraite


def test_pattern_remercie_as_closer() -> None:
    # `il/elle remercie` closes questions like "Elle le remercie de bien
    # vouloir lui indiquer…". Sanity-check that it's picked up as a closer.
    text = (
        "Mme Chose interroge le ministre sur un thème. Le sujet est complexe. "
        "Elle le remercie de bien vouloir lui indiquer les mesures prises."
    )
    p = parse(text)
    assert p.closer_label == "il/elle remercie"
    assert p.question_extraite is not None
    assert p.question_extraite.startswith("Elle le remercie")


def test_whitespace_only_returns_nulls() -> None:
    p = parse("   \n\t  ")
    assert p.est_rappel is False
    assert p.contexte_extrait is None
    assert p.question_extraite is None
    assert p.opener_label is None
    assert p.closer_label is None


def test_short_text_below_tail_window() -> None:
    # tail_offset must clamp to 0 when the text is shorter than
    # CLOSER_TAIL_CHARS — otherwise the question start index goes wrong.
    text = "M. X interroge Mme Y sur le climat. Il lui demande son avis."
    p = parse(text)
    assert p.contexte_extrait is not None
    assert p.question_extraite is not None
    assert p.question_extraite.startswith("Il lui demande")


def test_contexte_spans_full_body_between_opener_and_question() -> None:
    # contexte_extrait doit couvrir TOUT ce qui est entre l'ouverture et
    # la question — pas seulement la première phrase après "sur".
    text = (
        "Mme Chose interroge Mme la ministre sur la situation X. "
        "Contexte phrase deux. Contexte phrase trois. Contexte phrase quatre. "
        "Elle lui demande donc quelles mesures seront prises."
    )
    p = parse(text)
    assert p.contexte_extrait is not None
    # première phrase capturée
    assert "situation X" in p.contexte_extrait
    # ET le corps intermédiaire
    assert "phrase deux" in p.contexte_extrait
    assert "phrase quatre" in p.contexte_extrait
    # la question elle-même ne doit PAS être dans le contexte
    assert "Elle lui demande" not in p.contexte_extrait
    assert p.question_extraite is not None
    assert p.question_extraite.startswith("Elle lui demande")


def test_contexte_without_closer_reaches_end_of_text() -> None:
    # Si aucune question de clôture n'est détectée, le contexte va
    # jusqu'à la fin du texte (mieux que rien).
    text = (
        "Mme Chose interroge Mme la ministre sur un sujet. "
        "Corps du texte. Encore un peu de corps."
    )
    p = parse(text)
    assert p.contexte_extrait is not None
    assert "un sujet" in p.contexte_extrait
    assert "Corps du texte" in p.contexte_extrait
    assert p.question_extraite is None


def test_rappel_with_preamble_is_still_detected() -> None:
    # A rappel that starts with a short preamble (date, ref) must still be
    # caught — regression guard for the widened RE_RAPPEL search scope.
    preamble = "Réf. QE-2024-08765, séance du 12 mars 2024. " * 5
    text = (
        preamble
        + "Mme Dupont rappelle à M. le ministre les termes de sa question "
        "n° 12345 sur les crédits alloués, qui n'a pas reçu de réponse."
    )
    p = parse(text)
    assert p.est_rappel is True
