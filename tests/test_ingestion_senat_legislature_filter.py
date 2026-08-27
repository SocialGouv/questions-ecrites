import io

from qe.ingestion_senat import parse_senat_sql_dump

_COLUMNS = (
    "natquecod, legislature, numero, sorquecod, titre, nom, prenom, "
    "codequalite, circonscription, groupe, datejodepot, mindepotlib, "
    "minreplib1, datejorep1, txtque, themes, id"
)


def _row(legislature: int, numero: str, internal_id: str, txtque: str) -> str:
    return "\t".join(
        [
            "QE",
            str(legislature),
            numero,
            "1",
            "Titre",
            "Dupont",
            "Jean",
            "S",
            "Circ1",
            "Groupe1",
            "2023-01-01 00:00:00",
            "Min1",
            r"\N",
            r"\N",
            txtque,
            r"\N",
            internal_id,
        ]
    )


def _build_dump(rows: list[str]) -> bytes:
    lines = [
        f"COPY questions.tam_questions ({_COLUMNS}) FROM stdin;",
        *rows,
        r"\.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_drops_questions_before_legislature_14() -> None:
    dump = _build_dump([_row(13, "00100", "1", "Trop ancienne")])

    questions = parse_senat_sql_dump(lambda: io.BytesIO(dump))

    assert questions == []


def test_keeps_legislature_beyond_previously_hardcoded_range() -> None:
    # Legislature 25 doesn't exist yet, but the parser must not silently drop
    # it once it does — there's no upper bound on what's kept.
    dump = _build_dump(
        [
            _row(17, "00200", "2", "Legislature courante"),
            _row(25, "00300", "3", "Legislature future"),
        ]
    )

    questions = parse_senat_sql_dump(lambda: io.BytesIO(dump))

    assert {q.id for q in questions} == {"SENAT-17-QE-200", "SENAT-25-QE-300"}


def test_joins_response_text_across_the_two_parsing_passes() -> None:
    # tam_reponses is parsed in a second pass, filtered to the ids kept from
    # tam_questions in the first pass — this exercises that the join between
    # the two passes still lines up correctly.
    lines = [
        f"COPY questions.tam_questions ({_COLUMNS}) FROM stdin;",
        _row(17, "00200", "42", "Question texte"),
        r"\.",
        "COPY questions.tam_reponses (idque, txtrep, datejorep, minreplib) FROM stdin;",
        "42\tRéponse texte\t2023-02-01 00:00:00\tMinistere X",
        r"\.",
    ]
    dump = ("\n".join(lines) + "\n").encode("utf-8")

    questions = parse_senat_sql_dump(lambda: io.BytesIO(dump))

    assert len(questions) == 1
    assert questions[0].texte_reponse == "Réponse texte"
    assert questions[0].ministre_reponse_libelle == "Ministere X"


def test_response_for_a_filtered_out_question_is_not_joined_elsewhere() -> None:
    # A response row whose idque belongs to a question dropped in pass 1 must
    # not leak onto some other, unrelated kept question.
    lines = [
        f"COPY questions.tam_questions ({_COLUMNS}) FROM stdin;",
        _row(13, "00100", "1", "Trop ancienne"),  # dropped: legislature < 14
        _row(17, "00200", "2", "Question courante"),  # kept, no response
        r"\.",
        "COPY questions.tam_reponses (idque, txtrep, datejorep, minreplib) FROM stdin;",
        "1\tRéponse orpheline\t2023-02-01 00:00:00\tMinistere X",
        r"\.",
    ]
    dump = ("\n".join(lines) + "\n").encode("utf-8")

    questions = parse_senat_sql_dump(lambda: io.BytesIO(dump))

    assert len(questions) == 1
    assert questions[0].texte_reponse is None
