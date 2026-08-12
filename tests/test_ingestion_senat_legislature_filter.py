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


def _build_dump(rows: list[str]) -> io.BytesIO:
    lines = [
        f"COPY questions.tam_questions ({_COLUMNS}) FROM stdin;",
        *rows,
        r"\.",
    ]
    return io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))


def test_drops_questions_before_legislature_14() -> None:
    dump = _build_dump([_row(13, "00100", "1", "Trop ancienne")])

    questions = parse_senat_sql_dump(dump)

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

    questions = parse_senat_sql_dump(dump)

    assert {q.id for q in questions} == {"SENAT-17-QE-200", "SENAT-25-QE-300"}
