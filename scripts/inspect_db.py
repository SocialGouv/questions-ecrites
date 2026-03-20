#!/usr/bin/env python3
"""Print a quick summary of every QE ingestion table.

Useful for checking that ingestion ran correctly.

Usage:
    poetry run python scripts/inspect_db.py
    poetry run python scripts/inspect_db.py --rows 10
"""

from __future__ import annotations

import argparse

from sqlalchemy import func, select

from qe.db import get_session
from qe.models import (
    IngestCursor,
    Ministere,
    Question,
    QuestionAttribution,
    QuestionStateChange,
)

_SEP = "─" * 72


def _section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _counts(session) -> None:
    _section("Row counts")
    tables = [
        ("ministeres", Ministere),
        ("questions", Question),
        ("question_state_changes", QuestionStateChange),
        ("question_attributions", QuestionAttribution),
        ("ingest_cursors", IngestCursor),
    ]
    for name, model in tables:
        count = session.execute(select(func.count()).select_from(model)).scalar()
        print(f"  {name:<30} {count:>8} rows")


def _ministeres(session, n: int) -> None:
    _section(f"ministeres — first {n} rows")
    rows = (
        session.execute(select(Ministere).order_by(Ministere.id).limit(n))
        .scalars()
        .all()
    )
    if not rows:
        print("  (empty)")
        return
    for m in rows:
        print(f"  [{m.id:>4}] {m.titre_jo}")


def _questions(session, n: int) -> None:
    _section(f"questions — first {n} rows (ordered by date_publication_jo desc)")
    rows = (
        session.execute(
            select(Question)
            .order_by(Question.date_publication_jo.desc().nulls_last())
            .limit(n)
        )
        .scalars()
        .all()
    )
    if not rows:
        print("  (empty)")
        return
    for q in rows:
        reponse_flag = "REP" if q.texte_reponse else "   "
        date_str = str(q.date_publication_jo) if q.date_publication_jo else "????-??-??"
        auteur = (q.auteur_nom or "")[:25]
        ministere = (q.ministre_attributaire_libelle or "")[:30]
        print(f"  {reponse_flag} {q.id:<22} {date_str}  {auteur:<26} {ministere}")


def _questions_by_state(session) -> None:
    _section("questions — breakdown by etat_question")
    rows = session.execute(
        select(Question.etat_question, func.count())
        .group_by(Question.etat_question)
        .order_by(func.count().desc())
    ).all()
    if not rows:
        print("  (empty)")
        return
    for etat, count in rows:
        print(f"  {etat:<20} {count:>8}")


def _questions_by_source(session) -> None:
    _section("questions — breakdown by source / legislature")
    rows = session.execute(
        select(Question.source, Question.legislature, func.count())
        .group_by(Question.source, Question.legislature)
        .order_by(Question.source, Question.legislature)
    ).all()
    if not rows:
        print("  (empty)")
        return
    for source, legislature, count in rows:
        print(f"  {source:<8} legislature {legislature:<4} {count:>8} questions")


def _questions_by_ministry(session) -> None:
    _section("questions — top 15 attributee ministries")
    rows = session.execute(
        select(Question.ministre_attributaire_libelle, func.count())
        .group_by(Question.ministre_attributaire_libelle)
        .order_by(func.count().desc())
        .limit(15)
    ).all()
    if not rows:
        print("  (empty)")
        return
    for libelle, count in rows:
        label = (libelle or "(none)")[:55]
        print(f"  {count:>6}  {label}")


def _cursors(session) -> None:
    _section("ingest_cursors")
    rows = (
        session.execute(select(IngestCursor).order_by(IngestCursor.cursor_name))
        .scalars()
        .all()
    )
    if not rows:
        print("  (empty)")
        return
    for c in rows:
        print(f"  {c.cursor_name:<30} last_date={c.last_date}  jeton={c.jeton!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect QE ingestion DB tables.")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        metavar="N",
        help="Number of sample rows to show per table (default: 5)",
    )
    args = parser.parse_args()

    with get_session() as session:
        _counts(session)
        _ministeres(session, args.rows)
        _questions(session, args.rows)
        _questions_by_state(session)
        _questions_by_source(session)
        _questions_by_ministry(session)
        _cursors(session)

    print(f"\n{_SEP}\n")


if __name__ == "__main__":
    main()
