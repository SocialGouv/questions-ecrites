"""Lance l'analyseur `qe.analysis.question_parser` sur des questions et
écrit le résultat dans les nouvelles colonnes de la table `questions`.

Trois modes :

    # Analyse une (ou plusieurs) questions par id (dry-run par défaut) :
    python scripts/analyze_questions.py --id AN-17-QE-9848 SENAT-17-QE-8064

    # Analyse toutes celles qui n'ont pas encore été analysées :
    python scripts/analyze_questions.py --backfill --commit

    # Ré-analyse tout le corpus (utile quand on modifie les patterns) :
    python scripts/analyze_questions.py --all --commit

Sans `--commit`, le script montre ce qu'il ferait sans toucher à la base.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

import psycopg2
import psycopg2.extras

from qe.analysis.question_parser import ParsedQuestion, parse


# Read from env; the fallback is the local dev DSN documented in the README.
# Any deployment sets DATABASE_URL to point at its own Postgres.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://qe:qe@localhost:5433/qe")


UPDATE_SQL = """
UPDATE questions
SET contexte_extrait     = %(contexte)s,
    question_extraite = %(question)s,
    est_rappel        = %(est_rappel)s,
    analyzed_at       = NOW()
WHERE id = %(id)s
"""


def _fetch(cur, *, ids: list[str], backfill: bool, all_rows: bool, limit: int | None) -> Iterator[tuple[str, str]]:
    """Yield (id, texte_question) rows, en fonction du mode."""
    if ids:
        cur.execute(
            "SELECT id, texte_question FROM questions "
            "WHERE id = ANY(%s) AND texte_question IS NOT NULL",
            (ids,),
        )
    elif backfill:
        q = (
            "SELECT id, texte_question FROM questions "
            "WHERE texte_question IS NOT NULL AND analyzed_at IS NULL "
            "ORDER BY date_publication_jo DESC NULLS LAST"
        )
        if limit:
            cur.execute(q + " LIMIT %s", (limit,))
        else:
            cur.execute(q)
    elif all_rows:
        q = "SELECT id, texte_question FROM questions WHERE texte_question IS NOT NULL"
        if limit:
            cur.execute(q + " LIMIT %s", (limit,))
        else:
            cur.execute(q)
    else:
        raise SystemExit("Choisir --id, --backfill ou --all")

    while True:
        rows = cur.fetchmany(1000)
        if not rows:
            return
        for r in rows:
            yield r[0], r[1]


def _print_dry_run(qid: str, texte: str, parsed: ParsedQuestion) -> None:
    print("=" * 78)
    print(f"{qid}   ({len(texte)} chars)")
    print("=" * 78)
    if parsed.est_rappel:
        print("*** RAPPEL ***")
    print(f"opener={parsed.opener_label or 'MISS'}  |  closer={parsed.closer_label or 'MISS'}")
    print(f"CONTEXTE ({len(parsed.contexte_extrait or '')} c) : {(parsed.contexte_extrait or '(none)')[:200]}")
    print(f"QUESTION ({len(parsed.question_extraite or '')} c) : {(parsed.question_extraite or '(none)')[:300]}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--id", nargs="+", help="IDs à analyser (test rapide)")
    grp.add_argument("--backfill", action="store_true",
                     help="analyse toutes les lignes où analyzed_at IS NULL")
    grp.add_argument("--all", action="store_true", help="ré-analyse tout le corpus")
    ap.add_argument("--limit", type=int, help="limite pour --backfill / --all")
    ap.add_argument("--commit", action="store_true",
                    help="écrit en base (sinon dry-run affichant les résultats)")
    args = ap.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    # withhold=True keeps the named cursor valid across commits — otherwise
    # committing the write batch invalidates the still-open read cursor.
    read_cur = conn.cursor(name="analyze_read", withhold=True)
    write_cur = conn.cursor()

    stats = {"total": 0, "rappels": 0, "topic": 0, "question": 0, "both": 0}
    batch: list[dict] = []
    BATCH_SIZE = 500

    try:
        for qid, texte in _fetch(
            read_cur,
            ids=args.id or [],
            backfill=args.backfill,
            all_rows=args.all,
            limit=args.limit,
        ):
            parsed = parse(texte)
            stats["total"] += 1
            if parsed.est_rappel:
                stats["rappels"] += 1
            if parsed.contexte_extrait:
                stats["topic"] += 1
            if parsed.question_extraite:
                stats["question"] += 1
            if parsed.contexte_extrait and parsed.question_extraite:
                stats["both"] += 1

            if args.commit:
                batch.append({
                    "id": qid,
                    "contexte": parsed.contexte_extrait,
                    "question": parsed.question_extraite,
                    "est_rappel": parsed.est_rappel,
                })
                if len(batch) >= BATCH_SIZE:
                    psycopg2.extras.execute_batch(write_cur, UPDATE_SQL, batch)
                    conn.commit()
                    batch.clear()
                    print(f"  ... {stats['total']} analysées", file=sys.stderr)
            else:
                _print_dry_run(qid, texte, parsed)

        if args.commit and batch:
            psycopg2.extras.execute_batch(write_cur, UPDATE_SQL, batch)
            conn.commit()
    finally:
        read_cur.close()
        write_cur.close()
        conn.close()

    n = stats["total"]
    if n:
        print("\n=== Récap ===")
        print(f"  analysées         : {n}")
        print(f"  rappels détectés  : {stats['rappels']:>6}  ({100*stats['rappels']/n:.1f} %)")
        print(f"  contexte capté       : {stats['topic']:>6}  ({100*stats['topic']/n:.1f} %)")
        print(f"  question captée   : {stats['question']:>6}  ({100*stats['question']/n:.1f} %)")
        print(f"  les deux          : {stats['both']:>6}  ({100*stats['both']/n:.1f} %)")
        if not args.commit:
            print("\nDRY RUN — rien écrit en base. Relance avec --commit pour persister.")


if __name__ == "__main__":
    main()
