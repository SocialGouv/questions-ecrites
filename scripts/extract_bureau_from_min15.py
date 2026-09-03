#!/usr/bin/env python3
"""Extract real bureau (direction + sous-direction + bureau) from MIN15
workflow data into `question_bureau_extract`.

Rationale
---------
`question_real_attributions.bureau_reel_id` is populated only for DGCS
(~94 % coverage) and partially for DSS (~49 %). All other directions
(DGOS, DGS, DGEFP, DGT) have 0 bureau coverage — the vector-based
attribution algorithm has no human examples to learn from at bureau
granularity.

The MIN15 workflow exports contain a richer signal: each `poste_etape`
of a `Pour rédaction` step names the exact bureau that handled the
drafting. Parsing this gives us bureau-level attributions for the
directions that don't appear in the DGCS Excel.

Extraction rule
---------------
For each QE, collect the `Pour rédaction` (or `Pour rédaction interfacée`)
steps whose `poste_etape` is a bureau-level drill-down :

    poste_etape has >= 3 segments (split on ' - ')
    AND first segment ∈ KNOWN_DIRECTIONS

Then group by (question_id, direction), keeping the LATEST step per
group — the deepest drill-down before response is the one that mattered.

`BDC XXX`, `CABINET`, `SGG`, `DDC` are ignored (admin / political
transit, not drafting).

Usage
-----
    poetry run python scripts/extract_bureau_from_min15.py
        [--dry-run]    print sample rows, no DB write
        [--reset]      truncate the target table before insert (idempotent)
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter

from sqlalchemy import text as sqltext

from qe import db
from qe.attributions import refresh_attributions_all_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Known drafting directions in the SAS ministerial perimeter + a few
# extras seen in the MIN15 dumps (DGE, DGCCRF, DGALN, DGPR, DGAMPA
# appear for cross-ministry files).
KNOWN_DIRECTIONS = {
    "DGCS", "DGOS", "DSS", "DGS", "DGEFP", "DGT",
    "DFAS", "DGE", "DGCCRF", "DGALN", "DGPR", "DGAMPA",
    "DAJ", "DRH", "DNS", "DREES", "DIPLP", "DARES",
}

# Steps that plausibly reveal the bureau in charge.
#
# - "Pour attribution" is critical for DSS: their workflow tracks bureau
#   at attribution time but sends drafting to a central pool. Without
#   this type, DSS coverage is 31; with it, 1 276.
# - "Pour rédaction" / "Pour rédaction interfacée" are the natural
#   signal for DGOS, DGS, DGCS.
# - "Pour visa" is not included: it's a validation step, often reused
#   by cross-bureau reviewers, so it's noisier than the two above.
BUREAU_SIGNAL_TYPES = (
    "Pour rédaction",
    "Pour rédaction interfacée",
    "Pour attribution",
)

# Some poste_etape have double spaces or trailing spaces around ' - '.
_SPLIT_RE = re.compile(r"\s+-\s+")


def _split_poste(poste: str) -> list[str]:
    """Split on ' - ' tolerating extra whitespace. Trim segments."""
    return [s.strip() for s in _SPLIT_RE.split(poste.strip()) if s.strip()]


def _extract_direction(seg0: str) -> str | None:
    """Normalise the first segment to a known direction acronym.

    Handles:
    - Exact match: 'DGOS' → 'DGOS'
    - Trailing space: 'DGOS ' → 'DGOS'
    - Composite: 'CAB SFAH' → None (cabinet, not a drafting direction)
    - Pool: 'DGOS QE ministères sociaux' → None (not a bureau-level poste)
    """
    tok = seg0.strip().upper().split()[0] if seg0.strip() else ""
    return tok if tok in KNOWN_DIRECTIONS else None


def _parse_poste(poste: str) -> tuple[str, str | None, str | None, str | None] | None:
    """Return (direction, sous_dir, bureau, bureau_full) or None if the
    poste isn't a bureau-level drill-down."""
    segs = _split_poste(poste)
    if len(segs) < 3:
        return None
    direction = _extract_direction(segs[0])
    if direction is None:
        return None
    # segs[0] may have trailing text (e.g. 'DGOS QE ministères sociaux')
    # but if it survived the direction check it's a real direction line.
    # We only accept the exact acronym match — otherwise reject.
    if segs[0].strip().upper() != direction:
        return None
    sous_dir = segs[1] if len(segs) >= 2 else None
    bureau = segs[2] if len(segs) >= 3 else None
    bureau_full = " - ".join(segs[2:]) if len(segs) >= 3 else None
    return direction, sous_dir, bureau, bureau_full


# Legislature is derived from date_jo_question at query time — MIN15 has
# no `legislature` column, and hard-coding '17' would break silently on
# any future ingest of MIN14/16 archives. The start-date thresholds
# match `_LEGISLATURE_START_DATES` in qe/ingestion_an.py.
SELECT_SQL = sqltext("""
    SELECT
        e.id                                                    AS etape_id,
        e.parlement || '-' || (
            CASE
                WHEN e.date_jo_question >= DATE '2024-07-18' THEN 17
                WHEN e.date_jo_question >= DATE '2022-06-22' THEN 16
                WHEN e.date_jo_question >= DATE '2017-06-21' THEN 15
                WHEN e.date_jo_question >= DATE '2012-06-20' THEN 14
                ELSE 13
            END
        ) || '-QE-' || e.numero_question                        AS question_id,
        e.poste_etape                                           AS poste,
        e.type_etape                                            AS type_etape,
        e.date_debut_etape                                      AS date_debut
    FROM reponses_extract_etapes e
    WHERE e.type_etape = ANY(:signal_types)
      AND e.poste_etape IS NOT NULL
      AND e.date_debut_etape IS NOT NULL
      AND e.date_jo_question IS NOT NULL
    ORDER BY e.date_debut_etape ASC
""")


UPSERT_SQL = sqltext("""
    INSERT INTO question_bureau_extract
        (question_id, direction_txt, sous_direction, bureau, bureau_full,
         source_etape_id, date_debut_etape)
    VALUES (:qid, :dir, :sd, :bur, :burf, :eid, :date_debut)
    ON CONFLICT (question_id, direction_txt) DO UPDATE
        SET sous_direction  = EXCLUDED.sous_direction,
            bureau          = EXCLUDED.bureau,
            bureau_full     = EXCLUDED.bureau_full,
            source_etape_id = EXCLUDED.source_etape_id,
            date_debut_etape = EXCLUDED.date_debut_etape,
            extracted_at    = NOW()
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print 20 sample rows, no DB write")
    ap.add_argument("--reset", action="store_true",
                    help="TRUNCATE question_bureau_extract before insert")
    args = ap.parse_args()

    logger.info("Scanning reponses_extract_etapes …")
    # Stream rows via server-side cursor so we don't materialise the whole
    # result set — safe on tables that may grow to millions of steps.
    latest: dict[tuple[str, str], dict] = {}
    rejected_no_bureau = 0
    rejected_unknown_direction = 0
    n_scanned = 0
    with db.get_session() as session:
        stream = session.execute(
            SELECT_SQL, {"signal_types": list(BUREAU_SIGNAL_TYPES)}
        ).yield_per(1000)
        for r in stream:
            n_scanned += 1
            parsed = _parse_poste(r.poste)
            if parsed is None:
                segs = _split_poste(r.poste)
                if len(segs) < 3:
                    rejected_no_bureau += 1
                else:
                    rejected_unknown_direction += 1
                continue
            direction, sd, bur, burf = parsed
            latest[(r.question_id, direction)] = {
                "qid": r.question_id,
                "dir": direction,
                "sd": sd,
                "bur": bur,
                "burf": burf,
                "eid": r.etape_id,
                "date_debut": r.date_debut,
            }

    logger.info(
        "Scanned %d step rows — kept %d (qid, direction) pairs — rejected %d no-bureau, %d unknown-direction",
        n_scanned, len(latest), rejected_no_bureau, rejected_unknown_direction,
    )

    by_dir = Counter(v["dir"] for v in latest.values())
    logger.info("Coverage per direction: %s", dict(by_dir.most_common()))

    if args.dry_run:
        logger.info("Dry-run — 20 sample rows:")
        for i, v in enumerate(list(latest.values())[:20]):
            print(f"  {i+1:>2}. {v['qid']}  |  {v['dir']} / {v['sd']} / {v['bur']}")
        return

    with db.get_session() as session:
        if args.reset:
            logger.info("TRUNCATE question_bureau_extract …")
            session.execute(sqltext("TRUNCATE question_bureau_extract RESTART IDENTITY"))
        n = 0
        for v in latest.values():
            session.execute(UPSERT_SQL, v)
            n += 1
        session.commit()
    logger.info("Upserted %d rows into question_bureau_extract", n)

    logger.info("Refreshing question_attributions_all …")
    refresh_attributions_all_view()


if __name__ == "__main__":
    main()
