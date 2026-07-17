#!/usr/bin/env python3
"""Restore historical allotments hidden in the LEGACY reponses.

Context: the ingestion of AN legs 14/15/16 (and possibly others) created
one row per question in the `reponses` table with a synthetic id like
`AN-LEGACY-<question_id>`. When several questions were historically
answered together by the JO on the same page, their reponses ended up
with the SAME `texte_reponse` but DIFFERENT ids — so the SQL
`GROUP BY reponse_id` view of allotments returned 0 groups for these
legislatures.

This script dedupes: for each cluster of LEGACY reponses sharing the
exact same text, we pick the min id as canonical, redirect every
question in the cluster to it, then delete the now-orphan copies.

Effect (measured on current DB, dry-run):
  - 138 105 LEGACY reponses become ~67 691 unique canonical rows
  - 13 448 groups (>= 2 questions) become visible again
  - covers ~70 000 questions previously invisible for allotment metrics

Idempotent: running twice is a no-op since the min id is already
canonical.

Usage:
    poetry run python scripts/dedupe_legacy_reponses.py            # dry-run
    poetry run python scripts/dedupe_legacy_reponses.py --commit   # apply
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from qe import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def dedupe(commit: bool) -> None:
    with db.get_session() as session:
        # ------------------------------------------------------------------
        # 1. Diagnose current state
        # ------------------------------------------------------------------
        before = session.execute(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM reponses WHERE id LIKE 'AN-LEGACY-%%') AS legacy_reponses,
                  (SELECT COUNT(*) FROM (
                    SELECT MD5(texte_reponse) AS h, COUNT(*) n
                    FROM reponses WHERE id LIKE 'AN-LEGACY-%%'
                    GROUP BY MD5(texte_reponse)
                    HAVING COUNT(*) >= 2
                  ) t) AS hidden_groups,
                  (SELECT SUM(n) FROM (
                    SELECT COUNT(*) n
                    FROM reponses WHERE id LIKE 'AN-LEGACY-%%'
                    GROUP BY MD5(texte_reponse)
                    HAVING COUNT(*) >= 2
                  ) t) AS questions_in_hidden_groups
                """
            )
        ).one()
        logger.info(
            "Before: %d LEGACY reponses, %d hidden groups covering %d questions",
            before[0], before[1] or 0, before[2] or 0,
        )

        # ------------------------------------------------------------------
        # 2. Build the mapping old_id → canonical_id
        # ------------------------------------------------------------------
        # We keep the min id per text hash as canonical. `mapping` only
        # contains rows where old_id != canonical_id — the rows that need
        # to be redirected + deleted.
        session.execute(text("DROP TABLE IF EXISTS _legacy_dedup_map"))
        session.execute(
            text(
                """
                CREATE TEMP TABLE _legacy_dedup_map AS
                WITH canonical AS (
                  SELECT MD5(texte_reponse) AS text_hash, MIN(id) AS canonical_id
                  FROM reponses WHERE id LIKE 'AN-LEGACY-%%'
                  GROUP BY MD5(texte_reponse)
                )
                SELECT r.id AS old_id, c.canonical_id
                FROM reponses r
                JOIN canonical c ON MD5(r.texte_reponse) = c.text_hash
                WHERE r.id LIKE 'AN-LEGACY-%%' AND r.id <> c.canonical_id
                """
            )
        )
        n_to_redirect = session.execute(
            text("SELECT COUNT(*) FROM _legacy_dedup_map")
        ).scalar_one()
        logger.info("Will redirect %d LEGACY reponses to their canonical peer.", n_to_redirect)

        # ------------------------------------------------------------------
        # 3. Update questions to point at the canonical id
        # ------------------------------------------------------------------
        n_questions_updated = session.execute(
            text(
                """
                UPDATE questions
                SET reponse_id = m.canonical_id
                FROM _legacy_dedup_map m
                WHERE questions.reponse_id = m.old_id
                """
            )
        ).rowcount
        logger.info("  → %d questions.reponse_id updates staged.", n_questions_updated)

        # ------------------------------------------------------------------
        # 4. Delete the now-orphan reponses
        # ------------------------------------------------------------------
        n_deleted = session.execute(
            text(
                """
                DELETE FROM reponses
                WHERE id IN (SELECT old_id FROM _legacy_dedup_map)
                """
            )
        ).rowcount
        logger.info("  → %d orphan reponses staged for deletion.", n_deleted)

        # ------------------------------------------------------------------
        # 5. Verify by re-counting groups after the (uncommitted) change
        # ------------------------------------------------------------------
        after = session.execute(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM reponses WHERE id LIKE 'AN-LEGACY-%%') AS legacy_reponses,
                  (SELECT COUNT(*) FROM (
                    SELECT reponse_id FROM questions
                    WHERE reponse_id LIKE 'AN-LEGACY-%%'
                    GROUP BY reponse_id
                    HAVING COUNT(*) >= 2
                  ) t) AS visible_groups,
                  (SELECT SUM(n) FROM (
                    SELECT COUNT(*) n FROM questions
                    WHERE reponse_id LIKE 'AN-LEGACY-%%'
                    GROUP BY reponse_id
                    HAVING COUNT(*) >= 2
                  ) t) AS questions_in_visible_groups
                """
            )
        ).one()
        logger.info(
            "After%s: %d LEGACY reponses, %d visible groups covering %d questions",
            "" if commit else " (uncommitted)",
            after[0], after[1] or 0, after[2] or 0,
        )

        if commit:
            session.commit()
            logger.info("Committed.")
        else:
            session.rollback()
            logger.info("Dry run — nothing written. Re-run with --commit to persist.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--commit", action="store_true", help="Persist changes (default: dry run)")
    args = parser.parse_args()
    dedupe(commit=args.commit)


if __name__ == "__main__":
    main()
