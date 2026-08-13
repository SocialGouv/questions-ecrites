"""Read-time view unifying bureau sources — question_attributions_all.

Sources are NEVER mixed at write time: `question_real_attributions` keeps the
human/Excel attributions, `question_bureau_extract` keeps the MIN15
workflow extraction (with provenance). This view is the single read-time
merge point, with a `source` column for auditability. Dropping the view
reverts everything.

Semantics mirror `scripts/eval_bureau_with_min15.py` (the A/B eval that
measured +51 pts top-1 bureau on the independent MIN15 test set):

  - one row per question;
  - human attribution wins when both sources know the question;
  - MIN15 only fills questions absent from `question_real_attributions`;
  - `bureau_key` is the canonical vote/display key.

Two deliberate divergences from the eval, documented for reviewers:

  1. The eval's attribution-side canonicalisation only accepted
     DGCS-style codes (`[SDx/y]`), silently dropping the 22 bureaux
     whose code has another shape (`[SD2A]` DSS, `[BAEI]`, …). Prod
     votes on those today, so the view generalises to *any* `[CODE]`
     (uppercased, spaces stripped). DGCS keys are unchanged.
  2. The eval picked an arbitrary MIN15 row for multi-direction
     questions; the view deterministically picks the most recent step
     (date_debut_etape DESC, id ASC tie-break).

Known limit (follow-up, see docs/bureau_extract_min15.md): keys are not
yet unified across sources for DSS (`SD2A` attribution-side vs
`SD2A/<bureau>` MIN15-side) — the referential mapping to `bureaux.id`
is a separate chantier.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_SQL = r"""
CREATE VIEW question_attributions_all AS
WITH attribution_rows AS (
    -- question_id is the PRIMARY KEY of question_real_attributions, so this
    -- CTE is one-row-per-question by construction.
    SELECT
        qa.question_id,
        UPPER(REPLACE((regexp_match(b.nom, '^\s*\[([^\]]+)\]'))[1], ' ', ''))
            AS bureau_key,
        b.nom AS bureau_label,
        d.nom AS direction_label,
        'attribution'::text AS source
    FROM question_real_attributions qa
    JOIN bureaux b ON b.id = qa.bureau_reel_id
    LEFT JOIN directions d ON d.id = b.direction_id
    WHERE qa.bureau_reel_id IS NOT NULL
      AND b.nom ~ '^\s*\[[^\]]+\]'
),
min15_rows AS (
    SELECT DISTINCT ON (e.question_id)
        e.question_id,
        UPPER(REPLACE(BTRIM(e.sous_direction), ' ', ''))
          || CASE
               WHEN e.bureau IS NULL OR BTRIM(e.bureau) = '' THEN ''
               WHEN (regexp_match(e.bureau, '^\s*Bureau\s+(\w+)', 'i'))
                    IS NOT NULL
                 THEN '/' || UPPER(
                     (regexp_match(e.bureau, '^\s*Bureau\s+(\w+)', 'i'))[1])
               ELSE COALESCE(
                 '/' || NULLIF(UPPER(
                     (regexp_split_to_array(BTRIM(e.bureau), '[\s/,-]+'))[1]),
                     ''),
                 '')
             END AS bureau_key,
        BTRIM(CONCAT_WS(' — ',
            NULLIF(BTRIM(e.sous_direction), ''),
            NULLIF(BTRIM(COALESCE(e.bureau_full, e.bureau)), '')))
            AS bureau_label,
        e.direction_txt AS direction_label,
        'min15'::text AS source
    FROM question_bureau_extract e
    WHERE e.sous_direction IS NOT NULL AND BTRIM(e.sous_direction) <> ''
      -- Anti-join against the BASE tables (PK lookup per row), not the
      -- attribution_rows CTE: referencing the CTE here as well would let
      -- the PG >= 12 planner inline it twice and re-scan the join for
      -- every MIN15 row. Same membership condition as attribution_rows.
      AND NOT EXISTS (
            SELECT 1
            FROM question_real_attributions qa2
            JOIN bureaux b2 ON b2.id = qa2.bureau_reel_id
            WHERE qa2.question_id = e.question_id
              AND b2.nom ~ '^\s*\[[^\]]+\]'
      )
    ORDER BY e.question_id, e.date_debut_etape DESC NULLS LAST, e.id
)
-- Outer guards are defence in depth: both CTEs already guarantee a
-- non-empty key today (regex-filtered on one side, WHERE-filtered on the
-- other) and bureaux.nom is NOT NULL — but the view is a voting surface
-- for prod, so we make the invariant explicit rather than implicit.
SELECT question_id,
       bureau_key,
       COALESCE(NULLIF(bureau_label, ''), bureau_key) AS bureau_label,
       direction_label,
       source
FROM (
    SELECT * FROM attribution_rows
    UNION ALL
    SELECT * FROM min15_rows
) unioned
WHERE bureau_key IS NOT NULL AND bureau_key <> ''
"""


def upgrade() -> None:
    op.execute(VIEW_SQL)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS question_attributions_all")
