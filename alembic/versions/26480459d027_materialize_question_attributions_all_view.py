"""Materialize question_attributions_all so bureau kNN voting can use an index.

The bureau-attribution vote (`qe-front`'s `GET /api/questions/[id]/attributions`)
joins the HNSW candidate stream against this view on `question_id`. As a plain
view it can't be indexed, so Postgres materializes the whole ~7k-row view once
per request and does a Nested Loop with a linear Join Filter scan against it
for every candidate (measured: ~121 candidates x ~6.9k view rows, ~840k row
comparisons, 0.35s-7.66s per request, stdev 2.29s on a 3.4s mean). Its sibling
`direction_attributions` runs the identical KNN-vote shape in ~30ms because it
joins a real table (`question_real_attributions`) with a primary-key index
instead.

Converting to a MATERIALIZED VIEW with a unique index on `question_id` lets
the planner do an indexed lookup per candidate instead, matching
`direction_attributions`'s plan shape. `question_id` is unique in the view's
output by construction: `attribution_rows` is one-row-per-question (PK on
`question_real_attributions.question_id`), and `min15_rows` is deduplicated
via `DISTINCT ON (question_id)` and anti-joined against `attribution_rows`'s
membership condition — so the `UNION ALL` never produces two rows for the
same question_id.

The unique index is also required for `REFRESH MATERIALIZED VIEW
CONCURRENTLY`, which callers use (see `qe/attributions.py` and
`src/lib/attributions/refresh-view.ts`) after writing to
`question_real_attributions`, `bureaux`, or `question_bureau_extract` so the
view doesn't go stale — this trades a request-time cost that scaled with the
candidate x pool product for a write-time refresh that scales with the
(much smaller, much less frequently written) attribution pool alone.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "26480459d027"
down_revision: Union[str, Sequence[str], None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Identical SELECT to the view created in d4e5f6a7b8c9_add_question_attributions_all_view.py —
# only the CREATE [MATERIALIZED] VIEW wrapper differs. Keep semantics in sync if that
# view's SELECT is ever revisited; there is no shared source of truth between the two
# (each Alembic migration is a frozen, standalone script by convention).
SELECT_SQL = r"""
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

ORIGINAL_VIEW_SQL = "CREATE VIEW question_attributions_all AS" + SELECT_SQL


def upgrade() -> None:
    op.execute("DROP VIEW IF EXISTS question_attributions_all")
    op.execute("CREATE MATERIALIZED VIEW question_attributions_all AS" + SELECT_SQL)
    op.execute(
        "CREATE UNIQUE INDEX question_attributions_all_question_id_idx "
        "ON question_attributions_all (question_id)"
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS question_attributions_all")
    op.execute(ORIGINAL_VIEW_SQL)
