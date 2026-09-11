"""Build MIN15 bureau keys at bureau level in question_attributions_all.

A MIN15 poste reads "<direction> - <segment 2> - <segment 3>", but which
segment names the bureau depends on the direction: "DGCS - SD2 - Bureau 2B"
puts it in segment 3, while "DSS - SD1 B - REDACTEURS" and
"DGOS - SDRH1 - Chef de bureau" put it in segment 2 and use segment 3 for the
author's role. Appending segment 3 unconditionally gave one bureau as many keys
as roles, none matching the referential's code (SD1B), so bureau votes split.

    "SD2 - Bureau 2B"           -> SD2/2B   (unchanged)
    "SD1 B - REDACTEURS"        -> SD1B
    "SDRH1 - Chef de bureau"    -> SDRH1
    "SD1 - MCGRM - REDACTEURS"  -> MCGRM
    "DACI - REDACTEURS"         -> DACI
    "SDAS - Sous-Direction"     -> no row: names a sous-direction, not a bureau
    "CAB - Cabinet"             -> no row
    anything else               -> unchanged ("Centralisateur - MDI")

Rows without a bureau-level key are dropped before picking each question's
latest step. Bureau suggestion feedback recorded against a MIN15 key moves to
the new key when the mapping is unambiguous; downgrade restores the previous
view but leaves those feedback targets on the new keys.
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "5b1c9e2d7a4f"
down_revision: Union[str, Sequence[str], None] = "d3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# First word of segment 3 that names a role or a level rather than a unit.
ROLE_TOKENS = (
    "('REDACTEURS', 'VALIDEURS', 'CHEF', 'CHEFFE', 'CM', 'CHARGÉ', 'CHARGÉS', "
    "'CHARGEE', 'CHARGEES', 'COORDINATION', 'MISSION', 'CABINET', 'DIRECTION', "
    "'SOUS', 'MAJ', 'GOUV', 'CAB')"
)
SOUS_DIRECTION = "UPPER(REPLACE(BTRIM(e.sous_direction), ' ', ''))"
BUREAU_CODE = r"(regexp_match(e.bureau, '^\s*Bureau\s+(\w+)', 'i'))[1]"
FIRST_WORD = r"(regexp_split_to_array(BTRIM(COALESCE(e.bureau, '')), '[\s/,-]+'))[1]"

PREVIOUS_KEY_SQL = rf"""
    {SOUS_DIRECTION}
      || CASE
           WHEN e.bureau IS NULL OR BTRIM(e.bureau) = '' THEN ''
           WHEN {BUREAU_CODE} IS NOT NULL THEN '/' || UPPER({BUREAU_CODE})
           ELSE COALESCE('/' || NULLIF(UPPER({FIRST_WORD}), ''), '')
         END"""

BUREAU_KEY_SQL = rf"""
    CASE
      WHEN {BUREAU_CODE} IS NOT NULL THEN {SOUS_DIRECTION} || '/' || UPPER({BUREAU_CODE})
      WHEN {SOUS_DIRECTION} ~ '^SD[0-9]+[A-Z]$' OR {SOUS_DIRECTION} ~ '^SD[A-Z]+[0-9]+$'
        THEN {SOUS_DIRECTION}
      WHEN {SOUS_DIRECTION} ~ '^SD([0-9]+|[A-Z]+)$' THEN
        CASE WHEN {FIRST_WORD} ~ '^[A-Z]{{2,}}$' AND {FIRST_WORD} NOT IN {ROLE_TOKENS}
             THEN {FIRST_WORD} END
      WHEN {SOUS_DIRECTION} = 'CAB' THEN NULL
      WHEN UPPER({FIRST_WORD}) IN {ROLE_TOKENS} THEN {SOUS_DIRECTION}
      ELSE {SOUS_DIRECTION} || COALESCE('/' || NULLIF(UPPER({FIRST_WORD}), ''), '')
    END"""

ATTRIBUTION_ROWS_SQL = r"""
attribution_rows AS (
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
)"""

MIN15_SOURCE_FILTER_SQL = r"""
    e.sous_direction IS NOT NULL AND BTRIM(e.sous_direction) <> ''
      AND NOT EXISTS (
            SELECT 1
            FROM question_real_attributions qa2
            JOIN bureaux b2 ON b2.id = qa2.bureau_reel_id
            WHERE qa2.question_id = e.question_id
              AND b2.nom ~ '^\s*\[[^\]]+\]'
      )"""

MIN15_LABEL_SQL = """
    BTRIM(CONCAT_WS(' — ',
        NULLIF(BTRIM(e.sous_direction), ''),
        NULLIF(BTRIM(COALESCE(e.bureau_full, e.bureau)), '')))"""

OUTER_SELECT_SQL = """
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

BUREAU_LEVEL_SELECT_SQL = f"""
WITH {ATTRIBUTION_ROWS_SQL},
min15_keyed AS (
    SELECT e.question_id, e.date_debut_etape, e.id,
           {BUREAU_KEY_SQL} AS bureau_key,
           {MIN15_LABEL_SQL} AS bureau_label,
           e.direction_txt AS direction_label
    FROM question_bureau_extract e
    WHERE {MIN15_SOURCE_FILTER_SQL}
),
min15_rows AS (
    SELECT DISTINCT ON (question_id)
        question_id, bureau_key, bureau_label, direction_label, 'min15'::text AS source
    FROM min15_keyed
    WHERE bureau_key IS NOT NULL AND bureau_key <> ''
    ORDER BY question_id, date_debut_etape DESC NULLS LAST, id
)
{OUTER_SELECT_SQL}"""  # noqa: S608 -- interpolating module-level constants, not input

PREVIOUS_SELECT_SQL = f"""
WITH {ATTRIBUTION_ROWS_SQL},
min15_rows AS (
    SELECT DISTINCT ON (e.question_id)
        e.question_id,
        {PREVIOUS_KEY_SQL} AS bureau_key,
        {MIN15_LABEL_SQL} AS bureau_label,
        e.direction_txt AS direction_label,
        'min15'::text AS source
    FROM question_bureau_extract e
    WHERE {MIN15_SOURCE_FILTER_SQL}
    ORDER BY e.question_id, e.date_debut_etape DESC NULLS LAST, e.id
)
{OUTER_SELECT_SQL}"""  # noqa: S608 -- interpolating module-level constants, not input

REMAP_FEEDBACK_SQL = f"""
WITH keys AS (
    SELECT DISTINCT {PREVIOUS_KEY_SQL} AS previous_key, {BUREAU_KEY_SQL} AS bureau_key
    FROM question_bureau_extract e
    WHERE e.sous_direction IS NOT NULL AND BTRIM(e.sous_direction) <> ''
),
mapping AS (
    SELECT previous_key, MIN(bureau_key) AS bureau_key
    FROM keys
    WHERE bureau_key IS NOT NULL AND bureau_key <> previous_key
    GROUP BY previous_key
    HAVING COUNT(DISTINCT bureau_key) = 1
)
UPDATE suggestion_feedback f
SET suggestion_target = m.bureau_key
FROM mapping m
WHERE f.suggestion_kind = 'attribution_bureau'
  AND f.suggestion_target = m.previous_key
"""  # noqa: S608 -- interpolating module-level constants, not input

UNIQUE_INDEX_SQL = (
    "CREATE UNIQUE INDEX question_attributions_all_question_id_idx "
    "ON question_attributions_all (question_id)"
)


def _recreate_view(select_sql: str) -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS question_attributions_all")
    op.execute("CREATE MATERIALIZED VIEW question_attributions_all AS" + select_sql)
    op.execute(UNIQUE_INDEX_SQL)


def upgrade() -> None:
    _recreate_view(BUREAU_LEVEL_SELECT_SQL)
    if sa.inspect(op.get_bind()).has_table("suggestion_feedback"):
        op.execute(REMAP_FEEDBACK_SQL)


def downgrade() -> None:
    _recreate_view(PREVIOUS_SELECT_SQL)
