"""catch-up for pre-squash databases: recreate bureau-attribution view chain

Some databases have their `alembic_version` stamped past `e5f6c7d8a9b1`
(allotissements_jo), `c1d2e3f4a5b6` (question_bureau_extract) and/or
`d4e5f6a7b8c9` (question_attributions_all) without those migrations' DDL
ever having actually run — same root cause as the pre-squash catch-ups in
`e7f8a9b0c1d2` and `b2c3d4e5f6a7`. Confirmed on the Atlas Sandbox database:
`alembic_version` was already at `b2c3d4e5f6a7`, yet `to_regclass(...)`
returned NULL for all three objects while later-in-chain objects
(direction_algo_id, the JO parser columns, the Auth.js tables) were
present. Alembic never re-runs a migration once it believes the stamp is
past it, so a plain `alembic upgrade head` cannot self-heal this — hence
this catch-up, which every already-consistent database (this branch's own
dev/CI databases included) runs as a no-op.

Everything below is guarded to be safe on a database that already has
these objects (post-squash, or a database where these migrations did run
normally):

  - `allotissements_jo`: `CREATE OR REPLACE VIEW`, verbatim from
    `e5f6c7d8a9b1` — already idempotent by construction.
  - `question_bureau_extract`: `CREATE TABLE IF NOT EXISTS` +
    `CREATE INDEX IF NOT EXISTS`, same shape as `c1d2e3f4a5b6` on main
    (`source_etape_id` has no foreign key — it's a deliberate soft
    reference to the externally-loaded `reponses_extract_etapes`, never
    an FK; see that migration's comment).
  - `question_attributions_all`: `CREATE OR REPLACE VIEW`, verbatim from
    `d4e5f6a7b8c9`. Requires `question_bureau_extract` to exist (it's
    referenced in the view's MIN15-side CTE), hence the ordering below.

downgrade() is intentionally a no-op: on a database where this migration
was a no-op upgrade (objects pre-existed), downgrading would destroy
objects that predate this migration and that other code may depend on.
There's no way from here to tell "created by this migration" apart from
"already there" after the fact, so the safe default is to leave state
alone — consistent with the same trade-off made in `e7f8a9b0c1d2`.

Revision ID: 6cee3d0e7f23
Revises: 4bcdeece2262
Create Date: 2026-08-21 18:40:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6cee3d0e7f23"
down_revision: Union[str, Sequence[str], None] = "4bcdeece2262"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ALLOTISSEMENTS_JO_VIEW_DDL = """
CREATE OR REPLACE VIEW allotissements_jo AS
SELECT
    r.id                    AS allotissement_id,
    r.source                AS source,
    r.date_reponse_jo       AS date_jo,
    r.no_publication        AS no_publication,
    r.ministre_reponse_libelle AS ministre_reponse,
    COUNT(q.id)             AS n_questions,
    ARRAY_AGG(q.id ORDER BY q.id) AS question_ids,
    r.texte_reponse         AS texte_reponse
FROM reponses r
JOIN questions q ON q.reponse_id = r.id
GROUP BY r.id, r.source, r.date_reponse_jo, r.no_publication,
         r.ministre_reponse_libelle, r.texte_reponse
HAVING COUNT(q.id) >= 2
"""

QUESTION_BUREAU_EXTRACT_DDL = """
CREATE TABLE IF NOT EXISTS question_bureau_extract (
    id SERIAL PRIMARY KEY,
    question_id VARCHAR(100) NOT NULL
        REFERENCES questions(id) ON DELETE CASCADE,
    direction_txt TEXT NOT NULL,
    sous_direction TEXT,
    bureau TEXT,
    bureau_full TEXT,
    source_etape_id INTEGER,
    date_debut_etape DATE,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_question_bureau_extract_qid_dir
        UNIQUE (question_id, direction_txt)
);
CREATE INDEX IF NOT EXISTS ix_question_bureau_extract_qid
    ON question_bureau_extract (question_id);
CREATE INDEX IF NOT EXISTS ix_question_bureau_extract_direction
    ON question_bureau_extract (direction_txt);
COMMENT ON TABLE question_bureau_extract IS
    'Bureau reel extrait des workflows MIN15 (outil Reponses). '
    'Source complementaire a question_real_attributions.bureau_reel_id '
    'qui n''est peuplee que pour DGCS. Voir '
    'scripts/extract_bureau_from_min15.py pour la regle d''extraction.'
"""

QUESTION_ATTRIBUTIONS_ALL_VIEW_DDL = r"""
CREATE OR REPLACE VIEW question_attributions_all AS
WITH attribution_rows AS (
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
      AND NOT EXISTS (
            SELECT 1
            FROM question_real_attributions qa2
            JOIN bureaux b2 ON b2.id = qa2.bureau_reel_id
            WHERE qa2.question_id = e.question_id
              AND b2.nom ~ '^\s*\[[^\]]+\]'
      )
    ORDER BY e.question_id, e.date_debut_etape DESC NULLS LAST, e.id
)
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
    """Upgrade schema."""
    op.execute(ALLOTISSEMENTS_JO_VIEW_DDL)
    op.execute(QUESTION_BUREAU_EXTRACT_DDL)
    op.execute(QUESTION_ATTRIBUTIONS_ALL_VIEW_DDL)


def downgrade() -> None:
    """Downgrade schema."""
    pass
