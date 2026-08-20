"""add allotissements_jo view + documentation of the detection method

Persists our derived "allotissements du JO" as a first-class VIEW so
downstream consumers (eval scripts, dashboards, external users) can
query them by name without having to reconstruct the aggregation.

## What is `allotissements_jo`?

An "allotissement" is a group of parliamentary questions (QE) that
received EXACTLY the same response text in the same JO issue.  In the
official JO PDF, grouped questions are marked with an asterisk (*) on
the author name in the table of contents — this VIEW reproduces that
grouping.

**Detection method** (implemented in `qe/ingestion_an.py`):

    reponse_id = f"AN-{YYYYMMDD}-{sha1(texte_reponse)[:12]}"

Two QE share a `reponse_id` iff their `texte_reponse` is byte-identical
AND their `date_reponse_jo` matches.  Cross-checked against the PDF
`jo_anq_202310.pdf` (JO 2023-03-07):
  - QE5369 (soignants non vaccinés, no asterisk) got its own id
  - QE5690, 5834, 5835, 5839, 5843 (5 kiné, all asterisked) share one id

## Validation of the resulting GT

On the 13 506 groups this view exposes (as of ingestion 2026-07-28),
a two-LLM cross-validation on leg 17 (815 groups × Mistral-medium +
GPT-oss-120b via Albert) gave ~98 % of groups classified as legitimate
thematic allotments, ~2 % as batch admin (ministry applied a template
to unrelated questions).

Revision ID: e5f6c7d8a9b1
Revises: b2c3d4e5f6a7
Create Date: 2026-07-28
"""

from typing import Sequence, Union

from alembic import op


revision: str = "e5f6c7d8a9b1"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_DDL = """
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


def upgrade() -> None:
    op.execute(VIEW_DDL)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS allotissements_jo")
