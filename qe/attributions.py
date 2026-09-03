"""Refresh helper for the `question_attributions_all` materialized view.

The view (see the Alembic migration that materialized it) is a read-time
merge of `question_real_attributions` and `question_bureau_extract`, joined
via `bureaux`/`directions`. It only needs to be refreshed after a write to
one of those tables — CONCURRENTLY keeps it readable (by qe-front's bureau
kNN vote) while the refresh runs, at the cost of requiring the view's unique
index on `question_id`.
"""

from __future__ import annotations

from sqlalchemy import text

from qe.db import get_engine


def refresh_attributions_all_view() -> None:
    """Refresh `question_attributions_all` after writing to one of its sources."""
    with get_engine().begin() as conn:
        conn.execute(
            text("REFRESH MATERIALIZED VIEW CONCURRENTLY question_attributions_all")
        )
