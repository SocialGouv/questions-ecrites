"""Refresh helper for the `question_attributions_all` materialized view.

Call after writing `question_real_attributions`, `bureaux`, `directions`,
or `question_bureau_extract`. `CONCURRENTLY` keeps the view readable while
it refreshes (requires the unique index on `question_id`).
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
