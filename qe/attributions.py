"""Refresh helpers for attribution-derived state: the `question_attributions_all`
materialized view, and the `has_direction_attribution`/`has_bureau_attribution`
flags that back the partial HNSW indexes used by direction/bureau kNN voting.

Call `refresh_attributions_all_view` after writing `question_real_attributions`,
`bureaux`, `directions`, or `question_bureau_extract`. `CONCURRENTLY` keeps the
view readable while it refreshes (requires the unique index on `question_id`).

Call `resync_bureau_attribution_flags` after a batch write to
`question_bureau_extract` (e.g. `extract_bureau_from_min15.py`) — it can
change which questions qualify without qe-front's per-question write paths
knowing which ones, so this resyncs the flag for every row.

Call `sync_attribution_flags_for_points` after (re-)embedding a batch of
questions — a question can already have an attribution before it's ever
embedded (the two happen independently), so a freshly-inserted row needs
its flags computed once rather than defaulting to false forever.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import text

from qe.db import get_engine


def refresh_attributions_all_view() -> None:
    """Refresh `question_attributions_all` after writing to one of its sources."""
    with get_engine().begin() as conn:
        conn.execute(
            text("REFRESH MATERIALIZED VIEW CONCURRENTLY question_attributions_all")
        )


def resync_bureau_attribution_flags() -> None:
    """Resync `has_bureau_attribution` for every row from the (refreshed) view."""
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE vec_questions_opendata v
                SET has_bureau_attribution = EXISTS (
                      SELECT 1 FROM question_attributions_all va
                      WHERE va.question_id = v.payload ->> 'question_id'
                    )
                WHERE has_bureau_attribution IS DISTINCT FROM EXISTS (
                      SELECT 1 FROM question_attributions_all va
                      WHERE va.question_id = v.payload ->> 'question_id'
                    )
            """)
        )


def sync_attribution_flags_for_points(point_ids: Sequence[str]) -> None:
    """Set both attribution flags for exactly these `vec_questions_opendata.id`s.

    Point ids are the table's primary key, so this is an indexed lookup —
    safe to call after every embed batch regardless of table size.
    """
    if not point_ids:
        return
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE vec_questions_opendata v
                SET has_direction_attribution = EXISTS (
                      SELECT 1 FROM question_real_attributions qa
                      WHERE qa.question_id = v.payload ->> 'question_id'
                        AND qa.direction_reelle_id IS NOT NULL
                    ),
                    has_bureau_attribution = EXISTS (
                      SELECT 1 FROM question_attributions_all va
                      WHERE va.question_id = v.payload ->> 'question_id'
                    )
                WHERE v.id = ANY(:ids)
            """),
            {"ids": list(point_ids)},
        )
