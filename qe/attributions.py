"""Refresh helpers for attribution-derived state: the `question_attributions_all`
materialized view, and the `has_direction_attribution`/`has_bureau_attribution`
flags that back the partial HNSW indexes used by direction/bureau kNN voting.

The two flags are computed from base tables (`question_real_attributions`,
`question_bureau_extract`), not from `question_attributions_all`, on purpose:
a flag computed from the view would depend on the view refresh succeeding
first, and a failed/late refresh would silently write a permanently-wrong
`false` (the row drops out of the partial index for good, since nothing else
ever recomputes it). Computing from base tables removes that ordering
dependency entirely, at the cost of being a superset of the view's actual
membership (`question_real_attributions.bureau_reel_id` doesn't require the
matching `bureaux.nom` to carry a `[CODE]` prefix the way the view's
`attribution_rows` CTE does) — harmless, since the vote query's join against
`question_attributions_all` silently drops any candidate that isn't actually
in the view, rather than miscounting it.

Call `refresh_attributions_all_view` after writing `question_real_attributions`,
`bureaux`, `directions`, or `question_bureau_extract` — needed for the vote's
displayed labels and vote weights, independent of the flags above.

Call `resync_bureau_attribution_flags` / `resync_direction_attribution_flags`
after a batch write that can change which questions qualify without a
per-question write path knowing which ones — a MIN15 extraction run
(bureau), a bulk attribution import, or a DB restored from a dump taken
before this module existed. Both are guarded (`IS DISTINCT FROM`), so
they're cheap to call defensively — measured at ~0.4s over the full table
when nothing actually changes.

Call `sync_attribution_flags_for_points` after (re-)embedding a batch of
questions — a question can already have an attribution before it's ever
embedded (the two happen independently), so a freshly-inserted row needs
its flags computed once rather than defaulting to false forever.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import text

from qe.db import get_engine

# `has_bureau_attribution`'s single source of truth — see the module
# docstring for why this reads base tables rather than the view.
_BUREAU_EXISTS_SQL = """EXISTS (
        SELECT 1 FROM question_real_attributions qa
        WHERE qa.question_id = v.payload ->> 'question_id'
          AND qa.bureau_reel_id IS NOT NULL
    ) OR EXISTS (
        SELECT 1 FROM question_bureau_extract qbe
        WHERE qbe.question_id = v.payload ->> 'question_id'
    )"""

_DIRECTION_EXISTS_SQL = """EXISTS (
        SELECT 1 FROM question_real_attributions qa
        WHERE qa.question_id = v.payload ->> 'question_id'
          AND qa.direction_reelle_id IS NOT NULL
    )"""


def refresh_attributions_all_view() -> None:
    """Refresh `question_attributions_all` after writing to one of its sources."""
    with get_engine().begin() as conn:
        conn.execute(
            text("REFRESH MATERIALIZED VIEW CONCURRENTLY question_attributions_all")
        )


def resync_bureau_attribution_flags() -> None:
    """Resync `has_bureau_attribution` for every row from its base tables."""
    sql = f"""
        UPDATE vec_questions_opendata v
        SET has_bureau_attribution = {_BUREAU_EXISTS_SQL}
        WHERE has_bureau_attribution IS DISTINCT FROM ({_BUREAU_EXISTS_SQL})
    """  # noqa: S608 -- interpolating module-level constants, not input
    with get_engine().begin() as conn:
        conn.execute(text(sql))


def resync_direction_attribution_flags() -> None:
    """Resync `has_direction_attribution` for every row from `question_real_attributions`."""
    sql = f"""
        UPDATE vec_questions_opendata v
        SET has_direction_attribution = {_DIRECTION_EXISTS_SQL}
        WHERE has_direction_attribution IS DISTINCT FROM ({_DIRECTION_EXISTS_SQL})
    """  # noqa: S608 -- interpolating module-level constants, not input
    with get_engine().begin() as conn:
        conn.execute(text(sql))


def sync_attribution_flags_for_points(point_ids: Sequence[str]) -> None:
    """Set both attribution flags for exactly these `vec_questions_opendata.id`s.

    Point ids are the table's primary key, so this is an indexed lookup —
    safe to call after every embed batch regardless of table size. Guarded
    the same way as the resync helpers: a freshly-upserted row usually has
    no attribution yet (the common case), so most calls are a no-op write.
    """
    if not point_ids:
        return
    sql = f"""
        UPDATE vec_questions_opendata v
        SET has_direction_attribution = {_DIRECTION_EXISTS_SQL},
            has_bureau_attribution = {_BUREAU_EXISTS_SQL}
        WHERE v.id = ANY(:ids)
          AND (
            v.has_direction_attribution IS DISTINCT FROM ({_DIRECTION_EXISTS_SQL})
            OR v.has_bureau_attribution IS DISTINCT FROM ({_BUREAU_EXISTS_SQL})
          )
    """  # noqa: S608 -- interpolating module-level constants, not input
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"ids": list(point_ids)})
