"""Offline replay of qe-front's direction / bureau attribution vote.

Mirrors the production implementation — ``qe-front``'s
``src/lib/direction/attributionAlgo.ts::computeDirectionVotes`` and
``app/api/questions/[id]/attributions/route.ts`` — closely enough to be
scored against ground truth without going through the HTTP API.

Why replay rather than call the endpoint:

* ``GET /api/questions/[id]/direction-attributions`` **writes** — it
  fire-and-forgets ``writeDirectionAlgoCache`` into
  ``questions.direction_algo_id``. Scoring the full ground-truth set live
  would mutate thousands of preprod rows; replay touches none.
* The sibling-leak sensitivity run needs a voter-exclusion the API has no
  parameter for.

The constants here are mirrored **by hand** from TypeScript, which
``docs/direction-bureau-attribution.md`` already lists as a standing
risk. ``compare_against_live`` exists to catch a desync: it diffs this
replay against the deployed endpoint on a sample, and no replay number
should be published without it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

from qe.hashing import stable_question_point_id

# --- mirrored from qe-front (keep in sync; see compare_against_live) ---
DIRECTION_KNN = 15
BUREAU_KNN = 25
HNSW_EF_SEARCH = 1000
SERVICE_POOL_K = 50
RESPONSE_TOP_K = 3
# knn-vote.ts::PROXIMITY_VOTE_TEMPERATURE
PROXIMITY_VOTE_TEMPERATURE = 0.03


@dataclass(frozen=True)
class Neighbour:
    key: str
    similarity: float
    label: str = ""


@dataclass(frozen=True)
class Vote:
    key: str
    vote: float


def proximity_vote(
    neighbours: Sequence[Neighbour],
    temperature: float = PROXIMITY_VOTE_TEMPERATURE,
) -> list[Vote]:
    """Aggregate neighbours into one vote per key, highest first.

    Mirrors ``knn-vote.ts::proximityVote``: each neighbour weighs
    ``exp((similarity - best) / temperature)``, so a few very close
    neighbours outweigh many merely nearby ones — a plain sum of
    similarities would let whichever class has the most labelled
    questions win on numbers alone.
    """
    if not neighbours:
        return []
    best = max(n.similarity for n in neighbours)
    totals: dict[str, float] = {}
    for n in neighbours:
        weight = math.exp((n.similarity - best) / temperature)
        totals[n.key] = totals.get(n.key, 0.0) + weight
    # Python's sort is stable, so equal votes keep insertion (i.e.
    # closest-neighbour-first) order, matching the TS comparator.
    return sorted(
        (Vote(key=k, vote=v) for k, v in totals.items()),
        key=lambda v: v.vote,
        reverse=True,
    )


def vote_shares(votes: Sequence[Vote]) -> list[float]:
    """Vote as a share of the offered total — ``voteToScores``' relevance.

    The denominator is the votes actually returned to the caller, not
    every neighbour polled: direction divides by its top-3, bureau by its
    top-``SERVICE_POOL_K``.
    """
    total = sum(v.vote for v in votes)
    return [v.vote / total if total else 0.0 for v in votes]


_DIRECTION_SQL = """
    WITH src AS (SELECT vector FROM vec_questions_opendata WHERE id = :point_id)
    SELECT d.nom AS key,
           1 - (v.vector <=> (SELECT vector FROM src)) AS similarity
    FROM vec_questions_opendata v
    JOIN question_real_attributions qa
      ON qa.question_id = v.payload ->> 'question_id'
    JOIN directions d ON d.id = qa.direction_reelle_id
    WHERE v.id <> :point_id
      AND v.has_direction_attribution
      AND qa.direction_reelle_id IS NOT NULL
      {extra}
    ORDER BY v.vector <=> (SELECT vector FROM src)
    LIMIT :knn
"""

_BUREAU_SQL = """
    WITH src AS (SELECT vector FROM vec_questions_opendata WHERE id = :point_id)
    SELECT va.bureau_key AS key,
           va.bureau_label AS label,
           1 - (v.vector <=> (SELECT vector FROM src)) AS similarity
    FROM vec_questions_opendata v
    JOIN question_attributions_all va
      ON va.question_id = v.payload ->> 'question_id'
    WHERE v.id <> :point_id
      AND v.has_bureau_attribution
      {extra}
    ORDER BY v.vector <=> (SELECT vector FROM src)
    LIMIT :knn
"""

_EXCLUDE_CLAUSE = "AND NOT (v.payload ->> 'question_id' = ANY(:excluded))"


def fetch_neighbours(
    session: Session,
    question_id: str,
    feature: str,
    *,
    knn: int | None = None,
    exclude_question_ids: Iterable[str] = (),
) -> list[Neighbour]:
    """Run the production KNN walk for one question.

    ``exclude_question_ids`` drops voters from the walk — used by the
    sibling-leak sensitivity run. It changes the approximate-search
    regime (the filter is applied inside the ``LIMIT``), so a run using
    it is only comparable against another run using it, never against
    the unfiltered headline.
    """
    excluded = list(exclude_question_ids)
    if feature == "direction":
        template, default_knn = _DIRECTION_SQL, DIRECTION_KNN
    elif feature == "bureau":
        template, default_knn = _BUREAU_SQL, BUREAU_KNN
    else:
        raise ValueError(f"unknown attribution feature: {feature!r}")

    sql = template.format(extra=_EXCLUDE_CLAUSE if excluded else "")
    params: dict[str, object] = {
        "point_id": stable_question_point_id(question_id),
        "knn": knn or default_knn,
    }
    if excluded:
        params["excluded"] = excluded

    # Both settings are transaction-scoped in production too: ef_search
    # for recall on the partial index, enable_seqscan as the planner
    # safety net right after a bulk write.
    session.execute(sqltext(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"))
    session.execute(sqltext("SET LOCAL enable_seqscan = off"))
    rows = session.execute(sqltext(sql), params).mappings()
    return [
        Neighbour(
            key=r["key"],
            similarity=float(r["similarity"]),
            label=r.get("label") or "",
        )
        for r in rows
    ]


def predict(
    session: Session,
    question_id: str,
    feature: str,
    *,
    knn: int | None = None,
    exclude_question_ids: Iterable[str] = (),
) -> tuple[list[str], list[float]]:
    """Return the top-3 predicted keys and their relevance shares."""
    neighbours = fetch_neighbours(
        session,
        question_id,
        feature,
        knn=knn,
        exclude_question_ids=exclude_question_ids,
    )
    if feature == "bureau":
        # route.ts sorts by similarity desc, then label asc, so a key's
        # displayed label is stable across equal distances.
        neighbours = sorted(neighbours, key=lambda n: (-n.similarity, n.label))
    votes = proximity_vote(neighbours)
    # The relevance denominator differs per feature: direction divides by
    # its top-3, bureau by the wider SERVICE_POOL_K slice it offers.
    offered = votes[:SERVICE_POOL_K] if feature == "bureau" else votes[:RESPONSE_TOP_K]
    shares = vote_shares(offered)
    return (
        [v.key for v in offered[:RESPONSE_TOP_K]],
        shares[:RESPONSE_TOP_K],
    )
