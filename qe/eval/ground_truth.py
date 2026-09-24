"""Ground-truth builders for the feature-quality benchmark.

Every truth here is reconstructed from data the administration itself
produced (the JO, and the MIN15 workflow extract) — never from the
algorithm's own output.

Answer-text clusters
--------------------
Both list features share one primitive: the **answer-text cluster**, the
set of questions whose published answer has the same text. Within a
cluster, the answer's JO date splits the two features apart:

* same ``date_reponse_jo``  -> **allotissement** (one joint answer)
* different ``date_reponse_jo`` -> **EDR** (an existing answer reused later)

Clustering on the text rather than on ``reponse_id`` is not a detail: the
two chambers encode a joint answer differently, and ``reponse_id`` only
captures the AN form. Measured on the full preprod corpus, every
shared-``reponse_id`` group is AN and every same-date distinct-``reponse_id``
group is SENAT. A ``reponse_id``-based ground truth therefore scores AN
alone while reading like a whole-system number — which is what
``scripts/eval_realistic_encours.py`` and ``scripts/eval_rank_distribution.py``
currently do.

Answers shorter than ``MIN_ANSWER_CHARS`` are dropped: they are
procedural boilerplate ("la question est caduque…") that would cluster
unrelated questions together. On the preprod corpus this excludes 367
questions, 92 of which sit in a shared-``reponse_id`` group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session

# Below this length an answer is procedural boilerplate, not a real
# answer, and clustering on it produces spurious mates.
MIN_ANSWER_CHARS = 200


@dataclass(frozen=True)
class ClusterMember:
    """One question inside an answer-text cluster."""

    question_id: str
    source: str
    legislature: int
    date_publication_jo: date | None
    date_reponse_jo: date


@dataclass(frozen=True)
class AnswerCluster:
    """Questions sharing one answer text, across all of its JO dates."""

    text_hash: str
    members: tuple[ClusterMember, ...]

    @property
    def dates(self) -> frozenset[date]:
        return frozenset(m.date_reponse_jo for m in self.members)

    def same_date_as(self, member: ClusterMember) -> tuple[ClusterMember, ...]:
        """Co-answered questions — the allotissement mates of *member*."""
        return tuple(
            m
            for m in self.members
            if m.question_id != member.question_id
            and m.date_reponse_jo == member.date_reponse_jo
        )

    def answered_before(self, member: ClusterMember) -> tuple[ClusterMember, ...]:
        """Questions whose identical answer was published earlier.

        The EDR mates of *member*: the answer an agent drafting for
        *member* could have reused, because it already existed.
        """
        return tuple(
            m
            for m in self.members
            if m.question_id != member.question_id
            and m.date_reponse_jo < member.date_reponse_jo
        )


_CLUSTER_SQL = sqltext(
    """
    WITH answered AS (
        SELECT
            q.id                    AS question_id,
            q.source                AS source,
            q.legislature           AS legislature,
            q.date_publication_jo   AS date_publication_jo,
            r.date_reponse_jo       AS date_reponse_jo,
            md5(r.texte_reponse)    AS text_hash
        FROM questions q
        JOIN reponses r ON r.id = q.reponse_id
        WHERE r.texte_reponse IS NOT NULL
          AND length(r.texte_reponse) > :min_chars
          AND r.date_reponse_jo IS NOT NULL
    )
    SELECT question_id, source, legislature, date_publication_jo,
           date_reponse_jo, text_hash
    FROM answered
    WHERE text_hash IN (
        SELECT text_hash FROM answered GROUP BY text_hash HAVING count(*) >= 2
    )
    ORDER BY text_hash, date_reponse_jo, question_id
    """
)


def fetch_answer_clusters(
    session: Session, min_answer_chars: int = MIN_ANSWER_CHARS
) -> list[AnswerCluster]:
    """Load every answer-text cluster of size >= 2 from the database."""
    rows = session.execute(_CLUSTER_SQL, {"min_chars": min_answer_chars}).mappings()

    by_hash: dict[str, list[ClusterMember]] = {}
    for row in rows:
        by_hash.setdefault(row["text_hash"], []).append(
            ClusterMember(
                question_id=row["question_id"],
                source=row["source"],
                legislature=row["legislature"],
                date_publication_jo=row["date_publication_jo"],
                date_reponse_jo=row["date_reponse_jo"],
            )
        )
    return [
        AnswerCluster(text_hash=h, members=tuple(members))
        for h, members in by_hash.items()
    ]


@dataclass(frozen=True)
class GroundTruthCase:
    """One benchmark case: a source question and the mates it should surface."""

    member: ClusterMember
    mate_ids: frozenset[str]

    @property
    def question_id(self) -> str:
        return self.member.question_id

    @property
    def n_mates(self) -> int:
        return len(self.mate_ids)


def allotissement_cases(
    clusters: Iterable[AnswerCluster], *, require_published_at_source: bool = True
) -> list[GroundTruthCase]:
    """Cases where the mates were answered jointly, on the same JO date.

    ``require_published_at_source`` drops mates published after the source
    question, mirroring ``edr_cases``: ``as_of_predicate("allotissement")``
    filters the pool on ``q.date_publication_jo <= :as_of``, so such a mate
    is absent from the pool and scores as a miss no ranking could avoid.
    The relation is symmetric, so keeping them makes roughly half of every
    cross-publication-date joint answer unwinnable.
    """
    cases: list[GroundTruthCase] = []
    for cluster in clusters:
        for member in cluster.members:
            mates = cluster.same_date_as(member)
            if require_published_at_source:
                cutoff = member.date_publication_jo
                mates = tuple(
                    m
                    for m in mates
                    if cutoff is not None
                    and m.date_publication_jo is not None
                    and m.date_publication_jo <= cutoff
                )
            if mates:
                cases.append(
                    GroundTruthCase(
                        member=member,
                        mate_ids=frozenset(m.question_id for m in mates),
                    )
                )
    return cases


def edr_cases(
    clusters: Iterable[AnswerCluster], *, require_available_at_publication: bool = True
) -> list[GroundTruthCase]:
    """Cases where an existing answer was reused for a later question.

    ``require_available_at_publication`` keeps only mates already answered
    by the time the source question was published — the answer an agent
    could actually have found when the question landed on their desk.
    Turning it off widens the truth to every earlier reuse, including
    answers published after the source question arrived.
    """
    cases: list[GroundTruthCase] = []
    for cluster in clusters:
        for member in cluster.members:
            mates = cluster.answered_before(member)
            if require_available_at_publication:
                cutoff = member.date_publication_jo
                mates = tuple(
                    m
                    for m in mates
                    if cutoff is not None and m.date_reponse_jo <= cutoff
                )
            if mates:
                cases.append(
                    GroundTruthCase(
                        member=member,
                        mate_ids=frozenset(m.question_id for m in mates),
                    )
                )
    return cases


def cases_for_pool(
    clusters: Iterable[AnswerCluster], feature: str, pool: str
) -> list[GroundTruthCase]:
    """Build the truth matching the candidate pool the run will search.

    Both features' date filters mirror ``as_of_predicate``. Against an
    unrestricted pool they delete mates the search reaches — measured on
    preprod, 72% (allotissement) and 80% (EDR) of the mates they drop are
    actually retrieved there, each one then scored as a miss. Selecting
    the variant lives here rather than in the caller so the two features
    cannot drift apart on it.
    """
    if pool not in ("as-of-t", "unrestricted"):
        raise ValueError(f"unknown pool: {pool!r}")
    as_of = pool == "as-of-t"
    clusters = list(clusters)
    if feature == "allotissement":
        return allotissement_cases(clusters, require_published_at_source=as_of)
    if feature == "edr":
        return edr_cases(clusters, require_available_at_publication=as_of)
    raise ValueError(f"unknown feature: {feature!r}")


# ---------------------------------------------------------------------------
# Attribution ground truth
# ---------------------------------------------------------------------------

# MIN15 writes a direction as "DGOS (direction générale de l'offre de soins)";
# the `directions` referential stores the bare acronym.
_MIN15_DIRECTION_SQL = sqltext(
    """
    SELECT DISTINCT ON (e.question_id)
        e.question_id AS question_id,
        upper(split_part(btrim(e.direction_txt), ' ', 1)) AS direction
    FROM question_bureau_extract e
    WHERE e.direction_txt IS NOT NULL AND btrim(e.direction_txt) <> ''
    ORDER BY e.question_id, e.date_debut_etape DESC NULLS LAST, e.id
    """
)

_HUMAN_DIRECTION_SQL = sqltext(
    """
    SELECT qra.question_id AS question_id, d.nom AS direction
    FROM question_real_attributions qra
    JOIN directions d ON d.id = qra.direction_reelle_id
    WHERE qra.direction_reelle_id IS NOT NULL
    """
)

# One row per question, guaranteed: the materialized view carries a UNIQUE
# index on question_id (migration 5b1c9e2d7a4f), and its min15 half excludes
# any question already carrying an attribution row. A refresh producing a
# duplicate would fail rather than make the collapse below order-dependent.
_BUREAU_SQL = sqltext(
    """
    SELECT va.question_id AS question_id,
           va.bureau_key  AS bureau_key,
           va.source      AS source
    FROM question_attributions_all va
    """
)


def fetch_direction_truth(session: Session, origin: str) -> dict[str, str]:
    """Map question id -> direction label.

    ``origin="min15"`` is the only *uncontaminated* direction truth:
    ``attributionAlgo.ts::computeDirectionVotes`` votes exclusively over
    ``question_real_attributions``, so a MIN15 label is never one of the
    voters deciding the answer.

    ``origin="human"`` reads ``question_real_attributions`` — which *is*
    the voter pool. Its score is an optimistic bound, not a measurement,
    and callers must label it as such.
    """
    if origin == "min15":
        stmt = _MIN15_DIRECTION_SQL
    elif origin == "human":
        stmt = _HUMAN_DIRECTION_SQL
    else:
        raise ValueError(f"unknown direction truth origin: {origin!r}")
    return {r["question_id"]: r["direction"] for r in session.execute(stmt).mappings()}


#: ``question_attributions_all.source`` values — the human half is
#: labelled ``attribution``, not ``human``.
BUREAU_SOURCES = ("attribution", "min15")


def fetch_bureau_truth(session: Session, origin: str | None = None) -> dict[str, str]:
    """Map question id -> canonical bureau key, optionally one source only.

    No uncontaminated bureau truth exists: the vote reads
    ``question_attributions_all``, which unions both sources. The route
    excludes the question's own row, so the direct self-vote is gone, but
    every label here is still a voter for its neighbours.
    """
    if origin is not None and origin not in BUREAU_SOURCES:
        raise ValueError(
            f"unknown bureau truth source: {origin!r} (expected one of {BUREAU_SOURCES})"
        )
    rows = session.execute(_BUREAU_SQL).mappings()
    return {
        r["question_id"]: r["bureau_key"]
        for r in rows
        if origin is None or r["source"] == origin
    }


# ---------------------------------------------------------------------------
# As-of-t candidate pool
# ---------------------------------------------------------------------------


def as_of_predicate(kind: str) -> str:
    """SQL predicate restricting candidates to the pool an agent saw at time t.

    ``t`` is the source question's ``date_publication_jo``, bound as
    ``:as_of``. Aliases assumed: ``q`` for questions, ``r`` for its
    ``reponses`` row (LEFT JOIN).

    * ``allotissement`` — published by t, and not yet answered at t. This
      is the constraint ``scripts/eval_realistic_encours.py`` documents:
      the live route's ``etat_question = 'EN_COURS'`` filter reads today's
      status, so it excludes every ground-truth mate (all since answered)
      while admitting questions that did not exist at t.
    * ``edr`` — published by t and already answered by t.
    """
    if kind == "allotissement":
        return (
            "q.date_publication_jo <= :as_of "
            "AND (r.date_reponse_jo IS NULL OR r.date_reponse_jo > :as_of)"
        )
    if kind == "edr":
        return (
            "q.date_publication_jo <= :as_of "
            "AND r.date_reponse_jo IS NOT NULL AND r.date_reponse_jo <= :as_of"
        )
    raise ValueError(f"unknown as-of pool kind: {kind!r}")


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratify_key(case: GroundTruthCase) -> tuple[str, int]:
    return (case.member.source, case.member.legislature)


def stratified_sample(
    cases: Sequence[GroundTruthCase],
    total: int,
    rng,
    *,
    floor_per_stratum: int = 20,
) -> list[GroundTruthCase]:
    """Sample ~``total`` cases, proportional to stratum size with a floor.

    The truth is heavily skewed towards old legislatures (AN-14 alone
    holds most of it), so a proportional-only sample would leave the
    legislature-17 strata — the ones describing what agents see today —
    too small to read.
    """
    strata: dict[tuple[str, int], list[GroundTruthCase]] = {}
    for case in cases:
        strata.setdefault(stratify_key(case), []).append(case)
    if not strata:
        return []

    population = len(cases)
    quotas = {
        key: max(floor_per_stratum, round(total * len(group) / population))
        for key, group in strata.items()
    }
    sampled: list[GroundTruthCase] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda c: c.question_id)
        take = min(quotas[key], len(group))
        sampled.extend(rng.sample(group, take))
    return sampled
