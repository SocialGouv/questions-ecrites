"""Ground-truth construction for the quality benchmark.

The cases that matter are the ones where the two chambers' encodings and
the two features' definitions could be confused with each other.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from qe.eval.ground_truth import (
    AnswerCluster,
    ClusterMember,
    GroundTruthCase,
    allotissement_cases,
    as_of_predicate,
    cases_for_pool,
    edr_cases,
    stratified_sample,
)


def member(
    qid: str,
    *,
    answered: date,
    published: date = date(2024, 1, 1),
    source: str = "AN",
    legislature: int = 17,
) -> ClusterMember:
    return ClusterMember(
        question_id=qid,
        source=source,
        legislature=legislature,
        date_publication_jo=published,
        date_reponse_jo=answered,
    )


def case(qid: str, *, legislature: int = 17) -> GroundTruthCase:
    return GroundTruthCase(
        member=member(qid, answered=date(2024, 6, 1), legislature=legislature),
        mate_ids=frozenset({"other"}),
    )


# An AN joint answer: one reponse row, several questions, one JO date.
AN_JOINT = AnswerCluster(
    text_hash="h-an",
    members=(
        member("AN-17-QE-1", answered=date(2024, 6, 1)),
        member("AN-17-QE-2", answered=date(2024, 6, 1)),
        member("AN-17-QE-3", answered=date(2024, 6, 1)),
    ),
)

# The SENAT encoding of the same thing: distinct reponse rows carrying
# identical text, published on one JO date.
SENAT_JOINT = AnswerCluster(
    text_hash="h-senat",
    members=(
        member("SENAT-17-QE-1", answered=date(2024, 6, 1), source="SENAT"),
        member("SENAT-17-QE-2", answered=date(2024, 6, 1), source="SENAT"),
    ),
)

# A reuse: the same answer text published again, later.
REUSE = AnswerCluster(
    text_hash="h-reuse",
    members=(
        member("AN-17-QE-10", answered=date(2023, 1, 10), published=date(2022, 6, 1)),
        member("AN-17-QE-11", answered=date(2024, 5, 20), published=date(2024, 2, 1)),
    ),
)


def test_allotissement_covers_both_chamber_encodings():
    cases = allotissement_cases([AN_JOINT, SENAT_JOINT])
    by_id = {c.question_id: c for c in cases}
    assert by_id["AN-17-QE-1"].mate_ids == {"AN-17-QE-2", "AN-17-QE-3"}
    # The SENAT form has no shared reponse_id at all — a reponse_id ground
    # truth scores zero here while still reading as a whole-system number.
    assert by_id["SENAT-17-QE-1"].mate_ids == {"SENAT-17-QE-2"}


def test_a_later_reuse_is_not_an_allotissement():
    assert allotissement_cases([REUSE]) == []


def test_edr_mates_are_only_answers_that_already_existed():
    cases = edr_cases([REUSE])
    assert len(cases) == 1
    # Only the later question has something to reuse, and its mate is the
    # earlier one — never the other way round.
    assert cases[0].question_id == "AN-17-QE-11"
    assert cases[0].mate_ids == {"AN-17-QE-10"}


@pytest.mark.parametrize("require_available", [True, False])
def test_same_date_cluster_is_allotissement_and_never_edr(require_available):
    """The split that keeps the two features' truths disjoint.

    Both flag values are checked on purpose: with
    ``require_available_at_publication`` on, the publication-date filter
    can hide a broken date comparison by rejecting the leaked mates for
    an unrelated reason.
    """
    clusters = [AN_JOINT, SENAT_JOINT]
    assert edr_cases(clusters, require_available_at_publication=require_available) == []
    assert len(allotissement_cases(clusters)) == 5


def test_edr_excludes_an_answer_published_after_the_source_arrived():
    # Answered before the source's own answer, but only after the source
    # question was published: an agent drafting at t could not find it.
    cluster = AnswerCluster(
        text_hash="h",
        members=(
            member("A", answered=date(2024, 3, 1), published=date(2023, 1, 1)),
            member("B", answered=date(2024, 9, 1), published=date(2024, 1, 1)),
        ),
    )
    assert edr_cases([cluster]) == []
    relaxed = edr_cases([cluster], require_available_at_publication=False)
    assert relaxed[0].mate_ids == {"A"}


def test_allotissement_excludes_a_mate_published_after_the_source():
    # Co-answered on one JO date, but B only reached the JO after A: the
    # as-of-t pool (date_publication_jo <= as_of) cannot contain B when
    # searching from A, so keeping it would score an unwinnable miss.
    cluster = AnswerCluster(
        text_hash="h",
        members=(
            member("A", answered=date(2024, 6, 1), published=date(2023, 1, 1)),
            member("B", answered=date(2024, 6, 1), published=date(2024, 1, 1)),
        ),
    )
    cases = allotissement_cases([cluster])
    assert [c.question_id for c in cases] == ["B"]
    assert cases[0].mate_ids == {"A"}
    relaxed = allotissement_cases([cluster], require_published_at_source=False)
    assert {c.question_id: c.mate_ids for c in relaxed} == {
        "A": {"B"},
        "B": {"A"},
    }


def test_as_of_allotissement_pool_admits_the_not_yet_answered():
    sql = as_of_predicate("allotissement")
    assert "date_reponse_jo IS NULL OR r.date_reponse_jo > :as_of" in sql


def test_as_of_edr_pool_requires_an_answer_by_t():
    sql = as_of_predicate("edr")
    assert "r.date_reponse_jo IS NOT NULL" in sql
    assert "r.date_reponse_jo <= :as_of" in sql


def test_as_of_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        as_of_predicate("similar")


def test_stratified_sample_keeps_a_readable_floor_for_small_strata():
    # 400 AN-14 cases against 5 AN-17 ones: proportional-only sampling
    # would round the AN-17 stratum away entirely.
    cases = [case(f"old-{i}", legislature=14) for i in range(400)]
    cases += [case(f"new-{i}", legislature=17) for i in range(5)]
    sampled = stratified_sample(
        cases, total=50, rng=random.Random(0), floor_per_stratum=3
    )
    assert {c.member.legislature for c in sampled} == {14, 17}
    assert sum(1 for c in sampled if c.member.legislature == 17) == 3


def test_stratified_sample_floor_never_exceeds_the_stratum():
    sampled = stratified_sample(
        [case("only")], total=50, rng=random.Random(0), floor_per_stratum=20
    )
    assert len(sampled) == 1


def test_stratified_sample_is_reproducible_for_a_fixed_seed():
    cases = [case(f"q-{i}") for i in range(100)]
    first = stratified_sample(cases, 20, random.Random(1234))
    second = stratified_sample(cases, 20, random.Random(1234))
    assert [c.question_id for c in first] == [c.question_id for c in second]


def test_edr_relaxed_truth_is_a_superset_of_the_as_of_truth():
    """The two variants exist to match two different candidate pools.

    An unrestricted search reaches mates the as-of-t filter drops, so
    scoring such a run against the as-of-t truth turns retrieved mates
    into misses. Measured on preprod, 72 % (allotissement) and 80 % (EDR)
    of the dropped mates are in fact retrieved there.
    """
    cluster = AnswerCluster(
        text_hash="h",
        members=(
            member("A", answered=date(2024, 3, 1), published=date(2023, 1, 1)),
            member("B", answered=date(2024, 9, 1), published=date(2024, 1, 1)),
        ),
    )
    assert edr_cases(cluster_list := [cluster]) == []
    relaxed = edr_cases(cluster_list, require_available_at_publication=False)
    assert {c.question_id: c.mate_ids for c in relaxed} == {"B": frozenset({"A"})}


def test_allotissement_relaxed_truth_is_a_superset_of_the_as_of_truth():
    cluster = AnswerCluster(
        text_hash="h",
        members=(
            member("A", answered=date(2024, 6, 1), published=date(2023, 1, 1)),
            member("B", answered=date(2024, 6, 1), published=date(2024, 1, 1)),
        ),
    )
    strict = {c.question_id: c.mate_ids for c in allotissement_cases([cluster])}
    relaxed = {
        c.question_id: c.mate_ids
        for c in allotissement_cases([cluster], require_published_at_source=False)
    }
    assert strict == {"B": frozenset({"A"})}
    assert relaxed == {"A": frozenset({"B"}), "B": frozenset({"A"})}
    for qid, mates in strict.items():
        assert mates <= relaxed[qid]


CROSS_DATE = AnswerCluster(
    text_hash="h-cross",
    members=(
        member("A", answered=date(2024, 6, 1), published=date(2023, 1, 1)),
        member("B", answered=date(2024, 6, 1), published=date(2024, 1, 1)),
    ),
)
LATER_REUSE = AnswerCluster(
    text_hash="h-later",
    members=(
        member("C", answered=date(2024, 3, 1), published=date(2023, 1, 1)),
        member("D", answered=date(2024, 9, 1), published=date(2024, 1, 1)),
    ),
)


@pytest.mark.parametrize(
    "feature,clusters,strict_ids,relaxed_ids",
    [
        ("allotissement", [CROSS_DATE], {"B"}, {"A", "B"}),
        ("edr", [LATER_REUSE], set(), {"D"}),
    ],
)
def test_cases_for_pool_widens_the_truth_for_an_unrestricted_pool(
    feature, clusters, strict_ids, relaxed_ids
):
    """The wiring, not just the underlying filters.

    An unrestricted run scored against as-of-t truth counts retrieved
    mates as misses, so the pool has to reach the variant selection.
    """
    strict = cases_for_pool(clusters, feature, "as-of-t")
    relaxed = cases_for_pool(clusters, feature, "unrestricted")
    assert {c.question_id for c in strict} == strict_ids
    assert {c.question_id for c in relaxed} == relaxed_ids


def test_cases_for_pool_rejects_an_unknown_pool_or_feature():
    with pytest.raises(ValueError):
        cases_for_pool([], "allotissement", "today")
    with pytest.raises(ValueError):
        cases_for_pool([], "similar", "as-of-t")
