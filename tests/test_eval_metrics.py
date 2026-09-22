"""Benchmark metrics, checked against hand-computed values.

Every expected number below is written out by hand rather than derived
from the implementation — a metric that only agrees with itself proves
nothing about the report it produces.
"""

from __future__ import annotations

import pytest

from qe.eval.metrics import (
    Prediction,
    RankedOutcome,
    attainable_recall_at_k,
    calibration,
    first_hit_rank,
    hit_at_k,
    judge_false_drop_rate,
    majority_baseline,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    stage_recall,
    summarise_classification,
    summarise_ranked,
)

# Ranked list of 5, with true mates at positions 2 and 4.
RANKED = ("a", "b", "c", "d", "e")
MATES = frozenset({"b", "d"})


def outcome(qid, ranked, mates):
    return RankedOutcome(qid, tuple(ranked), frozenset(mates), ("AN", 17))


# ---------------------------------------------------------------------------
# Ranked-list primitives
# ---------------------------------------------------------------------------


def test_first_hit_rank_is_one_based():
    assert first_hit_rank(RANKED, MATES) == 2


def test_first_hit_rank_is_none_when_no_mate_is_present():
    assert first_hit_rank(RANKED, frozenset({"z"})) is None


@pytest.mark.parametrize("k,expected", [(1, False), (2, True), (5, True)])
def test_hit_at_k(k, expected):
    assert hit_at_k(RANKED, MATES, k) is expected


@pytest.mark.parametrize("k,expected", [(1, 0.0), (2, 0.5), (3, 0.5), (4, 1.0)])
def test_recall_at_k(k, expected):
    assert recall_at_k(RANKED, MATES, k) == expected


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, MATES) == 0.5
    assert reciprocal_rank(RANKED, frozenset({"z"})) == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, MATES, 4) == 0.5
    assert precision_at_k((), MATES, 4) == 0.0


def test_no_mates_scores_zero_rather_than_dividing_by_zero():
    assert recall_at_k(RANKED, frozenset(), 5) == 0.0
    assert attainable_recall_at_k(RANKED, frozenset(), 5) == 0.0


def test_attainable_recall_matches_plain_recall_when_slots_suffice():
    # 2 mates, 5 slots: nothing is out of reach, so both agree.
    assert attainable_recall_at_k(RANKED, MATES, 5) == recall_at_k(RANKED, MATES, 5)


def test_attainable_recall_does_not_punish_an_oversized_cluster():
    # 100 mates, the 3 reachable ones all ranked first: a perfect ranking.
    mates = frozenset(f"m{i}" for i in range(100))
    ranked = ("m0", "m1", "m2")
    assert attainable_recall_at_k(ranked, mates, 3) == 1.0
    assert recall_at_k(ranked, mates, 3) == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Ranked-list summary
# ---------------------------------------------------------------------------


def test_summarise_ranked_averages_over_questions():
    outcomes = [
        outcome("q1", ["a", "b"], {"a"}),  # first mate at rank 1
        outcome("q2", ["a", "b"], {"b"}),  # first mate at rank 2
    ]
    summary = summarise_ranked(outcomes, ks=(1, 2), min_results=2)
    assert summary.n == 2
    assert summary.hit_at[1] == 0.5
    assert summary.hit_at[2] == 1.0
    assert summary.mrr == pytest.approx(0.75)  # (1/1 + 1/2) / 2
    assert summary.median_first_hit_rank == 1.5


def test_summarise_ranked_counts_short_and_empty_lists():
    outcomes = [
        outcome("q1", [], {"a"}),
        outcome("q2", ["a"], {"a"}),
        outcome("q3", ["a", "b"], {"a"}),
    ]
    summary = summarise_ranked(outcomes, ks=(1,), min_results=2)
    assert summary.empty_rate == pytest.approx(1 / 3)
    assert summary.short_list_rate == pytest.approx(2 / 3)


def test_summarise_ranked_records_misses_in_the_histogram():
    summary = summarise_ranked([outcome("q1", ["x"], {"a"})], ks=(1,))
    assert summary.rank_histogram == {"not found": 1}
    assert summary.median_first_hit_rank is None


def test_summarise_ranked_bands_the_first_hit_rank():
    outcomes = [
        outcome("q1", ["m"], {"m"}),  # rank 1
        outcome("q2", ["x", "x", "m"], {"m"}),  # rank 3 -> band 2-3
    ]
    summary = summarise_ranked(outcomes, ks=(1,))
    assert summary.rank_histogram == {"1": 1, "2-3": 1}


def test_summarise_ranked_on_empty_input_does_not_divide_by_zero():
    assert summarise_ranked([]).n == 0


def test_stage_recall_reports_where_a_mate_is_lost():
    mates = frozenset({"m"})
    stages = {
        "retrieve": [RankedOutcome("q", ("m", "x"), mates, ("AN", 17))],
        "judge": [RankedOutcome("q", ("x",), mates, ("AN", 17))],
    }
    result = stage_recall(stages, k=20)
    assert result["retrieve"]["hit_at_k"] == 1.0
    assert result["judge"]["hit_at_k"] == 0.0


def test_judge_false_drop_rate_counts_only_removed_mates():
    mates = frozenset({"m1", "m2"})
    displayed = [RankedOutcome("q", ("m1", "m2", "x"), mates, ("AN", 17))]
    judged = [RankedOutcome("q", ("m1", "x"), mates, ("AN", 17))]
    result = judge_false_drop_rate(displayed, judged)
    assert result["mates_displayed"] == 2
    assert result["mates_dropped"] == 1
    assert result["false_drop_rate"] == 0.5


def test_judge_false_drop_skips_a_question_the_judge_never_answered():
    """A fail-open run must not be scored as 'the judge dropped nothing'."""
    displayed = [RankedOutcome("q", ("m",), frozenset({"m"}), ("AN", 17))]
    result = judge_false_drop_rate(displayed, [])
    assert result["mates_displayed"] == 0
    assert result["false_drop_rate"] == 0.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_against_a_hand_computed_confusion_matrix():
    # A: 2 correct. B: 1 predicted as A. C: 1 correct.
    predictions = [
        Prediction("q1", "A", ("A", "B", "C")),
        Prediction("q2", "A", ("A", "C", "B")),
        Prediction("q3", "B", ("A", "B", "C")),
        Prediction("q4", "C", ("C", "A", "B")),
    ]
    s = summarise_classification(predictions)
    assert s.top1_accuracy == 0.75
    assert s.top3_accuracy == 1.0
    # recall — A 2/2, B 0/1, C 1/1 -> mean 2/3
    assert s.balanced_accuracy == pytest.approx(2 / 3)
    # precision A = 2/3, recall A = 1 -> f1 = 0.8
    assert s.per_class["A"]["f1"] == pytest.approx(0.8)
    assert s.per_class["B"]["f1"] == 0.0
    assert s.per_class["C"]["f1"] == 1.0
    assert s.macro_f1 == pytest.approx((0.8 + 0.0 + 1.0) / 3)
    assert s.confusion["B"]["A"] == 1
    assert s.confusion["A"]["A"] == 2


def test_top3_accuracy_counts_a_truth_below_rank_one():
    s = summarise_classification([Prediction("q1", "C", ("A", "B", "C"))])
    assert s.top1_accuracy == 0.0
    assert s.top3_accuracy == 1.0


def test_top3_accuracy_ignores_a_truth_past_rank_three():
    s = summarise_classification([Prediction("q1", "D", ("A", "B", "C", "D"))])
    assert s.top3_accuracy == 0.0


def test_abstention_lowers_coverage_and_recall():
    predictions = [
        Prediction("q1", "A", ("A",)),
        Prediction("q2", "A", ()),
    ]
    s = summarise_classification(predictions)
    assert s.coverage == 0.5
    assert s.top1_accuracy == 0.5
    assert s.per_class["A"]["recall"] == 0.5


def test_a_label_only_ever_predicted_does_not_drag_the_macro_average():
    """Averaging over predicted-but-never-true labels invents a class.

    "Z" has no support: it is not a class the algorithm was asked to get
    right, so its empty recall must not enter balanced accuracy or
    macro-F1 — otherwise a single hallucinated label silently halves
    both headline numbers.
    """
    predictions = [
        Prediction("q1", "A", ("A",)),
        Prediction("q2", "A", ("Z",)),
    ]
    s = summarise_classification(predictions)
    assert s.per_class["Z"]["support"] == 0
    assert s.balanced_accuracy == 0.5  # recall(A) alone, not (0.5 + 0) / 2
    assert s.macro_f1 == pytest.approx(2 / 3)  # f1(A) alone


def test_single_class_predictions_do_not_crash():
    predictions = [
        Prediction("q1", "A", ("A",)),
        Prediction("q2", "B", ("A",)),
    ]
    s = summarise_classification(predictions)
    assert s.top1_accuracy == 0.5
    assert s.balanced_accuracy == 0.5


def test_majority_baseline_exposes_an_imbalanced_truth():
    predictions = [Prediction(f"q{i}", "A", ("B",)) for i in range(9)]
    predictions.append(Prediction("q9", "B", ("B",)))
    baseline = majority_baseline(predictions)
    # Always answering "A" scores 90% while learning nothing — which is
    # why accuracy alone cannot be the headline.
    assert baseline.top1_accuracy == 0.9
    assert baseline.balanced_accuracy == 0.5


def test_classification_on_empty_input_does_not_divide_by_zero():
    assert summarise_classification([]).n == 0


def test_perfect_confidence_has_a_zero_brier_score():
    predictions = [
        Prediction("q1", "A", ("A",), confidence=1.0),
        Prediction("q2", "A", ("B",), confidence=0.0),
    ]
    assert calibration(predictions)["brier"] == pytest.approx(0.0)


def test_overconfidence_shows_up_in_the_reliability_curve():
    predictions = [Prediction(f"q{i}", "A", ("B",), confidence=0.95) for i in range(10)]
    result = calibration(predictions)
    assert result["brier"] == pytest.approx(0.9025)
    bin_ = result["curve"][0]
    assert bin_["mean_confidence"] == pytest.approx(0.95)
    assert bin_["observed_accuracy"] == 0.0


def test_calibration_ignores_predictions_without_a_confidence():
    assert calibration([Prediction("q1", "A", ("A",))])["n"] == 0


# ---------------------------------------------------------------------------
# Attribution vote (mirrored from qe-front's knn-vote.ts)
# ---------------------------------------------------------------------------


def neighbour(key: str, similarity: float, label: str = ""):
    from qe.eval.attribution_replay import Neighbour

    return Neighbour(key=key, similarity=similarity, label=label)


def test_proximity_vote_lets_one_close_neighbour_beat_several_distant_ones():
    """The property the exponential weighting exists for.

    A plain sum of similarities would hand the vote to whichever class
    has the most labelled questions; at temperature 0.03 a neighbour
    0.1 further away is worth e^-3.3 ≈ 3.6% of the closest one, so three
    distant B's still lose to one close A.
    """
    from qe.eval.attribution_replay import proximity_vote

    votes = proximity_vote(
        [
            neighbour("A", 0.90),
            neighbour("B", 0.80),
            neighbour("B", 0.80),
            neighbour("B", 0.80),
        ]
    )
    assert [v.key for v in votes] == ["A", "B"]


def test_proximity_vote_sums_within_a_key():
    from qe.eval.attribution_replay import proximity_vote

    votes = proximity_vote([neighbour("A", 0.9), neighbour("A", 0.9)])
    assert len(votes) == 1
    assert votes[0].vote == pytest.approx(2.0)  # both at the best similarity


def test_proximity_vote_normalises_against_the_closest_neighbour():
    """Weights are relative, so shifting every similarity changes nothing."""
    from qe.eval.attribution_replay import proximity_vote

    a = proximity_vote([neighbour("A", 0.9), neighbour("B", 0.87)])
    b = proximity_vote([neighbour("A", 0.5), neighbour("B", 0.47)])
    assert [v.key for v in a] == [v.key for v in b]
    assert a[1].vote == pytest.approx(b[1].vote)


def test_proximity_vote_of_nothing_is_empty():
    from qe.eval.attribution_replay import proximity_vote

    assert proximity_vote([]) == []


def test_vote_shares_sum_to_one():
    from qe.eval.attribution_replay import Vote, vote_shares

    shares = vote_shares([Vote("A", 3.0), Vote("B", 1.0)])
    assert shares == pytest.approx([0.75, 0.25])
    assert sum(shares) == pytest.approx(1.0)


def test_vote_shares_of_nothing_do_not_divide_by_zero():
    from qe.eval.attribution_replay import Vote, vote_shares

    assert vote_shares([]) == []
    assert vote_shares([Vote("A", 0.0)]) == [0.0]
