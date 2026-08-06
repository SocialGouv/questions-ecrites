"""Tests for the attribution relevance blend.

`_to_relevance` (api/questions.py) blends two signals — see root CLAUDE.md,
"Attribution relevance scoring": 70% absolute (`sigmoid(agg_score) * 100`)
plus 30% pool-median-centred relative adjustment. This is the single most
"clever" piece of business logic in the API response and has no coverage
today; a silent regression here would make every returned relevance score
wrong without any error being raised.
"""

from __future__ import annotations

import math

import pytest

from api.questions import _to_relevance


def _sigmoid_pct(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def test_single_candidate_pool_falls_back_to_absolute_only():
    """With fewer than 2 pool scores there's no median to compare against,
    so relevance is just the raw sigmoid of the candidate's own score."""
    assert _to_relevance(5.0, [5.0]) == round(_sigmoid_pct(5.0), 1)
    assert _to_relevance(5.0, []) == round(_sigmoid_pct(5.0), 1)


def test_score_at_pool_median_matches_pure_absolute():
    """When a candidate's score equals the pool median, the relative
    component contributes exactly the median's absolute value, so the blend
    collapses to the same value as the absolute-only score."""
    pool = [1.0, 5.0, 9.0]
    assert _to_relevance(5.0, pool) == round(_sigmoid_pct(5.0), 1)


def test_well_separated_scores_produce_a_visible_gap():
    """A candidate well above the pool median and one well below it should
    end up far apart, not compressed together."""
    pool = [-10.0, 0.0, 10.0]
    high = _to_relevance(10.0, pool)
    low = _to_relevance(-10.0, pool)

    assert high - low > 50
    assert high > 90
    assert low < 10


def test_relative_component_clamps_within_0_100():
    """A candidate far from the pool median would push the raw linear
    relative term outside [0, 100]; the final blended relevance must stay
    within bounds regardless. Uses rerank-logit-scale values (tens, not
    thousands) since sigmoid overflows for extreme magnitudes."""
    pool = [0.0, 0.0, 20.0]

    assert 0.0 <= _to_relevance(20.0, pool) <= 100.0
    assert 0.0 <= _to_relevance(-20.0, pool) <= 100.0


@pytest.mark.parametrize("agg_score,pool", [(0.0, [0.0, 0.0]), (3.0, [3.0, 3.0, 3.0])])
def test_tightly_clustered_scores_stay_close_to_each_other(agg_score, pool):
    """Sanity check on the 'tightly clustered raw scores -> nearly identical
    relevance' guarantee documented on _to_relevance."""
    result = _to_relevance(agg_score, pool)
    assert result == pytest.approx(round(_sigmoid_pct(agg_score), 1), abs=0.1)


@pytest.mark.parametrize("agg_score", [-1000.0, 1000.0])
def test_extreme_agg_score_does_not_overflow(agg_score):
    """`_sigmoid_pct`'s naive `math.exp(-x)` raises OverflowError for |x| well
    past 700; `_to_relevance` must degrade gracefully to 0/100 instead of
    propagating that crash, even though real Albert rerank logits never
    reach this magnitude."""
    assert _to_relevance(agg_score, [agg_score]) in (0.0, 100.0)


def test_extreme_pool_median_does_not_overflow():
    """The pool-median branch calls the same sigmoid on `median_score`; a
    pool containing an extreme outlier must not crash either."""
    result = _to_relevance(5.0, [5.0, -1000.0, 1000.0])
    assert 0.0 <= result <= 100.0
