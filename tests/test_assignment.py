"""Tests for per-office score aggregation.

`aggregate_matches` (qe/assignment.py) is the sum-of-top-N-per-office logic
that turns a flat list of reranked chunk matches into the per-office scores
the API returns. It's pure dict-in/dict-out with no I/O, but has no test
coverage despite feeding directly into every attribution response.
"""

from __future__ import annotations

from qe.assignment import aggregate_matches


def _match(office_id: str, score: float) -> dict:
    return {"office_id": office_id, "score": score}


def test_caps_chunks_per_office_and_sums_only_the_kept_top_n():
    matches = [
        _match("office-a", 0.9),
        _match("office-a", 0.8),
        _match("office-a", 0.1),  # should be dropped by the cap
    ]

    kept, score_by_office = aggregate_matches(matches, max_chunks_per_office=2)

    assert len(kept) == 2
    assert {m["score"] for m in kept} == {0.9, 0.8}
    assert score_by_office["office-a"] == 0.9 + 0.8


def test_skips_matches_with_missing_or_empty_office_id():
    matches = [
        _match("office-a", 0.5),
        {"office_id": None, "score": 0.9},
        {"office_id": "", "score": 0.9},
    ]

    kept, score_by_office = aggregate_matches(matches, max_chunks_per_office=2)

    assert len(kept) == 1
    assert list(score_by_office.keys()) == ["office-a"]


def test_kept_matches_sorted_descending_with_sequential_rank():
    matches = [
        _match("office-a", 0.3),
        _match("office-b", 0.9),
        _match("office-c", 0.6),
    ]

    kept, _ = aggregate_matches(matches, max_chunks_per_office=2)

    assert [m["score"] for m in kept] == [0.9, 0.6, 0.3]
    assert [m["rank"] for m in kept] == [1, 2, 3]


def test_offices_are_aggregated_independently():
    matches = [
        _match("office-a", 0.9),
        _match("office-a", 0.8),
        _match("office-b", 0.5),
    ]

    _, score_by_office = aggregate_matches(matches, max_chunks_per_office=2)

    assert score_by_office["office-a"] == 0.9 + 0.8
    assert score_by_office["office-b"] == 0.5
