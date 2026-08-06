"""Tests for deterministic vector-row ID hashing.

`stable_question_point_id` is re-implemented independently in
`qe-front/src/lib/similarity/point-id.ts` (`questionPointId`) so the web app
can resolve a question's vector row without a DB round-trip. The pinned
golden values here must match the pinned values in that TypeScript test —
any drift between the two implementations would silently break similarity
joins between the two systems.
"""

from __future__ import annotations

from qe.hashing import stable_answer_point_id, stable_question_point_id


def test_stable_question_point_id_is_deterministic():
    assert stable_question_point_id("QANR5L17QE1") == stable_question_point_id(
        "QANR5L17QE1"
    )


def test_stable_question_point_id_differs_by_input():
    assert stable_question_point_id("QANR5L17QE1") != stable_question_point_id(
        "QANR5L17QE42"
    )


def test_stable_question_point_id_golden_value():
    """Pinned against qe-front's point-id.test.ts — keep both in sync."""
    assert (
        stable_question_point_id("QANR5L17QE1")
        == "b9a8486a-597b-54cd-0678-5f265ac5aa8c"
    )
    assert (
        stable_question_point_id("QANR5L17QE42")
        == "1ce8db0d-d9f0-c63a-aba3-d470c9a6eecb"
    )


def test_stable_answer_point_id_is_deterministic_and_distinct():
    assert stable_answer_point_id("REP123") == stable_answer_point_id("REP123")
    assert stable_answer_point_id("REP123") != stable_answer_point_id("REP456")


def test_question_and_answer_ids_do_not_collide_for_same_string():
    """Both functions hash the raw string the same way, so passing the same
    string to each yields the same UUID — callers must not mix up which ID
    space a string belongs to."""
    assert stable_question_point_id("SAME") == stable_answer_point_id("SAME")
