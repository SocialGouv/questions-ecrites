"""Argument guards of the attribution quality benchmark.

The sibling-exclusion run is only interpretable when both arms over-fetch
identically, and nothing in the report would reveal that they did not.
"""

from __future__ import annotations

import sys

import pytest

from scripts.bench_quality_attribution import SIBLING_OVERFETCH, main


def _run(argv: list[str]) -> str:
    sys.argv = ["bench_quality_attribution.py", *argv]
    with pytest.raises(SystemExit) as exc:
        main()
    return str(exc.value)


def test_exclude_siblings_without_an_explicit_knn_is_rejected(capsys):
    _run(
        [
            "--feature",
            "direction",
            "--exclude-siblings",
            "--output",
            "/dev/null",
        ]
    )
    err = capsys.readouterr().err
    assert "--exclude-siblings needs an explicit --knn" in err
    assert str(SIBLING_OVERFETCH) in err


def test_gt_all_is_rejected_for_direction(capsys):
    _run(["--feature", "direction", "--gt", "all", "--output", "/dev/null"])
    assert "--gt all applies to bureau only" in capsys.readouterr().err
