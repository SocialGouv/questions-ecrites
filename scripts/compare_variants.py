#!/usr/bin/env python3
"""Compare multiple eval variants side-by-side as a markdown table.

Reads JSON reports produced by `eval_question_similarity.py` and prints a
condensed comparison, one row per variant. Meant to be run after all
variant embeddings + evals are done.

Usage:
    poetry run python scripts/compare_variants.py \\
        data/eval_baseline.json:baseline \\
        data/eval_q_only.json:q_only \\
        data/eval_q_only_no_rappel.json:q_only_no_rappel

Output (stdout, markdown):

    | Variante              | Recall@0.8 | Hit@0.8 | n queries | delta Recall vs baseline |
    |-----------------------|-----------:|--------:|----------:|---------------------:|
    | baseline              |      0.612 |   0.834 |    11 032 |                    — |
    | q_only                |      0.687 |   0.891 |    11 032 |               +0.075 |
    | q_only_no_rappel      |      0.693 |   0.895 |    10 891 |               +0.081 |

The baseline is always the FIRST variant listed. Diffs are printed against it.

Colour: emits ANSI green/red on Δ for readability in the terminal — suppress
with --no-color for piping into a file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass(frozen=True)
class Variant:
    label: str
    path: Path
    recall: float
    hit: float
    n_queries: int
    score_threshold: float


def _parse_arg_pair(raw: str) -> tuple[Path, str]:
    """`path:label` or just `path` (label = filename stem).

    On Windows the path may itself start with a drive letter and a colon
    (`C:\\…`), so we look for a colon that isn't at position 1 (drive)
    before splitting.
    """
    idx = raw.rfind(":")
    # A colon at position 1 is the drive letter separator, not our sep.
    if idx > 1:
        return Path(raw[:idx]), raw[idx + 1:]
    p = Path(raw)
    return p, p.stem


def _load_variant(path: Path, label: str) -> Variant:
    with path.open(encoding="utf-8") as f:
        report = json.load(f)
    s = report["summary"]
    return Variant(
        label=label,
        path=path,
        recall=s["recall_at_threshold"],
        hit=s["hit_at_threshold"],
        n_queries=s["total_query_questions"],
        score_threshold=s["score_threshold"],
    )


def _fmt_diff(diff: float, use_color: bool) -> str:
    sign = "+" if diff >= 0 else ""
    txt = f"{sign}{diff:.3f}"
    if not use_color:
        return txt
    # Green when the variant beats baseline, red when worse.
    if diff > 0.001:
        return f"\033[32m{txt}\033[0m"
    if diff < -0.001:
        return f"\033[31m{txt}\033[0m"
    return txt


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    padded = []
    for cell, w in zip(cells, widths, strict=True):
        # Strip ANSI codes first — otherwise "\033[32m+0.075\033[0m"
        # starts with '\033' instead of '+' and never matches, and
        # `str.rjust` counts the escape bytes as visible characters.
        bare = _ANSI_RE.sub("", cell)
        is_numeric = (
            bare.strip("+- ").replace(".", "").replace(",", "").isdigit()
            and any(c.isdigit() for c in bare)
        ) or (
            bare.startswith(("+", "-")) and any(c.isdigit() for c in bare)
        )
        # Compute padding on the visible length so ANSI-coloured cells
        # still line up with plain ones.
        pad = max(0, w - len(bare))
        padded.append((" " * pad) + cell if is_numeric else cell + (" " * pad))
    return "| " + " | ".join(padded) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eval variant reports as a markdown table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "variants",
        nargs="+",
        help="path[:label] pairs. First is the baseline.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour codes.",
    )
    args = parser.parse_args()

    variants: list[Variant] = []
    for raw in args.variants:
        path, label = _parse_arg_pair(raw)
        if not path.is_file():
            sys.exit(f"error: {path} not found")
        variants.append(_load_variant(path, label))

    # Sanity: all variants must share the same score_threshold, otherwise
    # the comparison is apples-to-oranges.
    thresholds = {v.score_threshold for v in variants}
    if len(thresholds) > 1:
        sys.exit(
            f"error: variants use different score_thresholds ({sorted(thresholds)}) — "
            "re-run evals with the same --score-threshold."
        )
    threshold = thresholds.pop()

    baseline = variants[0]
    use_color = not args.no_color and sys.stdout.isatty()

    header = [
        "Variante",
        f"Recall@{threshold}",
        f"Hit@{threshold}",
        "n queries",
        "delta Recall vs baseline",
    ]
    rows = [header]
    for v in variants:
        diff = v.recall - baseline.recall
        rows.append([
            v.label,
            f"{v.recall:.3f}",
            f"{v.hit:.3f}",
            f"{v.n_queries:,}".replace(",", " "),
            "-" if v is baseline else _fmt_diff(diff, use_color),
        ])

    # Compute column widths on the ANSI-stripped text (same pattern as
    # `_fmt_row` uses for numeric detection).
    def _visible_len(s: str) -> int:
        return len(_ANSI_RE.sub("", s))

    widths = [max(_visible_len(r[c]) for r in rows) for c in range(len(header))]

    print(_fmt_row(rows[0], widths))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows[1:]:
        print(_fmt_row(row, widths))


if __name__ == "__main__":
    main()
