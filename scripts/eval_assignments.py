#!/usr/bin/env python
"""Evaluate assignment quality against ground-truth attributions.

Metrics (per question and aggregate):
  - H@1   (Hit@1):  1 if the top-ranked user is an attributed user, 0 otherwise.
  - H@3   (Hit@3):  1 if any attributed user appears in the top 3 ranked users, 0 otherwise.
  - RR    (Reciprocal Rank): 1/rank of the first attributed user in the ranking.
            RR=1.0 means the right person was #1; RR=0.5 means they were #2, etc.
            The aggregate MRR (Mean RR) summarises overall ranking quality in a single number.
  - Share (Score share): fraction of the total cumulative score mass that falls on attributed
            users. Scores are normalised to sum to 1 per question before computing this.
            With 4 experts, a random baseline is ~0.25 (single attribution) or ~0.50 (two).
            A high Share means the system confidently favoured the right people.
  - Ratio (Score ratio): best attributed user's score divided by the top score overall.
            Ratio=1.0 means the right person leads; Ratio=0.5 means the right person scored
            only half of whoever ranked first. Distinguishes near-misses from confident errors.

Output is written to an Excel file. Pass --format csv or --format json for alternative outputs.

Usage:
  python scripts/eval_assignments.py
  python scripts/eval_assignments.py --summary data/assignments_summary.json \
      --attributions data/attributions.json --output data/eval_results.xlsx
  python scripts/eval_assignments.py --format csv --output data/eval_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

DEFAULT_SUMMARY = Path("data/assignments_summary.json")
DEFAULT_ATTRIBUTIONS = Path("data/attributions.json")
DEFAULT_OUTPUT = Path("data/eval_results.xlsx")
DEFAULT_FORMAT = "xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate assignment quality against ground-truth attributions."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to assignments_summary.json.",
    )
    parser.add_argument(
        "--attributions",
        type=Path,
        default=DEFAULT_ATTRIBUTIONS,
        help="Path to attributions.json (ground truth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write results (extension inferred from --format).",
    )
    parser.add_argument(
        "--format",
        choices=["xlsx", "csv", "json"],
        default=DEFAULT_FORMAT,
        help="Output format: xlsx (default), csv, or json.",
    )
    return parser.parse_args()


def reciprocal_rank(ranked_users: list[str], attributed: set[str]) -> float:
    for rank, user in enumerate(ranked_users, start=1):
        if user in attributed:
            return 1.0 / rank
    return 0.0


def attributed_score_share(
    scores_by_user: dict[str, float], attributed: set[str]
) -> float:
    """Fraction of total score mass on attributed users (normalized to [0, 1])."""
    total = sum(scores_by_user.values())
    if total <= 1e-9:
        return 0.0
    return sum(scores_by_user.get(u, 0.0) for u in attributed) / total


def score_ratio(scores_by_user: dict[str, float], attributed: set[str]) -> float:
    """Best attributed user's score divided by the top score overall."""
    top_score = max(scores_by_user.values(), default=0.0)
    if top_score <= 1e-9:
        return 0.0
    best_attributed = max((scores_by_user.get(u, 0.0) for u in attributed), default=0.0)
    return best_attributed / top_score


def build_results(
    summary: dict[str, list[dict]],
    attributions: dict[str, list[str]],
) -> tuple[list[dict], list[str]]:
    """Compute per-question metrics. Returns (rows, skipped)."""
    rows: list[dict] = []
    skipped: list[str] = []

    for question, attributed_users in attributions.items():
        if question not in summary:
            skipped.append(question)
            continue

        entries = summary[question]
        ranked = [entry["user"] for entry in entries]
        scores_by_user = {entry["user"]: entry["cumulative_score"] for entry in entries}
        attributed_set = set(attributed_users)

        rows.append(
            {
                "question": question,
                "attributed": ", ".join(attributed_users),
                "predicted_top3": ", ".join(ranked[:3]),
                "H@1": 1.0 if ranked and ranked[0] in attributed_set else 0.0,
                "H@3": 1.0 if any(u in attributed_set for u in ranked[:3]) else 0.0,
                "RR": reciprocal_rank(ranked, attributed_set),
                "Share": attributed_score_share(scores_by_user, attributed_set),
                "Ratio": score_ratio(scores_by_user, attributed_set),
            }
        )

    return rows, skipped


def write_results(df: pl.DataFrame, output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "xlsx":
        df.write_excel(output)
    elif fmt == "csv":
        df.write_csv(output)
    elif fmt == "json":
        df.write_json(output)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    args = parse_args()

    summary: dict[str, list[dict]] = json.loads(
        args.summary.read_text(encoding="utf-8")
    )
    attributions: dict[str, list[str]] = json.loads(
        args.attributions.read_text(encoding="utf-8")
    )

    # Normalize attribution keys and values to lowercase (defensive, in case)
    attributions = {k.lower(): [u.lower() for u in v] for k, v in attributions.items()}

    rows, skipped = build_results(summary, attributions)

    n = len(rows)
    if n == 0:
        print("No questions matched between summary and attributions.")
        return

    df = pl.DataFrame(rows)

    # Aggregate row
    aggregate = pl.DataFrame(
        [
            {
                "question": "AGGREGATE",
                "attributed": "",
                "predicted_top3": "",
                "H@1": df["H@1"].mean(),
                "H@3": df["H@3"].mean(),
                "RR": df["RR"].mean(),
                "Share": df["Share"].mean(),
                "Ratio": df["Ratio"].mean(),
            }
        ]
    )
    df_with_agg = pl.concat([df, aggregate])

    # --- Console output ---
    print("Metrics (per question and aggregate):")
    print(
        "  - H@1   (Hit@1):  1 if the top-ranked user is an attributed user, 0 otherwise."
    )
    print(
        "  - H@3   (Hit@3):  1 if any attributed user appears in the top 3 ranked users, 0 otherwise."
    )
    print(
        "  - RR    (Reciprocal Rank): 1/rank of the first attributed user in the ranking."
    )
    print(
        "            RR=1.0 means the right person was #1; RR=0.5 means they were #2, etc."
    )
    print(
        "            The aggregate MRR (Mean RR) summarises overall ranking quality in a single number."
    )
    print(
        "  - Share (Score share): fraction of the total cumulative score mass that falls on attributed"
    )
    print(
        "            users. Scores are normalised to sum to 1 per question before computing this."
    )
    print(
        "            With 4 experts, a random baseline is ~0.25 (single attribution) or ~0.50 (two)."
    )
    print(
        "            A high Share means the system confidently favoured the right people."
    )
    print(
        "  - Ratio (Score ratio): best attributed user's score divided by the top score overall."
    )
    print(
        "            Ratio=1.0 means the right person leads; Ratio=0.5 means the right person scored"
    )
    print(
        "            only half of whoever ranked first. Distinguishes near-misses from confident errors."
    )
    print()
    col_q = 24
    col_attr = 28
    col_pred = 28
    col_h1 = 5
    col_h3 = 5
    col_rr = 6
    col_share = 7
    col_ratio = 7

    header = (
        f"{'Question':<{col_q}}"
        f"{'Attributed':<{col_attr}}"
        f"{'Predicted (top 3)':<{col_pred}}"
        f"{'H@1':>{col_h1}}"
        f"{'H@3':>{col_h3}}"
        f"{'RR':>{col_rr}}"
        f"{'Share':>{col_share}}"
        f"{'Ratio':>{col_ratio}}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['question']:<{col_q}}"
            f"{row['attributed']:<{col_attr}}"
            f"{row['predicted_top3']:<{col_pred}}"
            f"{'Y' if row['H@1'] else 'N':>{col_h1}}"
            f"{'Y' if row['H@3'] else 'N':>{col_h3}}"
            f"{row['RR']:>{col_rr}.2f}"
            f"{row['Share']:>{col_share}.2f}"
            f"{row['Ratio']:>{col_ratio}.2f}"
        )

    print("-" * len(header))
    agg = aggregate.row(0, named=True)
    print(
        f"{'AGGREGATE':.<{col_q + col_attr + col_pred}}"
        f"{agg['H@1']:>{col_h1}.2f}"
        f"{agg['H@3']:>{col_h3}.2f}"
        f"{agg['RR']:>{col_rr}.2f}"
        f"{agg['Share']:>{col_share}.2f}"
        f"{agg['Ratio']:>{col_ratio}.2f}"
    )
    print(f"\n{n} questions evaluated", end="")
    if skipped:
        print(
            f", {len(skipped)} skipped (not in summary): {', '.join(skipped)}", end=""
        )
    print()

    # --- File output ---
    write_results(df_with_agg, args.output, args.format)
    print(f"\nResults written to {args.output} ({args.format})")


if __name__ == "__main__":
    main()
