#!/usr/bin/env python3
"""Quality benchmark: allotissement and EDR suggestion lists.

Runs in two halves, because the ground truth and the metrics belong in
this repo while the pipeline being measured lives in ``qe-front``:

``--emit-cases``  build the ground truth, sample it, and write the cases
                  to JSON. No API calls.
``--score``       read the per-stage ranked lists produced by
                  ``qe-front/scripts/bench-quality-similar.ts`` and turn
                  them into the report.

Splitting it this way keeps the measured pipeline the *real* one:
the TypeScript runner imports ``rerank``, ``judgeCandidates`` and
``computeDisplayCount`` from production modules instead of this repo
re-implementing them, which is how ``eval_realistic_encours.py`` and the
constants in ``docs/direction-bureau-attribution.md`` drifted apart.

Ground truth is the answer-text cluster (see ``qe.eval.ground_truth``):
same answer text on the same JO date is an allotissement, on a later
date it is an EDR reuse. Clustering on ``reponse_id`` instead would
score the Assemblée alone — the Sénat never shares one.

Usage:
    poetry run python scripts/bench_quality_similar.py --emit-cases \\
        --feature allotissement --sample 400 --output data/bench/cases_allot.json
    poetry run python scripts/bench_quality_similar.py --score \\
        --runs data/bench/runs_allot.json --output data/bench/allot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from sqlalchemy import text as sqltext

from qe import db
from qe.eval.ground_truth import (
    cases_for_pool,
    fetch_answer_clusters,
    stratified_sample,
)
from qe.eval.metrics import (
    RankedOutcome,
    judge_false_drop_rate,
    size_band,
    stage_recall,
    summarise_ranked,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Order matters: the report reads as a funnel, and each stage's recall is
# only interpretable next to the one before it.
STAGE_ORDER = ("retrieve", "rerank", "displayed", "judged")

# Above this share of fail-open runs the judge stage describes the
# judge's availability, not its quality, and must not be published.
FAIL_OPEN_LIMIT = 0.15


def emit_cases(args: argparse.Namespace) -> dict:
    with db.get_session() as session:
        session.execute(sqltext("SET TRANSACTION READ ONLY"))
        clusters = fetch_answer_clusters(session)
        logger.info("answer-text clusters: %d", len(clusters))

        cases = cases_for_pool(clusters, args.feature, args.pool)
        logger.info(
            "%s ground-truth cases (pool=%s): %d", args.feature, args.pool, len(cases)
        )

        # A source question with no vector cannot be searched from at all.
        ids = [c.question_id for c in cases]
        embedded = {
            r[0]
            for r in session.execute(
                sqltext(
                    "SELECT payload ->> 'question_id' FROM vec_questions_opendata "
                    "WHERE payload ->> 'question_id' = ANY(:ids)"
                ),
                {"ids": ids},
            )
        }
        cases = [c for c in cases if c.question_id in embedded]
        logger.info("with an embedding: %d", len(cases))

    sampled = stratified_sample(cases, args.sample, random.Random(args.seed))
    logger.info(
        "sampled %d across %d strata (seed %d)",
        len(sampled),
        len({(c.member.source, c.member.legislature) for c in sampled}),
        args.seed,
    )

    return {
        "feature": args.feature,
        # The runner refuses a cases file whose pool differs from the one
        # it is about to search — the mismatch is invisible downstream.
        "pool": args.pool,
        "seed": args.seed,
        "population": len(cases),
        "cases": [
            {
                "question_id": c.question_id,
                "source": c.member.source,
                "legislature": c.member.legislature,
                # The instant the agent would have seen this question.
                "as_of": c.member.date_publication_jo.isoformat()
                if c.member.date_publication_jo
                else None,
                "mates": sorted(c.mate_ids),
            }
            for c in sampled
        ],
    }


def _outcomes(runs: list[dict], stage: str) -> list[RankedOutcome]:
    out = []
    for run in runs:
        if stage not in run.get("stages", {}):
            continue
        out.append(
            RankedOutcome(
                question_id=run["question_id"],
                ranked=tuple(run["stages"][stage]),
                mates=frozenset(run["mates"]),
                stratum=(run["source"], run["legislature"]),
            )
        )
    return out


def score(args: argparse.Namespace) -> dict:
    payload = json.loads(args.runs.read_text())
    runs: list[dict] = payload["runs"]
    logger.info(
        "scoring %d runs (%s, pool=%s)", len(runs), payload["feature"], payload["pool"]
    )

    # The judge's effect is only measurable where it actually ran. A
    # throttled or timed-out judge fails open and returns the list
    # untouched, so counting those runs would credit the judge with
    # keeping everything exactly when it was unavailable — the metric
    # would look best when the judge was most broken.
    applied = [r for r in runs if r.get("judge_applied")]
    applied_ids = {r["question_id"] for r in applied}
    fail_open_rate = 1 - len(applied) / len(runs) if runs else 0.0
    if fail_open_rate > FAIL_OPEN_LIMIT:
        raise SystemExit(
            f"judge failed open on {fail_open_rate:.0%} of runs "
            f"({len(runs) - len(applied)}/{len(runs)}) — above the "
            f"{FAIL_OPEN_LIMIT:.0%} limit. Re-run at a lower --llm-rate "
            "rather than publishing a judge stage that mostly did not run."
        )

    stages = {s: _outcomes(runs, s) for s in STAGE_ORDER}
    # Every stage is restricted to the judged subset so the funnel is
    # comparable row to row: a `retrieve` averaged over all 422 runs and
    # a `judged` averaged over the 157 the judge answered for would show
    # a drop that is partly just a change of population.
    stages = {
        k: [o for o in v if o.question_id in applied_ids] for k, v in stages.items()
    }
    stages = {k: v for k, v in stages.items() if v}

    final = stages.get("judged") or stages.get("displayed") or []
    summary = summarise_ranked(final)
    # Same pipeline, judge removed — the comparison that says whether the
    # judge earns its place.
    pre_judge = summarise_ranked(stages.get("displayed", []))

    judge = judge_false_drop_rate(stages.get("displayed", []), stages.get("judged", []))

    by_band: dict[str, dict] = {}
    for band in ("1", "2-4", "5-9", "10-19", "20+"):
        group = [o for o in final if size_band(len(o.mates)) == band]
        if group:
            by_band[band] = summarise_ranked(group).as_dict()

    by_stratum: dict[str, dict] = {}
    for stratum in sorted({o.stratum for o in final}):
        group = [o for o in final if o.stratum == stratum]
        by_stratum[f"{stratum[0]}-{stratum[1]}"] = summarise_ranked(group).as_dict()

    logger.info(
        "hit@1 %.1f%% | hit@10 %.1f%% | MRR %.3f | judge false-drop %.1f%%",
        summary.hit_at.get(1, 0) * 100,
        summary.hit_at.get(10, 0) * 100,
        summary.mrr,
        judge["false_drop_rate"] * 100,
    )

    return {
        "feature": payload["feature"],
        "pool": payload["pool"],
        "judge_model": payload.get("judge_model"),
        "rerank_model": payload.get("rerank_model"),
        "n": len(runs),
        "judge_applied": len(applied_ids),
        "judge_failed_open": len(runs) - len(applied_ids),
        "rerank_failed_open": sum(1 for r in runs if r.get("rerank_failed")),
        "mate_count_distribution": dict(
            Counter(size_band(len(r["mates"])) for r in runs).most_common()
        ),
        "fail_open_rate": fail_open_rate,
        "summary": summary.as_dict(),
        "summary_without_judge": pre_judge.as_dict(),
        "stage_recall": stage_recall(stages),
        "judge_false_drop": judge,
        "by_cluster_size": by_band,
        "by_stratum": by_stratum,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-cases", action="store_true")
    mode.add_argument("--score", action="store_true")
    ap.add_argument("--feature", choices=("allotissement", "edr"))
    ap.add_argument(
        "--pool",
        choices=("as-of-t", "unrestricted"),
        default="as-of-t",
        help="Which candidate pool the run will search. Selects the matching "
        "ground truth: as-of-t drops mates outside the pool, unrestricted "
        "keeps them because the search can reach them.",
    )
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--runs", type=Path, help="runs JSON from the TS runner")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.emit_cases:
        if not args.feature:
            ap.error("--emit-cases needs --feature")
        report = emit_cases(args)
    else:
        if not args.runs:
            ap.error("--score needs --runs")
        report = score(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
