#!/usr/bin/env python3
"""Quality benchmark: direction and bureau attribution against ground truth.

Scores the KNN-vote attribution features on truths the administration
produced itself, rather than on the in-code accuracy claims (≈92% top-1
direction; 50-60%/75-80% bureau) that
``docs/direction-bureau-attribution.md`` flags as never independently
re-verified.

Ground truth, and how contaminated each one is
----------------------------------------------
``--gt min15``  ``question_bureau_extract.direction_txt``. For
                **direction** this is the only clean truth: the vote
                reads ``question_real_attributions`` alone, so a MIN15
                label never votes on itself or its neighbours.
``--gt human``  ``question_real_attributions``. This *is* the direction
                voter pool. Reported as an optimistic bound, labelled as
                such, never as the headline.

For **bureau** no clean truth exists at all — the vote reads
``question_attributions_all``, which unions both sources. The route
excludes the question's own row, so there is no direct self-vote, but
every label is still a voter for its neighbours.

``--exclude-siblings`` bounds the remaining leak: co-answered questions
are near-duplicate texts carrying human attributions, which makes the
prediction nearly free. Both arms of that comparison run with the same
over-fetch so the delta isn't confounded by a different approximate-search
regime (the design ``eval_direction_with_min15.py`` uses for the same
reason).

Read-only: every transaction is opened ``SET TRANSACTION READ ONLY``, so
the benchmark cannot write to preprod even by accident. That also keeps
it clear of the live direction endpoint's fire-and-forget
``writeDirectionAlgoCache`` write.

Usage:
    poetry run python scripts/bench_quality_attribution.py \\
        --feature direction --gt min15 --output data/bench_direction_min15.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from sqlalchemy import text as sqltext
from tqdm import tqdm

from qe import db
from qe.eval.attribution_replay import predict
from qe.eval.ground_truth import (
    fetch_answer_clusters,
    fetch_bureau_truth,
    fetch_direction_truth,
)
from qe.eval.resilient import with_reconnect
from qe.eval.metrics import (
    Prediction,
    calibration,
    majority_baseline,
    summarise_classification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Over-fetch used by BOTH arms of the sibling-exclusion comparison, so
# the two arms share one approximate-search regime.
SIBLING_OVERFETCH = 200


def _embedded_ids(session, question_ids: list[str]) -> set[str]:
    """Keep only questions that actually have a vector to search from."""
    rows = session.execute(
        sqltext(
            """
            SELECT payload ->> 'question_id' AS qid
            FROM vec_questions_opendata
            WHERE payload ->> 'question_id' = ANY(:ids)
            """
        ),
        {"ids": question_ids},
    )
    return {r[0] for r in rows}


_LABEL_SPACE_SQL = {
    "direction": """
        SELECT DISTINCT d.nom
        FROM question_real_attributions qra
        JOIN directions d ON d.id = qra.direction_reelle_id
    """,
    "bureau": "SELECT DISTINCT bureau_key FROM question_attributions_all",
}


def _label_space(session, feature: str) -> set[str]:
    """Labels the vote is capable of emitting at all.

    A truth label outside this set cannot be predicted no matter how good
    the retrieval is — there is no voter carrying it. Scoring those as
    ordinary mistakes conflates "ranked the wrong direction" with "that
    direction is not in the output vocabulary", which have entirely
    different fixes (tune the vote vs. import the missing attributions).
    """
    return {r[0] for r in session.execute(sqltext(_LABEL_SPACE_SQL[feature]))}


def _sibling_map(session) -> dict[str, set[str]]:
    """question id -> the other questions sharing its answer text."""
    siblings: dict[str, set[str]] = {}
    for cluster in fetch_answer_clusters(session):
        ids = [m.question_id for m in cluster.members]
        for qid in ids:
            siblings.setdefault(qid, set()).update(i for i in ids if i != qid)
    return siblings


def run(args: argparse.Namespace) -> dict:
    with db.get_session() as session:
        session.execute(sqltext("SET TRANSACTION READ ONLY"))

        if args.feature == "direction":
            truth = fetch_direction_truth(session, args.gt)
        else:
            truth = fetch_bureau_truth(session, None if args.gt == "all" else args.gt)
        if not truth:
            raise SystemExit(
                f"no ground truth for {args.feature}/{args.gt} — refusing to "
                "report a vacuous 0%"
            )
        logger.info(
            "ground truth (%s/%s): %d labels", args.feature, args.gt, len(truth)
        )

        question_ids = sorted(truth)
        embedded = _embedded_ids(session, question_ids)
        question_ids = [q for q in question_ids if q in embedded]
        logger.info("with an embedding: %d", len(question_ids))

        if args.limit and args.limit < len(question_ids):
            question_ids = random.Random(args.seed).sample(question_ids, args.limit)
            logger.info("sampled down to %d (seed %d)", len(question_ids), args.seed)

        label_space = _label_space(session, args.feature)
        siblings = _sibling_map(session) if args.exclude_siblings else {}
        # Both arms of the sibling comparison must over-fetch identically,
        # or the delta measures the approximate-search regime rather than
        # the leak. --knn is therefore passed explicitly to BOTH runs;
        # --exclude-siblings alone would silently widen only one of them.
        knn = args.knn or None

    # The vote loop runs outside the long-lived session above: it opens a
    # short session per question so a dropped tunnel costs one retry
    # rather than the whole run.
    predictions: list[Prediction] = []
    for qid in tqdm(question_ids, desc=f"{args.feature}/{args.gt}"):

        def vote(qid: str = qid) -> tuple[list[str], list[float]]:
            with db.get_session() as s:
                s.execute(sqltext("SET TRANSACTION READ ONLY"))
                return predict(
                    s,
                    qid,
                    args.feature,
                    knn=knn,
                    exclude_question_ids=siblings.get(qid, set()),
                )

        ranked, shares = with_reconnect(vote)
        predictions.append(
            Prediction(
                question_id=qid,
                truth=truth[qid],
                ranked=tuple(ranked),
                confidence=shares[0] if shares else None,
            )
        )

    summary = summarise_classification(predictions)
    baseline = majority_baseline(predictions)

    in_vocab = [p for p in predictions if p.truth in label_space]
    oov = [p for p in predictions if p.truth not in label_space]
    in_vocab_summary = summarise_classification(in_vocab)

    logger.info(
        "top-1 %.1f%% | top-3 %.1f%% | macro-F1 %.3f | balanced acc %.3f "
        "(majority baseline top-1 %.1f%%)",
        summary.top1_accuracy * 100,
        summary.top3_accuracy * 100,
        summary.macro_f1,
        summary.balanced_accuracy,
        baseline.top1_accuracy * 100,
    )
    if oov:
        logger.info(
            "out of vocabulary: %d/%d (%.1f%%) — %s; on the reachable rest, "
            "top-1 %.1f%% | macro-F1 %.3f",
            len(oov),
            len(predictions),
            len(oov) / len(predictions) * 100,
            ", ".join(sorted({p.truth for p in oov})),
            in_vocab_summary.top1_accuracy * 100,
            in_vocab_summary.macro_f1,
        )

    return {
        "feature": args.feature,
        "ground_truth": args.gt,
        "contaminated": not (args.feature == "direction" and args.gt == "min15"),
        "exclude_siblings": args.exclude_siblings,
        "knn": knn,
        "seed": args.seed,
        "truth_distribution": dict(Counter(p.truth for p in predictions).most_common()),
        "summary": summary.as_dict(),
        "majority_baseline": baseline.as_dict(),
        "out_of_vocabulary": {
            "n": len(oov),
            "share": len(oov) / len(predictions) if predictions else 0.0,
            "labels": dict(Counter(p.truth for p in oov).most_common()),
            "voter_label_space": sorted(label_space),
        },
        "in_vocabulary_summary": in_vocab_summary.as_dict(),
        "calibration": calibration(predictions),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", choices=("direction", "bureau"), required=True)
    ap.add_argument(
        "--gt",
        choices=("min15", "human", "attribution", "all"),
        default="min15",
        help="min15 is the only uncontaminated direction truth. For bureau, "
        "'attribution' is the human half of question_attributions_all and "
        "'all' unions both of its sources.",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = every truth row")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument(
        "--knn",
        type=int,
        default=0,
        help=f"Override the production KNN (use {SIBLING_OVERFETCH} on BOTH "
        "arms of an --exclude-siblings comparison). 0 = production value.",
    )
    ap.add_argument(
        "--exclude-siblings",
        action="store_true",
        help="Drop co-answered questions from the voter pool (sensitivity run).",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.feature == "direction" and args.gt == "all":
        ap.error("--gt all applies to bureau only")

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
