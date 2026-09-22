"""Metrics for the feature-quality benchmark.

Pure functions over already-computed rankings and predictions — no I/O,
no database, no API. Everything here is unit-testable against
hand-computed values.

Two families:

* ranked lists (allotissement, EDR) — hit@K, recall@K, MRR, rank
  histogram, per-stage recall;
* classification (direction, bureau) — macro-F1, balanced accuracy,
  per-class scores, confusion matrix, calibration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ranked lists
# ---------------------------------------------------------------------------


def first_hit_rank(ranked: Sequence[str], mates: frozenset[str]) -> int | None:
    """1-based rank of the first true mate, or None if none is present."""
    for i, candidate in enumerate(ranked, start=1):
        if candidate in mates:
            return i
    return None


def hit_at_k(ranked: Sequence[str], mates: frozenset[str], k: int) -> bool:
    return any(c in mates for c in ranked[:k])


def recall_at_k(ranked: Sequence[str], mates: frozenset[str], k: int) -> float:
    """Fraction of the question's mates present in the top K."""
    if not mates:
        return 0.0
    found = sum(1 for c in ranked[:k] if c in mates)
    return found / len(mates)


def attainable_recall_at_k(
    ranked: Sequence[str], mates: frozenset[str], k: int
) -> float:
    """Recall against what K slots could possibly have held.

    Raw recall@K is uninterpretable when cluster sizes vary by two orders
    of magnitude: the preprod corpus holds clusters of 291 co-answered
    questions, where recall@20 is capped at 6.9% no matter how perfect
    the ranking. Dividing by ``min(k, len(mates))`` scores the ranking
    rather than the cluster size.
    """
    if not mates:
        return 0.0
    found = sum(1 for c in ranked[:k] if c in mates)
    return found / min(k, len(mates))


def reciprocal_rank(ranked: Sequence[str], mates: frozenset[str]) -> float:
    rank = first_hit_rank(ranked, mates)
    return 1.0 / rank if rank else 0.0


def precision_at_k(ranked: Sequence[str], mates: frozenset[str], k: int) -> float:
    """Share of the top K that are true mates.

    A LOWER BOUND on usefulness, never a measure of noise: a candidate
    absent from the answer-text cluster may still be a perfectly relevant
    suggestion the administration simply answered separately.
    """
    head = ranked[:k]
    if not head:
        return 0.0
    return sum(1 for c in head if c in mates) / len(head)


def size_band(n_mates: int) -> str:
    """Cluster-size band, so wildly different problems aren't averaged."""
    if n_mates == 1:
        return "1"
    if n_mates <= 4:
        return "2-4"
    if n_mates <= 9:
        return "5-9"
    if n_mates <= 19:
        return "10-19"
    return "20+"


@dataclass
class RankedOutcome:
    """One source question's result, at one pipeline stage."""

    question_id: str
    ranked: tuple[str, ...]
    mates: frozenset[str]
    stratum: tuple[str, int]


@dataclass
class RankedSummary:
    n: int
    hit_at: dict[int, float]
    recall_at: dict[int, float]
    attainable_recall_at: dict[int, float]
    mrr: float
    precision_at: dict[int, float]
    rank_histogram: dict[str, int]
    median_first_hit_rank: float | None
    short_list_rate: float
    empty_rate: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "hit_at": {str(k): v for k, v in self.hit_at.items()},
            "recall_at": {str(k): v for k, v in self.recall_at.items()},
            "attainable_recall_at": {
                str(k): v for k, v in self.attainable_recall_at.items()
            },
            "mrr": self.mrr,
            "precision_at": {str(k): v for k, v in self.precision_at.items()},
            "rank_histogram": self.rank_histogram,
            "median_first_hit_rank": self.median_first_hit_rank,
            "short_list_rate": self.short_list_rate,
            "empty_rate": self.empty_rate,
        }


DEFAULT_KS = (1, 3, 5, 10, 20)
_RANK_BANDS = ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20))


def _rank_band(rank: int) -> str:
    for low, high in _RANK_BANDS:
        if low <= rank <= high:
            return f"{low}" if low == high else f"{low}-{high}"
    return "21+"


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarise_ranked(
    outcomes: Sequence[RankedOutcome],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    min_results: int = 20,
) -> RankedSummary:
    n = len(outcomes)
    if n == 0:
        return RankedSummary(0, {}, {}, {}, 0.0, {}, {}, None, 0.0, 0.0)

    histogram: Counter[str] = Counter()
    first_ranks: list[float] = []
    for o in outcomes:
        rank = first_hit_rank(o.ranked, o.mates)
        histogram[_rank_band(rank) if rank else "not found"] += 1
        if rank:
            first_ranks.append(rank)

    return RankedSummary(
        n=n,
        hit_at={
            k: sum(hit_at_k(o.ranked, o.mates, k) for o in outcomes) / n for k in ks
        },
        recall_at={
            k: sum(recall_at_k(o.ranked, o.mates, k) for o in outcomes) / n for k in ks
        },
        attainable_recall_at={
            k: sum(attainable_recall_at_k(o.ranked, o.mates, k) for o in outcomes) / n
            for k in ks
        },
        mrr=sum(reciprocal_rank(o.ranked, o.mates) for o in outcomes) / n,
        precision_at={
            k: sum(precision_at_k(o.ranked, o.mates, k) for o in outcomes) / n
            for k in ks
        },
        rank_histogram=dict(histogram),
        median_first_hit_rank=_median(first_ranks),
        short_list_rate=sum(1 for o in outcomes if len(o.ranked) < min_results) / n,
        empty_rate=sum(1 for o in outcomes if not o.ranked) / n,
    )


def group_by(outcomes: Sequence[RankedOutcome], key) -> dict:
    grouped: dict = {}
    for o in outcomes:
        grouped.setdefault(key(o), []).append(o)
    return grouped


def stage_recall(
    stages: Mapping[str, Sequence[RankedOutcome]], k: int = 20
) -> dict[str, dict[str, float]]:
    """Recall and hit rate after each pipeline stage, in the given order.

    The single most actionable output of the list benchmark: it says
    *which* stage loses a mate — HNSW retrieve, Albert rerank, the
    display cut, or the LLM judge — rather than only that it was lost.
    """
    return {
        name: {
            "hit_at_k": sum(hit_at_k(o.ranked, o.mates, k) for o in outcomes)
            / len(outcomes),
            "recall_at_k": sum(recall_at_k(o.ranked, o.mates, k) for o in outcomes)
            / len(outcomes),
        }
        for name, outcomes in stages.items()
        if outcomes
    }


def judge_false_drop_rate(
    displayed: Sequence[RankedOutcome], judged: Sequence[RankedOutcome]
) -> dict[str, float]:
    """How many true mates the LLM judge removed from the displayed list.

    Only counts questions where the judge actually ran: a fail-open
    judge (timeout, transport error) returns the list untouched, and
    scoring those as "dropped nothing" would flatter the judge exactly
    when it was unavailable. Callers must pass only applied-judge runs.
    """
    by_id = {o.question_id: o for o in judged}
    mates_displayed = 0
    mates_dropped = 0
    for before in displayed:
        after = by_id.get(before.question_id)
        if after is None:
            continue
        shown = {c for c in before.ranked if c in before.mates}
        kept = set(after.ranked)
        mates_displayed += len(shown)
        mates_dropped += len(shown - kept)
    return {
        "mates_displayed": mates_displayed,
        "mates_dropped": mates_dropped,
        "false_drop_rate": mates_dropped / mates_displayed if mates_displayed else 0.0,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass
class ClassificationSummary:
    n: int
    top1_accuracy: float
    top3_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    coverage: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "coverage": self.coverage,
            "per_class": self.per_class,
            "confusion": self.confusion,
        }


@dataclass
class Prediction:
    """One classification result. ``ranked`` is empty when the vote abstained."""

    question_id: str
    truth: str
    ranked: tuple[str, ...]
    confidence: float | None = None

    @property
    def top1(self) -> str | None:
        return self.ranked[0] if self.ranked else None


def summarise_classification(
    predictions: Sequence[Prediction],
) -> ClassificationSummary:
    n = len(predictions)
    if n == 0:
        return ClassificationSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    labels = sorted(
        {p.truth for p in predictions} | {p.top1 for p in predictions if p.top1}
    )
    confusion = {t: dict.fromkeys(labels, 0) for t in labels}
    tp = Counter[str]()
    fp = Counter[str]()
    fn = Counter[str]()
    support = Counter[str]()

    for p in predictions:
        support[p.truth] += 1
        if p.top1 is None:
            fn[p.truth] += 1  # an abstention still misses the truth
            continue
        confusion[p.truth][p.top1] += 1
        if p.top1 == p.truth:
            tp[p.truth] += 1
        else:
            fp[p.top1] += 1
            fn[p.truth] += 1

    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        precision = (
            tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        )
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
        }
        # Balanced accuracy and macro-F1 average over classes that exist
        # in the truth; a label only ever predicted has no recall to average.
        if support[label]:
            recalls.append(recall)
            f1s.append(f1)

    return ClassificationSummary(
        n=n,
        top1_accuracy=sum(1 for p in predictions if p.top1 == p.truth) / n,
        top3_accuracy=sum(1 for p in predictions if p.truth in p.ranked[:3]) / n,
        balanced_accuracy=sum(recalls) / len(recalls) if recalls else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        coverage=sum(1 for p in predictions if p.ranked) / n,
        per_class=per_class,
        confusion=confusion,
    )


def majority_baseline(predictions: Sequence[Prediction]) -> ClassificationSummary:
    """Score of always answering the most common truth label.

    Without it a headline accuracy is unreadable: on the preprod human
    attribution set DGCS is 6,326 of 11,262 labels, so a constant
    predictor already scores ~56%.
    """
    if not predictions:
        return summarise_classification([])
    most_common = Counter(p.truth for p in predictions).most_common(1)[0][0]
    return summarise_classification(
        [Prediction(p.question_id, p.truth, (most_common,), None) for p in predictions]
    )


def calibration(predictions: Iterable[Prediction], bins: int = 10) -> dict[str, object]:
    """Reliability curve and Brier score for the vote-share confidence.

    Answers "when the UI says 80%, is it right 80% of the time?" — which
    is what a 'hide the suggestion below X' rule needs.
    """
    scored = [p for p in predictions if p.confidence is not None and p.top1]
    if not scored:
        return {"n": 0, "brier": None, "curve": []}

    curve = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        in_bin = [
            p
            for p in scored
            if (low <= (p.confidence or 0) < high)
            or (i == bins - 1 and (p.confidence or 0) == 1.0)
        ]
        if not in_bin:
            continue
        curve.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "n": len(in_bin),
                "mean_confidence": sum(p.confidence or 0 for p in in_bin) / len(in_bin),
                "observed_accuracy": sum(1 for p in in_bin if p.top1 == p.truth)
                / len(in_bin),
            }
        )
    brier = sum(
        ((p.confidence or 0) - (1.0 if p.top1 == p.truth else 0.0)) ** 2 for p in scored
    ) / len(scored)
    return {"n": len(scored), "brier": brier, "curve": curve}
