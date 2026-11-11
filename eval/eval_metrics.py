"""Pure-Python metrics shared by the ablation and judge-agreement scripts."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold passages that appear in the top-k."""
    if not gold:
        raise ValueError("gold set is empty")
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if any gold passage is in the top-k."""
    return float(bool(set(ranked[:k]) & set(gold)))


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """1/rank of the first gold passage within the top-k, else 0 (MRR@k when averaged)."""
    gold_set = set(gold)
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval over per-item values."""
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return (
        stats[int((alpha / 2) * n_resamples)],
        stats[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
    )


def confusion(human: Sequence[bool], judge: Sequence[bool]) -> dict[str, int]:
    if len(human) != len(judge):
        raise ValueError("length mismatch")
    pairs = list(zip(human, judge, strict=True))
    return {
        "both_yes": sum(h and j for h, j in pairs),
        "both_no": sum((not h) and (not j) for h, j in pairs),
        "human_yes_judge_no": sum(h and not j for h, j in pairs),
        "human_no_judge_yes": sum((not h) and j for h, j in pairs),
    }


def cohen_kappa(human: Sequence[bool], judge: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters. Returns nan when chance agreement is 1."""
    n = len(human)
    c = confusion(human, judge)
    p_o = (c["both_yes"] + c["both_no"]) / n
    p_h, p_j = sum(human) / n, sum(judge) / n
    p_e = p_h * p_j + (1 - p_h) * (1 - p_j)
    if math.isclose(p_e, 1.0):
        return float("nan")
    return (p_o - p_e) / (1 - p_e)
