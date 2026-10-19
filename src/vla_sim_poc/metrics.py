"""Success-rate statistics for closed-loop evaluation.

All functions take plain Python/numpy inputs so they can be unit tested without
a simulator. Success is binary per episode; comparisons are *paired* because the
baseline and fine-tuned policies are evaluated on the same initial states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float

    def __str__(self) -> str:
        return f"{self.successes}/{self.n} = {self.rate:.1%} (95% CI {self.ci_low:.1%}-{self.ci_high:.1%})"


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval. Unlike the normal approximation it behaves at 0% and 100%."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion(outcomes: Sequence[bool]) -> Proportion:
    s, n = int(sum(bool(o) for o in outcomes)), len(outcomes)
    lo, hi = wilson_interval(s, n)
    return Proportion(s, n, s / n if n else float("nan"), lo, hi)


@dataclass(frozen=True)
class PairedComparison:
    n_pairs: int
    rate_a: float
    rate_b: float
    diff: float  # rate_b - rate_a
    only_a: int  # A succeeded, B failed
    only_b: int  # B succeeded, A failed
    mcnemar_p: float
    bootstrap_ci: tuple[float, float]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test (binomial test on discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(
    a: Sequence[bool], b: Sequence[bool], n_boot: int = 10_000, seed: int = 0
) -> PairedComparison:
    """Compare two policies evaluated on the same ordered list of initial states."""
    a_arr, b_arr = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Paired outcomes must align: {a_arr.shape} vs {b_arr.shape}")
    n = len(a_arr)
    only_a = int(np.sum(a_arr & ~b_arr))
    only_b = int(np.sum(~a_arr & b_arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b_arr[idx].mean(axis=1) - a_arr[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return PairedComparison(
        n_pairs=n,
        rate_a=float(a_arr.mean()),
        rate_b=float(b_arr.mean()),
        diff=float(b_arr.mean() - a_arr.mean()),
        only_a=only_a,
        only_b=only_b,
        mcnemar_p=mcnemar_exact(only_a, only_b),
        bootstrap_ci=(float(lo), float(hi)),
    )
