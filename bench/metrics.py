"""Scores and 95% confidence intervals."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def per_sample_scores(rows: list[dict]) -> dict[str, float]:
    """Average repeats per sample (avg@k). Rows without a score (pending) are ignored."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            acc[r["sample_id"]].append(float(r["score"]))
    return {sid: sum(v) / len(v) for sid, v in acc.items()}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def summarize(rows: list[dict]) -> dict:
    scores = per_sample_scores(rows)
    vals = list(scores.values())
    if not vals:
        return {"score": None, "ci_low": None, "ci_high": None, "n_scored": 0}
    mean = sum(vals) / len(vals)
    binary = all(v in (0.0, 1.0) for v in vals)
    lo, hi = wilson(sum(vals), len(vals)) if binary else bootstrap(vals)
    return {"score": mean, "ci_low": lo, "ci_high": hi, "n_scored": len(vals)}
