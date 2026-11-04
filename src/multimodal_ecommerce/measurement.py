"""Small transparent statistics for baseline latency and uncertainty reports."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def distribution(values: list[float]) -> dict:
    """Describe observed values without manufacturing empty-set estimates.

    Args:
        values: Values used by this operation.

    Returns:
        Mapping with fields: count, min, p05, p50, p95, max, mean.

    Raises:
        ValueError: Measurements must be finite.
    """
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Measurements must be finite")
    return {
        "count": len(values),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    """Return a Wilson 95% interval for an approximately independent proportion.

    Args:
        successes: Successes used by this operation.
        total: Total used by this operation.
        z: Z used by this operation.

    Returns:
        Mapping with fields: estimate, lower, upper, n, method.

    Raises:
        ValueError: Need 0 <= successes <= positive total.
    """
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Need 0 <= successes <= positive total")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "estimate": p,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
        "n": total,
        "method": "Wilson 95%; assumes approximately independent binary observations",
    }


def cluster_interval(values: list[float], groups: list[str], seed: int = 42) -> dict:
    """Bootstrap source groups, preserving dependence among related queries.

    Args:
        values: Values used by this operation.
        groups: Groups used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Returns:
        Mapping with fields: estimate, groups, interval, lower, upper, method.

    Raises:
        ValueError: Values and groups must be nonempty and aligned.
    """
    if not values or len(values) != len(groups):
        raise ValueError("Values and groups must be nonempty and aligned")
    clustered: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        clustered[group].append(value)
    units = list(clustered.values())
    if len(units) < 2:
        return {
            "estimate": float(np.mean(values)),
            "groups": len(units),
            "interval": None,
        }
    generator = np.random.default_rng(seed)
    means = []
    for _ in range(1000):
        draw = generator.integers(0, len(units), size=len(units))
        means.append(
            float(np.mean([value for index in draw for value in units[index]]))
        )
    return {
        "estimate": float(np.mean(values)),
        "groups": len(units),
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "method": "1000-draw percentile cluster bootstrap over split/pair_id; exploratory interval",
    }
