"""Transfer learned rank corrections without replacing the stronger base ranking."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import numpy as np


@dataclass(frozen=True)
class RankCorrectionRankCorrectionInputs:
    """Input values for rank correction.

    Attributes:
        base: Base used by this operation.
        reference: Reference used by this operation.
        adapted: Adapted used by this operation.
        weight: Weight used by this operation.
        anchors: Anchors used by this operation.
        pool_size: Pool size used by this operation.
    """

    base: list[str]
    reference: list[str]
    adapted: list[str]
    _: KW_ONLY
    weight: float = 0.5
    anchors: int = 2
    pool_size: int = 30


def rank_correction(options: RankCorrectionRankCorrectionInputs) -> list[str]:
    """Apply the adapted-minus-reference rank delta within one unchanged pool.

    Args:
        options: Validated input fields described by RankCorrectionRankCorrectionInputs.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Correction rankings must use the same unique gallery; Invalid rank
            correction controls; Nonfinite rank correction.
    """
    base = options.base
    reference = options.reference
    adapted = options.adapted
    weight = options.weight
    anchors = options.anchors
    pool_size = options.pool_size
    if (
        len(set(base)) != len(base)
        or set(reference) != set(base)
        or set(adapted) != set(base)
    ):
        raise ValueError("Correction rankings must use the same unique gallery")
    if not 0 <= weight <= 1 or not 0 <= anchors <= min(pool_size, len(base)):
        raise ValueError("Invalid rank correction controls")
    pool = base[:pool_size]
    positions = set(pool)
    original = {key: i for i, key in enumerate(k for k in reference if k in positions)}
    learned = {key: i for i, key in enumerate(k for k in adapted if k in positions)}
    editable = pool[anchors:]
    score = {
        key: -i + weight * (original[key] - learned[key]) for i, key in enumerate(pool)
    }
    if not np.isfinite(list(score.values())).all():
        raise ValueError("Nonfinite rank correction")
    return (
        pool[:anchors]
        + sorted(editable, key=lambda k: (-score[k], k))
        + base[len(pool) :]
    )
