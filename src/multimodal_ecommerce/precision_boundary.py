"""Refine only the uncertain ranking boundary, protecting earlier results."""

from __future__ import annotations

import numpy as np

from multimodal_ecommerce.precision_reranking import refine_pool


def boundary_refine(
    base: list[str],
    logits: np.ndarray,
    *,
    anchors: int,
    count: int = 10,
    blend: float = 0.5,
) -> list[str]:
    """Return boundary refine for this workflow.

    Args:
        base: Base used by this operation.
        logits: Logits used by this operation.
        anchors: Anchors used by this operation.
        count: Count used by this operation.
        blend: Blend used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Invalid editable ranking boundary; Only editable-window logits may
            enter the boundary.
    """
    if anchors < 0 or count < 1 or anchors + count > len(base):
        raise ValueError("Invalid editable ranking boundary")
    if np.asarray(logits).shape != (count,):
        raise ValueError("Only editable-window logits may enter the boundary")
    window = base[anchors : anchors + count]
    return (
        base[:anchors]
        + refine_pool(window, window, np.asarray(logits), blend=blend)
        + base[anchors + count :]
    )
