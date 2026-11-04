"""Limit a candidate's changes to its displayed head; retain baseline tail order."""

from __future__ import annotations


def stable_tail(anchor: list[str], candidate: list[str], k: int = 10) -> list[str]:
    """Keep the candidate's first k results and fill the tail in anchor order.

    This uses rankings alone. It preserves every metric defined only on the
    first k positions. AP and MRR can change, so neither is guaranteed to improve.

    Args:
        anchor: Anchor used by this operation.
        candidate: Candidate used by this operation.
        k: K used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: k must be an integer within the full ranking; Full unique rankings
            over the same candidates required.
    """
    if type(k) is not int or k < 1 or k > len(anchor):
        raise ValueError("k must be an integer within the full ranking")
    if (
        len(anchor) != len(candidate)
        or len(set(anchor)) != len(anchor)
        or set(anchor) != set(candidate)
    ):
        raise ValueError("Full unique rankings over the same candidates required")
    head = candidate[:k]
    selected = set(head)
    return head + [cid for cid in anchor if cid not in selected]
