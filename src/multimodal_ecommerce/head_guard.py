"""Protect a fixed baseline prefix without inspecting relevance judgments."""

from __future__ import annotations


def protected_prefix(baseline: list[str], candidate: list[str], keep: int) -> list[str]:
    """Retain the first one/two baseline results and reorder the remaining slots.

    This preserves Hit@1 for any possible labels. MRR protection additionally
    requires that the first relevant baseline item was inside the protected prefix.

    Args:
        baseline: Baseline used by this operation.
        candidate: Candidate used by this operation.
        keep: Keep used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: A fixed prefix of one or two and equal lists are required; Full
            unique rankings over the same candidates are required; Insufficient
            candidates for protected prefix.
    """
    if keep not in {1, 2} or len(baseline) != len(candidate):
        raise ValueError("A fixed prefix of one or two and equal lists are required")
    if len(set(baseline)) != len(baseline) or set(baseline) != set(candidate):
        raise ValueError("Full unique rankings over the same candidates are required")
    if len(baseline) < keep:
        raise ValueError("Insufficient candidates for protected prefix")
    head = baseline[:keep]
    fixed = set(head)
    return head + [cid for cid in candidate if cid not in fixed]
