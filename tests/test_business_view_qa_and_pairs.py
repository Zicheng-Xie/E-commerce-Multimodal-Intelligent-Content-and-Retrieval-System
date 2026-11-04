"""Verify that hard-pair mining cannot consume unknown or held-out cells."""

from __future__ import annotations

import numpy as np

from business_view_qa_and_pairs import verified_pairs


def test_hard_mining_keeps_all_positives_and_never_relabels_unknowns() -> None:
    """Verify that hard mining keeps all positives and never relabels unknowns."""
    labels = np.array([[1.0, 1.0, 0.0, np.nan, 0.0]])
    allowed = np.array([[True, True, True, True, False]])
    scores = np.array([[0.0, 0.1, 0.3, 1.0, 2.0]])
    result, known = verified_pairs(labels, allowed, scores)
    assert result[0]["positive_indices"] == [0, 1]
    assert result[0]["hard_negative_indices"] == [2]
    assert not known[0, 3] and not known[0, 4]
