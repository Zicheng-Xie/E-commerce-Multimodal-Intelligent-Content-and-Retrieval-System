"""Verify train-family exclusion and immutable real-scored candidate pools."""

from __future__ import annotations

import numpy as np
import pytest

from business_joint_model import blend, training_rows


def test_class_balancing_never_assigns_weight_to_unknown_or_held_cells() -> None:
    """Verify that class balancing never assigns weight to unknown or held cells."""
    y = np.array([[1.0, 1.0, 0.0, np.nan], [0.0, 1.0, np.nan, np.nan]])
    allowed = np.array([[True, True, True, True], [False, False, False, False]])
    known, weights = training_rows(y, allowed)
    assert known.tolist() == [[True, True, True, False], [False, False, False, False]]
    assert np.allclose(weights, [0.25, 0.25, 0.5])


def test_blend_cannot_add_unscored_candidates_or_change_tail() -> None:
    """Verify that blend cannot add unscored candidates or change tail."""
    assert blend(["b", "a"], ["a", "b", "c"], {"a": 0.0, "b": 1.0})[-1] == "c"
    with pytest.raises(ValueError):
        blend(["a", "z"], ["a", "b"], {"a": 1.0, "z": 1.0})
