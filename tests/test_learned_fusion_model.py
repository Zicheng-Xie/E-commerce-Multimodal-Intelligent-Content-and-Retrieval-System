"""Training exclusion includes scaling, negative mining and gradients."""

from __future__ import annotations

import numpy as np
import pytest

from learned_fusion_model import FEATURE_NAMES, fit, predict


@pytest.mark.parametrize("pairwise", [False, True])
def test_excluded_query_and_gallery_cannot_change_fitting(pairwise: bool) -> None:
    """Verify that excluded query and gallery cannot change fitting.

    Args:
        pairwise: Pairwise used by this operation.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 6, len(FEATURE_NAMES)))
    labels = np.tile([0, 1, 0, 1, 0, 1], (4, 1)).astype(float)
    allowed = np.ones((4, 6), dtype=bool)
    allowed[0] = False
    allowed[:, 0] = False
    base = rng.normal(size=labels.shape)
    before = fit(x, labels, allowed, base, pairwise=pairwise)
    changed_x, changed_y, changed_base = x.copy(), labels.copy(), base.copy()
    changed_x[~allowed] = 1e8
    changed_y[~allowed] = 1 - labels[~allowed]
    changed_base[~allowed] = 1e9
    after = fit(changed_x, changed_y, allowed, changed_base, pairwise=pairwise)
    assert before == after
    assert np.isfinite(predict(x, before)).all()


def test_unknown_cells_are_not_silent_negative_examples() -> None:
    """Verify that unknown cells are not silent negative examples."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 5, len(FEATURE_NAMES)))
    y = np.tile([0, 1, np.nan, 1, 0], (3, 1))
    mask = np.ones(y.shape, dtype=bool)
    base = rng.normal(size=y.shape)
    first = fit(x, y, mask, base, pairwise=True)
    x[:, 2] = 1e7
    base[:, 2] = 1e8
    assert first == fit(x, y, mask, base, pairwise=True)
    assert first["training_cells"] == 12
