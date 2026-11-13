"""The loss variant keeps full evidence and does not mutate its sealed parent."""

from __future__ import annotations

import numpy as np

import pointwise_recall_experiment as variant
from learned_fusion_model import FEATURE_NAMES as ORIGINAL_FEATURES
from learned_fusion_model import fit as original_fit


def test_balanced_pointwise_matches_original_loss_and_excludes_held_null_data() -> None:
    """Verify that balanced pointwise matches original loss and excludes held null
    data.
    """
    rng = np.random.default_rng(29)
    x = rng.normal(size=(4, 12, 20))
    y = np.tile([1.0, 0.0] * 6, (4, 1))
    y[2, 4] = np.nan
    allowed = np.ones(y.shape, bool)
    allowed[0] = False
    allowed[:, 0] = False
    base = rng.normal(size=y.shape)
    expected = original_fit(x, y, allowed, base, pairwise=False)
    model = variant.fit(x, y, allowed, base, names=ORIGINAL_FEATURES)
    np.testing.assert_allclose(
        model["weights"], expected["weights"][:-1], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        model["bias"], expected["weights"][-1], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        model["score_location"], expected["score_location"], rtol=0, atol=1e-14
    )
    changed = x.copy()
    changed[~allowed | ~np.isfinite(y)] = 1e10
    labels = y.copy()
    labels[~allowed] = 1 - y[~allowed]
    other = variant.fit(changed, labels, allowed, base * 1e6, names=ORIGINAL_FEATURES)
    assert model == other
    assert model["negative"] == int((allowed & np.isfinite(y) & (y == 0)).sum())


def test_runtime_bindings_are_isolated_and_keep_full_neural_implementation() -> None:
    """Verify that runtime bindings are isolated and keep full neural implementation."""
    old_root, old_methods, old_fit = (
        variant.parent.ROOT,
        variant.parent.METHODS,
        variant.parent.fit,
    )
    module = variant.runtime()
    assert module.ROOT == variant.ROOT and module.ROOT != old_root
    assert module.previous.ROOT == old_root and module.METHODS == variant.METHODS
    assert module.fit is variant.fit and module.predict is variant.predict
    assert module.infer.__code__.co_code == variant.parent.infer.__code__.co_code
    assert module.infer.__code__.co_names == variant.parent.infer.__code__.co_names
    assert (
        module.infer.__code__.co_filename == variant.parent.infer.__code__.co_filename
    )
    assert module.infer.__globals__ is module.__dict__
    assert variant.parent.ROOT == old_root and variant.parent.METHODS == old_methods
    assert variant.parent.fit is old_fit and variant.parent.fit is not variant.fit
