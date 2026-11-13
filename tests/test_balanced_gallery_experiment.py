"""Family-held inference and immutable exact-input neural-cache contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import balanced_gallery_experiment as experiment
from balanced_gallery_experiment import BalancedGalleryExperimentTransferScoresInputs


def test_old_family_uses_original_held_folds_new_family_uses_only_held_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that old family uses original held folds new family uses only held
    query.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    called = []

    def sentinel(x: np.ndarray, model: dict) -> np.ndarray:
        """Return sentinel for this workflow.

        Args:
            x: X used by this operation.
            model: Model used by this operation.

        Returns:
            Computed array in the order of the supplied records.
        """
        called.append(model)
        return np.full(len(x), model)

    monkeypatch.setattr(experiment, "predict", sentinel)
    models: dict = {
        f"{name}_q{q}_g{g}": q * 10 + g
        for name in ("pointwise", "ranknet")
        for q in range(3)
        for g in range(3)
    }
    folds: dict = dict(queries={"tune/image": 2}, gallery={"old": 0})
    scores = experiment.transfer_scores(
        BalancedGalleryExperimentTransferScoresInputs(
            np.ones((2, 20)), ["old", "new"], "tune/image", folds, models, ["new"]
        )
    )
    assert set(called) == {20, 21, 22}
    assert scores == {"pointwise": [20.0, 21.0], "ranknet": [20.0, 21.0]}
    # A candidate absent from the frozen technical-success set cannot silently
    # receive an all-development model or a fallback score.
    with pytest.raises(AssertionError):
        experiment.transfer_scores(
            BalancedGalleryExperimentTransferScoresInputs(
                np.ones((2, 20)),
                ["old", "unregistered"],
                "tune/image",
                folds,
                models,
                ["new"],
            )
        )
    with pytest.raises(AssertionError):
        experiment.transfer_scores(
            BalancedGalleryExperimentTransferScoresInputs(
                np.full((2, 20), np.nan),
                ["old", "new"],
                "tune/image",
                folds,
                models,
                ["new"],
            )
        )


def test_exact_cache_rejects_corruption_and_conflicting_real_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that exact cache rejects corruption and conflicting real logits.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    monkeypatch.setattr(experiment, "rel", lambda p: p.name)
    path = tmp_path / "cache.json"
    path.write_text("{}", encoding="utf-8")
    inputs: dict = dict(
        query_description="real description",
        candidate_id="candidate",
        candidate_image_sha256="originalpixels",
    )
    digest = hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    identity: dict = dict(inputs=inputs, sha256=digest)
    raw: dict = dict(
        pool=["candidate"], logits=[1.25], pair_identities={"candidate": identity}
    )
    cache: dict = {}
    experiment.add_cache(cache, raw, path)
    assert cache[digest]["logit"] == 1.25
    experiment.add_cache(cache, raw, path)
    with pytest.raises(AssertionError):
        experiment.add_cache(cache, dict(raw, logits=[1.26]), path)
    with pytest.raises(AssertionError):
        experiment.add_cache({}, dict(raw, logits=[float("nan")]), path)
    wrong: dict = dict(inputs=dict(inputs, query_description="changed"), sha256=digest)
    with pytest.raises(AssertionError):
        experiment.add_cache({}, dict(raw, pair_identities={"candidate": wrong}), path)
