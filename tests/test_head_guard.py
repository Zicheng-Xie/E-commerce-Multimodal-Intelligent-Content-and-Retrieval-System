"""Head protection cannot duplicate candidates or use evaluator labels."""

from __future__ import annotations

import pytest

from multimodal_ecommerce.head_guard import protected_prefix


@pytest.mark.parametrize("keep", [1, 2])
def test_prefix_is_exact_and_remaining_order_is_candidate_order(keep: int) -> None:
    """Verify that prefix is exact and remaining order is candidate order.

    Args:
        keep: Keep used by this operation.
    """
    baseline = ["a", "b", "c", "d"]
    candidate = ["d", "a", "c", "b"]
    result = protected_prefix(baseline, candidate, keep)
    assert result[:keep] == baseline[:keep]
    assert result[keep:] == [cid for cid in candidate if cid not in baseline[:keep]]
    assert len(result) == len(set(result)) == 4


def test_foreign_or_repeated_candidates_are_rejected() -> None:
    """Verify that foreign or repeated candidates are rejected."""
    with pytest.raises(ValueError, match="unique"):
        protected_prefix(["a", "b"], ["a", "a"], 1)


def test_image_only_guard_preserves_original_text_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that image only guard preserves original text branch.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("compare_region_prefix")
    methods: dict = dict(
        region_title=["a", "b", "c"],
        described_cross_pool20=["c", "a", "b"],
        cross_evidence_pool60=["c", "a", "b"],
    )
    original: dict = dict(text=dict(methods), image=dict(methods))
    result = module.route_guard(original, dict(text=False, image=True))
    assert result["text"][module.NAME] == methods["cross_evidence_pool60"]
    assert result["image"][module.NAME][:2] == methods["region_title"][:2]
    assert module.NAME not in original["text"]
