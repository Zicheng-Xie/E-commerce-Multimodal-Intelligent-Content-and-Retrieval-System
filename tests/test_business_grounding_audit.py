"""Capacity and consumed-exclusion checks for the new development audit."""

from __future__ import annotations

import pytest

from business_grounding_audit import capacity, check_exclusion


def test_capacity_preserves_fixed_k_and_joint_unknowns() -> None:
    """Verify that capacity preserves fixed k and joint unknowns."""
    value = capacity(3, 9)
    assert value["precision_upper"] == 1
    assert value["joint_capacity_points"][0]["precision_ceiling"] == 0.3
    assert value["joint_capacity_points"][-1]["recall_ceiling"] == 10 / 12
    assert capacity(37, 3)["recall_upper_defined"] == 10 / 37
    assert capacity(0, 0)["recall_upper_defined"] is None
    with pytest.raises(ValueError):
        capacity(-1, 0)


def test_exclusions_compare_source_ids_pixels_and_near_hashes() -> None:
    """Verify that exclusions compare source ids pixels and near hashes."""
    excluded = ({"product/test/1"}, {"1"}, {"a"}, {"b"}, {15})
    fp: dict = {"sha256": "c", "pixels": "d", "dhash": 15}
    assert check_exclusion("product/train/1", fp, excluded)["exact_consumed_overlap"]
    result = check_exclusion("product/train/2", fp, excluded)
    assert not result["exact_consumed_overlap"] and result["near_consumed_risk"]
    fp["pixels"] = "b"
    assert check_exclusion("product/train/2", fp, excluded)["exact_consumed_overlap"]
