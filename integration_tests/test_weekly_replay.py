"""Verify the weekly tables from full frozen orders and unknown AI grades."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from multimodal_ecommerce.safety import PROJECT_ROOT
from replay_weekly import load_case, verify_case

CASES = [
    "blind_300",
    "blind_96",
    "joint_tune",
    "joint_regression",
    "joint_fresh",
    "role_pool_tune",
    "role_pool_regression",
    "role_pool_fresh",
    "grounded_tune",
    "grounded_regression",
    "grounded_fresh",
]


@pytest.mark.parametrize("case", CASES)
def test_weekly_metric_bounds_match(case: str) -> None:
    """Recompute six metrics, all modes, target flags and full denominators.

    Args:
        case: One frozen experiment and cohort from the weekly report.
    """
    result = verify_case(case)
    assert result["verified"] and result["human_labels"] == 0
    assert result["formal_metrics"] is None


def test_blind_unknowns_and_full_galleries_are_preserved() -> None:
    """Keep all 18,000/2,304 relations and their original 59/16 unknowns."""
    for case, relations, unknown, gallery in [
        ("blind_300", 18000, 59, 300),
        ("blind_96", 2304, 16, 96),
    ]:
        rankings, labels, _, _ = load_case(case)
        assert len(labels) == relations
        assert sum(r["grade"] is None for r in labels) == unknown
        assert all(
            len(order) == gallery for m in rankings.values() for order in m.values()
        )


def test_modified_evidence_is_rejected(tmp_path: Path) -> None:
    """Reject an altered artifact before metric computation.

    Args:
        tmp_path: Isolated evidence copy modified for this integrity scenario.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ["weekly_replay.json.gz", "provenance.json"]:
        shutil.copy2(PROJECT_ROOT / "evidence" / name, evidence / name)
    path = evidence / "weekly_replay.json.gz"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_case("blind_300", tmp_path)
