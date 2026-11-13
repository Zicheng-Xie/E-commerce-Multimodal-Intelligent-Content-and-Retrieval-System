"""Target-capacity math, hybrid evidence and real review-export critical paths."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from multimodal_ecommerce.business_retrieval import hybrid_scores
from multimodal_ecommerce.precision_labels import validate_precision_labels
from multimodal_ecommerce.precision_target import (
    partial_precision_bounds,
    precision_capacity,
)


def test_precision_capacity_is_macro_fixed_k_including_zero_queries() -> None:
    """Verify that precision capacity is macro fixed k including zero queries."""
    result = precision_capacity([0, 10])
    assert result["precision_ceiling"] == 0.5
    assert result["precision_target_feasible"] is False
    assert result["zero_positive_queries"] == 1
    assert result["positive_query_count"] == 1
    assert result["recall_ceiling_on_positive_queries"] == 1


def test_sixty_percent_does_not_require_six_positives_on_every_query() -> None:
    """Verify that sixty percent does not require six positives on every query."""
    result = precision_capacity([2, 10])
    assert result["precision_target_feasible"] is True
    assert result["required_top_k_true_positives"] == 12
    assert result["oracle_top_k_true_positives"] == 12
    assert result["precision_ceiling"] == 0.6


def test_high_precision_capacity_does_not_imply_recall_target_reachable() -> None:
    """Verify that high precision capacity does not imply recall target reachable."""
    result = precision_capacity([100, 100])
    assert result["precision_ceiling"] == 1
    assert result["recall_ceiling_on_positive_queries"] == 0.1
    assert result["recall_ceiling_on_positive_queries"] < result["recall_target"]


def test_unknown_labels_keep_bounds_and_no_measured_scores() -> None:
    """Verify that unknown labels keep bounds and no measured scores."""
    result = partial_precision_bounds([[1, 0, None, None]])
    assert result["capacity_lower_bound"] == 0.1
    assert result["capacity_upper_bound"] == 0.3
    assert result["pending_relations"] == 2
    assert result["quality_metrics"] is None
    with pytest.raises(ValueError):
        partial_precision_bounds([[True, None]])


def test_zero_lexical_evidence_abstains_without_tie_order_bias() -> None:
    """Verify that zero lexical evidence abstains without tie order bias."""
    ids = ["a", "b", "c"]
    regional = np.array([3.0, 2.0, 1.0])
    title = np.array([1.0, 3.0, 2.0])
    result = hybrid_scores(
        ids, regional, title, text="unmatchedquery", titles=["电脑", "耳机", "衣服"]
    )
    expected = 0.5 / (60 + np.array([1, 2, 3])) + 0.3 / (60 + np.array([3, 1, 2]))
    assert np.allclose(result, expected)
    assert len(result) == len(ids)
    with pytest.raises(ValueError):
        hybrid_scores(ids, regional, title, text="电脑", titles=["wrong_count"])


def test_grid_bulk_review_exports_real_jsonl_preserving_pending_and_signatures(
    tmp_path: Path,
) -> None:
    """Verify that grid bulk review exports real jsonl preserving pending and
    signatures.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.

    Notes:
        Writes derived artifacts through source.write_text.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the actual offline browser export helper")
    templates = [
        {
            "id": str(i),
            "packet_id": "packet",
            "role": "development_training_only",
            "query_id": "query",
            "candidate_id": f"c{i}",
            "query_image": "query.jpg",
            "candidate_image": f"c{i}.jpg",
            "grade": None,
            "note": "",
            "reviewer": "",
            "review_origin": None,
        }
        for i in range(3)
    ]
    source = tmp_path / "packet.json"
    source.write_text(json.dumps(templates), encoding="utf-8")
    js = Path("assets/precision_review_grid.js").resolve()
    program = """const fs=require('fs');const helpers=require(process.argv[1]);const blank=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));let rows=blank.map(r=>({...r}));rows[0]={...rows[0],grade:1,reviewer:'synthetic-test',review_origin:'human'};rows=helpers.markVisiblePending(rows,['0','1'],'synthetic-test');helpers.validateReviewRows(blank,rows);process.stdout.write(helpers.reviewRowsToJsonl(rows));"""
    result = subprocess.run(
        [node, "-e", program, str(js), str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    reviewed = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(reviewed) == 3
    assert [r["grade"] for r in reviewed] == [1, 0, None]
    checked = validate_precision_labels(templates, reviewed)
    assert checked["completed"] == 2 and checked["pending"] == 1


def test_complete_business_scoring_keeps_zero_positive_queries() -> None:
    """Verify that complete business scoring keeps zero positive queries."""
    from multimodal_ecommerce.business_evaluation import score_business_development

    ids = [str(i) for i in range(10)]
    allowed: dict = {"empty": ids, "positive": ids}
    rankings: dict = {q: {"global": ids, "region_title": ids} for q in allowed}
    labels = [
        {"query_id": q, "candidate_id": c, "grade": int(q == "positive")}
        for q in allowed
        for c in ids
    ]
    result = score_business_development(
        rankings, allowed, labels, {"empty": "image", "positive": "image"}
    )
    assert result["quality_metrics"]["global"]["precision@10"] == 0.5
    assert result["quality_metrics"]["global"]["recall@10"] == 1
    assert (
        result["precision_effective_queries"] == 2
        and result["recall_effective_queries"] == 1
    )
    assert result["zero_positive_query_ids"] == ["empty"]
    labels[0]["grade"] = None
    with pytest.raises(ValueError, match="complete"):
        score_business_development(
            rankings, allowed, labels, {"empty": "image", "positive": "image"}
        )


def test_business_gate_reports_weak_modality_despite_aggregate_target() -> None:
    """Verify that business gate reports weak modality even when aggregate precision
    is
    sixty percent.
    """
    from multimodal_ecommerce.business_evaluation import score_business_development

    ids = [str(i) for i in range(20)]
    allowed: dict = {"text": ids, "image": ids}
    rankings: dict = {q: {"global": ids, "region_title": ids} for q in allowed}
    labels = [
        {
            "query_id": q,
            "candidate_id": c,
            "grade": int(int(c) < (10 if q == "text" else 2)),
        }
        for q in allowed
        for c in ids
    ]
    result = score_business_development(
        rankings, allowed, labels, {"text": "text", "image": "image"}
    )
    assert result["quality_metrics"]["global"]["precision@10"] == 0.6
    assert result["metrics_by_mode"]["image"]["global"]["precision@10"] == 0.2
    assert not result["development_gates"]["global"][
        "passes_development_targets_and_nonregression"
    ]
    assert result["formal_acceptance"] is False
