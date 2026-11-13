"""Human query approval cannot change scope, drop slots or invent provenance."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from multimodal_ecommerce.business_query_review import validate_query_authoring


def sample() -> tuple[list[dict], dict]:
    """Return sample for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    row: dict = dict(
        id="text_01",
        kind="text",
        fine_type_scope="手机",
        category_for_reporting="3C数码",
        image_path=None,
        text=None,
        relevance_rule="",
        approved=False,
        reviewer="",
        review_origin=None,
        note="",
    )
    value: dict = dict(
        release_id="r",
        role="acceptance_query_author",
        reviewer="human-test",
        declared_human=True,
        rows=[copy.deepcopy(row)],
    )
    return [row], value


def test_unwritten_query_cannot_be_approved() -> None:
    """Verify that unwritten query cannot be approved."""
    rows, value = sample()
    r = value["rows"][0]
    r.update(approved=True, reviewer="human-test", review_origin="human")
    with pytest.raises(ValueError, match="actual text"):
        validate_query_authoring(rows, value, "r")
    r.update(
        text="寻找适合旅行拍照的手机",
        relevance_rule="手机，非保护壳或充电器；不猜未证实的性能。",
    )
    result = validate_query_authoring(rows, value, "r")
    assert result["all_queries_approved"] is True
    assert result["model_inference_allowed"] is False
    assert result["formal_metrics"] is None


@pytest.mark.parametrize(
    "change", ["scope", "delete", "image_hint", "declaration", "identity"]
)
def test_query_rejects_scope_drop_and_false_provenance(change: object) -> None:
    """Verify that query rejects scope drop and false provenance.

    Args:
        change: Change used by this operation.
    """
    rows, value = sample()
    if change == "scope":
        value["rows"][0]["fine_type_scope"] = "耳机"
    if change == "delete":
        value["rows"] = []
    if change == "image_hint":
        rows[0].update(kind="image", image_path="q.jpg")
        value["rows"][0].update(kind="image", image_path="q.jpg", text="手机")
    if change == "declaration":
        value["declared_human"] = False
    if change == "identity":
        value["rows"][0]["text"] = "新需求"
    with pytest.raises(ValueError):
        validate_query_authoring(rows, value, "r")


def test_actual_browser_export_matches_python(tmp_path: Path) -> None:
    """Verify that actual browser export matches python.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.

    Notes:
        Writes derived artifacts through fixture.write_text.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    rows, value = sample()
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(dict(job=dict(release_id="r", rows=rows), value=value)),
        encoding="utf-8",
    )
    js = Path(__file__).resolve().parents[1] / "assets/business_query_author.js"
    code = "const fs=require('fs'),m=require(process.argv[1]),v=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));process.stdout.write(JSON.stringify(m.validateQueryExport(v.job,v.value)));"
    result = subprocess.run(
        [node, "-e", code, str(js), str(fixture)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    status = validate_query_authoring(rows, json.loads(result.stdout), "r")
    assert status["pending_queries"] == 1
    assert status["human_relevance_labels"] == 0
