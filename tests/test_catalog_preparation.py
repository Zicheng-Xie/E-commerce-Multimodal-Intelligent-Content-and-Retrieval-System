"""Catalog candidates and priority ordering must not become fake relevance truth."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from multimodal_ecommerce.business_review_jobs import validate_submission
from multimodal_ecommerce.catalog_preparation import (
    merge_catalog_qa,
    metadata_bucket,
    priority_order,
    validate_catalog_submission,
)


def template() -> dict:
    """Return template for this workflow.

    Returns:
        Mapping with fields: id, role, bucket, bucket_name, title, source_label,
        image, verdict, product_type, note, reviewer, review_origin.
    """
    return dict(
        id="g1",
        role="development_catalog_QA_only",
        bucket="headphones",
        bucket_name="耳机",
        title="无线耳机",
        source_label="耳机",
        image="image.jpg",
        verdict=None,
        product_type="",
        note="",
        reviewer="",
        review_origin=None,
    )


def submit(slot: str, reviewer: str, verdict: str = "usable") -> dict:
    """Return submit for this workflow.

    Args:
        slot: Slot used by this operation.
        reviewer: Reviewer used by this operation.
        verdict: Verdict used by this operation.

    Returns:
        Mapping with fields: release_id, review_slot, reviewer, declared_human, rows.
    """
    row = template()
    row.update(
        verdict=verdict,
        product_type="无线耳机本体",
        reviewer=reviewer,
        review_origin="human",
    )
    return dict(
        release_id="release",
        review_slot=slot,
        reviewer=reviewer,
        declared_human=True,
        rows=[row],
    )


def test_metadata_does_not_admit_accessories_or_wrong_sleeves() -> None:
    """Verify that metadata does not admit accessories or wrong sleeves."""
    assert metadata_bucket("蓝牙耳机", "蓝牙耳机左耳补配充电盒") is None
    assert metadata_bucket("耳机/耳麦", "无线蓝牙耳机本体") == "wireless_headphones"
    assert metadata_bucket("T恤", "长袖T恤") is None
    assert metadata_bucket("T恤", "男女短袖T恤") == "short_tshirts"
    assert metadata_bucket("键盘", "电脑键盘保护膜") is None


def test_priority_reordering_is_directly_accepted_by_original_importer() -> None:
    """Verify that priority reordering is directly accepted by original importer."""
    base: dict = dict(
        packet_id="p",
        role="development_training_only",
        query_id="q",
        candidate_id="g",
        query_image="",
        query_text="需求",
        query_rule="用途",
        candidate_image="i.jpg",
        candidate_title="商品",
        grade=None,
        note="",
        reviewer="",
        review_origin=None,
    )
    rows = [dict(base, id=str(i), candidate_id="g" + str(i)) for i in range(4)]
    ordered = priority_order(rows, {"2"})
    assert [r["id"] for r in ordered] == ["2", "0", "1", "3"]
    assert {r["id"]: r for r in rows} == {r["id"]: r for r in ordered}
    manifest: dict = dict(
        release_id="release", batches=[dict(id="b", row_ids=[r["id"] for r in rows])]
    )
    export: dict = dict(
        release_id="release",
        batch_id="b",
        review_slot="A",
        reviewer="synthetic unit fixture",
        declared_human=True,
        adjudication_of=None,
        rows=ordered,
    )
    assert validate_submission(manifest, rows, export) == ordered


def test_catalog_acceptance_never_creates_query_relevance_labels() -> None:
    """Verify that catalog acceptance never creates query relevance labels."""
    result = merge_catalog_qa(
        dict(release_id="release"),
        [template()],
        [submit("A", "fixture A"), submit("B", "fixture B")],
    )
    assert result["accepted_catalog_ids"] == ["g1"]
    assert (
        result["human_relevance_labels"] == 0
        and result["relevance_labels_created"] is False
    )
    assert (
        merge_catalog_qa(dict(release_id="release"), [template()], [])["pending"] == 1
    )
    with pytest.raises(ValueError, match="different actual"):
        merge_catalog_qa(
            dict(release_id="release"),
            [template()],
            [submit("A", "same"), submit("B", "same")],
        )


def test_catalog_export_rejects_machine_origin_and_immutable_edits() -> None:
    """Verify that catalog export rejects machine origin and immutable edits."""
    value = submit("A", "fixture")
    value["rows"][0]["review_origin"] = "model"
    with pytest.raises(ValueError):
        validate_catalog_submission(dict(release_id="release"), [template()], value)
    value = submit("A", "fixture")
    value["rows"][0]["title"] = "changed"
    with pytest.raises(ValueError, match="Immutable"):
        validate_catalog_submission(dict(release_id="release"), [template()], value)


def test_actual_javascript_export_schema_matches_python_validation() -> None:
    """Verify that actual javascript export schema matches python validation."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    job: dict = dict(release_id="release", review_slot="A", rows=[template()])
    value = submit("A", "synthetic unit fixture")
    program = "const h=require(process.argv[1]);process.stdout.write(JSON.stringify(h.validateCatalog(JSON.parse(process.argv[2]),JSON.parse(process.argv[3]))));"
    result = subprocess.run(
        [
            node,
            "-e",
            program,
            str(Path("assets/catalog_qa.js").resolve()),
            json.dumps(job),
            json.dumps(value),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert (
        validate_catalog_submission(
            dict(release_id="release"), [template()], json.loads(result.stdout)
        )
        == value["rows"]
    )
