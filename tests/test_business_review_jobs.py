"""Protect human provenance, disagreements, frozen batches and source-family splits."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import merge_values


@pytest.fixture
def packet() -> tuple[dict, list]:
    """Return packet for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    rows = [
        dict(
            id=str(i),
            packet_id="release",
            role="development_training_only",
            query_id="q",
            candidate_id="g" + str(i),
            query_image="",
            query_text="需求",
            query_rule="用途相符",
            candidate_image="image.jpg",
            candidate_title="商品",
            grade=None,
            note="",
            reviewer="",
            review_origin=None,
        )
        for i in range(3)
    ]
    manifest: dict = dict(
        release_id="release",
        batches=[dict(id="batch_001", row_ids=[r["id"] for r in rows])],
    )
    return manifest, rows


@dataclass(frozen=True)
class TestBusinessReviewJobsSubmissionInputs:
    """Input values for submission.

    Attributes:
        rows: Records processed by this operation.
        slot: Slot used by this operation.
        grades: Grades used by this operation.
        reviewer: Reviewer used by this operation.
        digest: Digest used by this operation.
        batch: Batch used by this operation.
    """

    __test__ = False

    rows: list[dict]
    slot: str
    grades: Sequence[object]
    reviewer: str | None = None
    digest: str | None = None
    batch: str = "batch_001"


def submission(options: TestBusinessReviewJobsSubmissionInputs) -> dict:
    """Return submission for this workflow.

    Args:
        options: Validated input fields described by
            TestBusinessReviewJobsSubmissionInputs.

    Returns:
        Mapping with fields: release_id, batch_id, review_slot, reviewer,
        declared_human, adjudication_of, rows.
    """
    rows = options.rows
    slot = options.slot
    grades = options.grades
    reviewer = options.reviewer
    digest = options.digest
    batch = options.batch
    reviewer = reviewer or dict(A="Alice", B="Bob", adjudicator="Carol")[slot]
    reviewed = copy.deepcopy(rows)
    for row, grade in zip(reviewed, grades):
        row.update(
            grade=grade,
            reviewer=reviewer,
            review_origin="human",
            note="实际理由" if slot == "adjudicator" else "",
        )
    return dict(
        release_id="release",
        batch_id=batch,
        review_slot=slot,
        reviewer=reviewer,
        declared_human=True,
        adjudication_of=digest,
        rows=reviewed,
    )


def test_missing_is_pending_and_disagreement_needs_third_person(packet: dict) -> None:
    """Verify that missing is pending and disagreement needs third person.

    Args:
        packet: Packet used by this operation.
    """
    manifest, rows = packet
    a = submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, None]))
    b = submission(TestBusinessReviewJobsSubmissionInputs(rows, "B", [1, 1, 0]))
    result = merge_values(manifest, rows, [a, b])
    assert [r["grade"] for r in result["rows"]] == [1, None, None]
    assert result["status"]["pending"] == 2
    assert result["status"]["conflicts"] == 1
    assert result["status"]["quality_metrics"] is None
    adjud = submission(
        TestBusinessReviewJobsSubmissionInputs(
            [rows[1]],
            "adjudicator",
            [0],
            digest=result["primary_digest"],
            batch="adjud_001",
        )
    )
    fixed = merge_values(manifest, rows, [a, b], [adjud])
    assert [r["grade"] for r in fixed["rows"]] == [1, 0, None]
    assert fixed["status"]["unresolved_conflicts"] == 0
    assert fixed["status"]["formal_acceptance"] is False


def test_one_slot_cannot_finalize_labels(packet: dict) -> None:
    """Verify that one slot cannot finalize labels.

    Args:
        packet: Packet used by this operation.
    """
    m, rows = packet
    result = merge_values(
        m,
        rows,
        [submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, 1]))],
    )
    assert result["status"]["completed"] == 0
    assert result["status"]["A_completed"] == 3
    assert result["status"]["B_completed"] == 0


def test_reject_same_human_and_conflicting_revisions(packet: dict) -> None:
    """Verify that reject same human and conflicting revisions.

    Args:
        packet: Packet used by this operation.
    """
    m, rows = packet
    a = submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, 1]))
    with pytest.raises(ValueError, match="different reviewers"):
        merge_values(
            m,
            rows,
            [
                a,
                submission(
                    TestBusinessReviewJobsSubmissionInputs(
                        rows, "B", [1, 0, 1], reviewer=" Ａｌｉｃｅ "
                    )
                ),
            ],
        )
    with pytest.raises(ValueError, match="Conflicting batch revisions"):
        merge_values(
            m,
            rows,
            [
                a,
                submission(
                    TestBusinessReviewJobsSubmissionInputs(rows, "A", [0, 0, 1])
                ),
            ],
        )


def test_reject_stale_self_adjudication_and_no_reason(packet: dict) -> None:
    """Verify that reject stale self adjudication and no reason.

    Args:
        packet: Packet used by this operation.
    """
    m, rows = packet
    inputs = [
        submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 1, 1])),
        submission(TestBusinessReviewJobsSubmissionInputs(rows, "B", [0, 1, 1])),
    ]
    digest = merge_values(m, rows, inputs)["primary_digest"]
    item = submission(
        TestBusinessReviewJobsSubmissionInputs(
            [rows[0]], "adjudicator", [0], digest=digest, batch="adjud_001"
        )
    )
    stale = copy.deepcopy(item)
    stale["adjudication_of"] = "old"
    with pytest.raises(ValueError, match="Stale"):
        merge_values(m, rows, inputs, [stale])
    same = submission(
        TestBusinessReviewJobsSubmissionInputs(
            [rows[0]],
            "adjudicator",
            [0],
            reviewer="Alice",
            digest=digest,
            batch="adjud_001",
        )
    )
    with pytest.raises(ValueError, match="third reviewer"):
        merge_values(m, rows, inputs, [same])
    item["rows"][0]["note"] = ""
    with pytest.raises(ValueError, match="reason"):
        merge_values(m, rows, inputs, [item])


@pytest.mark.parametrize("bad", [True, 0.5, "1", -1])
def test_reject_nonbinary_grades(packet: dict, bad: object) -> None:
    """Verify that reject nonbinary grades.

    Args:
        packet: Packet used by this operation.
        bad: Bad used by this operation.
    """
    m, rows = packet
    with pytest.raises(ValueError, match="Grade"):
        merge_values(
            m,
            rows,
            [
                submission(
                    TestBusinessReviewJobsSubmissionInputs(rows, "A", [bad, 0, 1])
                )
            ],
        )


def test_reject_immutable_fields_and_wrong_release(packet: dict) -> None:
    """Verify that reject immutable fields and wrong release.

    Args:
        packet: Packet used by this operation.
    """
    m, rows = packet
    value = submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, 1]))
    value["rows"][0]["query_rule"] = "放宽"
    with pytest.raises(ValueError, match="Immutable"):
        merge_values(m, rows, [value])
    value = submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, 1]))
    value["release_id"] = "other"
    with pytest.raises(ValueError, match="release"):
        merge_values(m, rows, [value])


def test_source_family_detects_cross_split_and_merchant_model() -> None:
    """Verify that source family detects cross split and merchant model."""
    base: dict = dict(
        id="product/train/123",
        text="示例 手机",
        url="http://img.alicdn.com/imgextra/i1/7788/foo.jpg?x=1",
        label="手机",
        attributes={"品牌": ["品牌甲"], "型号": ["X9"], "货号": ["AA"]},
    )
    keys = product_family_keys(base)
    other: dict = {
        **base,
        "id": "product/test/456",
        "text": "不同标题",
        "url": "https://img.alicdn.com/imgextra/i1/7788/other.jpg",
    }
    assert "brand_model:手机|品牌甲|x9" in keys & product_family_keys(other)
    assert "merchant_sku:手机|7788|aa" in keys & product_family_keys(other)
    cross: dict = {
        **base,
        "id": "product/test/123",
        "text": "全新标题",
        "url": "",
        "attributes": {},
    }
    assert "id:123" in keys & product_family_keys(cross)
    sameurl: dict = {
        **base,
        "id": "different",
        "text": "新",
        "url": base["url"].replace("http:", "https:").split("?")[0],
        "attributes": {},
    }
    assert any(k.startswith("url:") for k in keys & product_family_keys(sameurl))


def test_browser_real_export_roundtrip(packet: dict, tmp_path: Path) -> None:
    """Verify that browser real export roundtrip.

    Args:
        packet: Packet used by this operation.
        tmp_path: Isolated temporary directory supplied by pytest.

    Notes:
        Writes derived artifacts through source.write_text.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    _, rows = packet
    job: dict = dict(
        release_id="release",
        batch_id="batch_001",
        review_slot="A",
        adjudication_of=None,
        rows=rows,
    )
    source = tmp_path / "job.json"
    source.write_text(json.dumps(job), encoding="utf-8")
    js = Path(__file__).resolve().parents[1] / "assets/business_review_jobs.js"
    code = "const fs=require('fs'),m=require(process.argv[1]),j=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));const r=j.rows.map((x,i)=>({...x,grade:i?null:1,reviewer:'Alice',review_origin:'human'}));process.stdout.write(JSON.stringify(m.makeSubmission(j,r,'Alice',true)));"
    proc = subprocess.run(
        [node, "-e", code, str(js), str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(proc.stdout)
    manifest: dict = {
        "release_id": "release",
        "batches": [dict(id="batch_001", row_ids=[r["id"] for r in rows])],
    }
    result = merge_values(manifest, rows, [value])
    assert result["status"]["A_completed"] == 1
    assert result["status"]["pending"] == 3


def test_complete_agreement_can_compute_gallery_recall_but_not_formal_metrics(
    packet: dict,
) -> None:
    """Verify that complete agreement can compute gallery recall but not formal
    metrics.

    Args:
        packet: Packet used by this operation.
    """
    m, rows = packet
    result = merge_values(
        m,
        rows,
        [
            submission(TestBusinessReviewJobsSubmissionInputs(rows, "A", [1, 0, 1])),
            submission(TestBusinessReviewJobsSubmissionInputs(rows, "B", [1, 0, 1])),
        ],
    )
    assert result["status"]["pending"] == 0
    assert result["status"]["full_gallery_recall_computable"] is True
    assert result["status"]["quality_metrics"] is None
    assert result["status"]["formal_acceptance"] is False
