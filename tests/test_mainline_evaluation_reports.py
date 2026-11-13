"""Strict batch exports and immutable annotation-aware evaluation snapshots."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

import module2_development_benchmark as benchmark
import module2_mainline_reports as reports
from module2_human_review_inputs import import_precision_answers
from multimodal_ecommerce import evaluation_access, safety
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from simple_cycle_review import FIELDS


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> benchmark.Evidence:
    """Provide a guarded temporary development-only artifact tree.

    Args:
        tmp_path: Isolated temporary project root.
        monkeypatch: Fixture restoring module-root constants after each test.

    Returns:
        Development-only evidence reader, isolating all test artifacts.
    """
    for module in (benchmark, evaluation_access, safety):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    return benchmark.Evidence(
        dict(
            rules=[
                dict(path_prefix="", role="development"),
                dict(
                    path_prefix="reserved/",
                    role="reserved_acceptance",
                    development_or_training_use_allowed=False,
                ),
            ]
        )
    )


def _packet(folder: Path, size: int = 31) -> tuple[list, dict]:
    (folder / "answers").mkdir(parents=True)
    tasks = [
        dict(task_id=f"t{i}", query_id="q", candidate_id=str(i)) for i in range(size)
    ]
    return tasks, dict(
        q=dict(judgments={str(i): dict(grade=0, human=False) for i in range(size)})
    )


def _row(task: str, grade: str = "1") -> dict:
    return dict(
        task_id=task,
        reviewer_type="human",
        reviewer="test reviewer",
        reviewed_at="2026-11-09T10:00:00Z",
        human_grade=grade,
        reason="原图和标题用途匹配",
    )


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_partial_and_uncertain_batches_keep_explicit_gaps(
    evidence: benchmark.Evidence, tmp_path: Path
) -> None:
    """Record answered and binary coverage separately for both frozen batches.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated packet directory.
    """
    folder = tmp_path / "packet"
    tasks, q = _packet(folder)
    _csv(
        folder / "answers/primary_safeguards_human_batch_001.csv",
        [_row("t0"), _row("t1", "uncertain")],
    )
    r = import_precision_answers(evidence, folder, tasks, q)
    assert r["completed"] == 2 and r["binary_completed"] == 1 and not r["complete"]
    assert r["uncertain_task_ids"] == ["t1"] and len(r["missing_task_ids"]) == 29
    assert r["batches"][0]["expected"] == 30 and r["batches"][1]["expected"] == 1
    assert q["q"]["judgments"]["1"] == dict(grade=None, human=True)


def test_all_twelve_frozen_batches_import_together(
    evidence: benchmark.Evidence, tmp_path: Path
) -> None:
    """Accept 341 distinct answers in the exact 30-by-eleven plus eleven packet.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated packet directory.
    """
    folder = tmp_path / "packet"
    tasks, q = _packet(folder, 341)
    for index in range(12):
        _csv(
            folder / f"answers/primary_safeguards_human_batch_{index + 1:03}.csv",
            [_row(t["task_id"]) for t in tasks[index * 30 : (index + 1) * 30]],
        )
    result = import_precision_answers(evidence, folder, tasks, q)
    assert result["complete"] and result["binary_completed"] == 341
    assert [b["expected"] for b in result["batches"]] == [30] * 11 + [11]
    assert not result["missing_task_ids"] and not result["uncertain_task_ids"]
    assert all(row == dict(grade=1, human=True) for row in q["q"]["judgments"].values())


@pytest.mark.parametrize(
    "defect",
    [
        "foreign_blank",
        "duplicate_blank",
        "duplicate_file",
        "wrong_batch",
        "mixed_batches",
        "reviewer",
        "machine_reviewer",
        "timestamp",
        "reason",
        "header",
        "malformed",
    ],
)
def test_bad_export_identity_or_human_attestation_is_rejected(
    evidence: benchmark.Evidence, tmp_path: Path, defect: str
) -> None:
    """Reject errors before accepting any row into the effective human labels.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated packet directory.
        defect: Deliberate malformed or incorrectly bound export.
    """
    folder = tmp_path / "packet"
    tasks, q = _packet(folder)
    rows = [_row("t0")]
    name = "human_batch_001.csv"
    if defect == "foreign_blank":
        rows = [_row("foreign", "")]
    elif defect == "duplicate_blank":
        rows = [_row("t0", ""), _row("t0")]
    elif defect == "duplicate_file":
        _csv(folder / "answers/second_export.csv", rows)
    elif defect == "wrong_batch":
        name = "human_batch_002.csv"
    elif defect == "mixed_batches":
        rows.append(_row("t30"))
    elif defect == "reviewer":
        rows[0]["reviewer"] = ""
    elif defect == "machine_reviewer":
        rows[0]["reviewer_type"] = "AI"
    elif defect == "timestamp":
        rows[0]["reviewed_at"] = "not a timestamp"
    elif defect == "reason":
        rows[0]["reason"] = ""
    _csv(folder / "answers" / name, rows)
    if defect == "header":
        (folder / "answers" / name).write_text("task_id,human_grade\nt0,1\n")
    elif defect == "malformed":
        with (folder / "answers" / name).open("a") as f:
            f.write("t1\n")
    with pytest.raises(ValueError):
        import_precision_answers(evidence, folder, tasks, q)
    assert all(not row["human"] for row in q["q"]["judgments"].values())


def test_reserved_exports_are_rejected_before_reading(
    evidence: benchmark.Evidence, tmp_path: Path
) -> None:
    """A CSV-looking filename cannot bypass the reserved-data access registry.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated packet directory.
    """
    folder = tmp_path / "reserved/packet"
    tasks, q = _packet(folder)
    _csv(folder / "answers/human_batch_001.csv", [_row("t0")])
    with pytest.raises(PermissionError):
        import_precision_answers(evidence, folder, tasks, q)


def _status() -> dict:
    order = [str(i) for i in range(12)]
    q = dict(
        mode="image",
        expected_pairs=12,
        judgments={c: dict(grade=1, human=True) for c in order},
    )
    target = dict(k=10, precision_at_10_required=0.8, recall_at_10_preferred=0.9)
    human = evaluate_human_retrieval({"q": order}, {"q": q}, target)
    mixed = {
        scope: dict(
            ranges={
                "precision@10": row["precision_at_10"]["bounds"],
                "recall@10": row["recall_at_10"]["bounds"],
            }
        )
        for scope, row in human["scopes"].items()
    }
    return dict(
        mainline_id="test",
        products=12,
        queries=1,
        frozen_rankings_sha256="ranking-hash",
        target_contract_sha256="target-hash",
        label_fingerprints=reports.label_fingerprints({"q": order}, {"q": q}),
        human_evaluation=human,
        candidate=dict(scopes=mixed),
        confirmed_image_gain=None,
    )


def test_snapshot_reuse_is_immutable_and_corruption_fails(
    evidence: benchmark.Evidence, tmp_path: Path
) -> None:
    """Repeated status calls preserve a snapshot; a modified snapshot is never overwritten.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated output directory.
    """
    status = _status()
    out = tmp_path / "output"
    first = reports.publish_evaluation(out, status, [], {})
    report = Path(first["report_path"])
    before = report.read_bytes()
    second = reports.publish_evaluation(out, copy.deepcopy(status), [], {})
    assert first == second and report.read_bytes() == before
    report.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable evaluation snapshot was modified"):
        reports.publish_evaluation(out, status, [], {})


def test_new_labels_create_snapshot_without_crediting_a_model_gain(
    evidence: benchmark.Evidence, tmp_path: Path
) -> None:
    """Changing label evidence changes identity while the prediction fingerprint stays fixed.

    Args:
        evidence: Guarded temporary development reader.
        tmp_path: Isolated output directory.
    """
    status = _status()
    out = tmp_path / "output"
    a = reports.publish_evaluation(out, status, [], {})
    current = copy.deepcopy(status)
    current["label_fingerprints"]["human_labels_sha256"] = "new-human-labels"
    b = reports.publish_evaluation(out, current, [], {})
    assert a["id"] != b["id"]
    transition = reports.evaluation_transition(status, current)
    assert (
        transition["kind"] == "same_predictions_new_labels"
        and not transition["metric_delta_is_model_gain"]
    )
    current["frozen_rankings_sha256"] = "changed-rankings"
    assert (
        reports.evaluation_transition(status, current)["kind"]
        == "both_changed_require_same_label_replay"
    )
    same_labels = copy.deepcopy(status)
    same_labels["frozen_rankings_sha256"] = "new-predictions"
    transition = reports.evaluation_transition(status, same_labels)
    assert transition["model_gain_attribution_eligible"]
    assert not transition["metric_delta_is_model_gain"]


def test_label_fingerprint_does_not_change_when_only_order_changes() -> None:
    """Separate a permutation change from an update to relevance observations."""
    order = [str(i) for i in range(12)]
    q = dict(
        judgments={"0": dict(grade=1, human=True), "1": dict(grade=0, human=False)}
    )
    assert reports.label_fingerprints(
        {"q": order}, {"q": q}
    ) == reports.label_fingerprints({"q": list(reversed(order))}, {"q": q})
