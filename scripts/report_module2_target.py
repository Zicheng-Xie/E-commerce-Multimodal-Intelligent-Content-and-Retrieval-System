"""Report current goals from verified human consensus without rewriting history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.current_target import assess_current_target
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.human_business_pipeline import REVIEW, validate_consensus
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

OUT = PROJECT_ROOT / "artifacts/module2_target_p60_r85"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    contract_path = PROJECT_ROOT / "configs/module2_current_target.json"
    contract = load(contract_path)
    current_path = (
        PROJECT_ROOT
        / "artifacts/module2_continuation/human_pipeline/current_status.json"
    )
    status = load(current_path)
    revision = (PROJECT_ROOT / status["revision"]).resolve()
    revision.relative_to(current_path.parent / "revisions")
    receipt = load(revision / "receipt.json")
    verify_files(revision, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["implementation"])
    source_status = load(revision / "status.json")
    assert all(status[k] == v for k, v in source_status.items())
    merge = (PROJECT_ROOT / status["source_merge"]).resolve()
    merge.relative_to((REVIEW / "merges").resolve())
    assert sha256_file(merge / "judgments.jsonl") == receipt["source_label_sha256"]
    assert (
        sha256_file(merge / "input_provenance.json")
        == receipt["source_provenance_sha256"]
    )
    verify_files(REVIEW, load(REVIEW / "release_receipt.json")["files"])
    consensus = validate_consensus(
        load(REVIEW / "manifest.json"),
        read_rows(REVIEW / "judgments.blank.jsonl"),
        load(merge / "input_provenance.json"),
        read_rows(revision / "human_consensus.jsonl"),
    )["status"]
    assert consensus["completed"] == status["human_completed"]
    summary, by_mode = None, None
    if consensus["pending"] == 0:
        metrics_path = revision / "human_development_metrics.json"
        assert "human_development_metrics.json" in receipt["files"]
        metrics = load(metrics_path)
        assert metrics["formal_acceptance"] is False
        summary, by_mode = metrics["quality_metrics"], metrics["metrics_by_mode"]
    report = dict(
        assessment=assess_current_target(summary, by_mode, contract),
        human_completed=consensus["completed"],
        human_pending=consensus["pending"],
        quality_metrics=summary,
        metrics_by_mode=by_mode,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        acceptance_model_predictions=0,
    )
    inputs = dict(
        contract_sha256=sha256_file(contract_path),
        human_revision=revision.relative_to(PROJECT_ROOT).as_posix(),
        human_receipt_sha256=sha256_file(revision / "receipt.json"),
        implementation={
            name: sha256_file(PROJECT_ROOT / name)
            for name in [
                "src/multimodal_ecommerce/current_target.py",
                "scripts/report_module2_target.py",
            ]
        },
    )
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]
    target = OUT / "revisions" / key
    if (target / "receipt.json").exists():
        verify_files(target, load(target / "receipt.json")["files"])
        assert load(target / "report.json") == report
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "inputs.json", inputs)
        write_json(target / "contract.json", contract)
        write_json(target / "report.json", report)
        write_json(
            target / "receipt.json",
            dict(
                files={p.name: sha256_file(p) for p in target.iterdir() if p.is_file()}
            ),
        )
    write_json(
        OUT / "current_status.json",
        dict(report, revision=target.relative_to(PROJECT_ROOT).as_posix()),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
