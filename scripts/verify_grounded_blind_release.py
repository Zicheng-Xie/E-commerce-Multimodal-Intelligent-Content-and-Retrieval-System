"""Read-only full release verification, including both execution recoveries."""

from __future__ import annotations

import grounded_blind_dataset as dataset
import grounded_blind_execution_recovery as execution
import grounded_blind_report as report
import grounded_blind_source_recovery as source
import run  # noqa: F401
from learned_fusion_experiment import load
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    dataset.guard()
    source.guard()
    execution.guard()
    sources = load(dataset.ROOT / "additional_reproducibility_sources.json")
    verify_files(PROJECT_ROOT, sources["files"])
    verify_files(dataset.ROOT / "reproducibility_sources", sources["files"])
    receipt = load(dataset.ROOT / "source_range_execution_protocol.json")
    assert sha256_file(PROJECT_ROOT / receipt["source"]) == receipt["source_sha256"]
    report.verify()
    labels = dataset.checked("label_receipt.json")
    assert labels["AI_relations"] == 2304 and labels["human_labels"] == 0
    checks = load(dataset.ROOT / "visual_replay_check.json")
    assert checks["passed"] and checks["cards"] == 960
    print(
        dict(
            complete_release_source_data_rank_and_metric_replay_verified=True,
            formal_metrics=None,
            default_strategy_changed=False,
        )
    )


if __name__ == "__main__":
    verify()
