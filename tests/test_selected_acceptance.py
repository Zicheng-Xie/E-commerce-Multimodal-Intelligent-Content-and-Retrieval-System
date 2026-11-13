"""Synthetic rehearsal of independent labeling, one-shot consumption and exact resume."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from multimodal_ecommerce.safety import sha256_file
from multimodal_ecommerce.selected_acceptance import acceptance_run
from multimodal_ecommerce.selected_engine import SelectedEngine
from multimodal_ecommerce.selected_evaluation import evaluate_sealed_predictions
from multimodal_ecommerce.selected_gold import (
    FIELDS,
    freeze_gold,
    prepare_packet,
    text_template,
)
from multimodal_ecommerce.selected_release import acceptance_preflight, verify_release
from multimodal_ecommerce.selected_runtime_io import (
    RuntimeAccess,
    atomic_json,
    digest,
    load,
)
from test_selected_runtime import _Models, workspace  # noqa: F401


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def reservation(workspace: tuple) -> tuple:  # noqa: F811 - imported pytest fixture
    """Build entirely synthetic reserved media, a model freeze and blinded labels.

    Args:
        workspace: Temporary public inputs and patched project roots.

    Returns:
        Temporary root, policy, release, reservation and private gold directory.
    """
    root, policy, catalog, queries = workspace
    target = load(
        Path(__file__).resolve().parents[1] / "configs/module2_target_p80_r90.json"
    )
    holdout, gold = root / "artifacts/heldout", root / "artifacts/gold"
    holdout.mkdir(parents=True)
    gallery = []
    for i, row in enumerate(load(catalog)):
        image = holdout / f"media/{i}.png"
        image.parent.mkdir(exist_ok=True)
        shutil.copyfile(row["image_path"], image)
        gallery.append(
            dict(
                id=row["id"],
                text=row["title"],
                image_path=f"media/{i}.png",
                fingerprints=dict(sha256=sha256_file(image)),
            )
        )
    image = holdout / "media/query.png"
    shutil.copyfile(load(queries)[1]["image_path"], image)
    atomic_json(holdout / "gallery.reserved.json", gallery)
    atomic_json(
        holdout / "image_queries.reserved.json",
        [
            dict(
                image_path="media/query.png",
                fingerprints=dict(sha256=sha256_file(image)),
            )
        ],
    )
    fields = ["fine_type", "human_author", "independent_query_text", "approved_at"]
    _csv(
        holdout / "text_queries.human_pending.csv",
        fields,
        [dict.fromkeys(fields, "") | dict(fine_type="headphone")],
    )
    files = {
        str(p.relative_to(holdout)): sha256_file(p)
        for p in holdout.rglob("*")
        if p.is_file()
    }
    atomic_json(
        holdout / "seal.json",
        dict(
            files=files,
            gallery=24,
            image_queries=1,
            independently_authored_text_queries_pending=1,
            final_model_freeze_required=True,
            model_predictions=0,
            human_relevance_labels=0,
            AI_relevance_labels=0,
        ),
    )
    registry_path = root / "configs/evaluation_registry.json"
    registry = load(registry_path)
    registry["current_final_acceptance_root"] = "artifacts/heldout"
    atomic_json(registry_path, registry)
    release = dict(
        policy=policy,
        model_policy=_Models(policy).identity(),
        files_sha256={},
        target=target,
        relevance_instruction="Original relevance definition; no ranking or AI advice.",
    )
    release["release_id"] = digest(release)
    release_path = root / "artifacts/release.json"
    atomic_json(release_path, release)
    text_template(release_path, holdout, gold)
    _csv(
        gold / "text_queries.csv",
        fields,
        [
            dict(
                fine_type="headphone",
                human_author="independent tester",
                independent_query_text="耳机",
                approved_at="2026-11-10T12:00:00Z",
            )
        ],
    )
    prepare_packet(release_path, holdout, gold)
    return root, policy, release_path, holdout, gold


def _answers(gold: Path, uncertain: bool = False) -> None:
    tasks = load(gold / "packet.private.json")["tasks"]
    for start in range(0, len(tasks), 30):
        rows = []
        for i, task in enumerate(tasks[start : start + 30], start):
            rows.append(
                dict(
                    task_id=task["task_id"],
                    reviewer_type="human",
                    reviewer="Synthetic tester",
                    reviewed_at="2026-11-10T12:00:00Z",
                    human_grade=(
                        "uncertain"
                        if uncertain and i == 0
                        else str(int(int(task["candidate_id"][1:]) < 16))
                    ),
                    reason="Synthetic fixture only; no actual acceptance data",
                )
            )
        _csv(gold / f"review/answers/batch_{start // 30 + 1:03}.csv", FIELDS, rows)


def _worker(job: dict, policy: dict) -> None:
    access = RuntimeAccess(job["claim"], job["token"])
    claim = load(job["claim"])
    with pytest.raises(PermissionError, match="gold"):
        access.check(Path(claim["private_gold_root"]) / "gold.private.json")
    with pytest.raises(PermissionError, match="model differs"):
        access.validate_model(dict(other="substituted model"))
    work = job["work"]
    engine = SelectedEngine(
        policy, work / "cache", backend=_Models(policy), access=access
    )
    engine.index(work / "public_inputs/catalog.json", work / "index")
    engine.predict(
        work / "index", work / "public_inputs/queries.json", work / "predictions"
    )


def test_one_shot_consumes_before_inference_and_never_restarts(
    reservation: tuple,
) -> None:
    """Only an exact frozen run can execute, and completed calls only return receipts.

    Args:
        reservation: Completely synthetic acceptance fixture.
    """
    root, policy, release, holdout, gold = reservation
    _answers(gold)
    assert freeze_gold(release, holdout, gold)["binary_complete"]
    assert acceptance_preflight(release, holdout, gold)["ready"]
    calls = []

    def worker(job: dict) -> None:
        """Verify consumption before running the isolated synthetic engine.

        Args:
            job: Claimed one-shot worker inputs.
        """
        calls.append(job)
        registry = load(root / "configs/evaluation_registry.json")
        assert next(
            r for r in registry["rules"] if r["path_prefix"] == "artifacts/heldout/"
        )["consumed"]
        with pytest.raises(PermissionError):
            RuntimeAccess(job["claim"], "forged credential")
        _worker(job, policy)

    receipt = acceptance_run(release, holdout, gold, worker)
    assert receipt["dataset_consumed"] and not receipt["goal_met"]
    assert acceptance_run(release, holdout, gold, worker) == receipt
    assert len(calls) == 1
    assert (
        "dataset_already_consumed"
        in acceptance_preflight(release, holdout, gold)["reasons"]
    )
    with pytest.raises(PermissionError, match="relabeled"):
        freeze_gold(release, holdout, gold)
    changed = load(release)
    changed["policy"]["version"] = "another strategy"
    changed.pop("release_id")
    changed["release_id"] = digest(changed)
    other = root / "artifacts/another_release.json"
    atomic_json(other, changed)
    with pytest.raises(PermissionError, match="another model"):
        acceptance_run(other, holdout, gold, worker)


def test_failed_run_resumes_only_identical_job_and_detects_tampering(
    reservation: tuple,
) -> None:
    """The consumed claim survives interruption; credentials, outputs and gold stay bound.

    Args:
        reservation: Synthetic protected media and human packet.
    """
    root, policy, release, holdout, gold = reservation
    _answers(gold)
    freeze_gold(release, holdout, gold)
    seen = []

    def fail(job: dict) -> None:
        """Simulate an interruption while preserving the exact run credential.

        Args:
            job: Already claimed synthetic worker job.
        """
        seen.append(load(job["claim"])["token"])
        raise RuntimeError("interrupt before worker completion")

    with pytest.raises(RuntimeError, match="interrupt"):
        acceptance_run(release, holdout, gold, fail)
    assert (
        len(list((root / "artifacts/module2_acceptance_runs").glob("*/claim.json")))
        == 1
    )

    def resume(job: dict) -> None:
        """Continue using the original credential and unchanged model/data binding.

        Args:
            job: Same frozen job as the failed invocation.
        """
        assert seen == [job["token"]]
        _worker(job, policy)

    receipt = acceptance_run(release, holdout, gold, resume)
    assert receipt["state"] == "complete"
    path = next((root / "artifacts/module2_acceptance_runs").glob("*/metrics.json"))
    atomic_json(path, dict(changed=True))
    with pytest.raises(ValueError, match="altered"):
        acceptance_run(release, holdout, gold, resume)


def test_uncertain_gold_is_never_forced_binary_and_capacity_is_honest(
    reservation: tuple,
) -> None:
    """Complete uncertain answers produce ranges and cannot claim formal success.

    Args:
        reservation: Synthetic exhaustive human tasks.
    """
    _, _, release, holdout, gold = reservation
    _answers(gold, uncertain=True)
    status = freeze_gold(release, holdout, gold)
    assert status["all_answers_complete"] and not status["binary_complete"]
    assert status["unresolved"] == 1 and status["gold_sha256"]
    assert acceptance_preflight(release, holdout, gold)["ready"]
    packet = load(gold / "packet.private.json")
    order = [r["id"] for r in packet["catalog"]]
    predictions: dict = dict(
        queries={
            q["id"]: dict(
                ranking=order, baselines=dict(global_=order, region_title=order)
            )
            for q in packet["queries"]
        }
    )
    for row in predictions["queries"].values():
        row["baselines"]["global"] = row["baselines"].pop("global_")
    metrics = evaluate_sealed_predictions(
        predictions,
        packet,
        load(gold / "gold.private.json"),
        verify_release(release)["target"],
    )
    assert metrics["formal_metrics"]["text"]["precision@10"] is None
    assert metrics["metric_intervals"]["text"]["selected"]["precision@10"] == [0.9, 1.0]
    assert (
        metrics["human_evaluation"]["query_metrics"]["image_000"][
            "ideal_recall_at_10_upper"
        ]
        == 0.625
    )
    assert not metrics["goal_met"] and metrics["no_query_dropped"]


@pytest.mark.parametrize(
    "defect", ["duplicate", "mixed_batch", "not_human", "no_reason", "packet_tamper"]
)
def test_independent_answer_validation_rejects_invalid_exports(
    reservation: tuple, defect: str
) -> None:
    """Foreign or modified human exports fail before becoming frozen truth.

    Args:
        reservation: Synthetic independent annotation fixture.
        defect: Corruption or invalid human-answer declaration.
    """
    _, _, release, holdout, gold = reservation
    _answers(gold)
    path = gold / "review/answers/batch_001.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "mixed_batch":
        rows[-1]["task_id"] = load(gold / "packet.private.json")["tasks"][-1]["task_id"]
    elif defect == "not_human":
        rows[0]["reviewer_type"] = "model"
    elif defect == "no_reason":
        rows[0]["reason"] = ""
    else:
        atomic_json(gold / "packet.private.json", dict(changed=True))
    _csv(path, FIELDS, rows)
    with pytest.raises(ValueError):
        freeze_gold(release, holdout, gold)


def test_mutated_reservation_is_rejected_before_any_inference(
    reservation: tuple,
) -> None:
    """A changed sealed file cannot consume an acceptance attempt or start its worker.

    Args:
        reservation: Synthetic sealed media.
    """
    _, _, release, holdout, gold = reservation
    _answers(gold)
    freeze_gold(release, holdout, gold)
    (holdout / "media/0.png").write_bytes(b"changed media")
    with pytest.raises(ValueError, match="Reserved data changed"):
        acceptance_run(
            release, holdout, gold, lambda job: pytest.fail("must not start")
        )


def test_frozen_model_files_and_release_manifest_cannot_change(
    reservation: tuple,
) -> None:
    """The release checksum and actual bound assets are both enforced before inference.

    Args:
        reservation: Synthetic release, never real model checkpoints.
    """
    root, _, release, _, _ = reservation
    asset = root / "synthetic_weights.bin"
    asset.write_bytes(b"synthetic fixed weights")
    value = load(release)
    value.pop("release_id")
    value["files_sha256"] = {str(asset): sha256_file(asset)}
    value["release_id"] = digest(value)
    atomic_json(release, value)
    verify_release(release)
    asset.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="release file changed"):
        verify_release(release)
    value["target"]["precision_at_10_required"] = 0.5
    atomic_json(release, value)
    with pytest.raises(ValueError, match="identity changed"):
        verify_release(release)


def test_readiness_cannot_read_private_answers_or_leak_status_labels(
    reservation: tuple,
) -> None:
    """Every development readiness path rejects answer files and nonmetadata fields.

    Args:
        reservation: Synthetic public release and separate private answer directory.
    """
    _, _, release, holdout, gold = reservation
    _answers(gold)
    freeze_gold(release, holdout, gold)
    with pytest.raises(PermissionError):
        verify_release(gold / "gold.private.json")
    status_path = gold / "public_status.json"
    status = load(status_path)
    status["answers"] = [dict(grade=1)]
    atomic_json(status_path, status)
    with pytest.raises(PermissionError, match="only public counts"):
        acceptance_preflight(release, holdout, gold)
