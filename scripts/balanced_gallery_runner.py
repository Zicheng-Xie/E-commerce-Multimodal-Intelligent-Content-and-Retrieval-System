"""Continue the frozen supplement pipeline without evicting resident user work.

The already-running native review is observed through immutable local receipts.
No duplicate reviewer is started. A failure remains a failure and stops this
runner; it never becomes a negative/unknown relation or an alternate model.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import balanced_gallery_experiment as experiment
import balanced_pair_review as review
import balanced_supplement_features as features
import recover_balanced_media as recovery
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_runner"


def review_failure() -> dict | None:
    """Return review failure for this workflow.

    Returns:
        Mapping with fields: file, exception.
    """
    for path in sorted((review.ROOT / "attempts").rglob("*.json")):
        value = load(path)
        if value.get("exception") is not None:
            return dict(file=rel(path), exception=value["exception"])
    return None


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    experiment.guard()
    if (ROOT / "execution_source_manifest.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    assert load(ROOT / "source_pinning_checks.pre.json")["passed"]
    paths = [
        PROJECT_ROOT / "scripts/balanced_gallery_runner.py",
        PROJECT_ROOT / "tests/test_balanced_gallery_runner.py",
        ROOT / "engineering_checks.pre.json",
        ROOT / "source_pinning_checks.pre.json",
        experiment.ROOT / "protocol.json",
        features.ROOT / "protocol.json",
        review.ROOT / "protocol.json",
    ]
    files = dict(load(experiment.ROOT / "protocol.json")["files"])
    files.update({rel(p): sha256_file(p) for p in paths})
    # Pin the exact old public arrays read by the new process as well as the
    # containing seals. A changed referenced array must not pass merely because
    # the containing completion-receipt JSON has stayed unchanged.
    public = [
        experiment.DATA / "inputs.json",
        experiment.DATA / "feature_paths.json",
        experiment.DATA / "roi_feature_index.json",
        experiment.CROP / "crop_inputs.json",
    ]
    for mapping in load(experiment.DATA / "feature_paths.json").values():
        public += [PROJECT_ROOT / name for name in mapping.values()]
    public += [
        PROJECT_ROOT / r["file"]
        for r in load(experiment.DATA / "roi_feature_index.json").values()
    ]
    for cohort in experiment.COHORTS:
        public += [
            experiment.DATA / cohort / "queries.public.json",
            experiment.DATA / cohort / "catalog.public.json",
        ]
        for query in load(experiment.DATA / cohort / "queries.public.json"):
            public += [experiment.bank_path(cohort, query["id"]) / "vectors.npz"]
            if query["image_path"]:
                public += [
                    Path(query["image_path"]),
                    experiment.bank_path(cohort, query["id"]) / "description.json",
                    experiment.DATA
                    / "query_captions"
                    / cohort
                    / (query["id"] + ".json"),
                    experiment.DATA
                    / "new_query_features"
                    / cohort
                    / (query["id"] + ".npz"),
                ]
    files.update({rel(p): sha256_file(p) for p in public})
    once(
        ROOT / "execution_source_manifest.json",
        dict(
            files=files,
            native_review="Observe existing native reviewer receipt, never launch another reviewer",
            stages=[
                "candidate-only profiles",
                "full Qwen",
                "full DINO",
                "full FG",
                "derived exposure",
                "shared public pair plan",
                "full reranker/exact cache",
                "full rankings",
                "evaluation",
                "replay and seal",
            ],
            GPU="Finite nonnegative nvidia-smi memory below 1500MiB, then explicit empty Ollama resident-model list before admission; preserve all user work and keep-alive, including our own natural native expiry",
            native_errors="Preserve evidence and stop, no automatic schema relaxation or model replacement",
            defaults="Original; never auto promote or open independent acceptance",
            full_models_only=True,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "execution_source_manifest.json")["files"])
    experiment.guard()


@contextmanager
def locked() -> Iterator[bool]:
    """Return locked for this workflow.

    Yields:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = PROJECT_ROOT / "tmp/module2_balanced_runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        logging.getLogger(__name__).debug("Handled exception in locked", exc_info=True)
        handle.close()
        yield False
        return
    try:
        yield True
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def public_status(phase: str, **extra: object) -> None:
    """Apply the public status workflow step.

    Args:
        phase: Phase used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            updated_at=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            completed=phase == "completed",
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
            **extra,
        ),
    )


def preserve_partial_exposure() -> None:
    """Even a failed full-model stage must exclude its generated crop pixels."""
    paths = sorted((features.ROOT / "additions/index").rglob("*.png"))
    paths += sorted((features.ROOT / "detections").rglob("*.png"))
    registry = PROJECT_ROOT / "data/module2_balanced_partial_derived/records.jsonl"
    rows, errors = [], []
    for path in paths:
        try:
            rows.append(
                dict(
                    id="balanced_partial/" + rel(path),
                    image_path=str(path.resolve()),
                    fingerprints=fingerprints(path),
                    project_role="exposed_development_failure_derived_not_acceptance",
                )
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Handled exception in preserve_partial_exposure", exc_info=True
            )
            errors.append(
                dict(path=rel(path), sha256=sha256_file(path), error=str(exc))
            )
    if registry.exists():
        assert read_rows(registry) == rows
    else:
        write_rows(registry, rows)
    prior = review.ROOT / "post_review_exclusion_snapshot.json"
    if not prior.exists():
        prior = recovery.ROOT / "post_acquisition_exclusion_snapshot.json"
    snapshot = load(prior)
    once(
        ROOT / "failure_exclusion_snapshot.json",
        dict(
            snapshot,
            files=dict(snapshot["files"], **{rel(registry): sha256_file(registry)}),
            image_fingerprints=snapshot["image_fingerprints"]
            + [r["fingerprints"] for r in rows],
            undecodable_partial_images=errors,
            scope="All prior exposure, all raw supplements and every decodable generated view/crop, including failed stages; remaining corrupt partial paths recorded, independent acceptance remains closed",
        ),
    )


def sync_docs(phase: str) -> None:
    """Apply the sync docs workflow step.

    Args:
        phase: Phase used by this operation.

    Notes:
        Writes derived artifacts through p.write_text.
    """
    completed = phase == "completed"
    failure = phase == "engineering_failure"
    if completed:
        decision = load(experiment.ROOT / "decision.json")
        verdict = (
            "主方法尚未在全部开发范围通过Precision硬目标"
            if not decision["development_precision_target_met"]
            else "主方法在本轮已曝光开发范围通过Precision下界检查，独立验收尚未开始"
        )
        verdict += (
            "；Recall偏好未整体通过"
            if not decision["development_recall_preference_met"]
            else "；Recall偏好在本轮开发范围通过"
        )
        body = f"19件补库的760条AI关系与候选专用描述均已冻结；完整模型特征及315件同图库8方法、320份完整排序对照已完成，重放和封存验证通过。原11840条关系保持不变。{verdict}。全部数字只留内部报告，默认原版、正式指标null，本轮未启动新的独立验收。"
    elif failure:
        body = "新补库流水线发现工程阻塞并已停下，失败证据见artifacts/module2_balanced_runner/failure.json；原始输出、冻结参数、历史标签和原版默认均保留。不能把未完成评审或推理记作实验成绩，正式指标null，本轮未启动新的独立验收。"
    else:
        body = "19件补库的单查询原生双图AI评审及后续完整模型流程正在推进。候选专用描述、全模型编码、315件同图库8方法对照规则均已单独冻结，等待前置收据后串行接续；工程错误会保存原始证据并停下，不改成标签。尚无新的315件图库检索成绩，默认原版、正式指标null，本轮未启动新的独立验收。"
    for filename, link in [
        ("README.md", "docs/module2_learned_fusion.md"),
        ("docs/current_progress.md", "module2_learned_fusion.md"),
    ]:
        p = PROJECT_ROOT / filename
        s = p.read_text(encoding="utf-8")
        a = s.index("<!-- module2_learned_fusion -->")
        b = s.index("<!-- /module2_learned_fusion -->", a) + len(
            "<!-- /module2_learned_fusion -->"
        )
        block = f"<!-- module2_learned_fusion -->\n2026年10月24日：家族隔离有监督融合22方法880份排序已封存，默认方法尚未替换。\n\n{body} 见[进度与完整证据]({link})。\n<!-- /module2_learned_fusion -->"
        p.write_text(s[:a] + block + s[b:], encoding="utf-8")
    p = PROJECT_ROOT / "docs/module2_learned_fusion.md"
    s = p.read_text(encoding="utf-8")
    a = s.index("<!-- balanced_review_state -->")
    b = s.index("<!-- /balanced_review_state -->", a)
    block = f'<!-- balanced_review_state -->\n{body}\n\n运行入口为`{rel(PROJECT_ROOT / "scripts/balanced_gallery_runner.py")} run`，阶段与真实计数见`{rel(ROOT / "public_status.json")}`；执行源码清单每阶段校验，另有并发锁，不会重复启动原生评审。\n'
    p.write_text(s[:a] + block + s[b:], encoding="utf-8")
    p = PROJECT_ROOT / "configs/evaluation_registry.json"
    registry = load(p)
    registry["current_balanced_pipeline_status"] = rel(ROOT / "public_status.json")
    registry["current_balanced_pipeline_execution_manifest"] = rel(
        ROOT / "execution_source_manifest.json"
    )
    registry["current_prepared_development_state"] = phase
    registry["current_balanced_gallery_protocol"] = rel(
        experiment.ROOT / "protocol.json"
    )
    if completed:
        for key in (
            "current_development_experiment",
            "current_completed_development_experiment",
        ):
            registry[key] = rel(experiment.ROOT)
        registry["current_module2_progress_file"] = rel(
            experiment.ROOT / "public_status.json"
        )
        for key in (
            "current_module2_target_status_file",
            "current_development_decision_file",
        ):
            registry[key] = rel(experiment.ROOT / "decision.json")
        registry["current_preparation_exclusion_snapshot"] = rel(
            experiment.ROOT / "post_experiment_exclusion_snapshot.json"
        )
        registry["current_balanced_gallery_report"] = rel(
            experiment.ROOT / "internal_comparison.ai.development.md"
        )
    elif failure and (ROOT / "failure_exclusion_snapshot.json").exists():
        registry["current_preparation_exclusion_snapshot"] = rel(
            ROOT / "failure_exclusion_snapshot.json"
        )
    elif (features.ROOT / "post_features_exclusion_snapshot.json").exists():
        registry["current_preparation_exclusion_snapshot"] = rel(
            features.ROOT / "post_features_exclusion_snapshot.json"
        )
    elif (review.ROOT / "post_review_exclusion_snapshot.json").exists():
        registry["current_preparation_exclusion_snapshot"] = rel(
            review.ROOT / "post_review_exclusion_snapshot.json"
        )
    for prefix, role in [
        (
            rel(ROOT) + "/",
            "frozen_source_guarded_development_pipeline_status_not_acceptance",
        ),
        (
            rel(experiment.ROOT) + "/",
            "exposed_315_gallery_frozen_family_fusion_AI_development_not_acceptance",
        ),
        (
            "data/module2_balanced_partial_derived/",
            "exposed_partial_failure_crop_pixels_not_acceptance",
        ),
    ]:
        if not any(r["path_prefix"] == prefix for r in registry["rules"]):
            registry["rules"].append(
                dict(
                    path_prefix=prefix,
                    role=role,
                    formal_acceptance_claim_allowed=False,
                    reusable_as_unseen=False,
                    development_scores_allowed_in_public_status=False,
                )
            )
    p.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def step() -> str:
    """Return step for this workflow.

    Returns:
        Text produced by this operation.

    Notes:
        Writes derived artifacts through once.
    """
    failure = review_failure()
    if failure:
        raise RuntimeError(f"Native review failed; preserved evidence: {failure}")
    if not (review.ROOT / "label_receipt.json").exists():
        public_status(
            "waiting_existing_native_review",
            native_responses=len(list((review.ROOT / "native").glob("*.json"))),
            planned_native_requests=532,
        )
        return "waiting_existing_native_review"
    guard()
    labels = load(review.ROOT / "label_receipt.json")
    verify_files(review.ROOT, labels["files"])
    assert labels["AI_reviewed_products"] == 19 and labels["new_AI_relations"] == 760
    with recovery.locked() as acquired:
        if not acquired:
            return "waiting_other_pipeline_instance"
        if not (features.ROOT / "profile_receipt.json").exists():
            public_status("candidate_only_profiles_or_waiting_natural_GPU_release")
            if not features.profiles():
                return "waiting_natural_GPU_release_for_profiles"
        if not all(
            (features.ROOT / name).exists()
            for name in (
                "qwen_receipt.json",
                "detection_receipt.json",
                "fg_receipt.json",
            )
        ):
            public_status("full_models_or_waiting_natural_GPU_release")
            if not features.full_models():
                return "waiting_natural_GPU_release_for_full_models"
        if not (features.ROOT / "post_features_exclusion_snapshot.json").exists():
            features.exposure()
        guard()
        if not (experiment.ROOT / "plan_receipt.json").exists():
            public_status("public_pair_plan")
            experiment.plan()
        if not (experiment.ROOT / "reranker_receipt.json").exists():
            public_status("full_reranker_or_waiting_natural_GPU_release")
            if not experiment.infer():
                return "waiting_natural_GPU_release_for_reranker"
        if not (experiment.ROOT / "prediction_receipt.json").exists():
            public_status("complete_rankings")
            experiment.prediction()
        if not (experiment.ROOT / "evaluation_receipt.json").exists():
            public_status("development_evaluation")
            experiment.evaluation()
        if not (experiment.ROOT / "completion_receipt.json").exists():
            public_status("replay_and_seal")
            experiment.finish()
        experiment.checked("completion_receipt.json")
        public_status(
            "completed",
            AI_relations=12600,
            original_relations_unchanged=11840,
            added_relations=760,
            methods=8,
            rankings=320,
            gallery_per_cohort=315,
        )
        sync_docs("completed")
        once(
            ROOT / "completion_receipt.json",
            dict(
                experiment_receipt_sha256=sha256_file(
                    experiment.ROOT / "completion_receipt.json"
                ),
                execution_source_manifest_sha256=sha256_file(
                    ROOT / "execution_source_manifest.json"
                ),
                completed=True,
                formal_metrics=None,
                goal_met=False,
            ),
        )
    return "completed"


def run_pipeline(once_only: bool = False) -> None:
    """Return run pipeline for this workflow.

    Args:
        once_only: Once only used by this operation.

    Raises:
        RuntimeError: Preserve recorded engineering failure; inspect before any
            continuation.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    if (ROOT / "failure.json").exists():
        raise RuntimeError(
            "Preserve recorded engineering failure; inspect before any continuation"
        )
    previous = None
    while True:
        try:
            phase = step()
            if phase != previous:
                print(dict(phase=phase), flush=True)
                if phase != "completed":
                    sync_docs(phase)
                previous = phase
            if phase == "completed" or once_only:
                return
        except Exception as exc:
            once(
                ROOT / "failure.json",
                dict(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    formal_metrics=None,
                    original_defaults_and_seals_preserved=True,
                ),
            )
            preserve_partial_exposure()
            public_status("engineering_failure")
            sync_docs("engineering_failure")
            raise
        # Native receipt observation is local only. GPU admission retries at a
        # slower cadence; no polling of Ollama while nvidia-smi is busy.
        for _ in range(1 if phase == "waiting_existing_native_review" else 4):
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "once", "status"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "status":
        print(load(ROOT / "public_status.json"))
    else:
        with locked() as acquired:
            if acquired:
                run_pipeline(once_only=args.action == "once")
            else:
                print(dict(another_runner_holds_lock=True), flush=True)
