"""Query-only grounding repair on exposed development; immutable staged evidence."""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import balanced_gallery_experiment as original
import balanced_supplement as admission
import public_conflict_experiment as parent
import run  # noqa: F401
import subject_attribute_experiment as attributes
from finalize_balanced_gallery import report_cells
from grounded_subject_runtime import (
    OPTIONS,
    POLICY,
    PROMPT,
    REVIEW_PROMPT,
    REVIEW_SCHEMA,
    SCHEMA,
    GroundedSubjectRuntimeFuseInputs,
    fuse,
    localize,
    proposal,
)
from learned_fusion_experiment import load, np_once, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.machine_teacher import resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_subject_experiment import pair_identity
from querywise_paired_bounds import paired_querywise
from roi_semantic_gate_review import offline_model_manifest

ROOT = PROJECT_ROOT / "artifacts/module2_grounded_subject_repair"
DATA = original.ROOT
COHORTS = original.COHORTS
NEW = ("grounded_whole_shared_union", "grounded_zoom_shared_union", POLICY["primary"])
PRIMARY = NEW[2]
METHODS = (*parent.METHODS, *NEW)
BASELINES = (
    "global",
    "region_title",
    "cross20",
    "whole_crop_context_union",
    parent.PRIMARY,
)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [
    (PRIMARY, NEW[0]),
    (PRIMARY, NEW[1]),
]
SOURCES = [
    "scripts/grounded_subject_runtime.py",
    "scripts/grounded_subject_experiment.py",
    "tests/test_grounded_subject_runtime.py",
    "scripts/balanced_gallery_experiment.py",
    "scripts/querywise_paired_bounds.py",
    "scripts/primary_subject_experiment.py",
]
REGISTRY = PROJECT_ROOT / "data/module2_grounded_subject_derived/records.jsonl"


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(ROOT / name, dict(files={rel(p): sha256_file(p) for p in paths}, **extra))


def checked(name: str) -> Any:
    """Return checked for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    value = load(ROOT / name)
    verify_files(PROJECT_ROOT, value["files"])
    return value


def status(phase: str, **extra: object) -> None:
    """Apply the status workflow step.

    Args:
        phase: Phase used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase=phase,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
            **extra,
        ),
    )


def queries() -> list:
    """Return queries for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [
        (c, q)
        for c in COHORTS
        for q in load(DATA / c / "queries.public.json")
        if q["image_path"]
    ]


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    parent.verify()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    once(ROOT / "grounder_model_manifest.json", offline_model_manifest())
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    paths += [
        ROOT / "engineering_checks.pre.json",
        ROOT / "grounder_model_manifest.json",
        parent.ROOT / "completion_receipt.json",
        parent.ROOT / "post_experiment_exclusion_snapshot.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
        PROJECT_ROOT / "configs/local.json",
        PROJECT_ROOT / "configs/embedding_model.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
        PROJECT_ROOT / "vendor/qwen3_vl_embedding/qwen3_vl_embedding.py",
    ]
    for c in COHORTS:
        paths += [
            DATA / c / n
            for n in (
                "queries.public.json",
                "catalog.public.json",
                "judgments.ai.jsonl",
            )
        ]
        paths += [
            parent.ROOT / c / "predictions.json",
            parent.ROOT / c / "internal_metrics.ai.development.json",
        ]
    for c, q in queries():
        paths += [
            Path(q["image_path"]),
            DATA / c / "runs" / q["id"] / "features.json",
            DATA / c / "runs" / q["id"] / "caption_features.json",
            original.CROP / "query_rois" / c / q["id"] / "result.json",
        ]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            methods=METHODS,
            primary=PRIMARY,
            comparisons=COMPARISONS,
            policy=POLICY,
            prompt=PROMPT,
            schema=SCHEMA,
            review_prompt=REVIEW_PROMPT,
            review_schema=REVIEW_SCHEMA,
            options=OPTIONS,
            role="all-exposed AI development only; never independent acceptance",
            queries=40,
            image_queries=24,
            gallery_per_cohort=315,
            AI_relations=12600,
            K=10,
            human_labels=0,
            grounding="One native query image; full pinned Qwen3.5 model and projector. One new box, then same-image two-panel source/crop verification. No gallery, grades or ranking inputs. Invalid engineering response stops; semantic uncertainty preserves whole query.",
            DINO="All previous complete official query detection outputs preserved as diagnostics; no highest-score selection reused to choose the new query-only box.",
            FG="All prior full FG methods retained unchanged in parent evidence; this repair tests Qwen grounding/crop CE, not a new FG model or score.",
            pool="Stable union of old20, safe-caption20 and actual selected-query embedding top20; all original candidates retained, max60. Crop-rejected input is the full query.",
            base="0.375 old full-gallery rank +0.375 safe-caption rank+0.25 selected-query global cosine rank",
            CE="Exact complete full Qwen whole+safe-caption and zoom+same-caption scores for every pool item; 0.65 average context ranks+0.35 base. Whole/zoom single contexts are shared-pool ablations. Full common base tail retained.",
            text="exact parent cross20",
            models="Complete original checkpoints, layers, dimensions, pixels and context budgets; no score imputation",
            selection="Primary predeclared; no choosing ablation or changing rules after scores. All scopes and six metrics retained, formal acceptance still needs entirely new data.",
            formal_metrics=None,
            default_promotion_allowed=False,
        ),
    )
    status("protocol_frozen")


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    value = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, value["files"])
    parent.verify()
    assert value["policy"] == POLICY and value["primary"] == PRIMARY


def grounding() -> bool:
    """Return grounding for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    pending = [
        (c, q)
        for c, q in queries()
        if not (ROOT / "queries" / c / q["id"] / "result.json").exists()
    ]
    if pending:
        if not admission.gpu_available():
            status("waiting_natural_GPU_release_for_grounder")
            return False
        client = OllamaClient(timeout=360)
        try:
            model = resolve_local_teacher(
                client,
                Path(load(PROJECT_ROOT / "configs/local.json")["ollama_models_root"]),
            )
            assert (
                model["digest"] == load(ROOT / "grounder_model_manifest.json")["digest"]
            )
            once(ROOT / "native_runtime_model.json", model)
            status("query_only_grounding_and_verification")
            for c, q in pending:
                result = localize(
                    Path(q["image_path"]),
                    ROOT / "queries" / c / q["id"],
                    client,
                    once,
                    load,
                )
                print(
                    dict(grounded=f"{c}/{q['id']}", ROI_accepted=result["accepted"]),
                    flush=True,
                )
        finally:
            client.close()
    results = [
        dict(
            cohort=c,
            query_id=q["id"],
            **load(ROOT / "queries" / c / q["id"] / "result.json"),
        )
        for c, q in queries()
    ]
    once(ROOT / "grounding_results.json", results)
    receipt(
        "grounding_receipt.json",
        [
            ROOT / "grounding_results.json",
            ROOT / "native_runtime_model.json",
            *[p for p in (ROOT / "queries").rglob("*") if p.is_file()],
        ],
        queries=24,
        accepted=sum(r["accepted"] for r in results),
        fallback=sum(not r["accepted"] for r in results),
        gallery_or_labels_used=False,
    )
    return True


def encoding() -> bool:
    """Return encoding for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("grounding_receipt.json")
    engine = None
    try:
        for r in load(ROOT / "grounding_results.json"):
            p = ROOT / "query_vectors" / r["cohort"] / (r["query_id"] + ".npz")
            if p.exists():
                continue
            if engine is None:
                if not admission.gpu_available():
                    status("waiting_natural_GPU_release_for_encoder")
                    return False
                engine = QwenEmbedder()
                status("full_Qwen_query_encoding")
            vector = engine.encode(image=Path(r["selected_image"]), query=True)
            np_once(p, vector=vector)
        once(
            ROOT / "embedding_runtime.json",
            engine.diagnostics() if engine else dict(resumed_no_new_forward=True),
        )
    finally:
        if engine:
            engine.close()
    receipt(
        "encoding_receipt.json",
        [
            ROOT / "embedding_runtime.json",
            *sorted((ROOT / "query_vectors").rglob("*.npz")),
        ],
        actual_selected_query_vectors=24,
    )
    return True


def plan() -> None:
    """Apply the plan workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("encoding_receipt.json")
    proposals = []
    jobs = []
    source = PROJECT_ROOT / "artifacts/module2_pointwise_recall_learning"
    cache = load(source / "exact_cache.json")
    for p in sorted((source / "pairs").glob("*.json")):
        raw = load(p)
        cache[raw["identity"]["sha256"]] = dict(
            identity=raw["identity"],
            logit=raw["logit"],
            source_file=rel(p),
            source_sha256=sha256_file(p),
        )
    banks = {c: original.feature_banks(c) for c in ("tune", "regression")}
    for r in load(ROOT / "grounding_results.json"):
        c, qid = r["cohort"], r["query_id"]
        docs = load(DATA / c / "catalog.public.json")
        ids = [d["id"] for d in docs]
        old, new = [
            load(DATA / c / "runs" / qid / n)
            for n in ("features.json", "caption_features.json")
        ]
        with np.load(
            ROOT / "query_vectors" / c / (qid + ".npz"), allow_pickle=False
        ) as b:
            vector = b["vector"].copy()
        index = banks["tune" if c == "fresh" else c][0]
        assert ids == [d["id"] for d in index.records]
        cosine = index.vectors[:, 0] @ vector
        pool = proposal(ids, old["pool"], new["pool"], cosine)
        from multimodal_ecommerce.intent_retrieval import normalized_ranks

        base = (
            0.375 * normalized_ranks(ids, old["base"])
            + 0.375 * normalized_ranks(ids, new["base"])
            + 0.25 * normalized_ranks(ids, cosine)
        )
        proposals.append(
            dict(
                cohort=c,
                query_id=qid,
                ids=ids,
                pool=pool,
                base=base.tolist(),
                crop_cosine=cosine.tolist(),
                ROI_accepted=r["accepted"],
            )
        )
        for context, image in [
            ("whole", r["source_image"]),
            ("zoom", r["selected_image"]),
        ]:
            q = dict(id=qid, text=None, image_path=image)
            for doc in docs:
                if doc["id"] in pool:
                    jobs.append(
                        dict(
                            cohort=c,
                            query_id=qid,
                            context=context,
                            candidate_id=doc["id"],
                            query=q,
                            description=new["description"],
                            document=doc,
                            identity=pair_identity(q, new["description"], doc),
                        )
                    )
    once(ROOT / "proposals.public.json", proposals)
    once(ROOT / "pair_plan.json", jobs)
    once(ROOT / "exact_cache.json", cache)
    receipt(
        "plan_receipt.json",
        [
            ROOT / n
            for n in ("proposals.public.json", "pair_plan.json", "exact_cache.json")
        ],
        unions=24,
        context_positions=len(jobs),
        planned_missing_unique_pairs=len(
            {
                j["identity"]["sha256"]
                for j in jobs
                if j["identity"]["sha256"] not in cache
            }
        ),
        original_candidates_retained=True,
    )


def infer() -> bool:
    """Run model inference using the configured inputs.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("plan_receipt.json")
    cache = load(ROOT / "exact_cache.json")
    jobs = load(ROOT / "pair_plan.json")
    engine = None
    try:
        for i, j in enumerate(jobs):
            key = j["identity"]["sha256"]
            p = ROOT / "pairs" / (key + ".json")
            if not p.exists():
                if key in cache:
                    raw = cache[key]
                    assert (
                        raw["identity"] == j["identity"]
                        and sha256_file(PROJECT_ROOT / raw["source_file"])
                        == raw["source_sha256"]
                    )
                    once(p, dict(raw, reused=True))
                else:
                    if engine is None:
                        if not admission.gpu_available():
                            status("waiting_natural_GPU_release_for_reranker")
                            return False
                        engine = DescribedImageReranker()
                        status("full_Qwen_whole_and_zoom_reranking")
                    doc = j["document"]
                    pairs = [
                        dict(
                            id=doc["id"],
                            text=doc["title"],
                            image_path=doc["image_path"],
                        )
                    ]
                    values = (
                        engine.score_described(
                            pairs,
                            query_image=Path(j["query"]["image_path"]),
                            description=j["description"],
                        )
                        if j["description"]
                        else engine.score(
                            pairs,
                            query_image=Path(j["query"]["image_path"]),
                            query_text=None,
                        )
                    )
                    assert len(values) == 1 and math.isfinite(float(values[0]))
                    once(
                        p,
                        dict(
                            identity=j["identity"],
                            logit=float(values[0]),
                            reused=False,
                            timing=engine.adapter.timings[-1],
                        ),
                    )
            row = load(p)
            assert row["identity"] == j["identity"] and math.isfinite(row["logit"])
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(dict(context_positions=i + 1, total=len(jobs)), flush=True)
        once(
            ROOT / "reranker_runtime.json",
            engine.diagnostics() if engine else dict(no_new_forward=True),
        )
    finally:
        if engine:
            engine.close()
    paths = sorted((ROOT / "pairs").glob("*.json"))
    receipt(
        "reranker_receipt.json",
        [ROOT / "reranker_runtime.json", *paths],
        unique_pairs=len(paths),
        real_new_calls=sum(not load(p)["reused"] for p in paths),
        context_positions=len(jobs),
        missing_scores_imputed=0,
    )
    return True


def orders() -> dict:
    """Return orders for this workflow.

    Returns:
        Result mapping produced by the workflow.
    """
    plans = {
        (p["cohort"], p["query_id"]): p for p in load(ROOT / "proposals.public.json")
    }
    contexts: dict = {}
    for j in load(ROOT / "pair_plan.json"):
        contexts.setdefault((j["cohort"], j["query_id"]), {}).setdefault(
            j["context"], {}
        )[j["candidate_id"]] = load(
            ROOT / "pairs" / (j["identity"]["sha256"] + ".json")
        )[
            "logit"
        ]
    outputs = {}
    for c in COHORTS:
        output = load(parent.ROOT / c / "predictions.json")
        for q in load(DATA / c / "queries.public.json"):
            qid = q["id"]
            if q["image_path"]:
                p = plans[c, qid]
                scores = contexts[c, qid]
                extra = {
                    m: fuse(
                        GroundedSubjectRuntimeFuseInputs(
                            p["ids"],
                            p["pool"],
                            p["base"],
                            scores["whole"],
                            scores["zoom"],
                            mode=mode,
                        )
                    )
                    for m, mode in zip(NEW, ["whole", "zoom", "dual"])
                }
            else:
                extra = {m: list(output[qid]["cross20"]) for m in NEW}
            output[qid].update(extra)
        outputs[c] = output
    return outputs


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("reranker_receipt.json")
    outputs = orders()
    for c, value in outputs.items():
        once(ROOT / c / "predictions.json", value)
    receipt(
        "prediction_receipt.json",
        [ROOT / c / "predictions.json" for c in COHORTS],
        methods=22,
        rankings=880,
        unchanged_parent_rankings=760,
    )


def evaluated(c: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        c: C used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / c / "predictions.json")
    labels = read_rows(DATA / c / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(DATA / c / "queries.public.json")
    }
    return evaluate_intervals(ranks, labels, modes), paired_querywise(
        ranks, labels, modes, COMPARISONS
    )


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("prediction_receipt.json")
    paths = []
    for c in COHORTS:
        for name, value in zip(
            ["internal_metrics.ai.development.json", "internal_paired_querywise.json"],
            evaluated(c),
        ):
            p = ROOT / c / name
            once(p, value)
            paths.append(p)
    receipt(
        "evaluation_receipt.json",
        paths,
        AI_relations_unchanged=12600,
        human_labels=0,
        formal_metrics=None,
    )


def exposure() -> None:
    """Apply the exposure workflow step.

    Notes:
        Writes derived artifacts through REGISTRY.parent.mkdir, once, receipt,
        write_rows.
    """
    rows = [
        dict(
            id="grounded_subject/" + p.relative_to(ROOT / "queries").as_posix(),
            image_path=str(p.resolve()),
            fingerprints=fingerprints(p),
            project_role="exposed_query_only_development_derived_not_acceptance",
            reusable_as_unseen=False,
        )
        for p in sorted((ROOT / "queries").rglob("*.png"))
    ]
    if REGISTRY.exists():
        assert read_rows(REGISTRY) == rows
    else:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        write_rows(REGISTRY, rows)
    prior = load(parent.ROOT / "post_experiment_exclusion_snapshot.json")
    snapshot = dict(prior)
    snapshot.update(
        files=dict(prior["files"], **{rel(REGISTRY): sha256_file(REGISTRY)}),
        image_fingerprints=prior["image_fingerprints"]
        + [r["fingerprints"] for r in rows],
        scope="All previous exposures plus every native grounding crop/zoom/review pane; all excluded from independent heldout",
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    receipt(
        "exposure_receipt.json",
        [REGISTRY, ROOT / "post_experiment_exclusion_snapshot.json"],
        derived_images=len(rows),
        new_raw_images=0,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, p.write_text, receipt, write_json.
    """
    guard()
    checked("evaluation_receipt.json")
    outputs = orders()
    gates: dict = {}
    losses = []
    lines = [
        "# Query-only主售定位修复：内部AI开发对照",
        "",
        "315件图库、40查询、12600冻结AI关系、K10，全部已曝光开发。",
        "完整原模型与原19方法760份排序保持。query-only定位与二次同图审查不使用图库/标签/排名；不确定回退整图。共享候选池检验整图/局部放大CE及融合，不根据结果更换主方法。DINO/FG旧完整推理保留，不宣称本轮重新运行或训练注意力。",
        "",
    ]
    for c in COHORTS:
        assert outputs[c] == load(ROOT / c / "predictions.json")
        old = load(parent.ROOT / c / "predictions.json")
        metrics, paired = evaluated(c)
        assert metrics == load(
            ROOT / c / "internal_metrics.ai.development.json"
        ) and paired == load(ROOT / c / "internal_paired_querywise.json")
        assert all(
            all(ms[m] == old[qid][m] for m in parent.METHODS)
            and all(len(v) == len(set(v)) == 315 for v in ms.values())
            for qid, ms in outputs[c].items()
        )
        gates[c] = {}
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, ms in metrics["scopes"].items():
            for m in METHODS:
                lines.append("|" + "|".join([scope, m, *report_cells(ms[m])]) + "|")
            gates[c][scope] = {
                m: dict(
                    precision_60=ms[m]["precision_60_lower_bound_met"],
                    recall_85=ms[m]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in ("global", "region_title")
                        for metric, v in paired["scopes"][scope][m][b].items()
                        if metric != "precision@10"
                    ),
                )
                for m in NEW
            }
        for row in paired["query_pair_bounds"]:
            if row["candidate"] == PRIMARY:
                bad = [
                    m
                    for m, v in row["metrics"].items()
                    if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
                ]
                if bad:
                    losses.append(
                        dict(
                            cohort=c,
                            query_id=row["query_id"],
                            baseline=row["baseline"],
                            strict_regression_metrics=bad,
                        )
                    )
    primary = [s[PRIMARY] for group in gates.values() for s in group.values()]
    decision = dict(
        primary=PRIMARY,
        checks=gates,
        development_precision_target_met=all(v["precision_60"] for v in primary),
        development_recall_preference_met=all(v["recall_85"] for v in primary),
        development_all_metric_nonregression=all(v["nonregression"] for v in primary),
        goal_met=False,
        default_strategy_changed=False,
        independent_AI_blind_predictions=0,
        formal_metrics=None,
    )
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_strict_regressions.json", losses)
    p = ROOT / "internal_comparison.ai.development.md"
    text = "\n".join(lines) + "\n"
    if p.exists():
        assert p.read_text(encoding="utf-8") == text
    else:
        p.write_text(text, encoding="utf-8")
    exposure()
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_rankings=880,
            parent_rankings_unchanged=760,
            AI_relations_unchanged=12600,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "completion_summary.json",
        dict(completed=True, methods=22, rankings=880, formal_metrics=None),
    )
    receipt(
        "completion_receipt.json",
        [
            p
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and p.name not in {"completion_receipt.json", "public_status.json"}
        ],
        source_files={s: sha256_file(PROJECT_ROOT / s) for s in SOURCES},
        completed=True,
        formal_metrics=None,
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="grounded_subject_repair_completed",
            methods=22,
            rankings=880,
            gallery_per_cohort=315,
            AI_relations=12600,
            formal_metrics=None,
            quality_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    verify()
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    checked("completion_receipt.json")


def worker() -> None:
    """Apply the worker workflow step.

    Raises:
        RuntimeError: Existing subject-learning worker holds lock; do not duplicate.
    """
    with attributes.locked() as acquired:
        if not acquired:
            raise RuntimeError(
                "Existing subject-learning worker holds lock; do not duplicate"
            )
        guard()
        for name, stage in [
            ("grounding_receipt.json", grounding),
            ("encoding_receipt.json", encoding),
            ("plan_receipt.json", plan),
            ("reranker_receipt.json", infer),
            ("prediction_receipt.json", prediction),
            ("evaluation_receipt.json", evaluation),
            ("completion_receipt.json", finish),
        ]:
            if (ROOT / name).exists():
                checked(name)
                continue
            while stage() is False:
                print(dict(waiting_natural_GPU_release=True, stage=name), flush=True)
                time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "verify"])
    action = parser.parse_args().action
    try:
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            worker()
    except Exception:
        if not (ROOT / "completion_receipt.json").exists():
            once(
                ROOT / "failure.json",
                dict(traceback=traceback.format_exc(), formal_metrics=None),
            )
            status("engineering_failure")
        raise
