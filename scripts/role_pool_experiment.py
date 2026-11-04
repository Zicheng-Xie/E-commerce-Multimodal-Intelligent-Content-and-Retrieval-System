"""Complete Qwen four-arm development control of pool extension and sale roles."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

import run  # noqa: F401
from business_grounding_audit import COHORTS, DATA, PARENT, load, once, relative, rows
from grounded_subject_runtime import (
    GroundedSubjectRuntimeFuseInputs,
    fuse,
    parse_native,
)
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.visual_query import DescribedImageReranker
from pair_identity import pair_identity
from querywise_paired_bounds import paired_querywise
from role_pool_profiles import JOINT, OPTIONS, POLICY, ROOT, SCHEMA, checked, digest
from role_pool_profiles import guard as profile_guard
from role_pool_profiles import locked, profile_worker, request_identity, status

ARMS: tuple[str, ...] = tuple(POLICY["four_arms"])
PRIMARY = POLICY["main"]
BASELINES = ("global", "region_title", "cross20", "business_joint_attention_fusion")
METHODS = (*BASELINES, *ARMS)
COMPARISONS = list(
    dict.fromkeys(
        [(m, b) for m in ARMS for b in ("global", "region_title", "cross20")]
        + [
            (ARMS[1], ARMS[0]),
            (ARMS[2], ARMS[0]),
            (ARMS[3], ARMS[1]),
            (ARMS[3], ARMS[2]),
            (PRIMARY, ARMS[0]),
            (PRIMARY, BASELINES[3]),
        ]
    )
)
ROLE_INSTRUCTION = BUSINESS_INSTRUCTION + (
    " Separate each product's sold item from its host device, intended-use objects and display props. "
    + "The supplied own-item role descriptions are fallible observations, not relevance labels. "
    + "Check them against original images and original titles; do not match two products merely "
    + "because a host device or prop appears in both. Preserve the user's requested main-product "
    + "versus accessory/part distinction. If uncertain, rely on direct original visual and title evidence."
)


def extend_pool(ids: list[str], parent: list[str], channel: list[str]) -> list:
    """Return extend pool for this workflow.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        parent: Parent used by this operation.
        channel: Channel used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Duplicate gallery or original pool; Full aligned auxiliary ranking
            required.
    """
    if len(ids) != len(set(ids)) or len(parent) != len(set(parent)):
        raise ValueError("Duplicate gallery or original pool")
    if (
        len(channel) != len(ids)
        or set(channel) != set(ids)
        or not set(parent) <= set(ids)
    ):
        raise ValueError("Full aligned auxiliary ranking required")
    return list(dict.fromkeys([*parent, *channel[: POLICY["new_channel_budget"]]]))


def role_text(value: dict) -> str:
    """Return role text for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.

    Raises:
        ValueError: Role context exceeds full described-image interface; no truncation.
    """
    if value["primary_unclear"] or value["role"] == "unclear":
        return ""
    roles = dict(
        main="商品本体",
        accessory="配件",
        part="零件",
        consumable="耗材",
        set="组合套装",
    )
    text = "主售商品：" + value["sale_item"] + "；角色：" + roles[value["role"]]
    if value["host_object"]:
        text += "；配套或使用对象：" + value["host_object"]
    if value["function"]:
        text += "；用途：" + value["function"]
    if value["appearance"]:
        text += "；可见结构：" + value["appearance"]
    if len(text) > 350:
        raise ValueError(
            "Role context exceeds full described-image interface; no truncation"
        )
    return text


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(ROOT / name, dict(files={relative(p): sha256_file(p) for p in paths}, **extra))


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: Frozen new source snapshot differs.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir.
    """
    profile_guard()
    if (ROOT / "experiment_protocol.json").exists():
        guard()
        return
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_role_pool_experiment.py",
        PROJECT_ROOT / "scripts/querywise_paired_bounds.py",
        PROJECT_ROOT / "scripts/pair_identity.py",
        ROOT / "profile_protocol.json",
        JOINT / "frozen_parent_pools.json",
        PROJECT_ROOT
        / "artifacts/module2_subject_recall_learning/feature_metadata.json",
    ]
    paths += sorted((PARENT / "pairs").glob("*.json"))
    for c in COHORTS:
        paths += [PARENT / c / "predictions.json", JOINT / c / "predictions.json"]
    release = {}
    for source in (
        PROJECT_ROOT / "scripts/role_pool_profiles.py",
        PROJECT_ROOT / "scripts/role_pool_experiment.py",
        PROJECT_ROOT / "tests/test_role_pool.py",
        PROJECT_ROOT / "tests/test_role_pool_experiment.py",
    ):
        target = ROOT / "source_release" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise ValueError("Frozen new source snapshot differs")
        release[relative(source)] = dict(
            snapshot=relative(target), sha256=sha256_file(target)
        )
        paths.append(target)
    once(ROOT / "source_release_manifest.json", release)
    paths.append(ROOT / "source_release_manifest.json")
    once(
        ROOT / "experiment_protocol.json",
        dict(
            files={relative(p): sha256_file(p) for p in paths},
            policy=POLICY,
            methods=METHODS,
            comparisons=COMPARISONS,
            role_instruction=ROLE_INSTRUCTION,
            primary=PRIMARY,
            fixed_weights=True,
            after_result_parameter_search=False,
            original_control="must exactly replay grounded_whole_zoom_union in original pool",
            role_pair="same whole and accepted-zoom source contexts; query-only role text, candidate-only original title plus own-role text; full Qwen3-VL-Reranker-2B",
            uncertainty="if either own role unclear, exact original pair score reused; no fabricated logits",
            unknown_labels="unchanged nulls, exact interval metrics and shared-unknown paired deltas",
            pool_learning="joint proposals reuse nine source-family-masked OOF heads; development only, not a single deployable heldout model",
            new_backbone_training=False,
            new_masks=False,
            independent_acceptance=False,
            human_labels=0,
            formal_metrics=None,
            default_strategy_changed=False,
        ),
    )
    print(
        json.dumps(dict(four_arm_protocol_frozen=True, methods=len(METHODS))),
        flush=True,
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    profile_guard()
    for path, checksum in load(ROOT / "experiment_protocol.json")["files"].items():
        if sha256_file(PROJECT_ROOT / path) != checksum:
            raise ValueError("Frozen four-arm input/source changed: " + path)


def plan() -> None:
    """Return plan for this workflow.

    Raises:
        ValueError: Incomplete actual joint proposal scores; Original Qwen pool changed.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("profile_receipt.json")
    if (ROOT / "plan_receipt.json").exists():
        checked("plan_receipt.json")
        return
    profiles = {r["key"]: r for r in load(ROOT / "profiles.json")}
    joint_meta = load(
        PROJECT_ROOT / "artifacts/module2_subject_recall_learning/feature_metadata.json"
    )
    qi = {q: i for i, q in enumerate(joint_meta["query_ids"])}
    gi = {g: i for i, g in enumerate(joint_meta["gallery_ids"])}
    with np.load(JOINT / "OOF_scores.npz", allow_pickle=False) as bank:
        joint = bank["scores"].copy()
    templates: dict = {}
    for job in load(PARENT / "pair_plan.json"):
        templates.setdefault((job["cohort"], job["query_id"], job["context"]), job)
    old_pools = load(JOINT / "frozen_parent_pools.json")
    proposals, jobs, cache = [], [], {}
    for p in sorted((PARENT / "pairs").glob("*.json")):
        raw = load(p)
        cache[raw["identity"]["sha256"]] = dict(path=relative(p), sha256=sha256_file(p))
    for p in load(PARENT / "proposals.public.json"):
        c, qid, ids = p["cohort"], p["query_id"], p["ids"]
        key = c + "/" + qid
        score = {cid: float(joint[qi[key], gi[cid]]) for cid in ids}
        if not np.isfinite(list(score.values())).all():
            raise ValueError("Incomplete actual joint proposal scores")
        channel = sorted(ids, key=lambda cid: (-score[cid], cid))
        old = old_pools[key]
        extended = extend_pool(ids, old, channel)
        if set(old) != set(p["pool"]):
            raise ValueError("Original Qwen pool changed")
        proposals.append(
            dict(
                cohort=c,
                query_id=qid,
                ids=ids,
                old_pool=old,
                expanded_pool=extended,
                base=p["base"],
                auxiliary_top20=channel[:20],
                added=[cid for cid in extended if cid not in set(old)],
            )
        )
        docs = {d["id"]: d for d in load(DATA / c / "catalog.public.json")}
        query_role = role_text(profiles[key]["value"])
        for context in ("whole", "zoom"):
            template = templates[c, qid, context]
            query = template["query"]
            for cid in extended:
                doc = docs[cid]
                control_identity = pair_identity(query, template["description"], doc)
                control = dict(
                    cohort=c,
                    query_id=qid,
                    context=context,
                    candidate_id=cid,
                    variant="original",
                    query=query,
                    description=template["description"],
                    document=doc,
                    instruction=BUSINESS_INSTRUCTION,
                    identity=control_identity,
                )
                jobs.append(control)
                candidate_role = role_text(profiles[cid]["value"])
                if not query_role or not candidate_role:
                    role_job = dict(
                        control, variant="role", uncertainty_original_fallback=True
                    )
                else:
                    role_doc = dict(
                        doc,
                        title=doc["title"]
                        + "\n本商品自身角色观察（可能出错）："
                        + candidate_role,
                    )
                    inputs = dict(
                        pair_identity(query, query_role, role_doc)["inputs"],
                        instruction=ROLE_INSTRUCTION,
                        own_query_profile_raw_sha256=profiles[key]["raw_sha256"],
                        own_candidate_profile_raw_sha256=profiles[cid]["raw_sha256"],
                        method_namespace="role_pool_native_v1",
                    )
                    identity = dict(sha256=digest(inputs), inputs=inputs)
                    role_job = dict(
                        control,
                        variant="role",
                        description=query_role,
                        document=role_doc,
                        instruction=ROLE_INSTRUCTION,
                        identity=identity,
                        uncertainty_original_fallback=False,
                    )
                jobs.append(role_job)
    once(ROOT / "proposals.public.json", proposals)
    once(ROOT / "pair_plan.public.json", jobs)
    once(ROOT / "exact_original_cache.json", cache)
    keys = {j["identity"]["sha256"] for j in jobs}
    receipt(
        "plan_receipt.json",
        [
            ROOT / n
            for n in (
                "proposals.public.json",
                "pair_plan.public.json",
                "exact_original_cache.json",
            )
        ],
        queries=len(proposals),
        original_candidates_retained=True,
        contexts=len(jobs),
        unique_pairs=len(keys),
        planned_new_full_Qwen_pairs=len(keys - set(cache)),
        pool_used_judgments=False,
        formal_metrics=None,
    )
    print(
        json.dumps(
            dict(
                pool_plan_frozen=True,
                added_candidates=sum(len(p["added"]) for p in proposals),
                planned_new_full_pairs=len(keys - set(cache)),
            )
        ),
        flush=True,
    )


def infer() -> bool:
    """Run model inference using the configured inputs.

    Returns:
        Whether the condition checked by this operation holds.

    Raises:
        ValueError: Original full pair cache changed; Original full pair context
            identity mismatch; Full actual finite Qwen score required; Saved full Qwen
            pair differs.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("plan_receipt.json")
    if (ROOT / "reranker_receipt.json").exists():
        checked("reranker_receipt.json")
        return True
    jobs = load(ROOT / "pair_plan.public.json")
    cache = load(ROOT / "exact_original_cache.json")
    engine = None
    started = time.perf_counter()
    try:
        for i, job in enumerate(jobs):
            key = job["identity"]["sha256"]
            path = ROOT / "pairs" / (key + ".json")
            if not path.exists():
                if key in cache:
                    source = PROJECT_ROOT / cache[key]["path"]
                    if sha256_file(source) != cache[key]["sha256"]:
                        raise ValueError("Original full pair cache changed")
                    raw = load(source)
                    if raw["identity"] != job["identity"]:
                        raise ValueError("Original full pair context identity mismatch")
                    once(
                        path,
                        dict(
                            identity=job["identity"],
                            logit=raw["logit"],
                            reused_original=True,
                            source=cache[key],
                        ),
                    )
                else:
                    if engine is None:
                        from gpu_admission import gpu_available

                        available, info = gpu_available()
                        if not available:
                            status(
                                "waiting_natural_GPU_release_for_full_Qwen",
                                admission=info,
                            )
                            print(
                                json.dumps(dict(waiting=True, reason=info)), flush=True
                            )
                            return False
                        engine = DescribedImageReranker()
                    engine.adapter.config = dict(
                        engine.adapter.config, instruction=job["instruction"]
                    )
                    doc = job["document"]
                    documents = [
                        dict(
                            id=doc["id"],
                            text=doc["title"],
                            image_path=doc["image_path"],
                        )
                    ]
                    if job["description"]:
                        values = engine.score_described(
                            documents,
                            query_image=Path(job["query"]["image_path"]),
                            description=job["description"],
                        )
                    else:
                        values = engine.score(
                            documents, query_image=Path(job["query"]["image_path"])
                        )
                    if len(values) != 1 or not math.isfinite(float(values[0])):
                        raise ValueError("Full actual finite Qwen score required")
                    once(
                        path,
                        dict(
                            identity=job["identity"],
                            logit=float(values[0]),
                            reused_original=False,
                            timing=engine.adapter.timings[-1],
                            instruction=job["instruction"],
                        ),
                    )
            raw = load(path)
            if raw["identity"] != job["identity"] or not math.isfinite(raw["logit"]):
                raise ValueError("Saved full Qwen pair differs")
            if (i + 1) % 40 == 0 or i + 1 == len(jobs):
                status(
                    "complete_Qwen_role_and_expanded_pool_inference",
                    contexts_completed=i + 1,
                    contexts_total=len(jobs),
                )
                print(
                    json.dumps(dict(contexts_completed=i + 1, total=len(jobs))),
                    flush=True,
                )
        runtime = (
            engine.diagnostics() if engine else dict(no_new_forward_this_resume=True)
        )
        runtime["resume_wall_seconds"] = time.perf_counter() - started
        runtime["instruction_policy"] = dict(
            original=BUSINESS_INSTRUCTION, role=ROLE_INSTRUCTION
        )
        run_id = len(list((ROOT / "reranker_sessions").glob("*.json"))) + 1
        once(ROOT / "reranker_sessions" / f"{run_id:03d}.json", runtime)
    finally:
        if engine:
            engine.close()
    paths = sorted((ROOT / "pairs").glob("*.json"))
    files = paths + sorted((ROOT / "reranker_sessions").glob("*.json"))
    receipt(
        "reranker_receipt.json",
        files,
        unique_pairs=len(paths),
        actual_new_full_Qwen_calls=sum(not load(p)["reused_original"] for p in paths),
        reused_exact_original_pairs=sum(load(p)["reused_original"] for p in paths),
        contexts=len(jobs),
        imputed_scores=0,
        formal_metrics=None,
    )
    return True


def predictions() -> dict:
    """Return predictions for this workflow.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Original-pool original-sort control is not exact; Full same-gallery
            ranking required.
    """
    checked("reranker_receipt.json")
    proposals = {
        (p["cohort"], p["query_id"]): p for p in load(ROOT / "proposals.public.json")
    }
    scores: dict = {}
    for j in load(ROOT / "pair_plan.public.json"):
        key = (j["cohort"], j["query_id"], j["variant"], j["context"])
        scores.setdefault(key, {})[j["candidate_id"]] = load(
            ROOT / "pairs" / (j["identity"]["sha256"] + ".json")
        )["logit"]
    outputs: dict = {}
    for c in COHORTS:
        parent = load(PARENT / c / "predictions.json")
        joint = load(JOINT / c / "predictions.json")
        outputs[c] = {}
        for q in load(DATA / c / "queries.public.json"):
            qid = q["id"]
            out = {m: joint[qid][m] for m in BASELINES}
            if q["image_path"]:
                p = proposals[c, qid]
                for method, pool_name, variant in zip(
                    ARMS,
                    ("old_pool", "expanded_pool", "old_pool", "expanded_pool"),
                    ("original", "original", "role", "role"),
                ):
                    pool = p[pool_name]
                    whole, zoom = [
                        {cid: scores[c, qid, variant, ctx][cid] for cid in pool}
                        for ctx in ("whole", "zoom")
                    ]
                    out[method] = fuse(
                        GroundedSubjectRuntimeFuseInputs(
                            p["ids"], pool, p["base"], whole, zoom, mode="dual"
                        )
                    )
                if out[ARMS[0]] != parent[qid]["grounded_whole_zoom_union"]:
                    raise ValueError("Original-pool original-sort control is not exact")
            else:
                out.update({m: list(parent[qid]["cross20"]) for m in ARMS})
            ids = {d["id"] for d in load(DATA / c / "catalog.public.json")}
            if any(len(rank) != len(ids) or set(rank) != ids for rank in out.values()):
                raise ValueError("Full same-gallery ranking required")
            outputs[c][qid] = out
    return outputs


def evaluate() -> None:
    """Return evaluate for this workflow.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    if (ROOT / "evaluation_receipt.json").exists():
        checked("evaluation_receipt.json")
        return
    status("four_arm_interval_and_shared_unknown_evaluation")
    outputs = predictions()
    for c, rankings in outputs.items():
        modes = {
            q["id"]: "image" if q["image_path"] else "text"
            for q in load(DATA / c / "queries.public.json")
        }
        labels = rows(DATA / c / "judgments.ai.jsonl")
        once(ROOT / c / "predictions.json", rankings)
        once(
            ROOT / c / "internal_metrics.ai.development.json",
            evaluate_intervals(rankings, labels, modes),
        )
        once(
            ROOT / c / "internal_paired_querywise.json",
            paired_querywise(rankings, labels, modes, COMPARISONS),
        )
    receipt(
        "evaluation_receipt.json",
        [p for c in COHORTS for p in sorted((ROOT / c).glob("*.json"))],
        complete_rankings=40 * len(METHODS),
        AI_relations_unchanged=12600,
        formal_metrics=None,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, receipt, write_json.
    """
    guard()
    checked("evaluation_receipt.json")
    metrics = ("precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1")
    lines = [
        "# 内部AI开发：主售角色 × 候选扩池四组对照",
        "",
        "同图库315件/组、40查询、原12600条AI关系、K10。画像不是相关性标签，未知关系保持null；人工0，formal_metrics=null。所有当前资料均已曝光；OOF候选不可当作独立验收或单一部署模型。",
        "",
        "扩池固定并入双图注意力OOF前20，原候选全部保留。主售角色使用每件原图及自身原始标题；不确定回退真实原始分数。新掩码全局停用。",
        "",
        "各单元格为未知标签补全范围（百分比），不是置信区间或AI标注错误范围。",
        "",
    ]
    gates: dict
    effects: dict
    gates, regressions, effects = {}, [], {}
    for c in COHORTS:
        result = load(ROOT / c / "internal_metrics.ai.development.json")
        paired = load(ROOT / c / "internal_paired_querywise.json")
        gates[c], effects[c] = {}, {}
        for scope, methods in result["scopes"].items():
            lines += [
                f"## {c} / {scope}",
                "",
                "| 方法 | P@10 | R@10 | AP | NDCG@10 | MRR | Hit@1 |",
                "|---|---|---|---|---|---|---|",
            ]
            for m, values in methods.items():
                cells = [
                    " / ".join(
                        "undefined" if v is None else f"{v*100:.2f}"
                        for v in values["ranges"][metric]
                    )
                    for metric in metrics
                ]
                lines.append("| " + " | ".join([m, *cells]) + " |")
            lines.append("")
            gates[c][scope] = {
                m: dict(
                    precision_60=methods[m]["precision_60_lower_bound_met"],
                    recall_85=methods[m]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in ("global", "region_title")
                        for metric, v in paired["scopes"][scope][m][b].items()
                        if metric != "precision@10"
                    ),
                )
                for m in ARMS
            }
            effects[c][scope] = {
                name: paired["scopes"][scope][a][b]
                for name, a, b in (
                    ("pool_only", ARMS[1], ARMS[0]),
                    ("roles_only", ARMS[2], ARMS[0]),
                    ("roles_given_expanded_pool", ARMS[3], ARMS[1]),
                    ("pool_given_roles", ARMS[3], ARMS[2]),
                    ("joint_total", ARMS[3], ARMS[0]),
                )
            }
        for row in paired["query_pair_bounds"]:
            bad = [
                m
                for m, v in row["metrics"].items()
                if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
            ]
            if row["candidate"] == PRIMARY and bad:
                regressions.append(
                    dict(
                        cohort=c,
                        query_id=row["query_id"],
                        baseline=row["baseline"],
                        strict_regression_metrics=bad,
                    )
                )
    primary_gates = [s[PRIMARY] for groups in gates.values() for s in groups.values()]
    decision = dict(
        primary=PRIMARY,
        checks=gates,
        development_precision_target_met=all(r["precision_60"] for r in primary_gates),
        development_recall_preference_met=all(r["recall_85"] for r in primary_gates),
        development_all_metric_nonregression=all(
            r["nonregression"] for r in primary_gates
        ),
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
    )
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_factorial_effects.json", effects)
    once(ROOT / "internal_strict_regressions.json", regressions)
    save_text(ROOT / "internal_comparison.ai.development.md", "\n".join(lines) + "\n")
    visual_page()
    parent_path = JOINT / "post_experiment_exclusion_snapshot.json"
    snapshot = copy.deepcopy(load(parent_path))
    snapshot.update(
        parent_snapshot_sha256=sha256_file(parent_path),
        role_pool_profiled_sources=435,
        scope="All previous development and consumed blind exclusions plus current same-source role contexts; no new media, no independent acceptance",
    )
    snapshot["files"][relative(ROOT / "profile_receipt.json")] = sha256_file(
        ROOT / "profile_receipt.json"
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    verification = verify(recompute=True, sealed=False)
    once(ROOT / "verification.json", verification)
    files = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name
        not in ("completion_receipt.json", "public_status.json", "runner.lock")
    ]
    receipt("completion_receipt.json", files, completed=True, formal_metrics=None)
    write_json(
        ROOT / "public_status.json",
        dict(
            phase="four_arm_development_completed",
            completed=True,
            methods=len(METHODS),
            complete_rankings=40 * len(METHODS),
            human_labels=0,
            development_precision_target_met=decision[
                "development_precision_target_met"
            ],
            development_recall_preference_met=decision[
                "development_recall_preference_met"
            ],
            development_all_metric_nonregression=decision[
                "development_all_metric_nonregression"
            ],
            formal_metrics=None,
            quality_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            new_independent_AI_blind_predictions=0,
        ),
    )
    print(
        json.dumps(dict(four_arm_completed=True, formal_metrics=None, goal_met=False)),
        flush=True,
    )


def save_text(path: Path, text: str) -> None:
    """Persist text.

    Args:
        path: Path of the input or output artifact used by this operation.
        text: Text input used by the operation.

    Notes:
        Writes derived artifacts through path.write_text.
    """
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Immutable report changed: " + path.name)
    path.write_text(text, encoding="utf-8")


def visual_page() -> None:
    """Apply the visual page workflow step.

    Raises:
        ValueError: Broken local image reference.

    Notes:
        Writes derived artifacts through once.
    """
    markup = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>内部四组对照</title>',
        "<style>body{font:16px sans-serif;max-width:1450px;margin:24px auto}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.card{border:1px solid #ccc;padding:10px}img{width:100%;height:170px;object-fit:contain}summary{padding:12px}</style>",
        "<h1>主售角色与候选池：内部AI开发对照</h1><p>AI标注、人工0。全部开发数据，不能当作正式验收；完整指标见内部报告。主售不确定回退原分数，新掩码停用。分数为模型logit及秩融合，不是校准概率。</p>",
    ]
    references = []
    profiles = {r["key"]: r["value"] for r in load(ROOT / "profiles.json")}
    for c in COHORTS:
        docs = {d["id"]: d for d in load(DATA / c / "catalog.public.json")}
        queries = {q["id"]: q for q in load(DATA / c / "queries.public.json")}
        for qid, methods in load(ROOT / c / "predictions.json").items():
            q = queries[qid]
            markup.append(
                "<h2>"
                + html.escape(c + "/" + qid)
                + "</h2><p>"
                + html.escape(
                    q.get("text")
                    or role_text(profiles[c + "/" + qid])
                    or "主售不确定，保留原描述"
                )
                + "</p>"
            )
            for method, ranking in methods.items():
                markup.append(
                    "<details><summary>"
                    + html.escape(method)
                    + '</summary><div class="cards">'
                )
                for k, cid in enumerate(ranking[:10], 1):
                    d = docs[cid]
                    path = Path(d["image_path"])
                    if not path.is_file():
                        raise ValueError("Broken local image reference")
                    references.append(str(path))
                    src = quote(Path(os.path.relpath(path, ROOT)).as_posix(), safe="/")
                    markup.append(
                        '<div class="card"><b>#'
                        + str(k)
                        + '</b><img src="'
                        + src
                        + '"><div>'
                        + html.escape(d["title"])
                        + "</div><small>"
                        + html.escape(profiles[cid]["sale_item"])
                        + "</small></div>"
                    )
                markup.append("</div></details>")
    markup.append("</html>")
    save_text(ROOT / "internal_comparison.html", "\n".join(markup))
    once(
        ROOT / "visual_replay_check.json",
        dict(
            static_image_references_verified=len(references),
            browser_interaction_tested=False,
        ),
    )


def verify(*, recompute: bool = True, sealed: bool = True) -> dict:
    """Verify the recorded artifacts and their consistency.

    Args:
        recompute: Recompute used by this operation.
        sealed: Sealed used by this operation.

    Returns:
        Mapping with fields: passed, complete_rankings_replayed,
        exact_original_control_replayed, metric_and_shared_unknown_scopes_recomputed,
        AI_relations_unchanged, formal_metrics, default_strategy_changed,
        new_independent_AI_blind_predictions.

    Raises:
        ValueError: Own-only native profile context does not replay; Own-only native
            profile parsing differs; New source release cannot replay; Saved ranks do
            not replay; Interval metrics do not recompute; Shared unknown deltas do not
            recompute; Missing full same-gallery rankings; Previous exposed/consumed
            exclusions lost.
    """
    guard()
    checked("profile_receipt.json")
    checked("plan_receipt.json")
    checked("reranker_receipt.json")
    checked("evaluation_receipt.json")
    if sealed:
        checked("completion_receipt.json")
    records = {r["key"]: r for r in load(ROOT / "profile_inputs.public.json")}
    for result in load(ROOT / "profiles.json"):
        r = records[result["key"]]
        rawpath = ROOT / "profiles" / (r["opaque"] + ".native.json")
        request = load(ROOT / "profiles" / (r["opaque"] + ".request.json"))
        if request["identity"] != request_identity(r) or request[
            "identity_sha256"
        ] != digest(request_identity(r)):
            raise ValueError("Own-only native profile context does not replay")
        value = parse_native(load(rawpath), SCHEMA, OPTIONS)
        if value["role"] == "unclear" and not value["primary_unclear"]:
            value = dict(value, primary_unclear=True)
        if value != result["value"]:
            raise ValueError("Own-only native profile parsing differs")
    for source, r in load(ROOT / "source_release_manifest.json").items():
        if (
            sha256_file(PROJECT_ROOT / source) != r["sha256"]
            or sha256_file(PROJECT_ROOT / r["snapshot"]) != r["sha256"]
        ):
            raise ValueError("New source release cannot replay")
    count = 0
    for c, rankings in predictions().items():
        if load(ROOT / c / "predictions.json") != rankings:
            raise ValueError("Saved ranks do not replay")
        count += sum(len(v) for v in rankings.values())
        if recompute:
            modes = {
                q["id"]: "image" if q["image_path"] else "text"
                for q in load(DATA / c / "queries.public.json")
            }
            labels = rows(DATA / c / "judgments.ai.jsonl")
            if evaluate_intervals(rankings, labels, modes) != load(
                ROOT / c / "internal_metrics.ai.development.json"
            ):
                raise ValueError("Interval metrics do not recompute")
            if paired_querywise(rankings, labels, modes, COMPARISONS) != load(
                ROOT / c / "internal_paired_querywise.json"
            ):
                raise ValueError("Shared unknown deltas do not recompute")
    if count != 40 * len(METHODS):
        raise ValueError("Missing full same-gallery rankings")
    snapshot = load(ROOT / "post_experiment_exclusion_snapshot.json")
    parent = load(JOINT / "post_experiment_exclusion_snapshot.json")
    if not set(parent["ids"]) <= set(snapshot["ids"]):
        raise ValueError("Previous exposed/consumed exclusions lost")
    return dict(
        passed=True,
        complete_rankings_replayed=count,
        exact_original_control_replayed=24,
        metric_and_shared_unknown_scopes_recomputed=3 if recompute else 0,
        AI_relations_unchanged=12600,
        formal_metrics=None,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
    )


def worker() -> None:
    """Return worker for this workflow.

    Notes:
        Writes derived artifacts through write_json.
    """
    guard()
    if (ROOT / "completion_receipt.json").exists():
        print(json.dumps(verify()), flush=True)
        return
    with locked():
        try:
            if not profile_worker():
                return
            plan()
            # Own profile job expires naturally; allow one short admission interval.
            if not (ROOT / "reranker_receipt.json").exists():
                time.sleep(65)
            if not infer():
                return
            evaluate()
            finish()
        except Exception as error:
            write_json(
                ROOT / "failure.json",
                dict(
                    type=type(error).__name__, message=str(error), formal_metrics=None
                ),
            )
            status("four_arm_engineering_failure", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "plan", "worker", "verify"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "plan":
        plan()
    elif action == "verify":
        print(json.dumps(verify()), flush=True)
    else:
        worker()
