"""Internal primary-subject comparisons, exposure accounting and immutable seal."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np

import run  # noqa: F401
from acquire_primary_subjects import MEDIA, PARENT, QUEUE, ROOT
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from primary_subject_experiment import COHORTS, METHODS, guard
from subject_roi_experiment import load, once

METRICS = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
DISPLAY = {
    "global": "整图",
    "region_title": "区域与标题",
    "cross20": "原Qwen方法",
    "fgclip2_local_fusion20": "全模态FG融合（原查询描述）",
    "image_fusion20": "图片FG融合（原查询描述）",
    "image_roi_fusion20": "图片FG融合加ROI（原查询描述）",
    "subject_caption20": "新主售描述Qwen",
    "subject_guard20": "新主售描述Qwen加公开冲突扣分",
}
PRIMARY = "subject_guard20"
VIEW_METHODS = ("cross20", "subject_caption20", "subject_guard20")


def percent(bounds: dict) -> str:
    """Return percent for this workflow.

    Args:
        bounds: Bounds used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return "–".join("未定义" if v is None else f"{100 * v:.2f}" for v in bounds)


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Notes:
        Writes derived artifacts through once.
    """
    once(
        ROOT / "report_protocol.json",
        dict(
            source_sha256=sha256_file(Path(__file__)),
            primary_method=PRIMARY,
            methods=METHODS,
            all_methods_and_scopes_reported=True,
            guard_is_ROI_addition=False,
            formal_metrics=None,
            default_promotion_allowed=False,
        ),
    )


def report_guard() -> None:
    """Apply the report guard workflow step."""
    guard()
    assert load(ROOT / "report_protocol.json")["source_sha256"] == sha256_file(
        Path(__file__)
    )


def audit() -> dict:
    """Return audit for this workflow.

    Returns:
        Mapping with fields: full_Qwen_embedding_calls, query_caption_vectors,
        new_caption_Qwen_calls, unclear_query_caption_abstentions,
        full_local_query_caption_requests, new_full_reranker_pairs,
        exact_reused_reranker_pairs, full_reranker_pairs, full_detector_calls,
        accepted_ROIs, global_fallbacks, full_FG_vision_forward_calls,
        full_FG_text_forward_calls, full_FG_total_forward_calls, all_models_complete.

    Notes:
        Writes derived artifacts through once.
    """
    embedding = load(ROOT / "encoding_receipt.json")
    detector = load(ROOT / "detection_receipt.json")
    fg = load(ROOT / "fg_encoding_receipt.json")
    reranker = load(ROOT / "reranker_receipt.json")
    caption_files = sorted((ROOT / "query_captions").glob("*/*.json"))
    captions = [load(p) for p in caption_files]
    assert len(captions) == 24 and all(
        not r["candidates_or_labels_supplied"] for r in captions
    )
    ledger = read_rows(ROOT / "fg_call_ledger.jsonl")
    vision = sum(r["detail"].get("actual_vision_forward_calls", 0) for r in ledger)
    text = sum(r["detail"]["kind"] == "text" for r in ledger)
    assert vision == fg["actual_full_vision_forward_calls"]
    assert text == fg["actual_full_text_forward_calls"]
    assert vision + text == fg["actual_full_total_forward_calls"]
    qwen_ledger = read_rows(ROOT / "qwen_call_ledger.jsonl")
    assert len(qwen_ledger) == embedding["actual_full_embedding_calls"]
    raw_files = [p for c in COHORTS for p in (ROOT / c / "runs").glob("*/*cross.json")]
    raw = [load(p) for p in raw_files]
    new = sum(len(r["new_pair_ids"]) for r in raw)
    reused = sum(len(r["reused_pair_ids"]) for r in raw)
    assert new == reranker["new_full_reranker_pairs"]
    assert reused == reranker["exact_reused_pairs"]
    assert new + reused == reranker["total_pairs"] == 1280
    assert new == sum(len(r["timings"]) for r in raw)
    caption_vectors = [
        load(p) for p in sorted((ROOT / "new_query_features").glob("*/*.json"))
    ]
    assert len(caption_vectors) == 24
    value = dict(
        full_Qwen_embedding_calls=embedding["actual_full_embedding_calls"],
        query_caption_vectors=24,
        new_caption_Qwen_calls=sum(r["actual_model_call"] for r in caption_vectors),
        unclear_query_caption_abstentions=sum(r["abstain"] for r in caption_vectors),
        full_local_query_caption_requests=24,
        new_full_reranker_pairs=new,
        exact_reused_reranker_pairs=reused,
        full_reranker_pairs=new + reused,
        full_detector_calls=detector["actual_full_detector_calls"],
        accepted_ROIs=detector["accepted"],
        global_fallbacks=detector["fallback"],
        full_FG_vision_forward_calls=vision,
        full_FG_text_forward_calls=text,
        full_FG_total_forward_calls=vision + text,
        all_models_complete=True,
    )
    once(ROOT / "runtime_audit.json", value)
    return value


def exposure() -> None:
    """Apply the exposure workflow step.

    Raises:
        ValueError: Preserve conflicting derived exposure registration.

    Notes:
        Writes derived artifacts through once, write_rows.
    """
    report_guard()
    for name in ["encoding_receipt.json", "detection_receipt.json"]:
        verify_files(ROOT, load(ROOT / name)["files"])
    registry = PROJECT_ROOT / "data/module2_primary_subject_derived/records.jsonl"
    paths = sorted((ROOT / "additions/index/products").glob("*/views/[0-9]*.png"))
    paths += sorted((ROOT / "detections").glob("*/native.png"))
    paths += sorted((ROOT / "detections").glob("*/zoom.png"))
    rows = [
        dict(
            id="primary_subject_derived/" + p.relative_to(ROOT).as_posix(),
            image_path=os.path.relpath(p, registry.parent).replace("\\", "/"),
            fingerprints=fingerprints(p),
            reusable_as_unseen=False,
            project_role="exposed_query_aware_development_derived_not_acceptance",
        )
        for p in paths
    ]
    if registry.exists():
        if read_rows(registry) != rows:
            raise ValueError("Preserve conflicting derived exposure registration")
    else:
        write_rows(registry, rows)
    parent_path = ROOT / "post_acquisition_exclusion_snapshot.json"
    parent = load(parent_path)
    verify_files(PROJECT_ROOT, parent["files"])
    snapshot = dict(parent)
    snapshot.update(
        scope="Every prior exposure, all current candidate metadata and acquired images, and derived primary-subject development views/ROIs; none reusable as independent acceptance",
        parent_snapshot_sha256=sha256_file(parent_path),
        files=dict(
            parent["files"],
            **{registry.relative_to(PROJECT_ROOT).as_posix(): sha256_file(registry)},
        ),
        image_fingerprints=parent["image_fingerprints"]
        + [r["fingerprints"] for r in rows],
        primary_subject_derived_images=len(rows),
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    acquired = load(ROOT / "acquisition_receipt.json")
    once(
        ROOT / "exposure_receipt.json",
        dict(
            candidate_metadata=acquired["candidates"],
            acquired_images=acquired["downloaded"],
            technical_duplicates_retained_in_exposure=acquired["technical_duplicates"],
            derived_images=len(rows),
            parent_snapshot_sha256=sha256_file(parent_path),
            project_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [QUEUE, MEDIA, registry]
            },
            all_failed_metadata_excluded=True,
            pixels_and_dhash_excluded=True,
            independent_acceptance_predictions=0,
        ),
    )
    print(dict(derived_images_registered=len(rows)), flush=True)


def url(path: Path) -> str:
    """Return url for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/:")


def baseline_scores(features: dict, raw: dict) -> dict:
    """Return baseline scores for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        raw: Raw used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    ids, pool = features["ids"], raw["pool"]
    position = {cid: i for i, cid in enumerate(ids)}
    selected = np.array([position[cid] for cid in pool])
    score = 0.65 * normalized_ranks(pool, np.array(raw["logits"]))
    score += 0.35 * normalized_ranks(ids, np.array(features["base"]))[selected]
    if features["offsets"] is not None:
        score += np.array(features["offsets"])[selected]
    return dict(zip(pool, score.tolist()))


def viewer() -> None:
    """Apply the viewer workflow step.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.html').write_text, once.
    """
    rows = []
    for cohort in COHORTS:
        catalog = load(ROOT / cohort / "catalog.public.json")
        docs = {d["id"]: d for d in catalog}
        profiles = {
            r["id"]: r["product_type"]
            for r in read_rows(ROOT / cohort / "profiles.public.jsonl")
        }
        labels = {
            (r["query_id"], r["candidate_id"]): r["grade"]
            for r in read_rows(ROOT / cohort / "judgments.ai.jsonl")
        }
        ranks = load(ROOT / cohort / "predictions.json")
        details = load(ROOT / cohort / "runtime_scores.json")
        for query in load(ROOT / cohort / "queries.public.json"):
            qid = query["id"]
            folder = ROOT / cohort / "runs" / qid
            feature, raw = load(folder / "features.json"), load(folder / "cross.json")
            scores = {"cross20": baseline_scores(feature, raw)}
            old_description, new_description = feature["description"], ""
            if query["image_path"]:
                updated = load(folder / "caption_features.json")
                updated_raw = load(folder / "caption_cross.json")
                scores["subject_caption20"] = baseline_scores(updated, updated_raw)
                scores["subject_guard20"] = details[qid]["public_subject_guard"][
                    "pool_scores"
                ]
                new_description = load(
                    ROOT / "query_captions" / cohort / (qid + ".json")
                )["text"]
            else:
                scores.update(
                    subject_caption20=dict(scores["cross20"]),
                    subject_guard20=dict(scores["cross20"]),
                )
            results = {}
            for method in VIEW_METHODS:
                values = scores[method]
                assert (
                    sorted(values, key=lambda cid: (-values[cid], cid))
                    == ranks[qid][method][:20]
                )
                results[method] = [
                    dict(
                        id=cid,
                        title=docs[cid]["title"],
                        image=url(docs[cid]["image_path"]),
                        category=profiles[cid],
                        score=values[cid],
                        AI_grade=labels[(qid, cid)],
                    )
                    for cid in ranks[qid][method][:20]
                ]
            description = query["text"] or (
                "原描述："
                + old_description
                + "；新主售描述："
                + (new_description or "主体不明，弃权")
            )
            rows.append(
                dict(
                    id=cohort + "/" + qid,
                    query=description,
                    old_description=old_description,
                    new_description=new_description,
                    image=url(query["image_path"]) if query["image_path"] else None,
                    crop=None,
                    results=results,
                )
            )
    size = len(load(ROOT / "tune/catalog.public.json"))
    once(
        ROOT / "runtime_results.internal.json",
        dict(
            queries=rows,
            score_is_probability=False,
            metric_k=10,
            gallery=size,
            review_origin="model",
            human_label=False,
            public_conflict_guard_is_ROI_addition=False,
        ),
    )
    template = (PARENT / "internal_comparison.html").read_text(encoding="utf-8")
    begin = '<script id="data" type="application/json">'
    start = template.index(begin) + len(begin)
    end = template.index("</script>", start)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = template[:start] + payload + template[end:]
    names = "const names={cross20:'原Qwen方法',subject_caption20:'新主售描述Qwen',subject_guard20:'新主售描述Qwen加公开冲突扣分'};"
    html, substitutions = re.subn(
        r"const names=\{.*?\};", lambda _: names, html, count=1
    )
    assert substitutions == 1
    html = re.sub(
        r"<title>.*?</title>",
        f"<title>模块二{size}件同图库主售主体开发对照</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1>.*?</h1>",
        f"<h1>{size}件同图库：原Qwen、新主售描述、公开冲突扣分</h1>",
        html,
        count=1,
    )
    (ROOT / "internal_comparison.html").write_text(html, encoding="utf-8")


def paired_fields(value: dict, scope: str, candidate: object) -> dict:
    """Return paired fields for this workflow.

    Args:
        value: Input value to validate or transform.
        scope: Scope used by this operation.
        candidate: Candidate used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return value["scopes"][scope][candidate]["cross_evidence_pool60"]


def append_incremental(lines: list, title: str, data: dict, candidate: object) -> None:
    """Apply the append incremental workflow step.

    Args:
        lines: Lines used by this operation.
        title: Public product title used as input.
        data: Data used by this operation.
        candidate: Candidate used by this operation.
    """
    lines += [
        "",
        title,
        "",
        data.get("meaning", ""),
        "",
        "配对界类型：" + data.get("bound_type", "见原始配对文件"),
        "",
        "| 范围 | 指标 | 差值范围（百分点） | 不退化可证明 |",
        "| --- | --- | --- | --- |",
    ]
    for scope in data["scopes"]:
        for metric, value in paired_fields(data, scope, candidate).items():
            lines.append(
                f"| {scope} | {metric} | {percent(value['delta_range'])} | {value['nonregression_proven_for_shared_unknown_labels']} |"
            )


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, once.
    """
    report_guard()
    assert load(ROOT / "verification.json")["passed"]
    runtime = audit()
    acquisition = load(ROOT / "acquisition_receipt.json")
    label_receipt = load(ROOT / "label_receipt.json")
    size = len(load(ROOT / "tune/catalog.public.json"))
    lines = [
        "# 模块二：精确主体补库与同图库开发对照",
        "",
        "内部开发证据；人工标签0。40条现有查询全部已曝光，本轮不是独立验收。K固定10。单方法区间是未决相关性所有取值的精确范围；未知项较多时配对差值使用保守界，不能称为精确差值。这些区间不是置信区间，也不覆盖AI误判。",
        "",
        f"冻结{acquisition['candidates']}条新元数据，全部尝试采集；成功{acquisition['downloaded']}张、失败{acquisition['failed']}条，技术重复{acquisition['technical_duplicates']}张。{acquisition['selected']}件技术可用商品全部逐图AI评审并补库，未按相关性剔除。原7720条AI关系不变，新增{label_receipt['new_relations']}条，总{label_receipt['total_relations']}条；不确定项继续为null。",
        "",
        f"两组图库各{size}件，后8张历史命名fresh的查询共享开发A图库。相同图库、查询、标签下比较8种方法，共320份完整排序。主方法为新主售描述Qwen加公开主体冲突扣分；它没有叠加ROI。ROI仍是使用原查询描述的单独参考方法。",
        "",
        "[逐查询商品对照](internal_comparison.html) · [AI评审来源](native/review_provenance.json) · [运行审计](runtime_audit.json)",
        "",
    ]
    gates: dict
    capacities: dict
    gates, capacities = {}, {}
    for cohort in COHORTS:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        paired = load(ROOT / cohort / "internal_paired_differences.json")
        lines += [
            f"## {cohort}（已曝光开发数据）",
            "",
            "配对差值界类型：" + paired.get("bound_type", "见原始配对文件"),
            "",
            "| 范围 | 方法 | P@10 % | Recall@10 % | mAP % | NDCG@10 % | MRR % | Hit@1 % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        gates[cohort], capacities[cohort] = {}, {}
        for scope, methods in metrics["scopes"].items():
            gates[cohort][scope] = {}
            for method in METHODS:
                bounds = methods[method]["ranges"]
                lines.append(
                    "| "
                    + " | ".join(
                        [scope, DISPLAY[method], *[percent(bounds[m]) for m in METRICS]]
                    )
                    + " |"
                )
            for method in ["subject_caption20", PRIMARY]:
                fields = paired_fields(paired, scope, method)
                gates[cohort][scope][method] = dict(
                    precision_60=methods[method]["precision_60_lower_bound_met"],
                    recall_85=methods[method]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for v in fields.values()
                    ),
                    paired=fields,
                )
            rows = [
                q
                for q in metrics["query_capacity"]
                if scope == "all" or q["query_id"].startswith(scope + "_")
            ]
            capacities[cohort][scope] = dict(
                precision_oracle_ceiling=[
                    sum(q["precision_ceiling_range"][i] for q in rows) / len(rows)
                    for i in [0, 1]
                ]
            )
        append_incremental(
            lines, "### 主售描述变化：新描述Qwen减原Qwen", paired, "subject_caption20"
        )
        append_incremental(
            lines,
            "### 公开冲突扣分消融：主方法减新描述Qwen",
            load(ROOT / cohort / "internal_guard_incremental.json"),
            PRIMARY,
        )
        append_incremental(
            lines,
            "### ROI独立参考消融：ROI图片融合减图片融合",
            load(ROOT / cohort / "internal_roi_incremental.json"),
            PRIMARY,
        )
        lines += [
            "",
            "### 补库前后（图库和相关正例分母改变，不是算法净收益）",
            "",
            "| 范围 | 相同方法 | 原193件P@10 % | 新图库P@10 % | 原193件Recall@10 % | 新图库Recall@10 % |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        previous = load(PARENT / cohort / "internal_metrics.ai.development.json")
        for scope in metrics["scopes"]:
            for method in METHODS[:6]:
                a, b = (
                    previous["scopes"][scope][method]["ranges"],
                    metrics["scopes"][scope][method]["ranges"],
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            scope,
                            DISPLAY[method],
                            percent(a["precision@10"]),
                            percent(b["precision@10"]),
                            percent(a["recall@10"]),
                            percent(b["recall@10"]),
                        ]
                    )
                    + " |"
                )
        lines += [
            "",
            "### 每查询容量",
            "",
            "| 查询 | 已知正例 | 未决 | P@10理论上限范围 % |",
            "| --- | --- | --- | --- |",
        ]
        for query in metrics["query_capacity"]:
            lines.append(
                f"| {query['query_id']} | {query['known_positive']} | {query['unknown']} | {percent(query['precision_ceiling_range'])} |"
            )
        lines += [
            "",
            "Recall的分母为完整图库的相关商品数R。R>10时Top-10的最大Recall为10/R；零正例查询继续保留，Recall未定义而不是满分，其他指标继续计入。保持原AI标签与未决项，不通过修改分母或补成有利标签追目标。",
            "",
        ]
    primary_checks = [
        scopes[PRIMARY] for cohort in gates.values() for scopes in cohort.values()
    ]
    precision_met = all(v["precision_60"] for v in primary_checks)
    recall_met = all(v["recall_85"] for v in primary_checks)
    nonregression = all(v["nonregression"] for v in primary_checks)
    decision = dict(
        primary_method=PRIMARY,
        development_precision_target_met=precision_met,
        development_recall_preference_met=recall_met,
        development_six_metric_nonregression_proven=nonregression,
        scope_checks=gates,
        precision_oracle_capacity=capacities,
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        independent_blind_queries=0,
        recommendation=(
            "Candidate remains development-only. Reconcile all paired metrics and freeze before an entirely new-gallery/new-query independent acceptance."
        ),
    )
    once(ROOT / "decision.json", decision)
    lines += [
        "## 运行证据与结论",
        "",
        f"完整Qwen嵌入实际{runtime['full_Qwen_embedding_calls']}次，其中新查询caption文本编码{runtime['new_caption_Qwen_calls']}次、主体不明弃权{runtime['unclear_query_caption_abstentions']}条。完整本地视觉模型仅凭查询图生成24条新描述，不提供图库、标签或排名。完整Qwen精排新算{runtime['new_full_reranker_pairs']}对、输入身份一致复用{runtime['exact_reused_reranker_pairs']}对，共{runtime['full_reranker_pairs']}对。完整GroundingDINO调用{runtime['full_detector_calls']}次，接受ROI {runtime['accepted_ROIs']}张、回退{runtime['global_fallbacks']}张；新增完整FG视觉前向{runtime['full_FG_vision_forward_calls']}次、文本前向{runtime['full_FG_text_forward_calls']}次。",
        "",
        f"主方法全部组别及all/text/image的Precision下界达60%：{precision_met}；Recall下界达85%：{recall_met}；六指标配对不退化可证明：{nonregression}。所有原始方法、失败、未决和回退结果保留。没有从不同方法拼接最高数字；是否达正式目标仍由后续独立验收决定。",
        "",
        "新增两种主售主体方法均保持16条文本查询逐条等于本轮原Qwen排序。中文文本与图片入口、统一图文特征、原品类属性证据及完整Top-N结果保留。开发查看页显示原标题、原图、AI主体类型和真实排序分数；分数不是概率。浏览器交互及在线端到端时延验收尚未完成。",
        "",
        "全部本轮元数据、采集成功项、失败项和衍生图都属于开发曝光，不能复用为未见测试。默认方法不自动替换，正式质量字段为空。最终方法和评审协议冻结后，独立验收必须使用全新图库与全新查询，并按最新排除清单检查商品族、字节/像素重复及dHash近重复。",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    viewer()
    print(
        dict(
            development_precision_target_met=precision_met,
            development_recall_preference_met=recall_met,
            nonregression=nonregression,
            formal_metrics=None,
        ),
        flush=True,
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, release.mkdir, write_json.
    """
    report_guard()
    assert (
        load(ROOT / "verification.json")["passed"]
        and load(ROOT / "checks.json")["passed"]
    )
    assert (ROOT / "decision.json").exists()
    release = ROOT / "source_release"
    release.mkdir(exist_ok=True)
    sources = [
        PROJECT_ROOT / "scripts" / name
        for name in [
            "acquire_primary_subjects.py",
            "primary_subject_captions.py",
            "primary_subject_captions_safe.py",
            "review_primary_subjects.py",
            "encode_primary_subjects.py",
            "primary_subject_experiment.py",
            "primary_subject_ranker.py",
            "check_primary_subject_inputs.py",
            "check_primary_subject_results.py",
            "continue_primary_subjects.py",
            Path(__file__).name,
        ]
    ]
    sources.append(PROJECT_ROOT / "tests/test_primary_subject_ranker.py")
    manifest = {}
    for source in sources:
        target = release / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256_file(target) == sha256_file(source)
        manifest[relative_source(source)] = sha256_file(source)
    once(release / "manifest.json", manifest)
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name not in {"completion_receipt.json", "public_status.json"}
    }
    project_files = load(ROOT / "exposure_receipt.json")["project_files"]
    decision = load(ROOT / "decision.json")
    acquisition = load(ROOT / "acquisition_receipt.json")
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_primary_subject_development",
            candidates_attempted=acquisition["attempted"],
            acquired_and_AI_reviewed=acquisition["selected"],
            source_failed=acquisition["failed"],
            added_products=acquisition["selected"],
            gallery_per_cohort=len(load(ROOT / "tune/catalog.public.json")),
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            development_precision_target_met=decision[
                "development_precision_target_met"
            ],
            development_recall_preference_met=decision[
                "development_recall_preference_met"
            ],
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    # Publish final status before the final authority marker, so interruption
    # cannot leave an apparently sealed experiment with a pending status.
    once(
        ROOT / "completion_receipt.json",
        dict(
            files=files,
            source_files=manifest,
            project_files=project_files,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
        ),
    )
    print(dict(sealed_files=len(files), source_snapshots=len(manifest)), flush=True)


def relative_source(path: Path) -> str:
    """Return relative source for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "report", "exposure", "seal"])
    globals()[parser.parse_args().action]()
