"""Internal same-gallery tables, runtime evidence, exposure registration and seal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from subject_coverage_acquisition import MEDIA, PARENT, ROOT
from subject_coverage_experiment import COHORTS, guard
from subject_roi_experiment import load, once

METRICS = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]
NAMES = {
    "global": "整图",
    "region_title": "区域与标题",
    "cross20": "原Qwen方法",
    "fgclip2_local_fusion20": "全模态FG融合",
    "image_fusion20": "图片FG融合",
    "image_roi_fusion20": "图片FG融合加主体ROI",
}


def percent(bounds: dict) -> str:
    """Return percent for this workflow.

    Args:
        bounds: Bounds used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return "–".join("未定义" if x is None else f"{100 * x:.2f}" for x in bounds)


def audit() -> None:
    """Apply the audit workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    fg = load(ROOT / "fg_runtime.json")
    image_calls = [r for r in fg["calls"] if r["kind"] == "image_global_and_dense"]
    text_calls = [r for r in fg["calls"] if r["kind"] == "text"]
    assert len(image_calls) == 43 and len(text_calls) == 25
    vision = sum(r["actual_vision_forward_calls"] for r in image_calls)
    assert vision == 86
    # The acquisition/model receipts stay immutable. Correct this bookkeeping
    # discriminator explicitly; full model outputs themselves are unaffected.
    original_field = load(ROOT / "fg_encoding_receipt.json")["actual_forward_calls"]
    new = sum(
        len(load(p)["new_pair_ids"])
        for c in COHORTS
        for p in (ROOT / c / "runs").glob("*/cross.json")
    )
    reused = sum(
        len(load(p)["reused_pair_ids"])
        for c in COHORTS
        for p in (ROOT / c / "runs").glob("*/cross.json")
    )
    timings = [
        t
        for c in COHORTS
        for p in (ROOT / c / "runs").glob("*/cross.json")
        for t in load(p)["timings"]
    ]
    assert new + reused == 800 and new == len(timings)
    once(
        ROOT / "runtime_audit.json",
        dict(
            source_sha256=sha256_file(Path(__file__)),
            full_FG_image_requests=43,
            actual_full_FG_vision_forward_calls=vision,
            full_FG_text_forward_calls=len(text_calls),
            actual_full_FG_total_forward_calls=vision + len(text_calls),
            correction=dict(
                artifact="fg_encoding_receipt.json",
                field="actual_forward_calls",
                original_value=original_field,
                corrected_value=vision + len(text_calls),
                reason="kind is image_global_and_dense, not image; each official image request makes two full visual forwards.",
                original_artifact_preserved=True,
                prediction_outputs_changed=False,
            ),
            full_Qwen_embedding_calls=load(ROOT / "encoding_receipt.json")[
                "actual_full_embedding_calls"
            ],
            full_Qwen_new_reranker_pairs=new,
            full_Qwen_exact_reused_pairs=reused,
            full_detector_calls=load(ROOT / "detection_receipt.json")[
                "actual_full_detector_calls"
            ],
            all_models_complete=True,
        ),
    )


def exposure() -> None:
    """Apply the exposure workflow step.

    Raises:
        FileExistsError: Preserve completed exposure registration.

    Notes:
        Writes derived artifacts through once, write_rows.
    """
    guard()
    final_registry = (
        PROJECT_ROOT / "data/module2_subject_coverage_final_media/records.jsonl"
    )
    crop_registry = (
        PROJECT_ROOT / "data/module2_subject_coverage_exposure/records.jsonl"
    )
    if final_registry.exists() or crop_registry.exists():
        raise FileExistsError("Preserve completed exposure registration")
    final = read_rows(ROOT / "retry/all_attempts.final.jsonl")
    write_rows(final_registry, final)
    rows = []
    paths = sorted((ROOT / "additions/index/products").glob("*/views/[0-9]*.png"))
    paths += sorted((ROOT / "detections").glob("*/native.png"))
    paths += sorted((ROOT / "detections").glob("*/zoom.png"))
    for path in paths:
        rows.append(
            dict(
                id="coverage_derived/" + path.relative_to(ROOT).as_posix(),
                image_path=os.path.relpath(path, crop_registry.parent).replace(
                    "\\", "/"
                ),
                fingerprints=fingerprints(path),
                reusable_as_unseen=False,
                project_role="exposed_development_derived_not_acceptance",
            )
        )
    write_rows(crop_registry, rows)
    prior_path = PARENT / "next_coverage/post_preparation_exclusion_snapshot.json"
    prior = load(prior_path)
    raw = [r["fingerprints"] for r in final if r["status"] == "downloaded"]
    snapshot = dict(prior)
    snapshot.update(
        scope="All prior exposures, all eighty candidate metadata, every acquired image and derived query-aware development crop; none reusable as independent acceptance",
        parent_snapshot_sha256=sha256_file(prior_path),
        image_fingerprints=prior["image_fingerprints"]
        + raw
        + [r["fingerprints"] for r in rows],
        files=dict(
            prior["files"],
            **{
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [MEDIA, final_registry, crop_registry]
            },
        ),
        subject_coverage_candidates=80,
        subject_coverage_acquired=len(raw),
        subject_coverage_derived=len(rows),
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    once(
        ROOT / "exposure_receipt.json",
        dict(
            candidate_metadata=80,
            acquired_images=len(raw),
            derived_images=len(rows),
            project_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in [MEDIA, final_registry, crop_registry]
            },
            all_attempts_preserved=True,
            pixels_and_dhash_excluded=True,
            independent_acceptance_predictions=0,
        ),
    )
    print(dict(exposed_raw=len(raw), exposed_derived=len(rows)), flush=True)


def url(path: Path) -> str:
    """Return url for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/:")


def viewer() -> None:
    """Apply the viewer workflow step.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.html').write_text, once.
    """
    rows = []
    for cohort in COHORTS:
        docs = {d["id"]: d for d in load(ROOT / cohort / "catalog.public.json")}
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
            feature, raw = load(ROOT / cohort / "runs" / qid / "features.json"), load(
                ROOT / cohort / "runs" / qid / "cross.json"
            )
            ids, pool = feature["ids"], raw["pool"]
            pos = {cid: i for i, cid in enumerate(ids)}
            selected = np.array([pos[cid] for cid in pool])
            cross = normalized_ranks(pool, np.array(raw["logits"]))
            base = 0.35 * normalized_ranks(ids, np.array(feature["base"]))[selected]
            offset = (
                np.array(feature["offsets"])[selected]
                if feature["offsets"] is not None
                else 0
            )
            whole = np.array(details[qid]["whole_scores"]["fused_scores"])
            roi = (
                (
                    0.5 * normalized_ranks(ids, whole)
                    + 0.5
                    * normalized_ranks(
                        ids, np.array(details[qid]["roi_scores"]["fused_scores"])
                    )
                )
                if query["image_path"]
                else whole
            )
            scores = {
                "cross20": dict(zip(pool, (0.65 * cross + base + offset).tolist()))
            }
            for method, aux in [("image_fusion20", whole), ("image_roi_fusion20", roi)]:
                score = (
                    0.65 * (0.75 * cross + 0.25 * normalized_ranks(pool, aux[selected]))
                    + base
                    + offset
                )
                scores[method] = (
                    dict(zip(pool, score.tolist()))
                    if query["image_path"]
                    else scores["cross20"]
                )
            result = {}
            for method, values in scores.items():
                assert (
                    sorted(pool, key=lambda cid: (-values[cid], cid))
                    == ranks[qid][method][:20]
                )
                result[method] = [
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
            rows.append(
                dict(
                    id=cohort + "/" + qid,
                    query=query["text"] or feature["description"],
                    image=url(query["image_path"]) if query["image_path"] else None,
                    crop=None,
                    results=result,
                )
            )
    once(
        ROOT / "runtime_results.internal.json",
        dict(
            queries=rows,
            score_is_probability=False,
            metric_k=10,
            gallery=193,
            review_origin="model",
            human_label=False,
        ),
    )
    # Reuse the already verified display template, changing only data and heading.
    parent = (PARENT / "internal_comparison.html").read_text(encoding="utf-8")
    begin = '<script id="data" type="application/json">'
    start = parent.index(begin) + len(begin)
    end = parent.index("</script>", start)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = parent[:start] + payload + parent[end:]
    html = html.replace("模块二主体ROI开发对照", "模块二193件同图库开发对照").replace(
        "原方法、图片融合与主体ROI对照", "补库后193件同图库：原方法、图片融合与主体ROI"
    )
    (ROOT / "internal_comparison.html").write_text(html, encoding="utf-8")


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, once.
    """
    guard()
    assert load(ROOT / "verification.json")["passed"]
    once(
        ROOT / "report_protocol.json",
        dict(
            source_sha256=sha256_file(Path(__file__)),
            default_promotion=False,
            formal_metrics=None,
        ),
    )
    audit()
    acquisition = load(ROOT / "retry/receipt.json")
    runtime = load(ROOT / "runtime_audit.json")
    lines = [
        "# 模块二：80条候选处理与193件同图库开发对照",
        "",
        "内部开发证据；AI评审，人工标签0。40条查询均已曝光，不属于独立验收。K固定10，区间为未决相关性所有取值的精确范围，不是置信区间，也不覆盖AI误判。原有6720条标签逐条不变，新增1000条关系在新排序前冻结。",
        "",
        f"80条全部尝试，55条按原URL再试一次；最终成功{acquisition['downloaded']}张、失败{acquisition['failed']}条（HTTP 420），没有技术重复。25张可用图片全部逐图AI评审并补库，不按相关性删除。55条无图片，仅保留失败及元数据，不伪造像素评审。两组193件图库共享97件商品；历史名fresh的8条图片查询复用A图库，仍然是开发数据。",
        "",
        "[查看40条查询的真实商品结果](internal_comparison.html) · [AI评审记录](native/judgments.json) · [完整采集结果](retry/all_attempts.final.jsonl)",
        "",
    ]
    gates: dict
    bounds: dict
    gates, bounds = {}, {}
    for cohort in COHORTS:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        paired = load(ROOT / cohort / "internal_paired_differences.json")
        lines += [
            f"## {cohort}（已曝光开发数据）",
            "",
            "| 范围 | 方法 | P@10 % | Recall@10 % | mAP % | NDCG@10 % | MRR % | Hit@1 % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        gates[cohort], bounds[cohort] = {}, {}
        for scope, methods in metrics["scopes"].items():
            gates[cohort][scope] = {}
            for method, name in NAMES.items():
                values = methods[method]["ranges"]
                lines.append(
                    "| "
                    + " | ".join([scope, name, *[percent(values[m]) for m in METRICS]])
                    + " |"
                )
            for method in ["image_fusion20", "image_roi_fusion20"]:
                checks = paired["scopes"][scope][method]["cross_evidence_pool60"]
                gates[cohort][scope][method] = dict(
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for v in checks.values()
                    ),
                    precision_60=methods[method]["precision_60_lower_bound_met"],
                    recall_85=methods[method]["recall_85_lower_bound_met"],
                    paired=checks,
                )
            selected_queries = [
                q
                for q in metrics["query_capacity"]
                if scope == "all" or q["query_id"].startswith(scope + "_")
            ]
            capacity = [
                sum(q["precision_ceiling_range"][i] for q in selected_queries)
                / len(selected_queries)
                for i in [0, 1]
            ]
            bounds[cohort][scope] = dict(precision_oracle_ceiling=capacity)
        lines += [
            "",
            "### 补库前后（图库和标签分母变化，不能解释为算法净收益）",
            "",
            "| 范围 | 方法 | 补库前P@10 % | 补库后P@10 % | 补库前Recall@10 % | 补库后Recall@10 % |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        before = load(PARENT / cohort / "internal_metrics.ai.development.json")
        for scope in metrics["scopes"]:
            for method in ["cross20", "image_fusion20", "image_roi_fusion20"]:
                a, b = (
                    before["scopes"][scope][method]["ranges"],
                    metrics["scopes"][scope][method]["ranges"],
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            scope,
                            NAMES[method],
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
        for q in metrics["query_capacity"]:
            lines.append(
                f"| {q['query_id']} | {q['known_positive']} | {q['unknown']} | {percent(q['precision_ceiling_range'])} |"
            )
        lines += [
            "",
            "Recall使用全图库正例分母。正例数量R超过10时，其单查询最大Recall为10/R；零正例查询仍保留，Recall未定义，其他指标照常计入。未知标签保留null，不能按负例或按希望的目标填补。",
            "",
        ]
    nonregression = {
        m: all(v[m]["nonregression"] for c in gates.values() for v in c.values())
        for m in ["image_fusion20", "image_roi_fusion20"]
    }
    decision = dict(
        primary_method="image_roi_fusion20",
        development_nonregression=nonregression,
        scope_checks=gates,
        precision_oracle_capacity=bounds,
        formal_metrics=None,
        goal_met=False,
        default_changed=False,
        independent_blind_queries=0,
        recommendation="Keep original default. Repair source acquisition and actual subject coverage before another frozen comparison; later acceptance requires entirely new gallery and queries.",
    )
    once(ROOT / "decision.json", decision)
    lines += [
        "## 运行证据与结论",
        "",
        f"新增完整Qwen嵌入{runtime['full_Qwen_embedding_calls']}次；精排新算{runtime['full_Qwen_new_reranker_pairs']}对、输入身份校验后复用{runtime['full_Qwen_exact_reused_pairs']}对，共800对。完整GroundingDINO实际推理{runtime['full_detector_calls']}次，18张接受ROI，7张回退整图。新增完整FG视觉前向86次、文本前向25次，共111次。FG原收据有请求数与前向数混淆，原值保留，runtime_audit.json已明确更正计数，不改变模型输出。",
        "",
        f"两个实验方法是否在全部组别及all/text/image的六项指标上均能证明不退化：{nonregression}。图片融合与ROI方法的16条文本排序逐条等于本轮原方法。全部240份完整排序覆盖193件相同图库。",
        "",
        "Precision≥60%的整体目标仍未完成；不能挑选单组好数字或对未知标签作有利补全。此次是开发补库与同图库对照，默认方法保持原版，正式验收指标为空。",
        "",
        "采集失败覆盖全部8条杯类候选；成功的耳机多为耳塞或配件、包类多为电脑或相机内胆，未补足头戴/单耳通话耳机、手袋整理内胆等实际短板。已有图片的挂绳、鞋盒、对讲机等误采作为干扰商品保留。下一轮先解决公开源可访问性并使用结构化主体筛选，保持失败和未决记录，之后重新冻结；当前80条及其衍生图全部排除于未来独立验收。",
        "",
        "原始模块二接口范围保持图文统一特征、中文文本/图片两种入口、品类与属性排序及Top-N查看。此开发页提供标题、主图、AI主体类型与真实排序分数；分数不是概率。页面交互仅进行静态脚本验证，未完成真实浏览器交互验收或在线端到端延迟验收。",
    ]
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    viewer()
    print(dict(development_nonregression=nonregression, goal_met=False), flush=True)


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, release.mkdir, write_json.
    """
    guard()
    assert (
        load(ROOT / "verification.json")["passed"]
        and load(ROOT / "checks.json")["passed"]
    )
    sources = [
        PROJECT_ROOT / "scripts" / n
        for n in [
            "subject_coverage_acquisition.py",
            "complete_subject_coverage_acquisition.py",
            "review_subject_coverage.py",
            "encode_subject_coverage.py",
            "subject_coverage_models.py",
            "subject_coverage_experiment.py",
            "verify_subject_coverage.py",
            "report_subject_coverage.py",
        ]
    ]
    manifest = {}
    release = ROOT / "source_release"
    release.mkdir(exist_ok=True)
    for source in sources:
        target = release / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256_file(source) == sha256_file(target)
        manifest[source.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(source)
    once(release / "manifest.json", manifest)
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name not in {"completion_receipt.json", "public_status.json"}
    }
    once(
        ROOT / "completion_receipt.json",
        dict(
            files=files,
            source_files=manifest,
            project_files=load(ROOT / "exposure_receipt.json")["project_files"],
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_development_subject_coverage",
            candidates_attempted=80,
            acquired_and_AI_reviewed=25,
            source_failed=55,
            added_products=25,
            gallery_per_cohort=193,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print(dict(sealed_files=len(files)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "exposure", "report", "seal"])
    globals()[parser.parse_args().action]()
