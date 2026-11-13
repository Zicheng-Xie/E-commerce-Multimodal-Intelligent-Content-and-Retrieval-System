"""Internal paired tables, ROI contact sheet and reviewable retrieval results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np

import run  # noqa: F401
from compare_fgclip2_retrieval import ROOT as FGROOT
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from report_fgclip2_comparison import percent
from subject_roi_experiment import (
    COHORTS,
    METHODS,
    ROOT,
    check,
    detection_path,
    load,
    once,
)

DISPLAY: dict[str, str] = {
    "cross20": "原Qwen方法",
    "fgclip2_local_fusion20": "上一轮全模态FG融合",
    "image_fusion20": "仅图片FG融合",
    "image_roi_fusion20": "仅图片FG融合加主体ROI",
}
METRICS = ["precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1"]


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Notes:
        Writes derived artifacts through once.
    """
    once(
        ROOT / "report_protocol.json",
        dict(
            files={
                Path(__file__)
                .resolve()
                .relative_to(PROJECT_ROOT)
                .as_posix(): sha256_file(Path(__file__))
            },
            main_method="image_roi_fusion20",
            report_all_methods=True,
            incremental_ablation="ROI versus image-only fusion under same unknown-label completions",
            automatic_promotion=False,
            formal_metrics=None,
        ),
    )


def image_url(path: Path) -> str:
    """Return image url for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/:")


def public_scores(
    data: dict, query: dict, orders: dict, whole: np.ndarray, roi: np.ndarray
) -> dict:
    """Return public scores for this workflow.

    Args:
        data: Data used by this operation.
        query: Query record or representation used by this operation.
        orders: Orders used by this operation.
        whole: Whole used by this operation.
        roi: Roi used by this operation.

    Returns:
        Mapping with fields: cross20.
    """
    f = load(PROJECT_ROOT / query["original_features"])
    raw = load(PROJECT_ROOT / query["original_cross"])
    ids = f["ids"]
    positions = {cid: i for i, cid in enumerate(ids)}
    pool = raw["pool"][:20]
    chosen = np.array([positions[cid] for cid in pool])
    base = 0.35 * normalized_ranks(ids, np.array(f["base"]))[chosen]
    offsets = np.array(f["offsets"])[chosen] if f["offsets"] is not None else 0
    cross = normalized_ranks(pool, np.array(raw["logits"][:20]))
    result = {"cross20": dict(zip(pool, (0.65 * cross + base + offsets).tolist()))}
    for method, aux in [("image_fusion20", whole), ("image_roi_fusion20", roi)]:
        if query["image_key"]:
            score = (
                0.65 * (0.75 * cross + 0.25 * normalized_ranks(pool, aux[chosen]))
                + base
                + offsets
            )
            result[method] = dict(zip(pool, score.tolist()))
        else:
            result[method] = result["cross20"]
        assert (
            sorted(pool, key=lambda cid: (-result[method][cid], cid))
            == orders[method][:20]
        )
    return result


def viewer(inputs: dict) -> None:
    """Apply the viewer workflow step.

    Args:
        inputs: Inputs used by this operation.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.html').write_text, write_json.
    """
    rows = []
    for cohort, data in inputs["cohorts"].items():
        ranks = load(ROOT / cohort / "predictions.json")
        new = load(ROOT / cohort / "runtime_scores.json")
        prior = load(FGROOT / cohort / "runtime_scores.json")
        ids = [d["id"] for d in data["gallery"]]
        docs = {d["id"]: d for d in data["gallery"]}
        grades = {
            (r["query_id"], r["candidate_id"]): r["grade"]
            for r in read_rows(PROJECT_ROOT / data["labels"])
        }
        for query in data["queries"]:
            qid = query["id"]
            whole = np.array(prior[qid]["scores"]["fused_scores"])
            combined = whole
            if query["image_key"]:
                combined = 0.5 * normalized_ranks(ids, whole) + 0.5 * normalized_ranks(
                    ids, np.array(new[qid]["roi_scores"]["fused_scores"])
                )
            scores = public_scores(data, query, ranks[qid], whole, combined)
            results: dict = {}
            for method in ["cross20", *METHODS]:
                results[method] = []
                for cid in ranks[qid][method][:20]:
                    d = docs[cid]
                    req = inputs["roi_requests"][d["roi_request"]]
                    results[method].append(
                        dict(
                            id=cid,
                            title=inputs["texts"][d["title_key"]],
                            image=image_url(
                                PROJECT_ROOT / inputs["images"][d["image_key"]]
                            ),
                            category=req["hint"],
                            score=scores[method][cid],
                            AI_grade=grades[(qid, cid)],
                        )
                    )
            row = dict(
                id=cohort + "/" + qid,
                query=inputs["texts"].get(query["text_key"], "未生成描述"),
                image=(
                    image_url(PROJECT_ROOT / inputs["images"][query["image_key"]])
                    if query["image_key"]
                    else None
                ),
                results=results,
            )
            if query["roi_request"]:
                detected = load(detection_path(query["roi_request"]))
                row["roi_state"] = detected["decision"]["status"]
                row["crop"] = (
                    image_url(
                        detection_path(query["roi_request"]).with_name("zoom.png")
                    )
                    if row["roi_state"] == "roi"
                    else row["image"]
                )
            rows.append(row)
    write_json(
        ROOT / "runtime_results.internal.json",
        dict(
            review_origin="model",
            human_label=False,
            score_is_probability=False,
            metric_K=10,
            queries=rows,
        ),
    )
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><meta charset="utf-8"><title>模块二主体ROI开发对照</title>
<style>body{font:15px system-ui;background:#f4f6fa;color:#17233b;margin:24px}select{padding:8px;max-width:60vw}header{background:white;padding:20px;border-radius:12px}#q img{height:140px;max-width:220px;object-fit:contain;margin-right:12px}.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}.column{min-width:0}.card{background:white;padding:12px;margin:10px 0;border-radius:10px;display:flex;gap:12px}.card img{width:100px;height:100px;object-fit:contain}.copy{min-width:0}.title{font-weight:600;overflow-wrap:anywhere}.detail{font-size:12px;color:#52627b;margin-top:6px}.grade{color:#985a00;font-size:12px}h2{font-size:18px}.note{color:#52627b;line-height:1.7}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<header><h1>原方法、图片融合与主体ROI对照</h1><p class="note">已曝光数据上的AI开发实验。排序分数不是相关概率；评测K固定为10。页面显示数量不改变实验指标。模型判断不代表人工真值。</p><select id="selector"></select> 显示<select id="count"><option>10</option><option>5</option><option>20</option></select>件 <label><input type="checkbox" id="labels">显示原AI关系</label><div id="q"></div></header><main id="cols" class="columns"></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('selector'),cols=document.getElementById('cols');
const names={cross20:'原Qwen方法',image_fusion20:'仅图片FG融合',image_roi_fusion20:'仅图片FG融合加主体ROI'};
for(const row of rows){let o=document.createElement('option');o.value=row.id;o.textContent=row.id+' '+row.query.slice(0,48);sel.append(o)}
function show(){const row=rows.find(r=>r.id===sel.value),q=document.getElementById('q');q.replaceChildren();let p=document.createElement('p');p.textContent=row.query;q.append(p);for(const url of [row.image,row.crop])if(url){let im=document.createElement('img');im.src=url;q.append(im)}cols.replaceChildren();for(const [method,name]of Object.entries(names)){let col=document.createElement('section');col.className='column';let h=document.createElement('h2');h.textContent=name;col.append(h);row.results[method].slice(0,Number(document.getElementById('count').value)).forEach((d,i)=>{let card=document.createElement('article');card.className='card';let im=document.createElement('img');im.src=d.image;card.append(im);let c=document.createElement('div');c.className='copy';let t=document.createElement('div');t.className='title';t.textContent=(i+1)+'. '+d.title;c.append(t);let x=document.createElement('div');x.className='detail';x.textContent='主体类型：'+d.category+' | 排序分数：'+d.score.toFixed(4);c.append(x);if(document.getElementById('labels').checked){let g=document.createElement('p');g.className='grade';g.textContent='原AI判断：'+(d.AI_grade===null?'未知':d.AI_grade===1?'相关':'不相关');c.append(g)}card.append(c);col.append(card)});cols.append(col)}}
sel.value=rows.find(r=>r.image).id;for(const id of ['selector','count','labels'])document.getElementById(id).addEventListener('change',show);show();
</script>""".replace("PAYLOAD", payload)
    (ROOT / "internal_comparison.html").write_text(html, encoding="utf-8")


def contact_sheet(inputs: dict) -> None:
    """Apply the contact sheet workflow step.

    Args:
        inputs: Inputs used by this operation.

    Notes:
        Writes derived artifacts through sheet.save.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    cases = [
        (c, q)
        for c, d in inputs["cohorts"].items()
        for q in d["queries"]
        if q["image_key"]
    ]
    sheet = Image.new("RGB", (4 * 360, 6 * 218), "#eef2f8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=13)
    for i, (cohort, q) in enumerate(cases):
        x, y = (i % 4) * 360, (i // 4) * 218
        result = load(detection_path(q["roi_request"]))
        source = PROJECT_ROOT / inputs["images"][q["image_key"]]
        first = (
            detection_path(q["roi_request"]).with_name("box.png")
            if result["decision"]["status"] == "roi"
            else source
        )
        second = (
            detection_path(q["roi_request"]).with_name("zoom.png")
            if result["decision"]["status"] == "roi"
            else source
        )
        for n, path in enumerate([first, second]):
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((170, 174))
                sheet.paste(
                    image,
                    (
                        x + n * 180 + (170 - image.width) // 2,
                        y + 22 + (174 - image.height) // 2,
                    ),
                )
        draw.text((x + 4, y + 2), cohort + "/" + q["id"], font=font, fill="#17233b")
        draw.text(
            (x + 4, y + 197),
            (result["prompt"] or "unsupported") + " / " + result["decision"]["status"],
            font=font,
            fill="#52627b",
        )
    sheet.save(ROOT / "query_roi_contact_sheet.png")


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, write_json.
    """
    inputs = check()
    assert load(ROOT / "verification.json")["passed"]
    guard = load(ROOT / "report_protocol.json")
    assert guard["files"][
        Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    ] == sha256_file(Path(__file__))
    lines = [
        "# 主体ROI与图片专用融合开发对照",
        "",
        "40条查询全部为已曝光开发数据，6720条原AI关系不变，K=10。区间是未知关系所有取值的精确范围，不是抽样置信区间，不覆盖AI误判。两组168件图库共享72件商品；最后8张图片查询复用开发组A图库。主方法与规则在新检测和排序前冻结，没有事后调权重。",
        "",
        "[打开逐查询商品对照](internal_comparison.html) · [查看24张查询的主体框与裁切](query_roi_contact_sheet.png)",
        "",
    ]
    gates: dict = {}
    for cohort in COHORTS:
        metrics = load(ROOT / cohort / "internal_metrics.ai.development.json")
        differences = load(ROOT / cohort / "internal_paired_differences.json")
        incremental = load(ROOT / cohort / "internal_roi_incremental_paired.json")
        gates[cohort] = {}
        lines.extend(
            [
                "## " + cohort + "（开发数据）",
                "",
                "| 范围 | 方法 | P@10 % | Recall@10 % | mAP % | NDCG@10 % | MRR % | Hit@1 % |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for scope, methods in metrics["scopes"].items():
            for method, name in DISPLAY.items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            scope,
                            name,
                            *[percent(methods[method]["ranges"][m]) for m in METRICS],
                        ]
                    )
                    + " |"
                )
            gates[cohort][scope] = {}
            for method in METHODS:
                fields = differences["scopes"][scope][method]["cross_evidence_pool60"]
                gates[cohort][scope][method] = dict(
                    nonregression=all(
                        r["nonregression_proven_for_shared_unknown_labels"]
                        for r in fields.values()
                    ),
                    paired=fields,
                )
        lines.extend(
            [
                "",
                "ROI增量：主方法减仅图片融合，单位为百分点，使用相同未决标签取值。",
                "",
                "| 范围 | 指标 | ROI增量范围 | 不退化可证明 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for scope, methods in incremental["scopes"].items():
            for metric, evidence in methods["image_roi_fusion20"][
                "cross_evidence_pool60"
            ].items():
                lines.append(
                    f"| {scope} | {metric} | {percent(evidence['delta_range'], delta=True)} | {evidence['nonregression_proven_for_shared_unknown_labels']} |"
                )
        lines.append("")
    detector = load(ROOT / "detection_receipt.json")
    encoder = load(ROOT / "encoding_receipt.json")
    nonregression = {
        m: all(
            scopes[m]["nonregression"]
            for cohorts in gates.values()
            for scopes in cohorts.values()
        )
        for m in METHODS
    }
    decision = dict(
        primary_method="image_roi_fusion20",
        development_nonregression=nonregression,
        scope_checks=gates,
        text_outputs_exactly_original=True,
        accepted_roi=detector["accepted_roi"],
        global_fallback=detector["global_fallback"],
        actual_detector_forward_calls=detector["model_forward_calls"],
        actual_new_full_FG_vision_forward_calls=encoder[
            "actual_full_FG_vision_forward_calls"
        ],
        new_text_embedding_calls=0,
        reused_original_Qwen_pairs=800,
        new_Qwen_pairs=0,
        formal_acceptance=False,
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        new_independent_AI_blind_predictions=0,
    )
    write_json(ROOT / "decision.json", decision)
    lines.extend(
        [
            "## 运行与结论边界",
            "",
            f"完整GroundingDINO实际前向{detector['model_forward_calls']}次，接受主体ROI {detector['accepted_roi']}个，回退整图{detector['global_fallback']}个。新增完整FG视觉前向{encoder['actual_full_FG_vision_forward_calls']}次，原图特征和文本特征原样复用。原Qwen 800对精排分数复用，新增Qwen精排0对。",
            "",
            f"六指标开发不退化门槛：{nonregression}。文本排序逐条等同原版。默认方案不自动替换，未新增独立验收结果。",
            "",
            "本轮保持统一图文主检索、中文文本与图片两种入口、原品类与属性偏移、完整Top-N排序；另提供开发结果查看页，展示标题、主图、AI主体类型和真实排序分数。页面的AI关系可显式开启，排序分数不作为相关概率。",
            "",
            "检测框置信度不是商品相关性概率。没有框真值，本轮不报告检测准确率。模型实际调用、图像尺寸与有效patch数逐项留档；端到端在线时延验收尚未完成。",
        ]
    )
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    viewer(inputs)
    contact_sheet(inputs)
    print(
        json.dumps(
            {k: v for k, v in decision.items() if k != "scope_checks"},
            ensure_ascii=False,
            indent=2,
        )
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, release.mkdir, write_json.
    """
    check()
    assert (
        load(ROOT / "verification.json")["passed"]
        and load(ROOT / "checks.json")["passed"]
    )
    source = [
        PROJECT_ROOT / "src/multimodal_ecommerce/subject_roi.py",
        PROJECT_ROOT / "scripts/prepare_grounding_model.py",
        PROJECT_ROOT / "scripts/subject_roi_experiment.py",
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests/test_subject_roi.py",
    ]
    release = ROOT / "source_release"
    release.mkdir(exist_ok=True)
    manifest = {}
    for p in source:
        target = release / p.name
        if not target.exists():
            shutil.copyfile(p, target)
        assert sha256_file(target) == sha256_file(p)
        manifest[p.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(p)
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
            formal_metrics=None,
            default_strategy_changed=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_development_semantic_ROI",
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print(json.dumps(dict(sealed_files=len(files))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "report", "seal"])
    globals()[parser.parse_args().action]()
