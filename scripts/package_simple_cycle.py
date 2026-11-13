"""Package the frozen cheap experiment as indexes and an honest local report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.safety import sha256_file, write_json
from simple_baseline_cycle import ROOT, guard

METHODS = {
    "whole_cosine": "原图召回",
    "crop_cosine": "主图裁切召回",
    "crop_top20_complete_reranker": "裁切召回＋Top20 完整精排",
}


def load(path: Path) -> Any:
    """Decode one frozen experiment artifact.

    Args:
        path: Existing local JSON receipt or results.

    Returns:
        Decoded JSON at this serialization boundary.
    """
    return json.loads(path.read_text())


def interval(values: list, scale: float = 100) -> str:
    """Format label-completion bounds without creating a point estimate.

    Args:
        values: Lower and upper bounds, possibly undefined.
        scale: Percentage or percentage-point display multiplier.

    Returns:
        Compact interval or an explicit undefined marker.
    """
    if any(value is None for value in values):
        return "未定义"
    return f"[{values[0] * scale:.2f}, {values[1] * scale:.2f}]"


def indexes(inputs: dict) -> dict:
    """Persist exact normalized gallery banks in the original candidate order.

    Args:
        inputs: Frozen development catalogs, query identities and image hashes.

    Returns:
        Index file checksums; no new encoding or model inference is performed.
    """
    files = {}
    receipt = load(ROOT / "encoding_receipt.json")
    for name, digest in receipt["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Encoded feature changed before index packaging")
    for cohort, info in inputs["cohorts"].items():
        folder = ROOT / cohort / "index"
        folder.mkdir(exist_ok=True)
        ids = info["gallery"]
        for kind in ["whole", "crop"]:
            matrix = np.stack(
                [
                    np.load(
                        ROOT
                        / "vectors"
                        / f"{hashlib.sha256(cid.encode()).hexdigest()[:24]}_{kind}.npy",
                        allow_pickle=False,
                    )
                    for cid in ids
                ]
            )
            if matrix.shape != (315, 2048) or not np.allclose(
                np.linalg.norm(matrix, axis=1), 1, atol=1e-5
            ):
                raise ValueError("Invalid normalized gallery matrix")
            path = folder / f"{kind}.npz"
            np.savez(path, ids=np.asarray(ids), vectors=matrix)
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        write_json(
            folder / "manifest.json",
            dict(
                role="exposed_development_only",
                candidates=len(ids),
                dimensions=2048,
                similarity="exact cosine of normalized vectors",
                query_input="unchanged whole image with official query instruction",
                files={
                    name: digest
                    for name, digest in files.items()
                    if name.startswith(cohort + "/")
                },
                encoding_receipt_sha256=sha256_file(ROOT / "encoding_receipt.json"),
                formal_metrics=None,
            ),
        )
    return files


def main() -> None:
    """Export actual indexes, component changes and a clearly scoped result page."""
    guard()
    summary = load(ROOT / "summary.json")
    inputs = load(ROOT / "inputs.json")
    files = indexes(inputs)
    absolute: list[str] = []
    increments: list[str] = []
    decision: dict = {}
    for cohort, methods in summary["results"].items():
        diagnostic = load(ROOT / cohort / "AI_diagnostic_metrics.json")
        for method, result in methods.items():
            ranges = result["ranges"]
            per_query = [
                row["ranges"]["precision@10"]
                for row in diagnostic["query_metrics"]
                if row["method"] == method
            ]
            floor = [min(row[i] for row in per_query) for i in [0, 1]]
            absolute.append(
                "<tr><td>"
                + cohort
                + "</td><td>"
                + METHODS[method]
                + "</td>"
                + "".join(
                    "<td>" + interval(ranges[metric]) + "</td>"
                    for metric in [
                        "precision@10",
                        "recall@10",
                        "ap",
                        "ndcg@10",
                        "mrr",
                        "hit@1",
                    ]
                )
                + "<td>"
                + interval(floor)
                + f"</td><td>{sum(row[0] >= 0.6 - 1e-12 for row in per_query)}/8</td>"
                + "</tr>"
            )
        paired = load(ROOT / cohort / "paired_increments.json")["scopes"]["all"]
        decision[cohort] = {}
        for candidate, baselines in paired.items():
            for baseline, metrics in baselines.items():
                values = {
                    metric: field["delta_range"] for metric, field in metrics.items()
                }
                decision[cohort][candidate + " - " + baseline] = values
                increments.append(
                    "<tr><td>"
                    + cohort
                    + "</td><td>"
                    + METHODS[candidate]
                    + " − "
                    + METHODS[baseline]
                    + "</td><td>"
                    + interval(values["precision@10"])
                    + "</td><td>"
                    + interval(values["recall@10"])
                    + "</td></tr>"
                )
    write_json(
        ROOT / "decision.json",
        dict(
            component_increments=decision,
            default_strategy_changed=False,
            pending_human_review=True,
            no_claim_of_uniform_improvement=True,
            formal_metrics=None,
        ),
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>模块二低成本对照</title>'
        + "<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:20px}"
        + "table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:8px}</style>"
        + "<h1>模块二：主图裁切与统一 Top20 精排</h1>"
        + "<p>2026-11-05。已曝光开发集：每组315件商品、8条图片查询，共411件唯一商品、"
        + "24条图片查询。所有411张商品主图均处理，361张有有效主体框并裁切，"
        + "50张无有效框时保留原图。原图与裁切在同一本机 MPS 上重新编码，查询保持原图。</p>"
        + "<p>三个方法合计72份完整排序。每条查询的裁切召回 Top20 都使用完整 "
        + "Qwen3-VL-Reranker-2B，输入原始查询图、候选原图及原标题，共480对实际分数；"
        + "尾部候选完整保留，没有掩码、描述生成、学习融合或新增训练。</p>"
        + "<p><strong>以下是AI标签下的开发诊断，不能作为人工准确率或最终验收成绩。</strong>"
        + "区间仅处理未判定标签，不覆盖AI错标，也不是置信区间。Recall使用完整图库正例分母。"
        + "默认策略尚未替换；需要检查逐组增量和人工复核后再决定。</p>"
        + "<h2>三组结果（%，mAP/MRR同乘100）</h2><table><tr><th>开发组</th>"
        + "<th>方法</th><th>P@10</th><th>R@10</th><th>mAP</th><th>NDCG@10</th>"
        + "<th>MRR</th><th>Hit@1</th><th>最差查询P@10</th><th>P@10下界≥60%条数</th></tr>"
        + "".join(absolute)
        + "</table><h2>共享未知标签下的成对增量（百分点）</h2><table>"
        + "<tr><th>开发组</th><th>变化</th><th>ΔP@10</th><th>ΔR@10</th></tr>"
        + "".join(increments)
        + '</table><h2>人工复核</h2><p><a href="human_audit/index.html">打开2520条人工复核任务</a>。'
        + "每批30条，填写姓名、确认本人独立判断、选择相关性并填理由，导出CSV至 "
        + "human_audit/answers/。同批修改请覆盖旧文件。原答案为空，"
        + "评分命令：<code>.venv-mac/bin/python scripts/simple_cycle_review.py score</code>。</p>"
        + "<p>完整复核前整体AI标注准确率为空。AI未判定与人工证据不足分开统计；"
        + "不足80%时需要扩充人工复核或重标注，不用准确率乘检索指标冒充修正。</p>"
        + "<h2>验收隔离</h2><p>两轮已消耗盲测继续封存，禁止用于调参。本轮新冻结300件商品"
        + "与30条图片查询，仅按元数据及图像指纹去重，不做模型推理或语义预览。"
        + "还需独立人工文本查询、相关性真值和单一最终模型冻结后才能正式验收。</p>"
        + '<p><a href="verification.json">完整性与隔离验证</a> · '
        + '<a href="summary.json">原始指标</a> · <a href="decision.json">逐组件增量</a></p>'
    )
    (ROOT / "report.html").write_text(page, encoding="utf-8")
    write_json(
        ROOT / "delivery_receipt.json",
        dict(
            files={
                **files,
                **{
                    name: sha256_file(ROOT / name)
                    for name in ["summary.json", "report.html", "decision.json"]
                },
            },
            protocol_sha256=sha256_file(ROOT / "protocol.json"),
            packaging_source_sha256=sha256_file(Path(__file__)),
            AI_development_only=True,
            formal_metrics=None,
        ),
    )
    print(json.dumps(dict(indexes=len(files), report=str(ROOT / "report.html"))))


if __name__ == "__main__":
    main()
