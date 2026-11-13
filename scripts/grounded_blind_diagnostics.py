"""Post-prediction capacity and visual replay; never selects a test method."""

from __future__ import annotations

import html
import os
from pathlib import Path

import grounded_blind_dataset as dataset
import grounded_blind_runtime as runtime
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

ROOT = dataset.ROOT


def image_tag(path: Path) -> str:
    """Return image tag for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    p = dataset.Path(path)
    assert p.is_file()
    relative = os.path.relpath(p, ROOT).replace("\\", "/")
    escaped = html.escape(relative, quote=True)
    return f'<img loading="lazy" src="{escaped}" alt="商品原图">'


def write() -> None:
    """Apply the write workflow step.

    Notes:
        Writes derived artifacts through note_path.write_text, once, p.write_text,
        target.parent.mkdir, target.write_bytes.
    """
    dataset.guard()
    dataset.checked("prediction_receipt.json")
    dataset.checked("label_receipt.json")
    ranks = load(ROOT / "predictions.json")
    qs = load(ROOT / "queries.public.json")
    docs = {d["id"]: d for d in load(ROOT / "catalog.public.json")}
    profiles = {d["id"]: d for d in read_rows(ROOT / "profiles.public.jsonl")}
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    byq = {
        q["id"]: {
            r["candidate_id"]: r["grade"] for r in labels if r["query_id"] == q["id"]
        }
        for q in qs
    }
    capacity = []
    for q in qs:
        values = list(byq[q["id"]].values())
        positive = sum(v == 1 for v in values)
        unknown = sum(v is None for v in values)
        capacity.append(
            dict(
                query_id=q["id"],
                mode="image" if q["image_path"] else "text",
                known_relevant=positive,
                unknown=unknown,
                known_label_oracle_precision_at_10=min(10, positive) / 10,
                optimistic_oracle_precision_at_10=min(10, positive + unknown) / 10,
                zero_confirmed_and_zero_possible_positive=positive + unknown == 0,
                oracle_is_not_measured_retrieval_quality=True,
            )
        )
    summary = {}
    for scope in ("all", "text", "image"):
        rows = [r for r in capacity if scope == "all" or r["mode"] == scope]
        summary[scope] = dict(
            queries=len(rows),
            zero_possible_relevant_queries=[
                r["query_id"]
                for r in rows
                if r["zero_confirmed_and_zero_possible_positive"]
            ],
            optimistic_oracle_macro_precision_at_10=sum(
                r["optimistic_oracle_precision_at_10"] for r in rows
            )
            / len(rows),
            cannot_reach_precision_60_even_under_optimistic_labels=sum(
                r["optimistic_oracle_precision_at_10"] for r in rows
            )
            / len(rows)
            < 0.6,
        )
    once(
        ROOT / "internal_capacity.ai.blind.json",
        dict(
            query_capacity=capacity,
            scopes=summary,
            interpretation="Post-prediction diagnostic only; not model performance. Insufficient or mismatched catalog coverage cannot be fixed by ranking, cannot justify swapping queries/products or relabeling this consumed test.",
            human_labels=0,
            formal_metrics=None,
        ),
    )
    notes = [
        "# 独立AI盲测的数据覆盖与解释",
        "",
        "本次独立的是新图库、新查询及预测前冻结的方法和AI关系；不是独立人工评审，也不是保证存在充足相关商品的流量代表性样本。12个采集桶来自元数据词面，不应表述为12种真实主体均正确、均衡。",
        "",
        "逐图评审发现印油、中式盘扣服装、圆柱形护颈枕等真实查询和部分清洁剂、拆机工具等候选；全部按冻结协议保留，不改为原采集词，不按标签、覆盖或分数替换样本。相关性仍按PDF业务类型、用途与关键可见形态判断，不是同SKU。",
        "",
        "以下为全图库理想排序且未知项均作最有利补全时的Precision容量界，不是实测检索成绩，也不是置信区间或AI误标界。",
    ]
    for scope, row in summary.items():
        notes.append(
            f"- {scope}: {row['queries']}条查询，乐观容量上界{row['optimistic_oracle_macro_precision_at_10']:.6f}；仍低于0.60：{row['cannot_reach_precision_60_even_under_optimistic_labels']}；无任何可能正例的查询：{row['zero_possible_relevant_queries']}。"
        )
    notes += [
        "",
        "容量不足不能靠重排补齐；实测与容量之间的差距仍是检索方法的改进空间。两者应分别诊断，不能将全部低分归因于数据，也不能通过改标签、分母、K或删查询追分。后续改进仅回开发集；这份已消耗盲测不能用于调参后的再次验收。",
        "",
        "完整向量与配对均使用冻结完整模型，参数顺序恢复只修正调用接口，不修改数学方法。标签文件仅作完整性哈希校验，标签值未进入特征、候选选择或模型请求。默认方法保留原版，formal_metrics=null。",
    ]
    note_path = ROOT / "internal_dataset_coverage_notes.ai.blind.md"
    note_text = "\n".join(notes) + "\n"
    if note_path.exists():
        assert note_path.read_text(encoding="utf-8") == note_text
    else:
        note_path.write_text(note_text, encoding="utf-8")
    embedding = load(ROOT / "embedding_runtime.json")
    reranker = load(ROOT / "reranker_runtime.json")
    pairs = [load(p) for p in sorted((ROOT / "pairs").glob("*.json"))]
    native = []
    for p in sorted((ROOT / "query_preparation").rglob("*.json")):
        value = load(p)
        if (
            isinstance(value, dict)
            and isinstance(value.get("raw"), dict)
            and value["raw"].get("model")
        ):
            native.append(
                dict(
                    file=rel(p),
                    seconds=value.get("seconds"),
                    prompt_tokens=value["raw"].get("prompt_eval_count"),
                    output_tokens=value["raw"].get("eval_count"),
                )
            )
    once(
        ROOT / "resource_audit.json",
        dict(
            actual_native_query_requests=len(native),
            native_query_requests=native,
            actual_Qwen_embedding_forward_calls=len(embedding.get("calls", [])),
            actual_Qwen_reranker_unique_pairs=len(pairs),
            context_positions=load(ROOT / "plan_receipt.json")["context_positions"],
            sum_unique_pair_model_seconds=sum(p["timing"]["seconds"] for p in pairs),
            embedding_peak_allocated_mib=embedding.get("peak_allocated_mib"),
            embedding_peak_reserved_mib=embedding.get("peak_reserved_mib"),
            reranker_peak_allocated_mib=reranker.get("peak_allocated_mib"),
            reranker_peak_reserved_mib=reranker.get("peak_reserved_mib"),
            Ollama_label_annotation_requests=0,
            label_origin="Codex direct native vision, 2304 AI relations, human 0",
            latency_scope="Actual saved forward-call timings, not end-to-end service latency or online benchmark",
            model_budget_reduced=False,
            user_model_unloaded=False,
            formal_metrics=None,
        ),
    )
    sources = {
        p
        for p in load(ROOT / "release_protocol.json")["files"]
        if p.startswith(("scripts/", "src/", "vendor/", "configs/"))
    }
    sources.update(
        [
            "scripts/grounded_blind_source_recovery.py",
            "scripts/grounded_blind_execution_recovery.py",
            "scripts/author_grounded_blind_judgments.py",
            "scripts/grounded_blind_diagnostics.py",
            "scripts/verify_grounded_blind_release.py",
            "tests/test_grounded_blind_execution_recovery.py",
        ]
    )
    manifest = {}
    for name in sorted(sources):
        source = PROJECT_ROOT / name
        target = ROOT / "reproducibility_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
        manifest[name] = sha256_file(source)
    once(
        ROOT / "additional_reproducibility_sources.json",
        dict(
            files=manifest,
            source_snapshots=len(manifest),
            root=rel(ROOT / "reproducibility_sources"),
            formal_metrics=None,
        ),
    )
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模块二独立AI盲测回放</title><style>body{font:15px/1.6 system-ui,"Microsoft Yahei",sans-serif;background:#f6f8fc;color:#17233c;margin:24px}h1{font-size:25px}.note{background:#fff1d7;padding:18px;border-radius:10px}details{background:white;padding:16px;margin:16px 0;border-radius:12px}summary{cursor:pointer;font-weight:650}.query img{width:220px;height:220px;object-fit:contain;background:white;margin:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px}.card{border:1px solid #d9e0ec;border-radius:9px;padding:10px}.card img{width:100%;height:170px;object-fit:contain}.title{font-size:13px;overflow-wrap:anywhere}small{color:#58677e}.grade{font-weight:bold}a{color:#235bac}</style><h1>模块二 · 独立AI盲测回放</h1><p class="note">96件全新图库、24查询。方法及标签在预测前冻结；AI评审，人工0。已消耗的独立数据，不作开发调参或新验收复用。formal_metrics=null。这里展示真实排序与冻结AI标签；标签可能有误，位次及模型分数不等于业务概率。</p><p><a href="internal_comparison.ai.blind.md">完整内部指标报告</a> · <a href="decision.json">判定</a> · <a href="internal_capacity.ai.blind.json">图库容量诊断（不是模型成绩）</a></p>'
    ]
    cards = 0
    for qi, q in enumerate(qs):
        label = q["text"] or "图片查询"
        parts.append(
            "<details"
            + (" open" if qi == 0 else "")
            + "><summary>"
            + html.escape(q["id"] + " · " + label)
            + "</summary>"
        )
        if q["image_path"]:
            selected = load(
                ROOT / "query_preparation" / q["id"] / "grounding/result.json"
            )
            parts.append(
                '<div class="query">原图'
                + image_tag(q["image_path"])
                + "实际选定输入"
                + image_tag(selected["selected_image"])
                + "</div>"
            )
        for method in runtime.METHODS:
            parts.append(
                "<details><summary>"
                + html.escape(method)
                + '</summary><div class="grid">'
            )
            for i, cid in enumerate(ranks[q["id"]][method][:10], 1):
                d = docs[cid]
                grade = byq[q["id"]][cid]
                text = {1: "AI相关", 0: "AI不相关", None: "AI未定"}[grade]
                parts.append(
                    '<article class="card"><b>Top '
                    + str(i)
                    + "</b>"
                    + image_tag(d["image_path"])
                    + '<p class="title">'
                    + html.escape(d["title"])
                    + "</p><small>AI主体："
                    + html.escape(profiles[cid]["product_type"])
                    + '</small><p class="grade">'
                    + text
                    + "</p></article>"
                )
                cards += 1
            parts.append("</div></details>")
        parts.append("</details>")
    parts.append("</html>")
    text = "\n".join(parts) + "\n"
    p = ROOT / "internal_comparison.html"
    if p.exists():
        assert p.read_text(encoding="utf-8") == text
    else:
        p.write_text(text, encoding="utf-8")
    assert cards == 960 and "\ufffd" not in text
    once(
        ROOT / "visual_replay_check.json",
        dict(
            passed=True,
            queries=24,
            methods=4,
            cards=960,
            local_image_paths_verified=True,
            static_HTML_no_JS=True,
            browser_interaction_not_claimed=True,
            quality_metrics=None,
            formal_metrics=None,
        ),
    )
    print(dict(capacity_diagnostic_saved=True, visual_cards=cards, formal_metrics=None))


if __name__ == "__main__":
    write()
