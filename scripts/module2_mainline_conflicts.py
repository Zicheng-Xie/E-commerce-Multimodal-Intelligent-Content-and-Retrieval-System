"""Replay public conflict controls after complete human Top10 review, preserving the mainline."""

from __future__ import annotations

import argparse
import hashlib

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_human_review_inputs import import_precision_answers
from module2_mainline import CONFIG, REVISION, _inputs
from module2_mainline_reports import label_fingerprints
from module2_shared_evaluation import precision_delta
from module2_shared_gallery_precision import worker_lock
from module2_shared_gallery_retrieval import _checked_media, _score
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from multimodal_ecommerce.public_conflict_guard import reorder_public_conflicts
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

OUTPUT = REVISION.parent / "public_conflicts_20261110"
SOURCES = (
    "scripts/module2_mainline_conflicts.py",
    "src/multimodal_ecommerce/public_conflict_guard.py",
)


def _capacity(rows: dict, scope: str, depth: int) -> float | None:
    current = [r for r in rows.values() if scope in ("all", r["mode"])]
    mandatory = [
        (min(depth, r["human_positive"]) / r["human_positive"],) * 2
        for r in current
        if r["human_positive"]
    ]
    optional = [
        (1.0, 1.0) for r in current if not r["human_positive"] and r["human_unresolved"]
    ]
    return optional_mean_bounds(mandatory, optional)[1]


def _human_queries(queries: dict) -> dict:
    return {
        key: {
            **q,
            "judgments": {c: r for c, r in q["judgments"].items() if r.get("human")},
        }
        for key, q in queries.items()
    }


def _swap_evidence(before: dict, after: dict, queries: dict) -> dict:
    rows = []
    for key, old in before.items():
        new = after[key]
        for i in range(10):
            if old[i] == new[i]:
                continue
            outgoing, incoming = old[i], new[i]
            j = old.index(incoming)
            labels = queries[key]["judgments"]
            label = labels.get(outgoing, {})
            rows.append(
                dict(
                    query_id=key,
                    sources=queries[key]["sources"],
                    outgoing=outgoing,
                    incoming=incoming,
                    outgoing_position=i + 1,
                    incoming_original_position=j + 1,
                    pure_position_swap=j >= 10 and new[j] == outgoing,
                    outgoing_human_negative=label.get("human") is True
                    and label.get("grade") == 0,
                    incoming_label=labels.get(incoming, {}),
                )
            )
    valid = bool(rows) and all(
        r["pure_position_swap"] and r["outgoing_human_negative"] for r in rows
    )
    changed = {
        (r["query_id"], p - 1)
        for r in rows
        for p in (r["outgoing_position"], r["incoming_original_position"])
    }
    valid = valid and all(
        old[i] == after[key][i] or (key, i) in changed
        for key, old in before.items()
        for i in range(len(old))
    )
    return dict(
        rows=rows,
        all_six_metrics_nondecreasing_for_all_human_label_completions=valid,
        basis="Each pure swap moves an explicitly human-negative item down and a positive or unresolved item up; all other positions remain fixed. This is a development-data proof, not an independent validation.",
    )


def _review(evidence: Evidence, manifest: dict, before: dict, after: dict) -> list:
    catalog = {r["id"]: r for r in manifest["catalog"]}
    tasks = []
    for key, query in manifest["query_manifest"].items():
        for cid in sorted(set(after[key][:10]) - set(before[key][:10])):
            label = query["judgments"].get(cid, {})
            if label.get("human") and label.get("grade") in (0, 1):
                continue
            doc = catalog[cid]
            tasks.append(
                dict(
                    task_id=hashlib.sha256(
                        f"{key}\n{cid}\npublic-conflict-matched-20261110".encode()
                    ).hexdigest()[:24],
                    query_id=key,
                    candidate_id=cid,
                    query=dict(
                        text=query.get("text"),
                        image_path=(
                            str(_checked_media(evidence, query))
                            if query["mode"] == "image"
                            else None
                        ),
                    ),
                    candidate=dict(
                        title=doc["title"],
                        image_path=str(_checked_media(evidence, doc)),
                    ),
                )
            )
    return sorted(tasks, key=lambda r: r["task_id"])


def _percent(value: float | None) -> str:
    return "未确定" if value is None else f"{100 * value:.2f}%"


def _report(result: dict) -> str:
    lines = [
        "# 完整人工精度与下一轮主体冲突对照",
        "",
        "2026-11-10。固定 845 件商品、55 条查询、原相关性定义和 K=10。341 对新答案与既有记录合并后，原主线的 550 个前十位置均有明确人工判断。",
        "",
        "## 完整人工开发成绩与容量",
        "",
        "| 范围 | 原主线人工 P@10 | 理想 R@10 乐观上限 | 达成 90% 的最小乐观深度，仅诊断 |",
        "|---|---:|---:|---:|",
    ]
    for scope, name in (("all", "总体"), ("image", "图搜"), ("text", "文搜")):
        h = result["human_metrics"]["baseline"]["scopes"][scope]
        lines.append(
            f"| {name} | {_percent(h['precision_at_10']['value'])} | {_percent(h['recall_capacity']['optimistic_ideal_macro_upper'])} | {result['capacity_diagnostics'][scope]['minimum_optimistic_depth_for_90']} |"
        )
    lines += [
        "",
        "人工 P@10 已超过 80%，但总体、图搜和文搜的 90% R@10 目标均与当前人工正例数量及固定 Top10 不兼容。真实 R@10 点值仍未确定；以上容量上限也不是当前召回成绩。深候选覆盖不能替代 R@10，目标、查询和固定分母不改变。",
        "",
        "## 同一标签快照下的离线优化",
        "",
        "只使用查询文字、已有查询图描述和候选原标题。先对照宽泛类型过滤，再验证明确冲突替换；标签、审核理由、查询来源与商品 ID 不参与排序规则。新候选只从原 Top20 选择，整个图库保留。已曝光开发标签用于分析与比较，结论不代表独立验收。",
        "",
        "| 版本 | 总体人工 P@10 边界 | 相对原版的配对 P@10 差值，百分点 |",
        "|---|---:|---:|",
    ]
    for name, human in result["human_metrics"].items():
        bounds = human["scopes"]["all"]["precision_at_10"]["bounds"]
        delta = result["paired_human_precision"][name]["all"]
        lines.append(
            f"| {name} | {_percent(bounds[0])}–{_percent(bounds[1])} | {100 * delta[0]:+.2f} 至 {100 * delta[1]:+.2f} |"
        )
    lines += [
        "",
        f"保守候选 `explicit_matched` 共发生 {len(result['candidate_swap_evidence']['rows'])} 次位置互换。逐项核验的六指标不退化证明：{result['candidate_swap_evidence']['all_six_metrics_nondecreasing_for_all_human_label_completions']}。证明依赖本开发集已确认的换出项为不相关，而不是以此标签驱动线上排序。其余位置完全不动。具体证据和全部指标边界见 [results.json](results.json)。",
        "",
        (
            f"仍有 {result['pending_review_pairs']} 个换入候选缺少明确人工标签；候选点值保持未确定。"
            if result["pending_review_pairs"]
            else f"换入候选均已明确作答，总体人工 P@10 点值为 {_percent(result['human_metrics']['explicit_matched']['scopes']['all']['precision_at_10']['value'])}。"
        )
        + "原主线及其完整人工成绩继续保留，新候选尚未替换默认配置。只准备这些实际换入候选的 [人工复核页](human_review/index.html)，原 341 对无需重做，也未追加整套 372 对 Top20 标注。",
        "",
        "钢琴查询十个结果均被人工判为不相关；当前图库原标题没有“钢琴／电子琴／电钢琴”商品词命中。这是目录覆盖线索，不能据标题扫描断言所有原图均不存在相关商品。保留该查询和固定分母；缺货识别或补库应另立实验，不能用删查询制造达标。",
        "",
        "输入与源码哈希、原版与候选的完整排序另存于本目录。主线配置、已封存验收数据和历史实验保持不变。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Evaluate public-input ablations and prepare only the new unresolved swaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("evaluate", "score-review"), nargs="?", default="evaluate"
    )
    phase = parser.parse_args().phase
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        manifest, _, original, tasks = _inputs(evidence, load(CONFIG))
        queries = manifest["query_manifest"]
        progress = import_precision_answers(
            evidence, REVISION / "human_review", tasks, queries
        )
        if not progress["complete"] or progress["binary_completed"] != len(tasks):
            raise ValueError(
                "Finish the original 341 binary decisions before this comparison"
            )
        catalog = {row["id"]: row["title"] for row in manifest["catalog"]}
        captions = evidence.json(
            REVISION.parent / "subject_recall/query_captions.json"
        )["rows"]
        target = load(PROJECT_ROOT / "configs/module2_target_p80_r90.json")
        variants = {"baseline": original}
        observations: dict = {}
        policies: dict[str, dict] = {
            "broad_control": dict(protect=0, broad=True, matched_only=False),
            "explicit_unrestricted_control": dict(protect=0, matched_only=False),
            "explicit_matched_keep2": dict(protect=2),
            "explicit_matched": dict(protect=0),
        }
        for name, options in policies.items():
            variants[name], observations[name] = {}, {}
            for key, query in queries.items():
                order, obs = reorder_public_conflicts(
                    original[key],
                    {cid: catalog[cid] for cid in original[key]},
                    text=query.get("text"),
                    caption=captions[key]["value"] if key in captions else None,
                    options=options,
                )
                if set(order) != set(original[key]) or len(order) != len(original[key]):
                    raise ValueError("All original eligible products must be retained")
                variants[name][key], observations[name][key] = order, obs
        pending = _review(evidence, manifest, original, variants["explicit_matched"])
        review_folder = OUTPUT / "human_review"
        if (review_folder / "manifest.json").exists():
            bound = evidence.json(review_folder / "manifest.json")
            if (
                bound["original_rankings_sha256"]
                != sha256_file(REVISION / "rankings.json")
                or bound["candidate_rankings_sha256"]
                != sha256_file(OUTPUT / "rankings.json")
                or load(OUTPUT / "rankings.json") != variants
                or evidence.json(
                    review_folder / "tasks.private.json", bound["tasks_sha256"]
                )
                != pending
            ):
                raise ValueError(
                    "Preserve the candidate ranking and its original review identities"
                )
        review_progress = import_precision_answers(
            evidence, review_folder, pending, queries
        )
        human_queries = _human_queries(queries)
        human = {
            name: evaluate_human_retrieval(ranks, queries, target)
            for name, ranks in variants.items()
        }
        if human["baseline"]["scopes"]["all"]["precision_at_10"]["judged"] != 550:
            raise ValueError(
                "The baseline must retain all 550 explicitly reviewed slots"
            )
        result = dict(
            role="exposed_development_public_conflict_controls",
            products=len(catalog),
            queries=len(queries),
            mainline_config_sha256=sha256_file(CONFIG),
            original_rankings_sha256=sha256_file(REVISION / "rankings.json"),
            target_contract_sha256=sha256_file(
                PROJECT_ROOT / "configs/module2_target_p80_r90.json"
            ),
            label_fingerprints=label_fingerprints(original, queries),
            human_metrics=human,
            paired_human_precision={
                name: precision_delta(original, ranks, human_queries)
                for name, ranks in variants.items()
            },
            human_only_six_metric_bounds={
                name: _score(ranks, human_queries) for name, ranks in variants.items()
            },
            mixed_label_diagnostics={
                name: _score(ranks, queries) for name, ranks in variants.items()
            },
            candidate_swap_evidence=_swap_evidence(
                original, variants["explicit_matched"], queries
            ),
            capacity_diagnostics={
                scope: dict(
                    ideal_upper_by_depth={
                        str(k): _capacity(human["baseline"]["query_metrics"], scope, k)
                        for k in (10, 20, 50, 100)
                    },
                    minimum_optimistic_depth_for_90=next(
                        (
                            k
                            for k in range(10, 846)
                            if (
                                _capacity(human["baseline"]["query_metrics"], scope, k)
                                or 0
                            )
                            >= 0.9
                        ),
                        None,
                    ),
                    diagnostic_only=True,
                )
                for scope in ("all", "image", "text")
            },
            piano_title_coverage=[
                cid
                for cid, title in catalog.items()
                if any(word in title for word in ("钢琴", "电子琴", "电钢琴"))
            ],
            review_packet_pairs=len(pending),
            pending_review_pairs=len(pending) - review_progress["binary_completed"],
            review_progress=review_progress,
            policies=policies,
            inputs=evidence.files,
            source_sha256={p: sha256_file(PROJECT_ROOT / p) for p in SOURCES},
            new_model_scores=0,
            default_changed=False,
            formal_metrics=None,
            formal_acceptance=False,
        )
        write_json(OUTPUT / "rankings.json", variants)
        write_json(OUTPUT / "public_observations.json", observations)
        write_json(OUTPUT / "human_review/tasks.private.json", pending)
        render_review(
            pending,
            OUTPUT / "human_review",
            title="模块二 · 明确主体冲突替换复核",
            instructions="这些是保守候选实际换入前十且尚无人工判断的商品。仍按原图、原标题、主体、用途与明确属性选择相关、不相关或证据不足，写理由并导出 CSV 到本目录 answers。已审核过的换出项不重复提问。",
        )
        if not (review_folder / "manifest.json").exists():
            write_json(
                review_folder / "manifest.json",
                dict(
                    role="new_top10_incoming_pairs_only",
                    original_rankings_sha256=result["original_rankings_sha256"],
                    candidate_rankings_sha256=sha256_file(OUTPUT / "rankings.json"),
                    tasks_sha256=sha256_file(
                        OUTPUT / "human_review/tasks.private.json"
                    ),
                    expected=len(pending),
                    inputs=evidence.files,
                ),
            )
        result["rankings_sha256"] = sha256_file(OUTPUT / "rankings.json")
        write_json(OUTPUT / "results.json", result)
        (OUTPUT / "report.md").write_text(_report(result), encoding="utf-8")
        print(
            f"{phase}: complete 55-query comparison; {result['pending_review_pairs']} pending human decisions; mainline unchanged."
        )
        print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
