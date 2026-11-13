"""Immutable, source-bound human evaluation reports for the fixed development mainline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import load
from multimodal_ecommerce.safety import (
    ensure_output_path,
    sha256_file,
    write_json,
)

FORMAT = "human_retrieval_evaluation_v1"
SOURCES = (
    "scripts/module2_mainline.py",
    "scripts/module2_mainline_reports.py",
    "scripts/module2_human_review_inputs.py",
    "src/multimodal_ecommerce/human_retrieval_evaluation.py",
    "src/multimodal_ecommerce/ai_review_metrics.py",
)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def label_fingerprints(rankings: dict, queries: dict) -> dict:
    """Fingerprint effective observations independently of their prediction order.

    Args:
        rankings: Full eligible product set per query.
        queries: Effective human/AI relation observations.

    Returns:
        Separate fingerprints for human evidence and the effective mixed labels.
    """
    mixed, human = {}, {}
    for key, order in sorted(rankings.items()):
        labels = queries[key]["judgments"]
        mixed[key] = [
            [
                cid,
                labels.get(cid, {}).get("grade"),
                labels.get(cid, {}).get("human", False),
            ]
            for cid in sorted(order)
        ]
        human[key] = [
            [cid, labels[cid].get("grade")]
            for cid in sorted(order)
            if labels.get(cid, {}).get("human")
        ]
    return dict(
        effective_labels_sha256=_digest(mixed), human_labels_sha256=_digest(human)
    )


def evaluation_transition(previous: dict, current: dict) -> dict:
    """Describe annotation and prediction changes without crediting labels as model gains.

    Args:
        previous: Previous current-status document, if available.
        current: Newly computed status with explicit label fingerprints.

    Returns:
        Change flags and the interpretation of cross-snapshot metric differences.
    """
    old = previous.get("label_fingerprints")
    if old is None:
        return dict(
            kind="initial_separated_human_evaluation", metric_delta_is_model_gain=False
        )
    labels = old != current["label_fingerprints"]
    predictions = (
        previous["frozen_rankings_sha256"] != current["frozen_rankings_sha256"]
    )
    return dict(
        kind=(
            "both_changed_require_same_label_replay"
            if labels and predictions
            else (
                "same_predictions_new_labels"
                if labels
                else (
                    "new_predictions_same_labels"
                    if predictions
                    else "same_predictions_and_labels"
                )
            )
        ),
        labels_changed=labels,
        predictions_changed=predictions,
        metric_delta_is_model_gain=False,
        model_gain_attribution_eligible=bool(predictions and not labels),
        previous_snapshot_id=previous.get("evaluation_snapshot", {}).get("id"),
        ranking_comparison_basis="Only paired predictions evaluated against the same current label snapshot can establish ranking gains",
    )


def _percent(value) -> str:
    return "未确定" if value is None else f"{value * 100:.2f}%"


def _bounds(values: list) -> str:
    return (
        "未定义"
        if values[0] is None
        else f"{_percent(values[0])}–{_percent(values[1])}"
    )


def _error_types(errors: list) -> list[dict]:
    definitions = {
        "subject_or_accessory": (
            "主体／配件混淆",
            "核对主售商品与用途，区分配件和替代品",
        ),
        "attribute_conflict": (
            "明确属性冲突",
            "保留人工指出的属性证据，再验证冲突处理",
        ),
        "unclassified": ("待归因", "保留原始理由，补充归因后再选择策略"),
    }
    groups = []
    for kind, (label, action) in definitions.items():
        selected = [
            row
            for row in errors
            if any(kind in item["error_types"] for item in row["errors"])
        ]
        groups.append(
            dict(
                error_type=kind,
                label=label,
                confirmed_errors=sum(
                    kind in item["error_types"]
                    for row in selected
                    for item in row["errors"]
                ),
                query_ids=[row["query_id"] for row in selected],
                next_step=action,
            )
        )
    return sorted(groups, key=lambda row: (-row["confirmed_errors"], row["error_type"]))


def _markdown(status: dict, errors: list, identity: str) -> str:
    human = status["human_evaluation"]
    scopes = human["scopes"]
    exact = scopes["all"]["precision_at_10"]["value"] is not None
    lines = [
        "# 模块二人工精度报告与召回容量审计",
        "",
        f"主线：`{status['mainline_id']}`。商品 {status['products']} 件、查询 {status['queries']} 条，固定 K=10。",
        f"快照：`{identity}`。预测 SHA256：`{status['frozen_rankings_sha256']}`。",
        "",
        (
            "本次总体人工 P@10 已完整计算。"
            if exact
            else "当前人工前十标签尚未补齐，人工 P@10 点值保持未确定；区间表示未决标签补全边界，不是置信区间。"
        ),
        "所有结果均为开发评测，正式独立验收未通过。本报告未运行模型或修改排序。",
        "",
        "## 一、仅使用明确人工标签",
        "",
        "| 范围 | 明确人工标签 / 前十位置数 | 人工 P@10 点值 | 人工 P@10 边界 | 人工 R@10 点值 | 人工 R@10 边界 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, label in (("all", "总体"), ("image", "图搜"), ("text", "文搜")):
        p, r = scopes[mode]["precision_at_10"], scopes[mode]["recall_at_10"]
        lines.append(
            f"| {label} | {p['judged']} / {p['expected']} | {_percent(p['value'])} | {_bounds(p['bounds'])} | {_percent(r['value'])} | {_bounds(r['bounds'])} |"
        )
    coverage = scopes["all"]["coverage"]
    lines += [
        "",
        f"完整图库共 {coverage['eligible_pairs']:,} 个查询—商品关系，已有 {coverage['human_positive'] + coverage['human_negative']:,} 个明确人工判断，{coverage['human_unresolved']:,} 个在人工视图中仍未确定（其中 {coverage['human_uncertain']} 个为人工选择证据不足）。机器标签不填入人工视图。",
        "",
        "R@10 的分母为整个可检索图库中的相关商品数。仅补齐前十不能确定分母；任一查询相关集未完整时，对应范围的真实 R@10 点值保持未确定。零相关查询的召回率未定义，报告保留其数量；它们仍计入 P@10 的固定分母。",
        "",
        "## 二、固定 Top10 的容量证据",
        "",
        "| 范围 | 保留现有人标时，理想排序的乐观 R@10 上限 | 目标 | 容量判断 |",
        "|---|---:|---:|---|",
    ]
    for mode, label in (("all", "总体"), ("image", "图搜"), ("text", "文搜")):
        c = scopes[mode]["recall_capacity"]
        verdict = (
            "与当前人工标签及固定 Top10 不兼容"
            if c["incompatible"]
            else "容量尚未排除目标，仍需完整真值与模型验证"
        )
        lines.append(
            f"| {label} | {_percent(c['optimistic_ideal_macro_upper'])} | {_percent(c['required'])} | {verdict} |"
        )
    lines += [
        "",
        "若一条查询已人工确认 16 件相关商品，前十最多找回十件，理想 R@10 上限即 10/16=62.5%。不通过删除查询、缩小相关集或改变 K 来制造达标。详见 [逐查询容量证据](capacity.json)。",
        "",
        "## 三、历史混合标签诊断，单独列示",
        "",
        "以下把已有 AI 标签与人工覆盖共同视为固定观察，边界仅处理其余未决关系，未计入 AI 错标风险，不能作为完整人工成绩。",
        "",
        "| 范围 | 混合标签 P@10 区间 | 混合标签 R@10 区间 |",
        "|---|---:|---:|",
    ]
    for mode, label in (("all", "总体"), ("image", "图搜"), ("text", "文搜")):
        mixed = status["candidate"]["scopes"][mode]["ranges"]
        lines.append(
            f"| {label} | {_bounds(mixed['precision@10'])} | {_bounds(mixed['recall@10'])} |"
        )
    lines += [
        "",
        "## 四、较深候选的已确认正例覆盖诊断",
        "",
        "仅统计人工确认的正例及包含这些正例的查询；标注选择会影响该诊断，它不等于完整相关集的 Recall@K，不能替代 R@10 验收。",
        "",
        "| 范围 | Top20 | Top50 | Top100 |",
        "|---|---:|---:|---:|",
    ]
    for mode, label in (("all", "总体"), ("image", "图搜"), ("text", "文搜")):
        values = scopes[mode]["documented_positive_coverage"]
        lines.append(
            f"| {label} | {_percent(values['20'])} | {_percent(values['50'])} | {_percent(values['100'])} |"
        )
    lines += [
        "",
        "## 五、人工已确认错误的优先清单",
        "",
        "以下按前十人工确认不相关数量排序；未判项不当作正确或错误。分类只整理人工理由，无法归因时保留待归因。完整 55 条见 [错误与候选位置清单](errors.json)。",
        "",
        "| 错误类型 | 已确认错误数 | 下一轮处理依据 |",
        "|---|---:|---|",
    ]
    for group in _error_types(errors):
        lines.append(
            f"| {group['label']} | {group['confirmed_errors']} | {group['next_step']} |"
        )
    lines += [
        "",
        "同一错误可以同时涉及主体与属性，分类数量不可相加作为总错误数。有人工正例在第 11–20 名时优先排查精排；已知正例在 Top20 外时排查召回。两者均为诊断线索，不能凭标签直接修改预测。",
        "",
        "| 查询 | 已确认前十错误 | 前十未决 | 已知相关在第11–20名 | 已知相关在Top20外 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in errors[:10]:
        label = "、".join(row["sources"]) or row["query_id"]
        lines.append(
            f"| {label.replace('|', '／')} | {row['confirmed_top10_error_count']} | {row['unresolved_top10']} | {len(row['known_relevant_in_top20_not_top10'])} | {len(row['known_relevant_outside_top20'])} |"
        )
    lines += [
        "",
        "人工补标会改变同一排序的评测数值，不能把这部分变化计作模型提升。模型收益使用同一最新标签下的配对比较单列，见 [指标及来源](metrics.json)。本次保留的排序与所有输入哈希见 [快照清单](manifest.json)。",
        f"本快照首次生成时的变更类型：`{status.get('evaluation_transition', {}).get('kind', 'initial_separated_human_evaluation')}`。该标记仅说明标签或预测是否改变，不代表获得了模型收益。",
        "",
    ]
    return "\n".join(lines)


def publish_evaluation(output: Path, status: dict, errors: list, inputs: dict) -> dict:
    """Write one immutable evaluation snapshot and a current-report pointer.

    Args:
        output: Development mainline report directory.
        status: Current metrics, label identities and fixed prediction binding.
        errors: All query-level human-confirmed errors and candidate coverage gaps.
        inputs: Verified source/CSV checksums from the guarded reader.

    Returns:
        Snapshot paths and fingerprints suitable for the current status document.
    """
    code = {
        name: sha256_file(Path(__file__).resolve().parents[1] / name)
        for name in SOURCES
    }
    identity = dict(
        format=FORMAT,
        rankings_sha256=status["frozen_rankings_sha256"],
        label_fingerprints=status["label_fingerprints"],
        target_contract_sha256=status["target_contract_sha256"],
        config_sha256=status.get("config_sha256"),
        input_sha256=dict(sorted(inputs.items())),
        code_sha256=code,
    )
    key = _digest(identity)[:20]
    folder = ensure_output_path(output / "evaluation_snapshots" / key)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = load(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Immutable evaluation snapshot identity differs")
        for name, digest in manifest["files"].items():
            if sha256_file(folder / name) != digest:
                raise ValueError("Immutable evaluation snapshot was modified")
    else:
        human = status["human_evaluation"]
        write_json(
            folder / "metrics.json",
            dict(
                mainline_id=status["mainline_id"],
                human_only=human,
                mixed_label_diagnostics={
                    scope: row["ranges"]
                    for scope, row in status["candidate"]["scopes"].items()
                },
                same_label_ranking_comparison=status.get("confirmed_image_gain"),
                transition_from_previous_snapshot=status.get("evaluation_transition"),
                formal_acceptance=False,
            ),
        )
        write_json(
            folder / "errors.json",
            dict(
                queries=errors,
                error_types=_error_types(errors),
                annotation_basis="binary human errors only; unresolved relations unclassified",
            ),
        )
        write_json(
            folder / "capacity.json",
            dict(
                scopes={
                    scope: row["recall_capacity"]
                    for scope, row in human["scopes"].items()
                },
                queries={
                    key: dict(
                        human_positive=row["human_positive"],
                        unresolved=row["human_unresolved"],
                        ideal_recall_at_10_upper=row["ideal_recall_at_10_upper"],
                        recall_status=row["recall_at_10"]["status"],
                    )
                    for key, row in human["query_metrics"].items()
                },
                target_unchanged=True,
                formal_acceptance=False,
            ),
        )
        (folder / "report.md").write_text(
            _markdown(status, errors, key), encoding="utf-8"
        )
        write_json(
            manifest_path,
            dict(
                identity=identity,
                files={
                    name: sha256_file(folder / name)
                    for name in (
                        "metrics.json",
                        "errors.json",
                        "capacity.json",
                        "report.md",
                    )
                },
            ),
        )
    latest = ensure_output_path(output / "evaluation_latest.md")
    p = status["human_evaluation"]["scopes"]["all"]["precision_at_10"]
    latest.write_text(
        f"# 当前人工评测入口\n\n主线：`{status['mainline_id']}`。前十明确人工标签：{p['judged']}/{p['expected']}；人工 P@10 点值：{_percent(p['value'])}。\n\n[打开完整人工精度与召回容量报告](evaluation_snapshots/{key}/report.md)\n\nAI 混合标签诊断在报告中单独列示，正式独立验收尚未通过。\n",
        encoding="utf-8",
    )
    return dict(
        id=key,
        report_path=str(folder / "report.md"),
        manifest_path=str(manifest_path),
        format=FORMAT,
    )
