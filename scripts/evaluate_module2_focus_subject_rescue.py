"""Measure subject rescue with paired non-regression checks on both cohorts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from evaluate_module2_focus_boundary import SUFFIX, reviewed_swaps, score_group
from module2_focus_subject_rescue import (
    BASELINES,
    FOCUS,
    FRESH,
    FRESH_METHOD,
    PARENT,
    ROOT,
    freeze,
    verified,
)
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import load

METHODS = ("evidence_only", "qwen_confirmed", "qwen_confirmed_strict")
REVIEW = ROOT / "human_review"


def read_results(protocol: dict, registry: dict) -> dict:
    """Validate the frozen scorer cache and derive label-free decisions.

    Args:
        protocol: Frozen per-query proposal set.
        registry: Evaluation data-role rules.

    Returns:
        Proposal diagnostics and eligibility per mechanism.
    """
    outcomes = {}
    digest = sha256_file(ROOT / "protocol.json")
    for job in protocol["proposals"]:
        qid, group = job["qid"], job["cohort"]
        decisions = dict.fromkeys(METHODS, False)
        record = dict(cohort=group, status=job["status"], decisions=decisions)
        if job["status"] == "score_pair":
            path = ROOT / "scores" / group / (qid + ".json")
            verified(path, "", registry)
            cache = load(path)
            logits = cache["logits"]
            if (
                cache["identity"]
                != dict(protocol_sha256=digest, candidates=job["candidates"])
                or len(logits) != 2
                or not all(map(math.isfinite, logits))
                or [item["document_id"] for item in cache["timings"]]
                != job["candidates"]
            ):
                raise ValueError(f"Invalid full-model checkpoint: {qid}")
            passed = logits[1] > logits[0]
            decisions.update(
                evidence_only=True,
                qwen_confirmed=passed,
                qwen_confirmed_strict=passed and job["strong"],
            )
            record.update(
                original_order=job["candidates"],
                original_ranks=job["original_ranks"],
                support=job["entrant_support"],
                logits=logits,
                cache_sha256=sha256_file(path),
            )
        outcomes[qid] = record
    return outcomes


def apply(original: list[str], event: dict) -> list[str]:
    """Exchange one matching incumbent and entrant across the Top10 boundary.

    Args:
        original: Frozen full baseline ordering.
        event: Frozen candidate identities and original positions.

    Returns:
        Complete gallery permutation.
    """
    out, incoming = (rank - 1 for rank in event["original_ranks"])
    if (
        not 5 <= out < 10 <= incoming
        or [original[out], original[incoming]] != event["original_order"]
    ):
        raise ValueError("The proposed swap no longer matches the baseline")
    result = original.copy()
    result[out], result[incoming] = result[incoming], result[out]
    return result


def compare(
    original: dict, names: tuple[str, ...], path: Path, outcomes: dict, method: str
) -> tuple[dict, dict, bool]:
    """Score one swap rule and check every metric on the same label completions.

    Args:
        original: Archived route rankings for one development group.
        names: Original route IDs within that group.
        path: Hash-verified AI labels.
        outcomes: Predicted swaps from both groups.
        method: Current label-free decision rule.

    Returns:
        Full rankings, complete diagnostics, and all-scopes non-regression.
    """
    rankings = {}
    for qid, row in original.items():
        rankings[qid] = {name: row[name] for name in names}
        event = outcomes.get(qid)
        for name in names:
            base = row[name]
            rankings[qid][name + SUFFIX] = (
                apply(base, event) if event and event["decisions"][method] else base
            )
    metrics = score_group(rankings, path, names)
    paired_scopes = metrics["paired"]["scopes"]
    passed = all(
        data["nonregression_proven_for_shared_unknown_labels"]
        for comparisons in paired_scopes.values()
        for paired_baselines in comparisons.values()
        for metric_rows in paired_baselines.values()
        for data in metric_rows.values()
    )
    return rankings, metrics, passed


def review_unknown_swaps(
    protocol: dict, label_paths: tuple[Path, ...], reviewed: dict
) -> dict:
    """Prepare human tasks for the swaps with unresolved AI relevance.

    Args:
        protocol: Frozen public image and candidate inputs.
        label_paths: Original hash-verified development label files.
        reviewed: Previously accepted human grades for applicable swaps.

    Returns:
        Review task count and private task digest when a packet is needed.
    """
    grades = {
        (row["query_id"], row["candidate_id"]): row["grade"]
        for path in label_paths
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }
    tasks = []
    for job in protocol["proposals"]:
        qid = job["qid"]
        docs = job["documents"]
        if not any(grades[qid, doc["id"]] is None for doc in docs):
            continue
        for direction, doc in zip(("outgoing", "incoming"), docs, strict=True):
            if reviewed.get(qid, {}).get(direction) in {"0", "1"}:
                continue
            cid = doc["id"]
            task_id = hashlib.sha256(f"subject-rescue|{qid}|{cid}".encode()).hexdigest()
            tasks.append(
                dict(
                    task_id=task_id[:24],
                    query_id=qid,
                    candidate_id=cid,
                    grade=grades[qid, cid],
                    query=dict(text=None, image_path=job["query_image"]),
                    candidate=dict(title=doc["text"], image_path=doc["image_path"]),
                )
            )
    if not tasks:
        return dict(count=0)
    tasks.sort(key=lambda row: row["task_id"])
    task_path = REVIEW / "tasks.private.json"
    if task_path.exists() and load(task_path) != tasks:
        raise ValueError("The frozen adjudication batch differs")
    render_review(
        tasks,
        REVIEW,
        title="模块二 · 主体替换抽查",
        instructions="只判断这两件商品的主体、用途和外形是否符合业务相关性。请把导出的 CSV 放到本目录 answers 文件夹。",
    )
    write_json(task_path, tasks)
    return dict(count=len(tasks), sample_sha256=sha256_file(task_path))


def main() -> None:
    """Reproduce the source freeze and compare the three fixed development rules."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    path = ROOT / "protocol.json"
    historical = FOCUS / "judgments.ai.jsonl"
    stress = FRESH / "judgments.ai.jsonl"
    receipt = FOCUS / "completion_receipt.json"
    fresh_protocol = FRESH / "inference_protocol.json"
    require_data_access([path, receipt, fresh_protocol], "development", registry)
    protocol = load(path)
    if freeze(registry) != protocol:
        raise ValueError("Subject-rescue freeze changed")
    verified(historical, load(receipt)["files"][historical.name], registry)
    verified(
        stress,
        load(fresh_protocol)["files"][str(stress.relative_to(PROJECT_ROOT))],
        registry,
    )
    outcomes = read_results(protocol, registry)
    outputs, rankings = {}, {}
    old, fresh = load(PARENT / "rankings.json"), load(FRESH / "rankings.json")
    for method in METHODS:
        old_rank, old_score, old_pass = compare(
            old, BASELINES, historical, outcomes, method
        )
        stress_rank, stress_score, stress_pass = compare(
            fresh, (FRESH_METHOD,), stress, outcomes, method
        )
        moves = {
            qid: dict(event, swapped=event["decisions"][method])
            for qid, event in outcomes.items()
        }
        human = reviewed_swaps(moves, registry)
        outputs[method] = dict(
            historical=old_score,
            stress=stress_score,
            nonregression_all_metrics_by_cohort=dict(
                historical=old_pass, stress=stress_pass
            ),
            swaps=dict(
                Counter(event["cohort"] for event in moves.values() if event["swapped"])
            ),
            human_diagnostic=human,
        )
        rankings[method] = dict(historical=old_rank, stress=stress_rank)
    review = review_unknown_swaps(
        protocol,
        (historical, stress),
        outputs["evidence_only"]["human_diagnostic"]["historical_swaps"],
    )
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            role="exposed_development",
            evaluated_image_queries=len(outcomes),
            actual_Qwen_scores=2
            * sum(event["status"] == "score_pair" for event in outcomes.values()),
            outcomes=outcomes,
            comparisons=outputs,
            targeted_review=review,
            protocol_sha256=sha256_file(path),
            evaluation_source_sha256=sha256_file(Path(__file__)),
            frozen_acceptance_model_calls=0,
            formal_metrics=None,
        ),
    )
    print(f"Evaluated {len(outcomes)} exposed image queries.")


if __name__ == "__main__":
    main()
