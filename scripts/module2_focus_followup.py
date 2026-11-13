"""Compare bounded pool expansion for the two retained module-two routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy
from paper_fusion_experiment import load
from querywise_paired_bounds import paired_querywise
from simple_cycle_review import FIELDS

FOCUS = PROJECT_ROOT / "artifacts/module2_focus_coverage"
ROOT = PROJECT_ROOT / "artifacts/module2_focus_followup_20261107"
REVIEW = ROOT / "human_review"
BASELINES = ("historical_text60", "blind_policy_text20")
VARIANTS = ("top30", "corroborated_boundary")


def candidate_order(base: list[str], wide: list[str], raw: list[str]) -> list[str]:
    """Promote at most two entrants supported by both complete Qwen contexts.

    Args:
        base: Original described Top20 plus the unscored gallery tail.
        wide: Described Top30 plus the unscored gallery tail.
        raw: Original-image context with the complete Top60 scored.

    Returns:
        Gallery permutation with only supported swaps at ranks nine and ten.
    """
    positions = {cid: i for i, cid in enumerate(raw)}
    order = base.copy()
    entrants = [cid for cid in wide[:10] if cid not in base[:10]]
    for slot in (9, 8):
        for cid in entrants:
            if (
                cid not in order[:10]
                and positions[cid] < 20
                and positions[cid] < positions[order[slot]]
            ):
                old = order.index(cid)
                order[slot], order[old] = order[old], order[slot]
                break
    return order


def replay(
    source: dict, image_features: dict, archived: dict, score_bank: dict
) -> dict:
    """Build both fixed route baselines and two image-only ablations.

    Args:
        source: Public historical development query and catalog identities.
        image_features: Frozen described-image pool and recall scores.
        archived: Earlier full rankings on the same development gallery.
        score_bank: Verified complete-model described-image score records.

    Returns:
        Full rankings keyed by query and route/method.
    """
    output = {}
    for query in source["queries"]:
        qid = query["id"]
        row = archived[qid]
        if query["text"]:
            output[qid] = {
                method + suffix: row[text]
                for method, text in zip(
                    BASELINES,
                    ("cross_evidence_pool60", "cross_evidence_pool20"),
                    strict=True,
                )
                for suffix in ("", *[f"_{variant}" for variant in VARIANTS])
            }
            continue
        features = image_features[qid]
        ids, pool = features["ids"], features["pool"]
        if len(pool) != 60 or set(score_bank[qid]) != set(pool):
            raise ValueError("Expected all 60 frozen described-image scores")
        base, wide = [
            rerank_complete(
                ids,
                np.array(features["fused_scores"]),
                pool[:n],
                np.array([score_bank[qid][cid] for cid in pool[:n]]),
            )
            for n in (20, 30)
        ]
        if base != row["described_cross_pool20"]:
            raise ValueError("Historical described Top20 did not replay exactly")
        guarded = candidate_order(base, wide, row["cross_evidence_pool60"])
        output[qid] = {
            method + suffix: order
            for method in BASELINES
            for suffix, order in zip(
                ("", *[f"_{variant}" for variant in VARIANTS]),
                (base, wide, guarded),
                strict=True,
            )
        }
    return output


def review_packet(
    source: dict, rankings: dict, labels: list[dict], registry: dict
) -> int:
    """Publish only original query/candidate evidence for boundary disagreements.

    Args:
        source: Public development query and catalog records.
        rankings: All frozen predictions, constructed before reading labels.
        labels: Prior AI judgments retained privately to check disagreement.
        registry: Access rules for the original image inputs.

    Returns:
        Count of blinded development pairs, including at least two per image query.
    """
    docs = {row["id"]: row for row in source["catalog"]}
    grades = {(row["query_id"], row["candidate_id"]): row["grade"] for row in labels}
    tasks = []
    for query in source["queries"]:
        if query["text"]:
            continue
        qid = query["id"]
        order = rankings[qid]
        base = order[BASELINES[0]]
        shifted = set()
        for variant in VARIANTS:
            shifted.update(
                set(order[f"{BASELINES[0]}_{variant}"][:10]) ^ set(base[:10])
            )
        selected = shifted | set(base[9:11])
        for cid in selected:
            task_id = hashlib.sha256(
                f"focus-followup-20261107|{qid}|{cid}".encode()
            ).hexdigest()[:24]
            tasks.append(
                dict(
                    task_id=task_id,
                    query_id=qid,
                    candidate_id=cid,
                    grade=grades[qid, cid],
                    query=query,
                    candidate=docs[cid],
                )
            )
    tasks.sort(key=lambda row: row["task_id"])
    paths = {
        project_path(task[key]["image_path"])
        for task in tasks
        for key in ("query", "candidate")
    }
    require_data_access(sorted(paths), "development", registry)
    if not all(path.is_file() for path in paths):
        raise ValueError("A human review original image is unavailable")
    frozen = REVIEW / "tasks.private.json"
    if frozen.exists() and load(frozen) != tasks:
        raise ValueError("Frozen human review tasks differ")
    render_review(
        tasks,
        REVIEW,
        title="模块二 · 主体精排复核",
        instructions=(
            "按商品主体、用途与可见外形选择相关、不相关或证据不足。"
            + "若候选只与查询商品配套，请按现行业务规则判断，并在理由里注明“配套”。"
            + "请把导出的 CSV 放到本目录 answers 文件夹。"
        ),
    )
    write_json(frozen, tasks)
    return len(tasks)


def evaluate() -> None:
    """Evaluate frozen development inputs and export a score-blinded review batch."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    receipt = FOCUS / "completion_receipt.json"
    require_data_access([receipt], "development", registry)
    hashes = load(receipt)["files"]
    pair_paths = sorted((FOCUS / "described_pairs").glob("*.json"))
    files = [
        FOCUS / "inputs.public.json",
        FOCUS / "query_enrichment/ranking_features.json",
        FOCUS / "rankings.json",
        FOCUS / "judgments.ai.jsonl",
        *pair_paths,
    ]
    require_data_access(files, "development", registry)
    for path in files:
        if sha256_file(path) != hashes[str(path.relative_to(FOCUS))]:
            raise ValueError(f"Historical development input changed: {path}")
    source, features, archived = (load(path) for path in files[:3])
    score_bank: dict[str, dict[str, float]] = {}
    for path in pair_paths:
        row = load(path)
        qid, cid = row["query_id"], row["candidate_id"]
        per_query = score_bank.setdefault(qid, {})
        if cid in per_query:
            raise ValueError("Duplicate cached pair")
        per_query[cid] = row["logit"]
    if set(features) != set(score_bank) or len(pair_paths) != 360:
        raise ValueError("Unexpected described-image scoring coverage")
    rankings = replay(source, features, archived, score_bank)
    grades = (
        json.loads(line) for line in files[3].read_text(encoding="utf-8").splitlines()
    )
    labels = [
        row
        for row in grades
        if row["candidate_id"] in rankings[row["query_id"]][BASELINES[0]]
    ]
    modes = {
        query["id"]: "text" if query["text"] else "image" for query in source["queries"]
    }
    comparisons = [
        (f"{method}_{variant}", method) for method in BASELINES for variant in VARIANTS
    ]
    metrics = evaluate_intervals(rankings, labels, modes)
    paired = paired_querywise(rankings, labels, modes, comparisons)
    count = review_packet(source, rankings, labels, registry)
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            scope="18 exposed historical queries; blind policy replayed on development",
            AI_development=metrics,
            paired_AI_development=paired,
            input_receipt_sha256=sha256_file(receipt),
            rankings_sha256=sha256_file(ROOT / "rankings.json"),
            cached_described_scores_verified=len(pair_paths),
            human_review_tasks=count,
            human_review_task_sha256=sha256_file(REVIEW / "tasks.private.json"),
            new_model_calls=0,
            frozen_acceptance_model_calls=0,
            formal_metrics=None,
            evaluation_source_sha256=sha256_file(Path(__file__)),
        ),
    )
    print(f"Evaluated {len(rankings)} queries. Blinded review tasks: {count}.")


def score_review() -> None:
    """Validate submitted human rows and count changes on reviewed Top10 slots."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    sample = REVIEW / "tasks.private.json"
    files = sorted((REVIEW / "answers").glob("*.csv"))
    rank_path = ROOT / "rankings.json"
    result_path = ROOT / "results.json"
    require_data_access(
        [sample, rank_path, result_path, *files], "development", registry
    )
    results = load(result_path)
    if (
        sha256_file(sample) != results["human_review_task_sha256"]
        or sha256_file(rank_path) != results["rankings_sha256"]
    ):
        raise ValueError("Review tasks or frozen rankings changed")
    tasks = load(sample)
    answers: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            answers.extend(reader)
    checked = human_accuracy(tasks, answers)
    by_id = {
        row["task_id"]: row["human_grade"]
        for row in answers
        if row["human_grade"].strip()
    }
    grades = {
        (task["query_id"], task["candidate_id"]): by_id.get(task["task_id"])
        for task in tasks
    }
    changes = {}
    rankings = load(rank_path)
    for variant in VARIANTS:
        added: list[str | None] = []
        removed: list[str | None] = []
        for qid, methods in rankings.items():
            original = set(methods[BASELINES[0]][:10])
            revised = set(methods[f"{BASELINES[0]}_{variant}"][:10])
            added.extend(grades[qid, cid] for cid in revised - original)
            removed.extend(grades[qid, cid] for cid in original - revised)
        known_delta = added.count("1") - removed.count("1")
        changes[variant] = dict(
            added_slots=len(added),
            removed_slots=len(removed),
            confirmed_relevant_added=added.count("1"),
            confirmed_relevant_removed=removed.count("1"),
            relevant_slot_delta_range=[
                known_delta - sum(value not in {"0", "1"} for value in removed),
                known_delta + sum(value not in {"0", "1"} for value in added),
            ],
        )
    write_json(
        REVIEW / "progress.json",
        dict(
            reviewed=checked["completed"],
            expected=checked["expected"],
            complete=checked["complete"],
            sampling="targeted_development_boundary_review",
            top10_changes=changes,
            formal_metrics=None,
        ),
    )
    print(f"Human review: {checked['completed']}/{checked['expected']}.")


def main() -> None:
    """Run development replay or import the returned local review CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=("evaluate", "score"), default="evaluate"
    )
    args = parser.parse_args()
    if args.action == "score":
        score_review()
    else:
        evaluate()


if __name__ == "__main__":
    main()
