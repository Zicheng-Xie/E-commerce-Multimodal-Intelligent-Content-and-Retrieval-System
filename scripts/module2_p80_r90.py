"""Tune a bounded development reranker and prepare a targeted human review packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy, stable_order
from paper_fusion_experiment import COHORTS, DATA
from paper_fusion_experiment import ROOT as PARENT
from paper_fusion_experiment import load
from simple_cycle_review import FIELDS

ROOT = PROJECT_ROOT / "artifacts/module2_p80_r90_20261106"
REVIEW = ROOT / "human_review"
CONTRACT = PROJECT_ROOT / "configs/module2_target_p80_r90.json"
OLD = "sid_cross_recall_complete_ce"
ANCHOR = "sid_cross_recall_consensus_anchor"
NEW = ("conditional_anchor", "cross_rank_blend25", "cross_rank_blend50")
PRIORITIES = [
    ("regression", "image_06", 15),
    ("fresh", "image_03", 10),
    ("tune", "text_01", 5),
]


def candidate_rankings(previous: dict, jobs: list[dict]) -> dict:
    """Construct fixed alternatives using the complete scored candidate heads.

    Args:
        previous: Full rankings from the real Qwen expansion experiment.
        jobs: Verified per-query pool identities for the completed scores.

    Returns:
        Full rankings for the controls and three new candidates.
    """
    by_query = {(row["cohort"], row["query"]["id"]): row for row in jobs}
    result: dict = {}
    for cohort, queries in previous.items():
        result[cohort] = {}
        for qid, methods in queries.items():
            base = methods[OLD]
            candidates = dict(methods)
            if qid.startswith("text_"):
                candidates.update({name: base for name in NEW})
            else:
                job = by_query[cohort, qid]
                size = len(job["head"]) + len(job["extra"])
                head = base[:size]
                if set(head) != set(job["head"] + job["extra"]):
                    raise ValueError("Scored Qwen head does not match the frozen pool")
                anchor = job["anchor"]
                anchored = (
                    [anchor] + [cid for cid in base if cid != anchor]
                    if anchor in head[:3]
                    else base
                )
                candidates[NEW[0]] = anchored
                allowed = set(head)
                cross = {
                    cid: i
                    for i, cid in enumerate(
                        cid for cid in methods["cross20"] if cid in allowed
                    )
                }
                for name, weight in zip(NEW[1:], [0.25, 0.5], strict=True):
                    scores = np.array(
                        [
                            -(1 - weight) * i - weight * cross[cid]
                            for i, cid in enumerate(head)
                        ]
                    )
                    candidates[name] = stable_order(head, scores) + base[size:]
            result[cohort][qid] = candidates
    return result


def oracle_capacity(rows: list[dict], contract: dict) -> dict:
    """Find exact optimistic Top10 joint capacity over unresolved AI labels.

    Args:
        rows: Known-positive and unresolved counts for every query in a scope.
        contract: Fixed K and simultaneous Precision and Recall thresholds.

    Returns:
        Precision and Recall ceilings, joint feasibility and ideal R90 cutoff.
    """
    k = contract["k"]
    p_target = contract["precision_at_10_required"]
    r_target = contract["recall_at_10_preferred"]
    states = {(0, 0): 0.0}
    positive_counts = []
    for row in rows:
        known, unknown = row["known_positive"], row["unknown"]
        if known or unknown:
            positive_counts.append(max(1, known))
        next_states: dict[tuple[int, int], float] = {}
        for total in range(known, known + unknown + 1):
            hits = min(k, total)
            for (defined, previous_hits), recall_sum in states.items():
                key = (defined + bool(total), previous_hits + hits)
                value = recall_sum + (hits / total if total else 0.0)
                next_states[key] = max(next_states.get(key, 0.0), value)
        states = next_states
    best_recall = max(
        (value / n for (n, _), value in states.items() if n), default=None
    )
    best_precision = max(hits for _, hits in states) / (k * len(rows))
    feasible = any(
        n
        and hits / (k * len(rows)) >= p_target - 1e-12
        and value / n >= r_target - 1e-12
        for (n, hits), value in states.items()
    )
    minimum_k = next(
        (
            cutoff
            for cutoff in range(1, max(positive_counts, default=0) + 1)
            if sum(min(cutoff, p) / p for p in positive_counts)
            >= r_target * len(positive_counts) - 1e-12
        ),
        None,
    )
    return dict(
        oracle_p10_ceiling=best_precision,
        oracle_r10_ceiling=best_recall,
        joint_oracle_feasible=feasible,
        optimistic_minimum_k_for_recall_target=minimum_k,
    )


def diagnostics(rankings: dict, labels: list[dict], contract: dict) -> dict:
    """Compute all six diagnostic intervals and the exact scope ceilings.

    Args:
        rankings: Candidate lists keyed by cohort-qualified query.
        labels: Original complete AI relation table for these queries.
        contract: Versioned 80/90 retrieval targets.

    Returns:
        Metrics and capacity for available scopes.
    """
    modes = {
        qid: "text" if qid.split("/")[-1].startswith("text_") else "image"
        for qid in rankings
    }
    measured = evaluate_intervals(rankings, labels, modes)
    result = {}
    for scope, methods in measured["scopes"].items():
        counts = [
            row
            for row in measured["query_capacity"]
            if scope == "all" or modes[row["query_id"]] == scope
        ]
        result[scope] = dict(
            capacity=oracle_capacity(counts, contract),
            queries=len(counts),
            methods={method: report["ranges"] for method, report in methods.items()},
        )
    return result


def review_tasks(ranks: dict, tables: dict, selected: str) -> list[dict]:
    """Select thirty targeted relations and retain original independent stimuli.

    Args:
        ranks: Complete development candidate ranks before human judgments.
        tables: Existing cohort tables, including queries, products and AI labels.
        selected: Tune-selected candidate variant.

    Returns:
        Three deliberately targeted strata for manual relevance adjudication.
    """
    tasks = []
    for cohort, qid, count in PRIORITIES:
        row = tables[cohort]
        grades = {
            r["candidate_id"]: r["grade"] for r in row["labels"] if r["query_id"] == qid
        }
        methods = ranks[cohort][qid]
        ranking = methods[selected]
        if cohort == "regression":
            order = [cid for cid in ranking if grades[cid] is None] + ranking
        elif cohort == "fresh":
            changed = set(ranking[:10]) ^ set(methods["simple_previous"][:10])
            order = [cid for cid in ranking if cid in changed] + ranking
        else:
            order = sorted(
                (cid for cid in ranking if grades[cid] == 1),
                key=lambda cid: hashlib.sha256(
                    f"p80-r90/{qid}/{cid}".encode()
                ).hexdigest(),
            )
        for cid in list(dict.fromkeys(order))[:count]:
            key = hashlib.sha256(
                f"priority-p80-r90/{cohort}/{qid}/{cid}".encode()
            ).hexdigest()[:24]
            tasks.append(
                dict(
                    task_id=key,
                    cohort=cohort,
                    query_id=qid,
                    candidate_id=cid,
                    grade=grades[cid],
                    query=row["queries"][qid],
                    candidate=row["catalog"][cid],
                )
            )
    if len(tasks) != 30:
        raise ValueError("Priority review requires exactly thirty distinct relations")
    return tasks


def prepare() -> None:
    """Select on tune images, measure all development scopes and build the review UI."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    paths = [
        PARENT / name
        for name in ("summary.json", "sid_protocol.json", "sid_rerank_rankings.json")
    ]
    for cohort in COHORTS:
        paths.extend(
            DATA / cohort / name
            for name in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
            )
        )
    require_data_access(paths, "development", registry)
    manifest = load(PARENT / "summary.json")
    for path in paths[1:3]:
        if sha256_file(path) != manifest["files"][path.name]:
            raise ValueError(f"Previous Qwen artifact changed: {path.name}")
    original = load(PARENT / "sid_rerank_rankings.json")
    ranks = candidate_rankings(original, load(PARENT / "sid_protocol.json")["jobs"])
    contract = load(CONTRACT)
    tables = {}
    for cohort in COHORTS:
        folder = DATA / cohort
        tables[cohort] = dict(
            catalog={r["id"]: r for r in load(folder / "catalog.public.json")},
            queries={r["id"]: r for r in load(folder / "queries.public.json")},
            labels=[
                json.loads(line)
                for line in (folder / "judgments.ai.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ],
        )

    def metrics(cohorts: tuple[str, ...], images_only: bool = False) -> dict:
        """Evaluate one fixed scope without changing relation denominators.

        Args:
            cohorts: Exposed development cohorts included in the report.
            images_only: Select only the image stratum used for model selection.

        Returns:
            Diagnostic metrics for the selected records.
        """
        orders = {
            f"{cohort}/{qid}": methods
            for cohort in cohorts
            for qid, methods in ranks[cohort].items()
            if not images_only or qid.startswith("image_")
        }
        labels = [
            dict(row, query_id=f"{cohort}/{row['query_id']}")
            for cohort in cohorts
            for row in tables[cohort]["labels"]
            if f"{cohort}/{row['query_id']}" in orders
        ]
        return diagnostics(orders, labels, contract)

    tune = metrics(("tune",), images_only=True)["image"]["methods"]
    options = [OLD, ANCHOR, *NEW]
    selected = max(
        options,
        key=lambda name: (
            min(
                tune[name]["precision@10"][0] / contract["precision_at_10_required"],
                tune[name]["recall@10"][0] / contract["recall_at_10_preferred"],
            ),
            tune[name]["precision@10"][0],
            tune[name]["ndcg@10"][0],
            tune[name]["mrr"][0],
        ),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        ROOT / "selection.json",
        dict(
            selected=selected,
            choices=options,
            fit_scope="tune/image only",
            objective="max min(P lower / 0.80, R lower / 0.90); ties by P, NDCG, MRR lower",
            tune=tune,
            contract_sha256=sha256_file(CONTRACT),
            inputs={
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths
            },
        ),
    )
    write_json(ROOT / "rankings.json", ranks)
    write_json(
        ROOT / "results.json",
        dict(
            selected=selected,
            combined=metrics(COHORTS),
            cohorts={cohort: metrics((cohort,)) for cohort in COHORTS},
            interpretation="AI development diagnosis, uncertainty only from unresolved labels",
            formal_metrics=None,
        ),
    )
    tasks = review_tasks(ranks, tables, selected)
    images = [
        project_path(record["image_path"])
        for task in tasks
        for record in (task["query"], task["candidate"])
        if record["image_path"]
    ]
    require_data_access(images, "development", registry)
    if not all(path.is_file() for path in images):
        raise ValueError("A review stimulus is missing")
    REVIEW.mkdir(exist_ok=True)
    sample = REVIEW / "tasks.private.json"
    if sample.exists() and load(sample) != tasks:
        raise ValueError("Previously issued human tasks differ from the new selection")
    write_json(sample, tasks)
    write_json(
        REVIEW / "manifest.json",
        dict(
            sample_sha256=sha256_file(sample),
            tasks=30,
            sampling="targeted_development_diagnosis",
            new_holdout_model_calls=0,
            formal_metrics=None,
        ),
    )
    render_review(
        tasks,
        REVIEW,
        title="模块二优先人工复核 · 30 对",
        instructions=(
            "本批覆盖 15 对低证据图片、10 对排序分歧图片、5 对文本相关性抽查。"
            + "请按查询商品主体、用途及明确属性判断。拿不准请选择证据不足。"
            + "导出后把 CSV 放入本目录 answers 文件夹，然后告诉我。"
        ),
    )
    print(
        f"Tune-selected candidate: {selected}; priority review: {REVIEW / 'index.html'}"
    )


def score() -> None:
    """Validate returned human CSVs and export only adjudicated development relations."""
    sample = REVIEW / "tasks.private.json"
    if sha256_file(sample) != load(REVIEW / "manifest.json")["sample_sha256"]:
        raise ValueError("Frozen priority tasks changed")
    tasks = load(sample)
    answers: list[dict] = []
    for path in sorted((REVIEW / "answers").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            answers.extend(reader)
    checked = human_accuracy(tasks, answers)
    by_id = {task["task_id"]: task for task in tasks}
    reviewed = [
        dict(
            cohort=by_id[row["task_id"]]["cohort"],
            query_id=by_id[row["task_id"]]["query_id"],
            candidate_id=by_id[row["task_id"]]["candidate_id"],
            original_ai_grade=by_id[row["task_id"]]["grade"],
            **row,
        )
        for row in answers
        if row["human_grade"].strip()
    ]
    write_json(REVIEW / "adjudications.json", reviewed)
    progress = dict(
        completed=checked["completed"],
        expected=checked["expected"],
        comparison_counts=checked["counts"],
        sampling="targeted_development_diagnosis",
        population_accuracy=None,
        formal_metrics=None,
    )
    write_json(REVIEW / "progress.json", progress)
    print(json.dumps(progress, ensure_ascii=False))


def main() -> None:
    """Prepare the target experiment and review, or ingest completed review CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "score"])
    {"prepare": prepare, "score": score}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
