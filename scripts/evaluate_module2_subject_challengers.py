"""Replay contested slots and evaluate the new sale-item instruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_subject_challengers import BASE, EXPANDED, ROOT, TARGET
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy
from paper_fusion_experiment import COHORTS, DATA, load
from querywise_paired_bounds import paired_querywise
from simple_cycle_review import FIELDS
from simple_cycle_review import ROOT as AUDIT

FULL = "sale_item_full_challengers"
BLEND = "sale_item_half_challengers"
CONTROLS = ("global", "region_title", "cross20")


def replace_slots(
    order: list[str], job: dict, scores: dict[str, float], weight: float
) -> list[str]:
    """Replace exactly eight selected positions using ranks within one instruction.

    Args:
        order: Full baseline gallery ranking.
        job: Frozen candidate identities and their zero-based positions.
        scores: Actual complete Qwen logits for the alternate instruction.
        weight: Fraction of alternate ordinal rank in the slot order.

    Returns:
        Permutation of the full gallery with the selected slots reranked.
    """
    ids, slots = job["extra"], job["original_slots"]
    if (
        len(slots) != 8
        or sorted(set(slots)) != slots
        or slots[0] < 0
        or slots[-1] >= 30
        or [order[i] for i in slots] != ids
        or set(scores) != set(ids)
        or not np.isfinite(list(scores.values())).all()
    ):
        raise ValueError(
            "Contested positions and real scores differ from the frozen job"
        )
    baseline = {cid: i for i, cid in enumerate(ids)}
    model_order = sorted(ids, key=lambda cid: (-scores[cid], baseline[cid]))
    model = {cid: i for i, cid in enumerate(model_order)}
    ranked = sorted(
        ids,
        key=lambda cid: (
            weight * model[cid] + (1 - weight) * baseline[cid],
            baseline[cid],
        ),
    )
    result = order.copy()
    for position, cid in zip(slots, ranked, strict=True):
        result[position] = cid
    return result


def score_replay(protocol: dict) -> dict:
    """Validate complete real pair caches and construct frozen method comparisons.

    Args:
        protocol: Original image/score plan with pinned inputs and instruction.

    Returns:
        Complete gallery rankings for every control and candidate.
    """
    previous = load(EXPANDED / "rankings.json")
    controls = load(TARGET / "rankings.json")
    rankings = {
        f"{cohort}/{qid}": {
            **{name: controls[cohort][qid][name] for name in CONTROLS},
            BASE: row[BASE],
            FULL: row[BASE],
            BLEND: row[BASE],
        }
        for cohort, queries in previous.items()
        for qid, row in queries.items()
    }
    digest = sha256_file(ROOT / "sid_protocol.json")
    for job in protocol["jobs"]:
        cohort, query = job["cohort"], job["query"]
        qid = query["id"]
        cache = load(ROOT / "sid_pairs" / cohort / f"{qid}.json")
        if cache["identity"] != dict(
            protocol_sha256=digest,
            query_sha256=query["source_sha256"],
            extra=job["extra"],
        ):
            raise ValueError("Score cache identity changed")
        row = rankings[f"{cohort}/{qid}"]
        row[FULL] = replace_slots(row[BASE], job, cache["scores"], 1.0)
        row[BLEND] = replace_slots(row[BASE], job, cache["scores"], 0.5)
    return rankings


def human_diagnostic(rankings: dict, answer_paths: list[Path]) -> dict:
    """Read the two distinct sets of real development human judgments.

    Args:
        rankings: Fixed full-gallery predictions made without human answers.
        answer_paths: The random audit CSV input files already imported.

    Returns:
        Reviewed relation counts and actual rank movement, by review design.
    """
    sample = AUDIT / "sample.private.json"
    if sha256_file(sample) != load(AUDIT / "sampling_manifest.json")["sample_sha256"]:
        raise ValueError("Random human sample changed")
    tasks = load(sample)
    by_id = {task["task_id"]: task for task in tasks}
    rows: list[dict] = []
    for path in answer_paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Human CSV schema changed")
            rows.extend(reader)
    human_accuracy(tasks, rows)
    targeted = load(TARGET / "human_review/adjudications.json")
    sources = {
        "random_incomplete": [
            {**by_id[row["task_id"]], "human_grade": row["human_grade"]}
            for row in rows
            if row["human_grade"].strip()
        ],
        "targeted": targeted,
    }
    report = {}
    for name, cases in sources.items():
        records = []
        for case in cases:
            key = f"{case['cohort']}/{case['query_id']}"
            cid = case["candidate_id"]
            records.append(
                dict(
                    query=key,
                    candidate_id=cid,
                    human_grade=case["human_grade"],
                    positions={
                        method: rankings[key][method].index(cid) + 1
                        for method in (BASE, FULL, BLEND)
                    },
                )
            )
        report[name] = dict(
            reviewed=len(records),
            image_top10={
                method: {
                    grade: sum(
                        "/image_" in row["query"]
                        and row["human_grade"] == grade
                        and row["positions"][method] <= 10
                        for row in records
                    )
                    for grade in ("0", "1")
                }
                for method in (BASE, FULL, BLEND)
            },
            records=records,
        )
    return report


def main() -> None:
    """Measure exact development intervals and real review-case movement."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    labels_paths = [DATA / c / "judgments.ai.jsonl" for c in COHORTS]
    csv_paths = [
        AUDIT / "human_answers.csv",
        *sorted((AUDIT / "answers").glob("*.csv")),
    ]
    protocol_path = ROOT / "sid_protocol.json"
    files = [
        protocol_path,
        ROOT / "runtime.json",
        EXPANDED / "rankings.json",
        TARGET / "rankings.json",
        TARGET / "selection.json",
        AUDIT / "sample.private.json",
        AUDIT / "sampling_manifest.json",
        TARGET / "human_review/adjudications.json",
        *labels_paths,
        *csv_paths,
        *sorted((ROOT / "sid_pairs").glob("*/*.json")),
    ]
    require_data_access(files, "development", registry)
    protocol = load(protocol_path)
    for name, digest in protocol["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen scorer input changed: {name}")
    label_receipt = load(TARGET / "selection.json")["inputs"]
    labels = []
    for path in labels_paths:
        if sha256_file(path) != label_receipt[str(path.relative_to(PROJECT_ROOT))]:
            raise ValueError("Original AI labels changed")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels.append(dict(row, query_id=f"{path.parent.name}/{row['query_id']}"))
    rankings = score_replay(protocol)
    modes = {key: "image" if "/image_" in key else "text" for key in rankings}
    absolute = evaluate_intervals(rankings, labels, modes)
    comparisons = [(name, base) for name in (FULL, BLEND) for base in (BASE, *CONTROLS)]
    paired = paired_querywise(rankings, labels, modes, comparisons)
    human = human_diagnostic(rankings, csv_paths)
    runtime = load(ROOT / "runtime.json")
    recorded = len(runtime.get("timings", []))
    if recorded > protocol["new_pairs"]:
        raise ValueError("Runtime receipt exceeds the frozen new scores")
    write_json(ROOT / "rankings.json", rankings)
    write_json(
        ROOT / "results.json",
        dict(
            role="exposed_development",
            complete_new_model_scores=protocol["new_pairs"],
            recorded_runtime_calls=recorded,
            AI_development=absolute,
            paired_AI_development=paired,
            human_review_diagnostic=human,
            formal_metrics=None,
            frozen_acceptance_model_calls=0,
            inputs={
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files
            },
            evaluation_source_sha256=sha256_file(Path(__file__)),
        ),
    )
    print(
        f"Replayed {len(rankings)} queries and {protocol['new_pairs']} real new scores."
    )


if __name__ == "__main__":
    main()
