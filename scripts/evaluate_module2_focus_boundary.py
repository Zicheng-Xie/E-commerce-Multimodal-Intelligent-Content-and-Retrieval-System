"""Evaluate full-gallery effects of the adjacent sale-item comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_focus_boundary import FRESH, FRESH_METHOD, ROOT, plan, verified
from module2_focus_followup import BASELINES, FOCUS, REVIEW
from module2_focus_followup import ROOT as PARENT
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy
from paper_fusion_experiment import load
from querywise_paired_bounds import paired_querywise
from simple_cycle_review import FIELDS

SUFFIX = "_sale_item_boundary"


def replay(protocol: dict, registry: dict) -> tuple[dict, dict, dict]:
    """Apply at most one strictly supported adjacent swap per image query.

    Args:
        protocol: Exact frozen pairs and description instruction.
        registry: Data roles checked before reading score files.

    Returns:
        Historical rankings, stress rankings, and actual score/rank movements.
    """
    historical = {
        qid: {name + suffix: row[name] for name in BASELINES for suffix in ("", SUFFIX)}
        for qid, row in load(PARENT / "rankings.json").items()
    }
    stress = {
        qid: {FRESH_METHOD + suffix: row[FRESH_METHOD] for suffix in ("", SUFFIX)}
        for qid, row in load(FRESH / "rankings.json").items()
    }
    digest = sha256_file(ROOT / "protocol.json")
    movements = {}
    for job in protocol["jobs"]:
        group, qid = job["cohort"], job["qid"]
        path = ROOT / "scores" / group / (qid + ".json")
        require_data_access([path], "development", registry)
        cache = load(path)
        values = cache["logits"]
        if (
            cache["identity"] != dict(protocol_sha256=digest, boundary=job["boundary"])
            or len(values) != 2
            or not np.isfinite(values).all()
            or [row["document_id"] for row in cache["timings"]] != job["boundary"]
        ):
            raise ValueError(f"Missing or changed real score pair: {qid}")
        group_rankings = historical if group == "historical" else stress
        methods = BASELINES if group == "historical" else (FRESH_METHOD,)
        swap = values[1] > values[0]
        for method in methods:
            original = group_rankings[qid][method]
            if original[9:11] != job["boundary"]:
                raise ValueError(f"Boundary differs from the frozen ranking: {qid}")
            if swap:
                group_rankings[qid][method + SUFFIX] = [
                    *original[:9],
                    original[10],
                    original[9],
                    *original[11:],
                ]
        movements[qid] = dict(
            cohort=group,
            original_order=job["boundary"],
            logits=values,
            swapped=swap,
            cache_sha256=sha256_file(path),
        )
    return historical, stress, movements


def score_group(rankings: dict, path: Path, names: tuple[str, ...]) -> dict:
    """Measure the unchanged complete-gallery AI diagnostic on one cohort.

    Args:
        rankings: Complete original and revised rankings.
        path: Verified source AI labels.
        names: Original method identities in this cohort.

    Returns:
        Absolute and paired intervals for all original ranking metrics.
    """
    allowed = {qid: set(row[names[0]]) for qid, row in rankings.items()}
    labels = [
        row
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        if row["query_id"] in allowed
        and row["candidate_id"] in allowed[row["query_id"]]
    ]
    modes = {qid: "text" if qid.startswith("text_") else "image" for qid in rankings}
    return dict(
        absolute=evaluate_intervals(rankings, labels, modes),
        paired=paired_querywise(
            rankings, labels, modes, [(method + SUFFIX, method) for method in names]
        ),
    )


def reviewed_swaps(movements: dict, registry: dict) -> dict:
    """Report effects only where the user independently judged both positions.

    Args:
        movements: Candidate movements already determined without answers.
        registry: Data roles for the submitted review artifacts.

    Returns:
        Human-verified entry and exit decisions and incomplete pair coverage.
    """
    tasks_path = REVIEW / "tasks.private.json"
    answers = sorted((REVIEW / "answers").glob("*.csv"))
    receipt = PARENT / "results.json"
    require_data_access([receipt], "development", registry)
    verified(tasks_path, load(receipt)["human_review_task_sha256"], registry)
    require_data_access(answers, "development", registry)
    tasks = load(tasks_path)
    rows: list[dict] = []
    for path in answers:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            rows.extend(reader)
    human_accuracy(tasks, rows)
    by_task = {
        row["task_id"]: row["human_grade"] for row in rows if row["human_grade"].strip()
    }
    grades = {
        (task["query_id"], task["candidate_id"]): by_task.get(task["task_id"])
        for task in tasks
    }
    swaps = {}
    for qid, event in movements.items():
        if event["swapped"] and event["cohort"] == "historical":
            out, incoming = event["original_order"]
            swaps[qid] = dict(
                outgoing=grades.get((qid, out)),
                incoming=grades.get((qid, incoming)),
            )
    complete = [row for row in swaps.values() if set(row.values()) <= {"1", "0"}]
    return dict(
        reviewed_pairs=len(by_task),
        fully_reviewed_swaps=len(complete),
        historical_swaps=swaps,
        relevant_slot_delta=sum(row["incoming"] == "1" for row in complete)
        - sum(row["outgoing"] == "1" for row in complete),
        inputs={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in answers
        },
    )


def main() -> None:
    """Check the scoring freeze and evaluate the two exposed cohorts separately."""
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    frozen = ROOT / "protocol.json"
    historical_labels = FOCUS / "judgments.ai.jsonl"
    stress_labels = FRESH / "judgments.ai.jsonl"
    source_receipt = FOCUS / "completion_receipt.json"
    stress_receipt = FRESH / "inference_protocol.json"
    require_data_access(
        [frozen, source_receipt, stress_receipt], "development", registry
    )
    protocol = load(frozen)
    if plan(registry) != protocol:
        raise ValueError("Frozen boundary inputs changed")
    historical, stress, movements = replay(protocol, registry)
    labels = {
        historical_labels: load(source_receipt)["files"][historical_labels.name],
        stress_labels: load(stress_receipt)["files"][
            str(stress_labels.relative_to(PROJECT_ROOT))
        ],
    }
    for path, digest in labels.items():
        verified(path, digest, registry)
    write_json(ROOT / "rankings.json", dict(historical=historical, stress=stress))
    write_json(
        ROOT / "results.json",
        dict(
            historical=score_group(historical, historical_labels, BASELINES),
            stress=score_group(stress, stress_labels, (FRESH_METHOD,)),
            moves=movements,
            human_diagnostic=reviewed_swaps(movements, registry),
            input_labels={
                str(p.relative_to(PROJECT_ROOT)): h for p, h in labels.items()
            },
            actual_new_pairs=2 * len(movements),
            protocol_sha256=sha256_file(frozen),
            evaluation_source_sha256=sha256_file(Path(__file__)),
            frozen_acceptance_model_calls=0,
            formal_metrics=None,
        ),
    )
    print(f"Evaluated {len(movements)} image boundary comparisons.")


if __name__ == "__main__":
    main()
