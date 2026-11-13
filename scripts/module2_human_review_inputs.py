"""Strict current-packet imports and provenance lookup for human-only reporting."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import REVIEW_GRADES, Evidence
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.simple_baseline import human_accuracy
from simple_cycle_review import FIELDS


def import_precision_answers(
    evidence: Evidence, folder: Path, tasks: list, queries: dict
) -> dict:
    """Import partial or complete batch exports without silently accepting bad identities.

    Args:
        evidence: Reader enforcing the development-only input registry.
        folder: Frozen packet directory with answers subdirectory.
        tasks: All frozen task identities in the original displayed batch order.
        queries: Effective relation observations updated with declared human answers.

    Returns:
        Completion, unresolved identities, and coverage for every original batch.
    """
    task_map = {t["task_id"]: t for t in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("Frozen human task IDs must be unique")
    batches = {t["task_id"]: i // 30 + 1 for i, t in enumerate(tasks)}
    seen, answers = set(), []
    for path in sorted((folder / "answers").glob("*.csv")):
        digest = evidence.digest(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            rows = list(reader)
        if sha256_file(path) != digest:
            raise ValueError("Human CSV changed while it was being read")
        found_batches = set()
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed human CSV row: {path.name}")
            task_id = row["task_id"]
            if task_id not in task_map or task_id in seen:
                raise ValueError(
                    "Unknown or duplicate human task, including blank rows"
                )
            seen.add(task_id)
            found_batches.add(batches[task_id])
            if row["human_grade"].strip():
                try:
                    datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(
                        "A valid human review timestamp is required"
                    ) from exc
            answers.append(row)
        named = re.search(r"human_batch_(\d{3})", path.name)
        if (
            len(found_batches) > 1
            or named
            and found_batches
            and found_batches != {int(named[1])}
        ):
            raise ValueError(
                f"Human export does not match its frozen batch: {path.name}"
            )
    progress = human_accuracy([dict(t, grade=None) for t in tasks], answers)
    completed, uncertain = set(), set()
    for row in answers:
        grade = row["human_grade"]
        if not grade.strip():
            continue
        task = task_map[row["task_id"]]
        value = REVIEW_GRADES[grade]
        queries[task["query_id"]]["judgments"][task["candidate_id"]] = dict(
            grade=value, human=True
        )
        completed.add(row["task_id"])
        if value is None:
            uncertain.add(row["task_id"])
    progress.update(
        binary_completed=len(completed - uncertain),
        uncertain_task_ids=sorted(uncertain),
        missing_task_ids=sorted(set(task_map) - completed),
        batches=[
            dict(
                batch=i,
                expected=sum(b == i for b in batches.values()),
                answered=sum(batches[t] == i for t in completed),
                binary=sum(batches[t] == i for t in completed - uncertain),
                uncertain=sum(batches[t] == i for t in uncertain),
            )
            for i in range(1, (len(tasks) + 29) // 30 + 1)
        ],
    )
    return progress


def annotation_reasons(evidence: Evidence, queries: dict) -> dict:
    """Resolve reasons only from already verified development task and CSV inputs.

    Args:
        evidence: Reader containing the frozen inputs already used by the mainline.
        queries: Canonical shared queries, used to bind historic image/text aliases.

    Returns:
        Human reason/source rows keyed by exact canonical query/product relation.
    """
    identities: dict[str, tuple[str, str]] = {}
    text_ids = {q.get("text"): key for key, q in queries.items() if q["mode"] == "text"}
    for name in list(evidence.files):
        if not name.endswith("tasks.private.json"):
            continue
        tasks = evidence.json(PROJECT_ROOT / name)
        for task in tasks:
            if "candidate_id" not in task:
                continue
            query = task.get("query", {})
            key = task.get("query_id")
            if key not in queries:
                key = (
                    "image/" + evidence.digest(project_path(query["image_path"]))
                    if query.get("image_path")
                    else text_ids.get(query.get("text"))
                )
            if key not in queries:
                continue
            pair = key, task["candidate_id"]
            previous = identities.get(task["task_id"])
            if previous is not None and previous != pair:
                raise ValueError(
                    "One human task ID refers to different query/product pairs"
                )
            identities[task["task_id"]] = pair
    output: dict = {}
    for name in list(evidence.files):
        if not name.endswith(".csv"):
            continue
        path = PROJECT_ROOT / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"task_id", "human_grade", "reason"} <= set(reader.fieldnames or []):
                continue
            for row in reader:
                resolved_pair = identities.get(row["task_id"])
                if resolved_pair is None or row["human_grade"] not in REVIEW_GRADES:
                    continue
                output.setdefault(resolved_pair, []).append(
                    dict(
                        grade=REVIEW_GRADES[row["human_grade"]],
                        reason=row["reason"].strip(),
                        source_csv=name,
                        task_id=row["task_id"],
                    )
                )
    return output
