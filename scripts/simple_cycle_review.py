"""Prepare an actual-human, AI-blinded 20% development annotation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json

import run  # noqa: F401
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy, sample_audit

ROOT = PROJECT_ROOT / "artifacts/module2_simple_baseline_20261105/human_audit"
DATA = PROJECT_ROOT / "artifacts/module2_balanced_gallery"
FIELDS = [
    "task_id",
    "reviewer_type",
    "reviewer",
    "reviewed_at",
    "human_grade",
    "reason",
]


def prepare() -> None:
    """Freeze 20% of development relations and create blinded local review pages."""
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "sampling_manifest.json").exists():
        return
    tasks = []
    sources = {}
    for cohort in ["tune", "regression", "fresh"]:
        folder = DATA / cohort
        path = folder / "judgments.ai.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        sources[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
        catalog = {
            r["id"]: r for r in json.loads((folder / "catalog.public.json").read_text())
        }
        queries = {
            r["id"]: r for r in json.loads((folder / "queries.public.json").read_text())
        }
        for row in sample_audit(rows):
            key = hashlib.sha256(
                f"{cohort}/{row['query_id']}/{row['candidate_id']}".encode()
            ).hexdigest()[:24]
            tasks.append(
                dict(
                    row,
                    task_id=key,
                    cohort=cohort,
                    query=queries[row["query_id"]],
                    candidate=catalog[row["candidate_id"]],
                )
            )
    write_json(ROOT / "sample.private.json", tasks)
    with (ROOT / "human_answers.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            dict(
                task_id=t["task_id"],
                reviewer_type="",
                reviewer="",
                reviewed_at="",
                human_grade="",
                reason="",
            )
            for t in tasks
        )
    write_json(
        ROOT / "sampling_manifest.json",
        dict(
            seed=2026110501,
            fraction=0.2,
            relations=12600,
            sampled=len(tasks),
            per_query="20%, proportional AI grade strata; null included; ranks/scores unused",
            sample_sha256=sha256_file(ROOT / "sample.private.json"),
            source_sha256=sources,
            actual_human_answers=0,
            AI_accuracy=None,
            reviewer="unassigned",
            threshold=0.8,
            no_consumed_blind_judgments_read=True,
        ),
    )

    render()


def render() -> None:
    """Refresh blinded forms and record the user's own human-review assignment."""
    manifest = json.loads((ROOT / "sampling_manifest.json").read_text())
    if sha256_file(ROOT / "sample.private.json") != manifest["sample_sha256"]:
        raise ValueError("Frozen audit tasks changed")
    tasks = json.loads((ROOT / "sample.private.json").read_text())
    render_review(tasks, ROOT)
    manifest["reviewer"] = "user_self"
    manifest["assignment_source"] = "User explicitly chose self review on 2026-11-05"
    manifest["answers_initially_blank"] = True
    write_json(ROOT / "sampling_manifest.json", manifest)


def score() -> None:
    """Validate actual human rows and report agreement without inventing truth."""
    manifest = json.loads((ROOT / "sampling_manifest.json").read_text())
    if sha256_file(ROOT / "sample.private.json") != manifest["sample_sha256"]:
        raise ValueError("Frozen audit tasks changed")
    tasks = json.loads((ROOT / "sample.private.json").read_text())
    answers: list[dict] = []
    paths = [ROOT / "human_answers.csv", *sorted((ROOT / "answers").glob("*.csv"))]
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV schema: {path.name}")
            answers.extend(reader)
    result = human_accuracy(tasks, answers)
    write_json(ROOT / "human_accuracy.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    """Prepare blinded packets or import and score the returned human CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "render", "score"])
    {"prepare": prepare, "render": render, "score": score}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
