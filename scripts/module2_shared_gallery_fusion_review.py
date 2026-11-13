"""Review high-ranked differences between full-image and subject-fusion retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import REVIEW_GRADES, Evidence, load
from module2_shared_gallery_precision import ROOT, reviewed_inputs
from module2_shared_gallery_retrieval import _checked_media, _score
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json
from multimodal_ecommerce.simple_baseline import human_accuracy
from simple_cycle_review import FIELDS

REVIEW = ROOT / "fusion_review"
TEXT_REVIEW = ROOT / "text_remaining/human_review"
FUSION = "whole_subject_equal_fusion"
K = 10
PER_GROUP = 6


def _candidates(
    orders: dict, queries: dict, mode: str, method: str
) -> list[tuple[str, str]]:
    other = "whole" if method == FUSION else FUSION
    rows = []
    for qid, query in queries.items():
        if query["mode"] != mode:
            continue
        other_head = set(orders[other][qid][:K])
        for rank, cid in enumerate(orders[method][qid][:K], 1):
            label = query["judgments"].get(cid, {})
            if (
                cid not in other_head
                and label.get("grade") is None
                and not label.get("human")
            ):
                rows.append((rank, qid, cid))
                break
    return [(qid, cid) for _, qid, cid in sorted(rows)[:PER_GROUP]]


def _tasks(
    evidence: Evidence, orders: dict, queries: dict, catalog: dict
) -> list[dict]:
    tasks = []
    for mode in ("image", "text"):
        for method in ("whole", FUSION):
            for qid, cid in _candidates(orders, queries, mode, method):
                query, doc = queries[qid], catalog[cid]
                image = _checked_media(evidence, query) if mode == "image" else None
                candidate = _checked_media(evidence, doc)
                task_id = hashlib.sha256(f"{qid}\n{cid}\nfusion".encode()).hexdigest()[
                    :24
                ]
                tasks.append(
                    dict(
                        task_id=task_id,
                        query_id=qid,
                        candidate_id=cid,
                        query=dict(
                            text=query.get("text"),
                            image_path=str(image) if image else None,
                        ),
                        candidate=dict(title=doc["title"], image_path=str(candidate)),
                    )
                )
    if len({row["task_id"] for row in tasks}) != 4 * PER_GROUP:
        raise ValueError(
            "Not enough distinct unreviewed differences for the balanced packet"
        )
    return sorted(tasks, key=lambda row: row["task_id"])


def _remaining_text_tasks(
    evidence: Evidence, orders: dict, queries: dict, catalog: dict
) -> list[dict]:
    tasks = []
    for qid, query in queries.items():
        if query["mode"] != "text":
            continue
        full = set(orders["whole"][qid][:K])
        fused = set(orders[FUSION][qid][:K])
        for cid in sorted(full ^ fused):
            label = query["judgments"].get(cid, {})
            if label.get("grade") is not None or label.get("human"):
                continue
            doc = catalog[cid]
            candidate = _checked_media(evidence, doc)
            task_id = hashlib.sha256(
                f"{qid}\n{cid}\ntext-remaining".encode()
            ).hexdigest()[:24]
            tasks.append(
                dict(
                    task_id=task_id,
                    query_id=qid,
                    candidate_id=cid,
                    query=dict(text=query["text"], image_path=None),
                    candidate=dict(title=doc["title"], image_path=str(candidate)),
                )
            )
    return sorted(tasks, key=lambda row: row["task_id"])


def _paired_precision(orders: dict, queries: dict) -> dict:
    output = {}
    for scope in ("all", "image", "text"):
        low, high, size = 0, 0, 0
        for key, query in queries.items():
            if scope not in ("all", query["mode"]):
                continue
            size += 1
            whole = set(orders["whole"][key][:K])
            fusion = set(orders[FUSION][key][:K])
            labels = query["judgments"]
            entering = [labels.get(cid, {}).get("grade") for cid in fusion - whole]
            leaving = [labels.get(cid, {}).get("grade") for cid in whole - fusion]
            known = entering.count(1) - leaving.count(1)
            low += known - leaving.count(None)
            high += known + entering.count(None)
        output[scope] = [low / (K * size), high / (K * size)]
    return output


def _apply_answers(
    evidence: Evidence, directory: Path, tasks: list[dict], queries: dict
) -> dict:
    answers: list[dict] = []
    for path in sorted((directory / "answers").glob("*.csv")):
        evidence.digest(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected human CSV fields: {path.name}")
            answers.extend(reader)
    progress = human_accuracy([dict(task, grade=None) for task in tasks], answers)
    by_id = {task["task_id"]: task for task in tasks}
    for row in answers:
        grade = row["human_grade"]
        if grade.strip():
            task = by_id[row["task_id"]]
            queries[task["query_id"]]["judgments"][task["candidate_id"]] = dict(
                grade=REVIEW_GRADES[grade], human=True
            )
    return progress


def _score_answers(
    evidence: Evidence, directory: Path, tasks: list[dict], orders: dict, queries: dict
) -> None:
    progress = _apply_answers(evidence, directory, tasks, queries)
    write_json(
        directory / "progress.json",
        dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            paired_precision_delta=_paired_precision(orders, queries),
            metrics={
                method: _score(orders[method], queries) for method in ("whole", FUSION)
            },
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print(f"Fusion review: {progress['completed']}/{progress['expected']}.")


def main() -> None:
    """Prepare or import a blinded review of the two frozen Top10 candidate sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-review", action="store_true")
    parser.add_argument("--remaining-text", action="store_true")
    args = parser.parse_args()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    manifest, baseline, _ = reviewed_inputs(evidence)
    receipt = evidence.json(ROOT / "crop_results.json")
    orders = evidence.json(ROOT / "crop_rankings.json", receipt["rankings_sha256"])
    if orders["whole"] != baseline:
        raise ValueError(
            "The full-image control must reproduce the frozen original ranks"
        )
    queries = manifest["query_manifest"]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    tasks = _tasks(evidence, orders, queries, catalog)
    directory = REVIEW
    if args.remaining_text:
        if evidence.json(REVIEW / "tasks.private.json") != tasks:
            raise ValueError("Original fusion review tasks differ")
        progress = _apply_answers(evidence, REVIEW, tasks, queries)
        if not progress["complete"]:
            raise ValueError(
                "Complete the original fusion review before resolving text differences"
            )
        directory = TEXT_REVIEW
        tasks = _remaining_text_tasks(evidence, orders, queries, catalog)
    path = directory / "tasks.private.json"
    if path.is_file() and evidence.json(path) != tasks:
        raise ValueError(
            "Existing fusion review tasks differ; preserve the frozen packet"
        )
    if args.score_review and not path.is_file():
        raise ValueError("Prepare the frozen review packet before importing")
    if not args.score_review:
        render_review(
            tasks,
            directory,
            title=(
                "模块二 · 文搜剩余换位复核"
                if args.remaining_text
                else "模块二 · 全图与主体融合换位复核"
            ),
            instructions=(
                "按原标准核对候选商品主体、用途与明确属性。"
                + "选相关、不相关或证据不足，写一句理由。导出 CSV 放进本目录 answers 文件夹。"
            ),
        )
        if not path.is_file():
            write_json(path, tasks)
            evidence.digest(path)
    _score_answers(evidence, directory, tasks, orders, queries)


if __name__ == "__main__":
    main()
