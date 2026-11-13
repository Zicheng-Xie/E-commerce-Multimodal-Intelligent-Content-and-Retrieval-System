"""Verified human review history and shared-label development metric comparisons."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import FOLLOWUP, REVIEW_GRADES, Evidence
from module2_shared_gallery_fusion_review import _apply_answers
from module2_shared_gallery_retrieval import _checked_media
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.simple_baseline import human_accuracy
from simple_cycle_review import FIELDS


def _answers(
    evidence: Evidence, folder: Path, digests: tuple[str, str | None]
) -> tuple[list, list]:
    tasks = evidence.json(folder / "tasks.private.json", digests[0])
    path = folder / "answers" / (folder.parent.name + "_human_batch_001.csv")
    evidence.digest(path, digests[1])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected historical human CSV fields: {path.name}")
        answers = list(reader)
    checked = human_accuracy(tasks, answers)
    if not checked["complete"] or checked["completed"] != len(answers):
        raise ValueError("Historical human review must contain each frozen task once")
    return tasks, answers


def merge_previous_reviews(evidence: Evidence, manifest: dict) -> dict:
    """Bind past relevance decisions to exact query and product original images.

    Args:
        evidence: Reader restricted to registered development inputs.
        manifest: Shared gallery and query relation records updated in place.

    Returns:
        Imported relation count and changes to the preceding evaluation snapshot.
    """
    focus = evidence.json(FOLLOWUP / "results.json")
    rescue = FOLLOWUP / "subject_rescue"
    result = evidence.json(rescue / "results.json")
    prior = evidence.json(rescue / "human_review/adjudication.json")
    remaining = rescue / "remaining_swaps/human_review"
    audit = evidence.json(remaining / "adjudication.json")
    evidence.digest(
        rescue / "human_review/adjudication.json", audit["prior_adjudication_sha256"]
    )
    evidence.digest(remaining / "tasks.private.json", prior["next_task_sha256"])
    packets = [
        (FOLLOWUP / "human_review", (focus["human_review_task_sha256"], None)),
        (
            rescue / "human_review",
            (result["targeted_review"]["sample_sha256"], prior["answer_sha256"]),
        ),
        (remaining, (audit["tasks_sha256"], audit["answer_sha256"])),
    ]
    documents = {row["id"]: row for row in manifest["catalog"]}
    queries = manifest["query_manifest"]
    imported: dict[tuple[str, str], int | None] = {}
    changed = 0
    for folder, digests in packets:
        tasks, answers = _answers(evidence, folder, digests)
        by_id = {row["task_id"]: row for row in tasks}
        for answer in answers:
            task = by_id[answer["task_id"]]
            query = task["query"]
            qid = "image/" + evidence.digest(project_path(query["image_path"]))
            cid = task["candidate_id"]
            image = project_path(task["candidate"]["image_path"])
            if (
                query["text"] is not None
                or qid not in queries
                or cid not in documents
                or evidence.digest(image) != documents[cid]["image_sha256"]
                or cid in queries[qid]["exclusions"]
            ):
                raise ValueError(
                    "Historical human review does not match shared original inputs"
                )
            grade = REVIEW_GRADES[answer["human_grade"]]
            key = qid, cid
            previous = queries[qid]["judgments"].get(cid, {})
            if (
                key in imported
                and imported[key] != grade
                or previous.get("human")
                and previous["grade"] != grade
            ):
                raise ValueError("Conflicting human decisions require adjudication")
            changed += key not in imported and previous.get("grade") != grade
            imported[key] = grade
            queries[qid]["judgments"][cid] = dict(grade=grade, human=True)
    return dict(imported_pairs=len(imported), changed_grades=changed)


def precision_delta(before: dict, after: dict, queries: dict) -> dict:
    """Bound the paired Top10 precision change under the same unresolved labels.

    Args:
        before: Original full-gallery order per query.
        after: New full-gallery order per query.
        queries: Query modes and shared relevance judgments.

    Returns:
        Exact paired precision ranges for overall, image, and text scopes.
    """
    output = {}
    for scope in ("all", "image", "text"):
        low, high, count = 0, 0, 0
        for key, query in queries.items():
            if scope not in ("all", query["mode"]):
                continue
            count += 1
            original, revised = set(before[key][:10]), set(after[key][:10])
            labels = query["judgments"]
            entering = [labels.get(cid, {}).get("grade") for cid in revised - original]
            leaving = [labels.get(cid, {}).get("grade") for cid in original - revised]
            fixed = entering.count(1) - leaving.count(1)
            low += fixed - leaving.count(None)
            high += fixed + entering.count(None)
        output[scope] = [low / (10 * count), high / (10 * count)]
    return output


def recall_ceiling(queries: dict) -> dict:
    """Compute the optimistic ideal Recall@10 under current full-gallery labels.

    Args:
        queries: Query modes, gallery sizes and observed relevance records.

    Returns:
        Ideal macro-recall ceilings with unresolved zero-positive queries optional.
    """
    output = {}
    for scope in ("all", "image", "text"):
        mandatory, optional = [], []
        for query in queries.values():
            if scope not in ("all", query["mode"]):
                continue
            grades = [row["grade"] for row in query["judgments"].values()]
            positive = grades.count(1)
            if positive:
                mandatory.append((min(10, positive) / positive,) * 2)
            elif query["expected_pairs"] > len(grades) - grades.count(None):
                optional.append((1.0, 1.0))
        output[scope] = optional_mean_bounds(mandatory, optional)[1]
    return output


def remaining_subject_tasks(
    evidence: Evidence, manifest: dict, before: dict, after: dict
) -> list[dict]:
    """Prepare every remaining unreviewed text Top10 change after the first packet.

    Args:
        evidence: Reader restricted to registered development inputs.
        manifest: Shared gallery and query records updated with verified answers.
        before: Original full-gallery ranking for each query.
        after: Frozen selected candidate ranking for each query.

    Returns:
        Complete, deterministically ordered text difference tasks with original evidence.
    """
    folder = (
        PROJECT_ROOT
        / "artifacts/module2_p80_r90_20261106/development_benchmark"
        / "subject_experiment/subject_recall/human_review"
    )
    progress = evidence.json(folder / "progress.json")
    path = folder / "tasks.private.json"
    tasks = evidence.json(
        path, progress["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
    )
    for path in (folder / "answers").glob("*.csv"):
        evidence.digest(
            path, progress["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
        )
    queries = manifest["query_manifest"]
    checked = _apply_answers(evidence, folder, tasks, queries)
    if not checked["complete"] or checked["completed"] != progress["reviewed"]:
        raise ValueError(
            "Finish the initial subject-recall review before preparing the remainder"
        )
    catalog = {row["id"]: row for row in manifest["catalog"]}
    pending = []
    for key, query in queries.items():
        if query["mode"] != "text":
            continue
        for cid in sorted(set(before[key][:10]) ^ set(after[key][:10])):
            if query["judgments"].get(cid, {}).get("human"):
                continue
            doc = catalog[cid]
            pending.append(
                dict(
                    task_id=hashlib.sha256(
                        f"{key}\n{cid}\nsubject-recall-complete".encode()
                    ).hexdigest()[:24],
                    query_id=key,
                    candidate_id=cid,
                    query=dict(text=query["text"], image_path=None),
                    candidate=dict(
                        title=doc["title"],
                        image_path=str(_checked_media(evidence, doc)),
                    ),
                )
            )
    return sorted(pending, key=lambda row: row["task_id"])


def confirmed_change_metrics(before: dict, after: dict, queries: dict) -> dict:
    """Measure fully human-reviewed changes with the shared full-gallery denominator.

    Args:
        before: Original full-gallery rankings.
        after: New full-gallery rankings on the same queries.
        queries: Query modes, gallery sizes, and relevance observations.

    Returns:
        Exact precision deltas, conditional recall ranges, and per-query hit changes.
    """
    rows = {}
    for key, query in queries.items():
        original, revised = set(before[key][:10]), set(after[key][:10])
        labels = query["judgments"]
        if any(
            not labels.get(cid, {}).get("human") or labels[cid]["grade"] not in (0, 1)
            for cid in original ^ revised
        ):
            raise ValueError(
                "Every changed Top10 relation must have a binary human judgment"
            )
        difference = sum(labels[cid]["grade"] for cid in revised - original) - sum(
            labels[cid]["grade"] for cid in original - revised
        )
        grades = [row["grade"] for row in labels.values()]
        positive = grades.count(1)
        unknown = query["expected_pairs"] - len(grades) + grades.count(None)
        endpoints = (
            sorted((difference / positive, difference / (positive + unknown)))
            if positive
            else [0.0, 0.0]
        )
        rows[key] = dict(
            mode=query["mode"],
            hit_delta=difference,
            known_positive=positive,
            unresolved=unknown,
            recall_delta=endpoints,
        )
    scopes = {}
    for scope in ("all", "image", "text"):
        current = [row for row in rows.values() if scope in ("all", row["mode"])]
        known = [tuple(row["recall_delta"]) for row in current if row["known_positive"]]
        optional = [
            tuple(row["recall_delta"])
            for row in current
            if not row["known_positive"] and row["unresolved"]
        ]
        net = sum(row["hit_delta"] for row in current)
        scopes[scope] = dict(
            queries=len(current),
            net_top10_hits=net,
            precision_at_10_delta=net / (10 * len(current)),
            recall_at_10_delta=optional_mean_bounds(known, optional),
        )
    return dict(
        scopes=scopes,
        query_changes=rows,
        recall_denominator="all observed mixed labels, with unknown completions",
    )


def unreviewed_image_changes(
    evidence: Evidence, manifest: dict, orders: tuple[dict, dict]
) -> list[dict]:
    """Collect every image Top10 change lacking an existing human decision.

    Args:
        evidence: Reader restricted to registered development inputs.
        manifest: Shared gallery and queries with all preceding human judgments.
        orders: Original and candidate full-gallery rankings.

    Returns:
        Deterministically ordered pairs displaying only exact originals and titles.
    """
    catalog = {row["id"]: row for row in manifest["catalog"]}
    before, after = orders
    tasks = []
    for key, query in manifest["query_manifest"].items():
        if query["mode"] != "image":
            continue
        for cid in sorted(set(before[key][:10]) ^ set(after[key][:10])):
            if query["judgments"].get(cid, {}).get("human"):
                continue
            doc = catalog[cid]
            tasks.append(
                dict(
                    task_id=hashlib.sha256(
                        f"{key}\n{cid}\nprimary-recall-complete".encode()
                    ).hexdigest()[:24],
                    query_id=key,
                    candidate_id=cid,
                    query=dict(
                        text=None, image_path=str(_checked_media(evidence, query))
                    ),
                    candidate=dict(
                        title=doc["title"],
                        image_path=str(_checked_media(evidence, doc)),
                    ),
                )
            )
    return sorted(tasks, key=lambda row: row["task_id"])
