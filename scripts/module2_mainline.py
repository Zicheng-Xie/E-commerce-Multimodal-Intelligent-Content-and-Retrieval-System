"""Verify and continue the user-selected module-two development candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_human_review_inputs import annotation_reasons, import_precision_answers
from module2_mainline_reports import (
    evaluation_transition,
    label_fingerprints,
    publish_evaluation,
)
from module2_primary_continuation import OUTPUT as REVISION
from module2_primary_continuation import reviewed_inputs, total_top10_tasks
from module2_shared_evaluation import (
    confirmed_change_metrics,
    precision_delta,
    recall_ceiling,
    unreviewed_image_changes,
)
from module2_shared_gallery_fusion_review import _apply_answers
from module2_shared_gallery_precision import ROOT, worker_lock
from module2_shared_gallery_retrieval import _score
from module2_shared_primary_recall import OUTPUT as SELECTED
from module2_shared_primary_recall import _data, _review
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.human_retrieval_evaluation import (
    evaluate_human_retrieval,
    retrieval_error_queue,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

CONFIG = PROJECT_ROOT / "configs/module2_development_mainline.json"
OUTPUT = ROOT / "mainline"


def _inputs(evidence: Evidence, config: dict) -> tuple[dict, dict, dict, list]:
    if config["artifact_root"] != SELECTED.relative_to(PROJECT_ROOT).as_posix():
        raise ValueError("This continuation entry is bound to primary_recall")
    for name, digest in config["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Selected development implementation changed: {name}")
    for name, digest in config["frozen_files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    result = evidence.json(SELECTED / "results.json")
    for name, digest in result["inputs"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    ranks = evidence.json(SELECTED / "rankings.json", result["rankings_sha256"])
    evidence.digest(SELECTED / "protocol.json", result["protocol_sha256"])
    manifest, baseline, _ = _data(evidence)
    queries = manifest["query_manifest"]
    if set(ranks) != set(queries):
        raise ValueError("The mainline must include every shared development query")
    for key, query in queries.items():
        order = ranks[key]
        if (
            len(order) != query["expected_pairs"]
            or len(order) != len(set(order))
            or set(order) != set(baseline[key])
            or order[:2] != baseline[key][:2]
            or query["mode"] == "text"
            and order != baseline[key]
        ):
            raise ValueError(
                "Selected complete-gallery and protected-head rules differ"
            )
    folder = SELECTED / "human_review"
    receipt = evidence.json(folder / "manifest.json")
    tasks = evidence.json(folder / "tasks.private.json", receipt["tasks_sha256"])
    if receipt["rankings_sha256"] != result[
        "rankings_sha256"
    ] or tasks != unreviewed_image_changes(evidence, manifest, (baseline, ranks)):
        raise ValueError("The frozen review does not match the selected predictions")
    for name, digest in receipt["inputs"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    if config.get("revision"):
        if (
            config["revision"]["artifact_root"]
            != REVISION.relative_to(PROJECT_ROOT).as_posix()
        ):
            raise ValueError("Unknown continuation of the selected primary version")
        manifest, baseline, _ = reviewed_inputs(evidence)
        result = evidence.json(REVISION / "results.json")
        for name, digest in result["inputs"].items():
            evidence.digest(PROJECT_ROOT / name, digest)
        evidence.digest(REVISION / "protocol.json", result["protocol_sha256"])
        ranks = evidence.json(REVISION / "rankings.json", result["rankings_sha256"])
        queries = manifest["query_manifest"]
        if set(ranks) != set(queries) or any(
            len(ranks[key]) != query["expected_pairs"]
            or len(ranks[key]) != len(set(ranks[key]))
            or set(ranks[key]) != set(baseline[key])
            or ranks[key][:2] != baseline[key][:2]
            or query["mode"] == "text"
            and ranks[key] != baseline[key]
            for key, query in queries.items()
        ):
            raise ValueError(
                "The continuation violates the selected gallery or head policy"
            )
        receipt = evidence.json(REVISION / "human_review/manifest.json")
        tasks = evidence.json(
            REVISION / "human_review/tasks.private.json", receipt["tasks_sha256"]
        )
        if receipt["rankings_sha256"] != result[
            "rankings_sha256"
        ] or tasks != total_top10_tasks(evidence, manifest, ranks):
            raise ValueError("The total precision review differs from the continuation")
    return manifest, baseline, ranks, tasks


def _status(evidence: Evidence, config: dict) -> None:
    previous = load(OUTPUT / "status.json") if (OUTPUT / "status.json").exists() else {}
    manifest, baseline, ranks, tasks = _inputs(evidence, config)
    queries = manifest["query_manifest"]
    folder = (REVISION if config.get("revision") else SELECTED) / "human_review"
    progress = (
        import_precision_answers(evidence, folder, tasks, queries)
        if config.get("revision")
        else _apply_answers(evidence, folder, tasks, queries)
    )
    binary_complete = progress["complete"] and all(
        queries[t["query_id"]]["judgments"][t["candidate_id"]]["grade"] in (0, 1)
        for t in tasks
    )
    candidate = _score(ranks, queries)
    changes_verified = all(
        queries[key]["judgments"].get(cid, {}).get("human")
        and queries[key]["judgments"][cid]["grade"] in (0, 1)
        for key in queries
        for cid in set(baseline[key][:10]) ^ set(ranks[key][:10])
    )
    target = load(PROJECT_ROOT / "configs/module2_target_p80_r90.json")
    human = evaluate_human_retrieval(ranks, queries, target)
    reasons = annotation_reasons(evidence, queries)
    errors = retrieval_error_queue(ranks, manifest, reasons)
    target_met = all(
        candidate["scopes"][scope]["ranges"][metric][0] is not None
        and candidate["scopes"][scope]["ranges"][metric][0] >= required
        for scope in ("all", "image", "text")
        for metric, required in (
            ("precision@10", target["precision_at_10_required"]),
            ("recall@10", target["recall_at_10_preferred"]),
        )
    )
    gaps = {}
    pending_pairs = {(t["query_id"], t["candidate_id"]) for t in tasks}
    for scope in ("all", "image", "text"):
        current = [key for key, q in queries.items() if scope in ("all", q["mode"])]
        gaps[scope] = dict(
            queries=len(current),
            candidate_top10_without_binary_human_label=sum(
                not (
                    queries[key]["judgments"].get(cid, {}).get("human")
                    and queries[key]["judgments"][cid]["grade"] in (0, 1)
                )
                for key in current
                for cid in ranks[key][:10]
            ),
            additional_top10_pairs_beyond_current_packet=sum(
                (key, cid) not in pending_pairs
                and not (
                    queries[key]["judgments"].get(cid, {}).get("human")
                    and queries[key]["judgments"][cid]["grade"] in (0, 1)
                )
                for key in current
                for cid in ranks[key][:10]
            ),
        )
    status = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        mainline_id=config["mainline_id"],
        role="user_selected_exposed_development_candidate",
        products=len(manifest["catalog"]),
        queries=len(queries),
        frozen_rankings_sha256=sha256_file(
            (REVISION if config.get("revision") else SELECTED) / "rankings.json"
        ),
        baseline=_score(baseline, queries),
        candidate=candidate,
        candidate_label_basis="mixed human overrides and AI labels; bounds omit AI labeling errors",
        human_evaluation=human,
        label_fingerprints=label_fingerprints(ranks, queries),
        target_contract_sha256=sha256_file(
            PROJECT_ROOT / "configs/module2_target_p80_r90.json"
        ),
        paired_precision=precision_delta(baseline, ranks, queries),
        review=dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            binary_complete=binary_complete,
            binary_completed=progress.get("binary_completed"),
            batches=progress.get("batches"),
            missing_task_ids=progress.get("missing_task_ids"),
            uncertain_task_ids=progress.get("uncertain_task_ids"),
            index=str(folder / "index.html"),
            answers=str(folder / "answers"),
        ),
        confirmed_image_gain=(
            confirmed_change_metrics(baseline, ranks, queries)
            if changes_verified
            else None
        ),
        annotation_gaps=gaps,
        optimistic_ideal_recall_at_10=recall_ceiling(queries),
        human_only_ideal_recall_at_10=recall_ceiling(
            {
                key: {
                    **q,
                    "judgments": {
                        cid: row
                        for cid, row in q["judgments"].items()
                        if row.get("human")
                    },
                }
                for key, q in queries.items()
            }
        ),
        observed_lower_bounds_meet_target=target_met,
        observed_lower_bounds_basis="legacy mixed-label diagnostic only; use human_evaluation for human evidence",
        formal_acceptance=False,
        formal_metrics=None,
        config_sha256=sha256_file(CONFIG),
        inputs=evidence.files,
    )
    status["evaluation_transition"] = evaluation_transition(previous, status)
    status["evaluation_snapshot"] = publish_evaluation(
        OUTPUT, status, errors, evidence.files
    )
    write_json(OUTPUT / "status.json", status)
    if config.get("revision"):
        write_json(
            folder / "progress.json",
            dict(
                reviewed=progress["completed"],
                expected=progress["expected"],
                binary_complete=binary_complete,
                metrics=candidate,
                human_evaluation=human,
                batches=progress.get("batches"),
                evaluation_snapshot=status["evaluation_snapshot"],
                inputs=evidence.files,
                formal_metrics=None,
            ),
        )
    print(
        f"Development mainline: {config['mainline_id']}; {len(manifest['catalog'])} products, {len(queries)} queries."
    )
    print(
        f"Human review: {progress['completed']}/{progress['expected']}; binary complete: {binary_complete}."
    )
    print(f"Status: {OUTPUT / 'status.json'}")
    precision = human["scopes"]["all"]["precision_at_10"]
    print(
        f"Human Top10 labels: {precision['judged']}/{precision['expected']}; exact Precision@10: {precision['value']}."
    )
    print(f"Human evaluation report: {status['evaluation_snapshot']['report_path']}")


def main() -> None:
    """Continue the pinned development version without changing frozen predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("status", "review", "score-review"),
        nargs="?",
        default="status",
    )
    phase = parser.parse_args().phase
    config = load(CONFIG)
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        manifest, _, ranks, tasks = _inputs(evidence, config)
        if phase != "status" and not config.get("revision"):
            _review(evidence, phase == "score-review")
        elif phase == "review":
            render_review(
                tasks,
                REVISION / "human_review",
                title="模块二 · 总集前十完整人工复核",
                instructions="本批补齐 55 条查询前十的剩余人工相关性判断。按原图和原标题判断并写理由，CSV 导出到本目录 answers。",
            )
        _status(evidence, config)


if __name__ == "__main__":
    main()
