"""Recall image candidates supported by public sale-item profiles and titles."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import FOCUS, Evidence, load
from module2_shared_evaluation import (
    confirmed_change_metrics,
    precision_delta,
    recall_ceiling,
    remaining_subject_tasks,
    unreviewed_image_changes,
)
from module2_shared_gallery_fusion_review import _apply_answers
from module2_shared_gallery_precision import ROOT, worker_lock
from module2_shared_gallery_retrieval import _checked_media, _score
from module2_shared_subject_recall import OUTPUT as PARENT
from module2_shared_subject_recall import _context, _read_scores, _scores
from multimodal_ecommerce.business_constraints import ENTITY_RULES, canonical
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.head_guard import protected_prefix
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.intent_retrieval import EXTRA_TYPES, main_type, public_entity
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head
from multimodal_ecommerce.visual_query import DescribedImageReranker, caption_text
from paper_fusion_experiment import ROLE

OUTPUT = ROOT / "primary_recall"
POOL = 20
ANCHORS = 10


def _data(evidence: Evidence) -> tuple[dict, dict, dict]:
    manifest, _ = _context(evidence)
    parent = evidence.json(PARENT / "results.json")
    protocol = evidence.json(PARENT / "protocol.json", parent["protocol_sha256"])
    baseline = evidence.json(PARENT / "rankings.json", parent["rankings_sha256"])
    queries = manifest["query_manifest"]
    catalog_ids = {row["id"] for row in manifest["catalog"]}
    if set(baseline) != set(queries) or any(
        len(baseline[key]) != query["expected_pairs"]
        or len(baseline[key]) != len(set(baseline[key]))
        or set(baseline[key]) != catalog_ids - set(query["exclusions"])
        for key, query in queries.items()
    ):
        raise ValueError("Every shared query must rank the complete eligible gallery")
    original = {key: job["baseline"] for key, job in protocol["jobs"].items()}
    expected = remaining_subject_tasks(evidence, manifest, original, baseline)
    folder = PARENT / "text_complete/human_review"
    progress = evidence.json(folder / "progress.json")
    path = folder / "tasks.private.json"
    tasks = evidence.json(
        path, progress["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
    )
    if tasks != expected:
        raise ValueError(
            "The completed text review refers to different candidate changes"
        )
    for path in (folder / "answers").glob("*.csv"):
        evidence.digest(
            path, progress["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
        )
    checked = _apply_answers(evidence, folder, tasks, manifest["query_manifest"])
    if not checked["complete"] or checked["completed"] != progress["reviewed"]:
        raise ValueError("All text difference reviews must be complete")
    return manifest, baseline, protocol


def _profiles(evidence: Evidence, catalog: dict) -> dict:
    pins = evidence.json(FOCUS / "completion_receipt.json")["files"]
    records = evidence.json(FOCUS / "index/records.json", pins["index/records.json"])
    descriptions = evidence.json(
        FOCUS / "index/profile_texts.json", pins["index/profile_texts.json"]
    )
    profiles = {}
    for row, text in zip(records, descriptions, strict=True):
        cid = row["id"]
        if (
            row["image_sha256"] != catalog[cid]["image_sha256"]
            or row["title"] != catalog[cid]["title"]
        ):
            raise ValueError(
                "Indexed sale-item observation has different original input"
            )
        profiles[cid] = main_type(text)
    paths = [ROLE / "profile_inputs.public.json", ROLE / "profiles.json"]
    role_pins = evidence.json(ROLE / "completion_receipt.json")["files"]
    sources, values = [
        evidence.json(path, role_pins[path.relative_to(PROJECT_ROOT).as_posix()])
        for path in paths
    ]
    identities = {row["key"]: row for row in sources if row["kind"] == "gallery"}
    for row in values:
        cid, value = row["key"], row["value"]
        if cid not in identities:
            continue
        original = identities[cid]
        if (
            original["source_sha256"] != catalog[cid]["image_sha256"]
            or original["public_title"] != catalog[cid]["title"]
        ):
            raise ValueError("Role profile is not aligned to the catalog")
        if not value["primary_unclear"]:
            profiles[cid] = public_entity(value["sale_item"])
    supported = {}
    for cid, kind in profiles.items():
        title = canonical(catalog[cid]["title"])
        if kind and any(
            kind == family and any(alias in title for alias in aliases)
            for family, aliases in [*EXTRA_TYPES, *ENTITY_RULES]
        ):
            supported[cid] = kind
    return supported


def _prepare(evidence: Evidence) -> None:
    manifest, baseline, parent = _data(evidence)
    for name, digest in parent["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    catalog = {row["id"]: row for row in manifest["catalog"]}
    kinds = _profiles(evidence, catalog)
    captions = evidence.json(PARENT / "query_captions.json")["rows"]
    jobs = {}
    for key, query in manifest["query_manifest"].items():
        if query["mode"] != "image":
            continue
        old = parent["jobs"][key]
        value = captions[key]["value"]
        description = caption_text(value)
        if (
            description != old["description"]
            or str(_checked_media(evidence, query)) != old["query_image"]
        ):
            raise ValueError(
                "The reused query context differs from the original scorer"
            )
        kind = public_entity(value["product"]) if description else None
        candidates = [
            cid for cid in baseline[key][ANCHORS:] if kind and kinds.get(cid) == kind
        ]
        head = baseline[key][:ANCHORS] + candidates[: POOL - ANCHORS]
        head += [cid for cid in baseline[key] if cid not in set(head)][
            : POOL - len(head)
        ]
        proposed = head + [cid for cid in baseline[key] if cid not in set(head)]
        jobs[key] = dict(
            query_image=old["query_image"],
            query_text=None,
            description=description,
            query_kind=kind,
            active=bool(candidates),
            original=baseline[key],
            proposed=proposed,
            documents=[
                dict(
                    id=cid,
                    image_path=str(_checked_media(evidence, catalog[cid])),
                    text=catalog[cid]["title"],
                )
                for cid in head
            ],
        )
    write_json(
        OUTPUT / "protocol.json",
        dict(
            role="exposed_development_primary_type_image_recall",
            policy="Keep original Top10; add up to ten nearest full-image candidates whose sale-item profile and title support the query type; fill from original order.",
            prefix_protected=2,
            pool_size=POOL,
            jobs=jobs,
            candidate_type_support=kinds,
            parent_protocol_sha256=sha256_file(PARENT / "protocol.json"),
            instruction=BUSINESS_INSTRUCTION,
            model_sha256=parent["model_sha256"],
            files=evidence.files,
        ),
    )
    proposed = {
        key: jobs[key]["proposed"] if key in jobs and jobs[key]["active"] else order
        for key, order in baseline.items()
    }
    write_json(
        OUTPUT / "recall_results.json",
        dict(
            baseline=_score(baseline, manifest["query_manifest"]),
            candidate_pool=_score(proposed, manifest["query_manifest"]),
            active_queries=sum(job["active"] for job in jobs.values()),
            supported_products=len(kinds),
            formal_metrics=None,
        ),
    )
    print(
        f"Frozen {len(jobs)} image jobs, {sum(j['active'] for j in jobs.values())} active, {len(kinds)} supported products."
    )


def _visual_head_order(original: list[str], selected: list[str]) -> list[str]:
    """Order the selected Top10 by original visual similarity without changing membership.

    Args:
        original: Original full-gallery visual order with protected leading results.
        selected: Full order whose Top10 was selected by the semantic reranker.

    Returns:
        The same selected Top10 in visual order, followed by the unchanged tail.
    """
    if (
        len(original) < 10
        or len(original) != len(set(original))
        or len(selected) != len(set(selected))
        or set(original) != set(selected)
        or original[:2] != selected[:2]
    ):
        raise ValueError("Unique same-gallery orders with protected first two required")
    head = set(selected[:10])
    return [cid for cid in original if cid in head] + selected[10:]


def _evaluate(evidence: Evidence, protocol: dict, scores: dict) -> None:
    manifest, baseline, parent = _data(evidence)
    ranks = dict(baseline)
    for key, job in protocol["jobs"].items():
        if job["active"]:
            ranks[key] = protected_prefix(
                baseline[key], rerank_head(job["proposed"], scores[key], POOL), 2
            )
    control = dict(ranks)
    ranks = {
        key: (
            _visual_head_order(baseline[key], order)
            if manifest["query_manifest"][key]["mode"] == "image"
            else order
        )
        for key, order in control.items()
    }
    control_path = OUTPUT / "reranker_rankings.json"
    write_json(control_path, control)
    path = OUTPUT / "rankings.json"
    write_json(path, ranks)
    write_json(
        OUTPUT / "results.json",
        dict(
            role="exposed_development_complete_primary_recall",
            protocol_sha256=sha256_file(OUTPUT / "protocol.json"),
            rankings_sha256=sha256_file(path),
            source_sha256=sha256_file(Path(__file__)),
            reranker_order_control_sha256=sha256_file(control_path),
            reranker_order_control=_score(control, manifest["query_manifest"]),
            selected_policy="Semantic reranker selects Top10 with original first two protected; display selected image results in original visual order. Text uses the verified subject-recall policy.",
            policy_selection="Selected on exposed development labels; requires complete image swap review and independent acceptance.",
            adoption="candidate_pending_complete_image_swap_review",
            products=len(manifest["catalog"]),
            queries=len(ranks),
            active_queries=sum(job["active"] for job in protocol["jobs"].values()),
            baseline=_score(baseline, manifest["query_manifest"]),
            candidate=_score(ranks, manifest["query_manifest"]),
            paired_precision=precision_delta(
                baseline, ranks, manifest["query_manifest"]
            ),
            optimistic_ideal_recall_at_10=recall_ceiling(manifest["query_manifest"]),
            human_only_ideal_recall_at_10=recall_ceiling(
                {
                    key: {
                        **query,
                        "judgments": {
                            cid: row
                            for cid, row in query["judgments"].items()
                            if row.get("human")
                        },
                    }
                    for key, query in manifest["query_manifest"].items()
                }
            ),
            confirmed_text_improvement=confirmed_change_metrics(
                {key: job["baseline"] for key, job in parent["jobs"].items()},
                baseline,
                manifest["query_manifest"],
            ),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )


def _run(evidence: Evidence) -> None:
    path = OUTPUT / "protocol.json"
    protocol = evidence.json(path)
    identity = sha256_file(path)
    if (
        protocol["model_sha256"]
        != sha256_file(PROJECT_ROOT / "configs/reranker_model.json")
        or protocol["instruction"] != BUSINESS_INSTRUCTION
    ):
        raise ValueError("Reranker configuration changed")
    for name, digest in protocol["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    parent = evidence.json(PARENT / "protocol.json", protocol["parent_protocol_sha256"])
    previous = _read_scores(evidence, parent, protocol["parent_protocol_sha256"])
    scores: dict = {}
    counts: Counter[str] = Counter()
    model = DescribedImageReranker()
    try:
        for key, job in protocol["jobs"].items():
            if not job["active"]:
                continue
            old = parent["jobs"][key]
            if any(
                job[field] != old[field]
                for field in ("query_image", "query_text", "description")
            ):
                raise ValueError("The reused query context changed")
            old_docs = {row["id"]: row for row in old["documents"]}
            cached = (
                OUTPUT / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            )
            if not cached.exists():
                reuse = {}
                for row in job["documents"]:
                    cid = row["id"]
                    if cid in old_docs:
                        if row != old_docs[cid]:
                            raise ValueError(
                                "The reused candidate image or title changed"
                            )
                        reuse[cid] = previous[key][cid]
                write_json(cached, dict(identity=identity, scores=reuse))
            require_data_access([cached], "development", evidence.registry)
            saved = load(cached)
            counts["new_scores"] += POOL - len(saved["scores"])
            counts["reused_scores"] += len(saved["scores"])
            scores[key] = _scores(
                evidence, model, job, (OUTPUT / "scores", identity), key
            )
            evidence.digest(cached)
            print(
                f"Full-model image Top20 complete: {len(scores)}/{sum(j['active'] for j in protocol['jobs'].values())}.",
                flush=True,
            )
        runtime = model.diagnostics()
    finally:
        model.close()
    write_json(OUTPUT / "runtime.json", dict(**counts, runtime=runtime))
    _evaluate(evidence, protocol, scores)


def _rescore(evidence: Evidence) -> None:
    protocol = evidence.json(OUTPUT / "protocol.json")
    identity = sha256_file(OUTPUT / "protocol.json")
    for name, digest in protocol["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    scores = {}
    for key, job in protocol["jobs"].items():
        if not job["active"]:
            continue
        path = OUTPUT / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        row = evidence.json(path)
        if (
            row["identity"] != identity
            or set(row["scores"]) != {doc["id"] for doc in job["documents"]}
            or not all(math.isfinite(v) for v in row["scores"].values())
        ):
            raise ValueError("Complete bound Top20 scores required")
        scores[key] = row["scores"]
    _evaluate(evidence, protocol, scores)


def _review(evidence: Evidence, score_only: bool) -> None:
    manifest, baseline, _ = _data(evidence)
    result = evidence.json(OUTPUT / "results.json")
    ranks = evidence.json(OUTPUT / "rankings.json", result["rankings_sha256"])
    folder = OUTPUT / "human_review"
    tasks = unreviewed_image_changes(evidence, manifest, (baseline, ranks))
    path = folder / "tasks.private.json"
    receipt_path = folder / "manifest.json"
    if path.exists():
        receipt = evidence.json(receipt_path)
        saved = evidence.json(path, receipt["tasks_sha256"])
        if (
            tasks != saved
            or receipt["rankings_sha256"] != result["rankings_sha256"]
            or receipt["baseline_rankings_sha256"]
            != sha256_file(PARENT / "rankings.json")
            or receipt["protocol_sha256"] != result["protocol_sha256"]
        ):
            raise ValueError("Preserve the frozen complete image difference review")
        for name, digest in receipt["inputs"].items():
            evidence.digest(PROJECT_ROOT / name, digest)
    elif score_only:
        raise ValueError("Prepare the complete image difference review first")
    else:
        write_json(path, tasks)
        write_json(
            receipt_path,
            dict(
                tasks_sha256=sha256_file(path),
                rankings_sha256=result["rankings_sha256"],
                baseline_rankings_sha256=sha256_file(PARENT / "rankings.json"),
                protocol_sha256=result["protocol_sha256"],
                expected=len(tasks),
                entering=sum(
                    t["candidate_id"] in ranks[t["query_id"]][:10] for t in tasks
                ),
                queries=len({task["query_id"] for task in tasks}),
                inputs={
                    name: digest
                    for name, digest in evidence.files.items()
                    if name
                    != (OUTPUT / "results.json").relative_to(PROJECT_ROOT).as_posix()
                },
            ),
        )
    evidence.digest(path)
    if not score_only:
        render_review(
            tasks,
            folder,
            title="模块二 · 图搜主售商品补召回完整换位复核",
            instructions="核对商品主体、用途与明确属性，选择相关、不相关或证据不足并写一句理由。所有图搜前十未人工核对的换位均在这里，CSV 导出到本目录 answers 文件夹。",
        )
    progress = _apply_answers(evidence, folder, tasks, manifest["query_manifest"])
    binary_complete = progress["complete"] and all(
        manifest["query_manifest"][t["query_id"]]["judgments"][t["candidate_id"]][
            "grade"
        ]
        in (0, 1)
        for t in tasks
    )
    write_json(
        folder / "progress.json",
        dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            binary_complete=binary_complete,
            baseline=_score(baseline, manifest["query_manifest"]),
            candidate=_score(ranks, manifest["query_manifest"]),
            paired_precision=precision_delta(
                baseline, ranks, manifest["query_manifest"]
            ),
            confirmed_change=(
                confirmed_change_metrics(baseline, ranks, manifest["query_manifest"])
                if binary_complete
                else None
            ),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print(
        f"Complete image swap review: {progress['completed']}/{progress['expected']}."
    )


def main() -> None:
    """Freeze, run or evaluate same-gallery image primary-type recall."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "run", "evaluate", "review", "score-review")
    )
    args = parser.parse_args()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        if args.phase == "prepare":
            if (OUTPUT / "protocol.json").exists():
                raise FileExistsError("Preserve the frozen primary recall plan")
            _prepare(evidence)
        elif args.phase == "run":
            _run(evidence)
        elif args.phase == "evaluate":
            _rescore(evidence)
        else:
            _review(evidence, args.phase == "score-review")


if __name__ == "__main__":
    main()
