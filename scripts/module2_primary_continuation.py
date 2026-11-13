"""Replay caption/anchor safeguards and prepare complete shared Top10 human labels."""

from __future__ import annotations

import argparse
import hashlib
import math

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_shared_evaluation import (
    confirmed_change_metrics,
    precision_delta,
    unreviewed_image_changes,
)
from module2_shared_gallery_fusion_review import _apply_answers
from module2_shared_gallery_precision import worker_lock
from module2_shared_gallery_retrieval import _checked_media, _score
from module2_shared_primary_recall import OUTPUT as PRIMARY
from module2_shared_primary_recall import PARENT, _data, _visual_head_order
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.retrieval_safeguards import (
    compound_caption_conflict,
    guarded_top20,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head

OUTPUT = PRIMARY.parent / "primary_safeguards"


def reviewed_inputs(evidence: Evidence) -> tuple[dict, dict, dict]:
    """Load the frozen primary version and all 144 explicitly validated human answers.

    Args:
        evidence: Guarded reader for current development files.

    Returns:
        Shared query/gallery manifest, pre-image-change baseline, and primary ranks.
    """
    manifest, baseline, _ = _data(evidence)
    result = evidence.json(PRIMARY / "results.json")
    for name, digest in result["inputs"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    ranks = evidence.json(PRIMARY / "rankings.json", result["rankings_sha256"])
    folder = PRIMARY / "human_review"
    receipt = evidence.json(folder / "manifest.json")
    tasks = evidence.json(folder / "tasks.private.json", receipt["tasks_sha256"])
    if tasks != unreviewed_image_changes(evidence, manifest, (baseline, ranks)):
        raise ValueError(
            "The 144 reviewed relations differ from the frozen image changes"
        )
    imported = evidence.json(folder / "import_receipt.json")
    for name, digest in imported["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    progress = _apply_answers(evidence, folder, tasks, manifest["query_manifest"])
    if not progress["complete"] or any(
        manifest["query_manifest"][t["query_id"]]["judgments"][t["candidate_id"]][
            "grade"
        ]
        not in (0, 1)
        for t in tasks
    ):
        raise ValueError("All original image swap decisions must be explicitly binary")
    return manifest, baseline, ranks


def total_top10_tasks(evidence: Evidence, manifest: dict, ranks: dict) -> list[dict]:
    """Collect every remaining unreviewed selected Top10 relation across all queries.

    Args:
        evidence: Reader validating exact original query and candidate images.
        manifest: Shared query/gallery manifest after historic human imports.
        ranks: Frozen selected full-gallery ranking for each query.

    Returns:
        Unique blinded tasks for the complete outstanding overall precision sample.
    """
    catalog = {row["id"]: row for row in manifest["catalog"]}
    tasks = []
    for key, query in manifest["query_manifest"].items():
        for cid in ranks[key][:10]:
            label = query["judgments"].get(cid, {})
            if label.get("human") and label.get("grade") in (0, 1):
                continue
            doc = catalog[cid]
            tasks.append(
                dict(
                    task_id=hashlib.sha256(
                        f"{key}\n{cid}\nshared-total-top10".encode()
                    ).hexdigest()[:24],
                    query_id=key,
                    candidate_id=cid,
                    query=dict(
                        text=query.get("text"),
                        image_path=(
                            str(_checked_media(evidence, query))
                            if query["mode"] == "image"
                            else None
                        ),
                    ),
                    candidate=dict(
                        title=doc["title"],
                        image_path=str(_checked_media(evidence, doc)),
                    ),
                )
            )
    return sorted(tasks, key=lambda row: row["task_id"])


def _replay(evidence: Evidence, manifest: dict, baseline: dict, current: dict) -> dict:
    protocol = evidence.json(PRIMARY / "protocol.json")
    for name, digest in protocol["files"].items():
        evidence.digest(PROJECT_ROOT / name, digest)
    captions = evidence.json(PARENT / "query_captions.json")["rows"]
    variants = {
        name: dict(current)
        for name in (
            "original",
            "caption_guard",
            "anchor_guard",
            "combined",
            "uniform_keep3",
        )
    }
    observations = []
    for key, job in protocol["jobs"].items():
        if not job["active"]:
            continue
        path = PRIMARY / "scores" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        saved = evidence.json(path)
        if (
            saved["identity"] != sha256_file(PRIMARY / "protocol.json")
            or set(saved["scores"]) != {d["id"] for d in job["documents"]}
            or not all(math.isfinite(v) for v in saved["scores"].values())
        ):
            raise ValueError("Actual bound Top20 scores must be complete")
        scores = saved["scores"]
        raw = rerank_head(job["proposed"], scores, 20)
        conflict = compound_caption_conflict(captions[key]["value"])
        anchor = guarded_top20(baseline[key], job["proposed"], scores)
        variants["anchor_guard"][key] = anchor
        prefix = baseline[key][:3]
        variants["uniform_keep3"][key] = _visual_head_order(
            baseline[key], prefix + [cid for cid in raw if cid not in set(prefix)]
        )
        if conflict:
            variants["caption_guard"][key] = baseline[key]
        variants["combined"][key] = baseline[key] if conflict else anchor
        observations.append(
            dict(
                query_id=key,
                caption_conflict=conflict,
                score_consistent_third=scores[baseline[key][2]]
                >= min(scores[cid] for cid in baseline[key][:2]),
                top10_changed=set(current[key][:10])
                != set(variants["combined"][key][:10]),
            )
        )
    write_json(OUTPUT / "decision_observations.json", observations)
    return variants


def _run(evidence: Evidence) -> None:
    manifest, baseline, current = reviewed_inputs(evidence)
    protocol = dict(
        role="exposed_development_after_complete_human_swap_review",
        caption_guard="If a noun ending in bag is parsed as a non-bag object while its appearance clause names a bag, abstain from typed augmentation and use original image order.",
        anchor_guard="Keep original first two; retain original third when its actual score is at least the smaller score of those protected two. Order selected Top10 by original visual order.",
        selection="Development-selected following human error analysis; not independent validation",
        parent_rankings_sha256=sha256_file(PRIMARY / "rankings.json"),
        source_sha256={
            name: sha256_file(PROJECT_ROOT / name)
            for name in (
                "scripts/module2_primary_continuation.py",
                "src/multimodal_ecommerce/retrieval_safeguards.py",
            )
        },
        human_inputs=dict(evidence.files),
        formal_metrics=None,
    )
    path = OUTPUT / "protocol.json"
    if path.exists():
        if evidence.json(path) != protocol:
            raise ValueError("Preserve the frozen continuation experiment")
    else:
        write_json(path, protocol)
    variants = _replay(evidence, manifest, baseline, current)
    selected = variants["combined"]
    queries = manifest["query_manifest"]
    metrics = {name: _score(order, queries) for name, order in variants.items()}
    old, new = metrics["original"], metrics["combined"]
    nonregression = all(
        new["scopes"][scope]["ranges"][metric][i] + 1e-12 >= interval[i]
        for scope in ("all", "image", "text")
        for metric, interval in old["scopes"][scope]["ranges"].items()
        for i in (0, 1)
    )
    if not nonregression:
        raise ValueError(
            "Retain the original candidate when an interval endpoint regresses"
        )
    write_json(OUTPUT / "variant_rankings.json", variants)
    write_json(OUTPUT / "rankings.json", selected)
    write_json(
        OUTPUT / "results.json",
        dict(
            role="exposed_development_caption_and_anchor_safeguards",
            products=len(manifest["catalog"]),
            queries=len(queries),
            protocol_sha256=sha256_file(path),
            rankings_sha256=sha256_file(OUTPUT / "rankings.json"),
            variants=metrics,
            candidate=metrics["combined"],
            baseline=_score(baseline, queries),
            confirmed_primary_gain=confirmed_change_metrics(baseline, current, queries),
            confirmed_revision_gain=confirmed_change_metrics(
                current, selected, queries
            ),
            confirmed_total_image_gain=confirmed_change_metrics(
                baseline, selected, queries
            ),
            paired_precision=precision_delta(current, selected, queries),
            all_interval_endpoints_nonregressing=nonregression,
            reused_model_scores=600,
            new_model_scores=0,
            inputs=dict(evidence.files),
            formal_metrics=None,
        ),
    )
    tasks = total_top10_tasks(evidence, manifest, selected)
    folder = OUTPUT / "human_review"
    task_path = folder / "tasks.private.json"
    if task_path.exists() and evidence.json(task_path) != tasks:
        raise ValueError("Preserve the frozen complete overall Top10 review")
    if not task_path.exists():
        write_json(task_path, tasks)
    write_json(
        folder / "manifest.json",
        dict(
            tasks_sha256=sha256_file(task_path),
            rankings_sha256=sha256_file(OUTPUT / "rankings.json"),
            expected=len(tasks),
            role="complete_remaining_overall_top10",
            tasks_by_mode={
                mode: sum(queries[t["query_id"]]["mode"] == mode for t in tasks)
                for mode in ("image", "text")
            },
            inputs=dict(evidence.files),
        ),
    )
    render_review(
        tasks,
        folder,
        title="模块二 · 总集前十完整人工复核",
        instructions="本批补齐所有 55 条查询前十中尚无明确人工判断的关系，用于计算总体 P@10。按商品主体、用途和明确属性判断并写理由，每批 CSV 保存到本目录 answers。证据不足可选 uncertain。",
    )
    print(
        f"Replayed safeguards on all {len(queries)} queries; complete remaining Top10 tasks: {len(tasks)}."
    )


def main() -> None:
    """Run a cache-only reviewed continuation and create its total precision packet."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    with worker_lock():
        _run(evidence)


if __name__ == "__main__":
    main()
