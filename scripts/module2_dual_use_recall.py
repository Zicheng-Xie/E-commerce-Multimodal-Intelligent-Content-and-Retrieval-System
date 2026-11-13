"""Compare whole-image recall and two-use filtering on an augmented development gallery."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import paper_sid_rerank as worker
import run  # noqa: F401
from module2_focus_boundary import FOCUS, FRESH, verified
from module2_poster_usage import ROOT as POSTER
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import stable_order
from paper_fusion_experiment import load

ROOT = POSTER / "dual_use_catalog"
REVIEW = ROOT / "human_review"
CAPABILITIES = ("game", "external_power", "same_device")
DEPTH = 20


def both_uses(observation: dict) -> bool:
    """Require explicit support for both functions on the same candidate device.

    Args:
        observation: Image-only judgments and visible evidence.

    Returns:
        Whether all three checks have affirmative evidence.
    """
    return all(observation[field] == "yes" for field in CAPABILITIES) and bool(
        observation["image_evidence"].strip()
    )


def input_plan(registry: dict) -> dict:
    """Freeze development acquisitions, their evidence, and the existing vector bank.

    Args:
        registry: Registered evaluation-data roles.

    Returns:
        Candidate originals, one frozen query, and pinned replay dependencies.
    """
    scan, acquisition, evidence = (
        ROOT / name
        for name in ("metadata_scan.json", "acquisition.json", "image_evidence.json")
    )
    receipt = FOCUS / "completion_receipt.json"
    poster = POSTER / "experiment_protocol.json"
    result = POSTER / "results.json"
    files = [scan, acquisition, evidence, receipt, poster, result]
    require_data_access(files, "development", registry)
    archive, images, descriptions = map(load, (scan, acquisition, evidence))
    if (
        images["metadata_sha256"] != sha256_file(scan)
        or descriptions["acquisition_sha256"] != sha256_file(acquisition)
        or descriptions["title_supplied"]
    ):
        raise ValueError(
            "Candidate metadata, acquisition, or evidence identity changed"
        )
    selected = [row for row in images["attempts"] if row["status"] == "downloaded"]
    by_id = {row["id"]: row for row in archive["selected"]}
    observed = {row["id"]: row for row in descriptions["rows"]}
    if (
        not selected
        or len(selected) != descriptions["completed"]
        or descriptions["completed"] != descriptions["expected"]
        or set(observed) != {row["id"] for row in selected}
    ):
        raise ValueError("Complete image-only evidence is required")
    candidates = []
    for row in selected:
        item = by_id[row["id"]]
        image = ROOT / "media" / row["path"]
        digest = verified(image, row["signature"]["sha256"], registry)
        proof = observed[item["id"]]
        if digest != proof["image_sha256"] or not item["explicit_dual"]:
            raise ValueError("Candidate image or title provenance differs")
        files.append(image)
        candidates.append(
            dict(
                id=item["id"],
                title=item["title"],
                image=str(image),
                eligible=both_uses(proof["value"]),
            )
        )
    protocol = load(poster)
    if len(protocol["jobs"]) != 1:
        raise ValueError("Expected the previously confirmed dual-use query")
    query = protocol["jobs"][0]
    replay = POSTER / (query["qid"] + "_replay.json")
    require_data_access([replay], "development", registry)
    if (
        load(replay)["protocol_sha256"] != sha256_file(poster)
        or query["qid"] not in load(result)["score_receipts"]
    ):
        raise ValueError("The confirmed poster query does not match its replay")
    files.append(replay)
    files.append(Path(query["image"]))
    verified(files[-1], query["image_sha256"], registry)
    hashes = load(receipt)["files"]
    for rel in ("index/manifest.json", "index/records.json", "index/vectors.npz"):
        path = FOCUS / rel
        verified(path, hashes[rel], registry)
        files.append(path)
    views = FRESH / "runs" / query["qid"] / "query_vectors.npz"
    verified(views, protocol["files"][str(views.relative_to(PROJECT_ROOT))], registry)
    files.append(views)
    return dict(
        query=query,
        candidates=candidates,
        query_views=str(views.relative_to(PROJECT_ROOT)),
        files={str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in files},
        role="query_aware_exposed_development",
        source_sha256=sha256_file(Path(__file__)),
        formal_metrics=None,
    )


def recall(plan: dict) -> dict:
    """Encode the new gallery entries and compare three fixed recall methods.

    Args:
        plan: Frozen image, title, and same-device evidence.

    Returns:
        Full augmented-gallery rankings, eligible counts, and real model receipts.
    """
    ids = [row["id"] for row in load(FOCUS / "index/records.json")]
    old_count = len(ids)
    added = [row["id"] for row in plan["candidates"]]
    if len(set(ids + added)) != len(ids) + len(added):
        raise ValueError("The new candidates overlap the original gallery")
    with np.load(FOCUS / "index/vectors.npz", allow_pickle=False) as data:
        whole, titles = data["regions"][:, 0].copy(), data["texts"].copy()
    with np.load(PROJECT_ROOT / plan["query_views"], allow_pickle=False) as data:
        query_image = data["views"][0].copy()
    model = QwenEmbedder()
    try:
        archived = load(FOCUS / "index/manifest.json")["model"]
        for field in (
            "revision",
            "vendor_sha256",
            "max_pixels",
            "query_instruction",
            "document_instruction",
        ):
            if model.config[field] != archived[field]:
                raise ValueError(f"Gallery embedding configuration differs: {field}")
        query_text = model.encode(text=plan["query"]["caption"], query=True)
        for item in plan["candidates"]:
            whole = np.vstack([whole, model.encode(image=Path(item["image"]))])
            titles = np.vstack([titles, model.encode(text=item["title"])])
        runtime = model.diagnostics()
    finally:
        model.close()
    ids.extend(added)
    scores = whole @ query_image
    visual = stable_order(ids, scores)
    fused = stable_order(
        ids,
        0.5 * normalized_ranks(ids, scores)
        + 0.5 * normalized_ranks(ids, titles @ query_text),
    )
    eligible = {row["id"] for row in plan["candidates"] if row["eligible"]}
    strict = [cid for cid in fused if cid in eligible] + [
        cid for cid in fused if cid not in eligible
    ]
    rankings = dict(
        whole_image=visual,
        whole_and_description=fused,
        conjunctive_support_then_fallback=strict,
    )
    return dict(
        original_gallery=old_count,
        added_candidates=len(added),
        image_supported_dual_use_count=len(eligible),
        top10_support={
            m: sum(cid in eligible for cid in order[:10])
            for m, order in rankings.items()
        },
        top20_support={
            m: sum(cid in eligible for cid in order[:DEPTH])
            for m, order in rankings.items()
        },
        rankings=rankings,
        runtime=runtime,
        final_ranking_metrics=None,
        formal_metrics=None,
    )


def review_packet(plan: dict) -> dict:
    """Create a blinded relevance batch covering every newly acquired candidate.

    Args:
        plan: Frozen query and all acquired candidate originals.

    Returns:
        Review task count and digest.
    """
    query = plan["query"]
    tasks = []
    for row in plan["candidates"]:
        digest = hashlib.sha256(
            f'dual-use-catalog|{query["qid"]}|{row["id"]}'.encode()
        ).hexdigest()
        tasks.append(
            dict(
                task_id=digest[:24],
                query_id=query["qid"],
                candidate_id=row["id"],
                grade=None,
                query=dict(
                    text="请判断候选商品是否是同一台既能玩游戏，又能给其他设备充电的商品。",
                    image_path=query["image"],
                ),
                candidate=dict(title=row["title"], image_path=row["image"]),
            )
        )
    tasks.sort(key=lambda row: row["task_id"])
    path = REVIEW / "tasks.private.json"
    if path.exists() and load(path) != tasks:
        raise ValueError("The frozen human review packet changed")
    render_review(
        tasks,
        REVIEW,
        title="模块二 · 双用途商品复核",
        instructions="同一台设备明确同时支持玩游戏和给其他设备充电才选相关。只给自己充电、分开的两件商品或配件选不相关，图片和标题无法判断时选证据不足。导出 CSV 后放入本页同目录 answers 文件夹。",
    )
    write_json(path, tasks)
    return dict(count=len(tasks), sample_sha256=sha256_file(path))


def main() -> None:
    """Run the isolated development recall comparison and prepare human confirmation."""
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        plan = input_plan(registry)
        path = ROOT / "recall_protocol.json"
        if path.exists() and load(path) != plan:
            raise ValueError("The dual-use recall protocol changed")
        if not path.exists():
            write_json(path, plan)
        data = recall(plan)
        write_json(
            ROOT / "recall_results.json",
            dict(
                **data,
                protocol_sha256=sha256_file(path),
                human_review=review_packet(plan),
                frozen_acceptance_model_calls=0,
            ),
        )
        print(data["top10_support"], data["top20_support"], flush=True)


if __name__ == "__main__":
    main()
