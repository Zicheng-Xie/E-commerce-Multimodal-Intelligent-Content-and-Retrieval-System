"""Find corroborated sale-item candidates and score a bounded development swap."""

from __future__ import annotations

import math
from pathlib import Path

import paper_sid_rerank as worker
import run  # noqa: F401
from module2_focus_boundary import FRESH, FRESH_METHOD, INSTRUCTION
from module2_focus_boundary import ROOT as BOUNDARY
from module2_focus_boundary import plan as boundary_plan
from module2_focus_boundary import verified
from module2_focus_followup import BASELINES, FOCUS
from module2_focus_followup import ROOT as PARENT
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.intent_retrieval import public_entity
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import ranked_ids
from multimodal_ecommerce.visual_query import DescribedImageReranker
from paper_fusion_experiment import load
from primary_subject_ranker import signature

ROOT = PARENT / "subject_rescue"
RECALL_LIMIT = 60
STRONG_RANK = 10
MIN_SLOT = 5
ALIASES = {
    "valve_cap": ("气门嘴帽", "气嘴帽", "气门帽", "气门嘴盖", "气嘴盖"),
    "suit_set": ("西装套装", "西服套装"),
    "phone_case": ("ipad保护壳", "ipad保护套", "平板保护壳", "平板保护套"),
}


def product_type(value: str) -> str | None:
    """Infer a sale-item family from the subject phrase or original title.

    Args:
        value: Frozen query description, original title, or item image profile.

    Returns:
        Known coarse item family, with compatible protective cases grouped.
    """
    head = value.removeprefix("商品主体：").split("。", 1)[0].casefold()
    for name, aliases in ALIASES.items():
        if any(term in head for term in aliases):
            return name
    return public_entity(head) or signature(head)["family"]


def proposal(order: list[str], fusion: dict, types: dict) -> dict:
    """Select at most one supported candidate and one corroborated mismatch.

    Args:
        order: Complete baseline gallery ranking.
        fusion: Archived local-view and description similarities.
        types: Title and image-profile type for each candidate.

    Returns:
        Abstention reason or deterministic candidate and incumbent evidence.
    """
    subject = product_type(fusion["description"])
    if subject is None:
        return dict(status="unknown_query_subject")
    if set(order) != set(fusion["ids"]):
        raise ValueError("Visual/description retrieval must cover the same gallery")
    roles = {
        cid: title if title == profile else None
        for cid, (title, profile) in types.items()
    }
    wrong = [
        cid
        for cid in reversed(order[MIN_SLOT:10])
        if roles[cid] is not None and roles[cid] != subject
    ]
    if not wrong:
        return dict(status="no_corroborated_top10_conflict", subject=subject)
    visual, caption = (
        {
            cid: rank
            for rank, cid in enumerate(ranked_ids(fusion["ids"], fusion[field]), 1)
        }
        for field in ("multi_scores", "caption_scores")
    )
    matching = [
        cid
        for cid in order[10:]
        if roles[cid] == subject and max(visual[cid], caption[cid]) <= RECALL_LIMIT
    ]
    if not matching:
        return dict(status="no_corroborated_replacement", subject=subject)
    incoming = min(
        matching,
        key=lambda cid: (
            max(visual[cid], caption[cid]),
            visual[cid] + caption[cid],
            cid,
        ),
    )
    out = wrong[0]
    return dict(
        status="score_pair",
        subject=subject,
        incumbent_type=roles[out],
        candidates=[out, incoming],
        original_ranks=[order.index(cid) + 1 for cid in (out, incoming)],
        entrant_support=[visual[incoming], caption[incoming]],
        strong=max(visual[incoming], caption[incoming]) <= STRONG_RANK,
    )


def freeze(registry: dict) -> dict:
    """Pin the same exposed queries plus the aligned full-gallery public evidence.

    Args:
        registry: Project role rules for all accessed data.

    Returns:
        Label-free proposals with original image identities and fixed policies.
    """
    prior = BOUNDARY / "protocol.json"
    receipt = FOCUS / "completion_receipt.json"
    require_data_access([prior, receipt], "development", registry)
    old = load(prior)
    if boundary_plan(registry) != old:
        raise ValueError("The previous twelve-query freeze changed")
    names = (
        "index/records.json",
        "index/profile_texts.json",
        "query_enrichment/ranking_features.json",
    )
    hashes = load(receipt)["files"]
    files = {
        FOCUS / name: verified(FOCUS / name, hashes[name], registry) for name in names
    }
    records, profiles, historical_fusion = (load(FOCUS / name) for name in names)
    by_id = {row["id"]: row for row in records}
    types = {
        doc["id"]: (product_type(doc["title"]), product_type(profile))
        for doc, profile in zip(records, profiles, strict=True)
    }
    old_ranks = load(PARENT / "rankings.json")
    stress_ranks = load(FRESH / "rankings.json")
    proposals = []
    for job in old["jobs"]:
        qid = job["qid"]
        if job["cohort"] == "historical":
            order = old_ranks[qid][BASELINES[0]]
            fusion = historical_fusion[qid]
        else:
            path = FRESH / "runs" / qid / "query_fusion.json"
            files[path] = verified(path, "", registry)
            fusion = load(path)
            order = stress_ranks[qid][FRESH_METHOD]
        if fusion["description"] != job["description"]:
            raise ValueError(f"Recall and rerank descriptions differ: {qid}")
        choice = proposal(order, fusion, types)
        docs = []
        for cid in choice.get("candidates", []):
            doc = by_id[cid]
            image = project_path(doc["image_path"])
            files[image] = verified(image, "", registry)
            docs.append(dict(id=cid, image_path=str(image), text=doc["title"]))
        proposals.append(
            dict(
                cohort=job["cohort"],
                qid=qid,
                query_image=job["query_image"],
                description=job["description"],
                documents=docs,
                **choice,
            )
        )
    files[prior] = sha256_file(prior)
    files[Path(__file__).resolve()] = sha256_file(Path(__file__))
    return dict(
        role="exposed_development",
        instruction=INSTRUCTION,
        policy=dict(
            first_protected=MIN_SLOT,
            recall_limit=RECALL_LIMIT,
            strict_support_rank=STRONG_RANK,
        ),
        proposals=proposals,
        files={str(p.relative_to(PROJECT_ROOT)): h for p, h in files.items()},
        expected_new_pairs=sum(len(p["documents"]) for p in proposals),
        frozen_acceptance_model_calls=0,
    )


def score(protocol: dict) -> None:
    """Run full local Qwen for each proposed swap, preserving completed scores.

    Args:
        protocol: Frozen proposed pairs and original model instruction.
    """
    client = None
    digest = sha256_file(ROOT / "protocol.json")
    try:
        for job in protocol["proposals"]:
            if job["status"] != "score_pair":
                continue
            path = ROOT / "scores" / job["cohort"] / (job["qid"] + ".json")
            identity = dict(protocol_sha256=digest, candidates=job["candidates"])
            if path.exists():
                cache = load(path)
                if (
                    cache["identity"] != identity
                    or len(cache["logits"]) != 2
                    or not all(map(math.isfinite, cache["logits"]))
                ):
                    raise ValueError(f"Invalid prior score cache: {path}")
                continue
            if client is None:
                client = DescribedImageReranker()
                client.adapter.config = {
                    **client.adapter.config,
                    "instruction": protocol["instruction"],
                }
            previous = len(client.adapter.timings)
            logits = client.score_described(
                job["documents"],
                query_image=Path(job["query_image"]),
                description=job["description"],
            )
            write_json(
                path,
                dict(
                    identity=identity,
                    logits=logits.tolist(),
                    timings=client.adapter.timings[previous:],
                ),
            )
            print(
                f"{job['cohort']}/{job['qid']}: verified candidate and incumbent scored",
                flush=True,
            )
        if client is not None:
            write_json(ROOT / "runtime.json", client.diagnostics())
    finally:
        if client is not None:
            client.close()


def main() -> None:
    """Prepare or resume real scoring for the frozen development proposal set."""
    ROOT.mkdir(parents=True, exist_ok=True)
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        path = ROOT / "protocol.json"
        protocol = freeze(registry)
        if path.exists() and load(path) != protocol:
            raise ValueError("Subject-rescue protocol changed")
        if not path.exists():
            write_json(path, protocol)
        print(f"Frozen {protocol['expected_new_pairs']} real pair scores.", flush=True)
        score(protocol)


if __name__ == "__main__":
    main()
