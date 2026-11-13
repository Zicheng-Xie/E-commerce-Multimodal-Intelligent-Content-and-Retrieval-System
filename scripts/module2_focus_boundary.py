"""Rescore only ranks ten and eleven of both retained image-search routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import paper_sid_rerank as worker
import run  # noqa: F401
from module2_focus_followup import BASELINES, FOCUS
from module2_focus_followup import ROOT as PARENT
from module2_subject_challengers import INSTRUCTION as SALE_ITEM
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.visual_query import DescribedImageReranker
from paper_fusion_experiment import load

ROOT = PARENT / "sale_item_boundary"
FRESH = PROJECT_ROOT / "artifacts/module2_rank_stability/fresh_query_check"
FRESH_METHOD = "described_cross_pool20"
INSTRUCTION = (
    SALE_ITEM
    + " Treat the query description as a clue to the sale item and check it against the image. "
    + "In a staged outfit or furniture scene, compare the main offered item, not another "
    + "visible part of the scene. Give precedence to the practical use and the object the buyer gets."
)


def verified(path: Path, expected: str, registry: dict) -> str:
    """Check data role before reading the bytes of an archived input.

    Args:
        path: Recorded development artifact or original image.
        expected: Expected SHA256, or empty when freezing a new input.
        registry: Active evaluation data policy.

    Returns:
        Current SHA256 of the allowed and verified file.
    """
    require_data_access([path], "development", registry)
    digest = sha256_file(path)
    if expected and digest != expected:
        raise ValueError(f"Archived input changed: {path}")
    return digest


def plan(registry: dict) -> dict:
    """Build twelve label-free jobs with a fixed adjacent rank comparison.

    Args:
        registry: Active evaluation data policy.

    Returns:
        Frozen scoring plan with fixed descriptions, original images and pairs.
    """
    prior_receipt = FOCUS / "completion_receipt.json"
    parent_receipt = PARENT / "results.json"
    fresh_receipt = FRESH / "evaluation_receipt.json"
    fresh_protocol = FRESH / "inference_protocol.json"
    require_data_access(
        [prior_receipt, parent_receipt, fresh_receipt, fresh_protocol],
        "development",
        registry,
    )
    archived, parent, fresh, protocol = map(
        load, (prior_receipt, parent_receipt, fresh_receipt, fresh_protocol)
    )
    features = FOCUS / "query_enrichment/ranking_features.json"
    source = FOCUS / "inputs.public.json"
    old_ranks = PARENT / "rankings.json"
    fresh_ranks = FRESH / "rankings.json"
    fresh_queries = FRESH / "queries.public.json"
    files = {
        source: archived["files"][source.name],
        features: archived["files"][str(features.relative_to(FOCUS))],
        old_ranks: parent["rankings_sha256"],
        fresh_ranks: fresh["files"][fresh_ranks.name],
        fresh_queries: protocol["files"][str(fresh_queries.relative_to(PROJECT_ROOT))],
    }
    for path, digest in files.items():
        verified(path, digest, registry)
    old, fusion, rankings, stress = map(
        load, (source, features, old_ranks, fresh_ranks)
    )
    documents = {row["id"]: row for row in old["catalog"]}
    jobs = []
    for cohort, queries in (
        ("historical", [query for query in old["queries"] if query["image_path"]]),
        ("stress", load(fresh_queries)),
    ):
        for query in queries:
            qid = query["id"]
            query_path = project_path(query["image_path"])
            if cohort == "historical":
                order = rankings[qid][BASELINES[0]]
                if order != rankings[qid][BASELINES[1]]:
                    raise ValueError("The two image baselines must agree")
                cross = fusion[qid]
                expected_image = ""
            else:
                order = stress[qid][FRESH_METHOD]
                cache = FRESH / "runs" / qid / "cross_encoder.json"
                files[cache] = verified(cache, "", registry)
                cross = load(cache)
                expected_image = protocol["files"][
                    str(query_path.relative_to(PROJECT_ROOT))
                ]
            pool = cross["pool"][:20]
            if len(pool) != 20 or set(order[:20]) != set(pool):
                raise ValueError(f"Frozen complete Top20 differs: {qid}")
            boundary = order[9:11]
            rows = []
            for cid in boundary:
                doc = documents[cid]
                path = project_path(doc["image_path"])
                files[path] = verified(path, "", registry)
                rows.append(dict(id=cid, image_path=str(path), text=doc["title"]))
            files[query_path] = verified(query_path, expected_image, registry)
            jobs.append(
                dict(
                    cohort=cohort,
                    qid=qid,
                    query_image=str(query_path),
                    description=cross["description"],
                    boundary=boundary,
                    documents=rows,
                )
            )
    if len(jobs) != 12 or len({(job["cohort"], job["qid"]) for job in jobs}) != 12:
        raise ValueError("Expected six historical and six stress queries")
    files[Path(__file__).resolve()] = sha256_file(Path(__file__))
    return dict(
        role="exposed_development",
        instruction=INSTRUCTION,
        jobs=jobs,
        files={str(p.relative_to(PROJECT_ROOT)): h for p, h in files.items()},
        total_new_pairs=2 * len(jobs),
        frozen_acceptance_model_calls=0,
    )


def main() -> None:
    """Freeze original evidence and checkpoint actual full-model boundary logits."""
    ROOT.mkdir(parents=True, exist_ok=True)
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = plan(registry)
        frozen = ROOT / "protocol.json"
        if frozen.exists() and load(frozen) != protocol:
            raise ValueError("The frozen boundary protocol changed")
        if not frozen.exists():
            write_json(frozen, protocol)
        digest = sha256_file(frozen)
        client = None
        try:
            for job in protocol["jobs"]:
                output = ROOT / "scores" / job["cohort"] / (job["qid"] + ".json")
                identity = dict(protocol_sha256=digest, boundary=job["boundary"])
                if output.exists():
                    record = load(output)
                    values = record["logits"]
                    if (
                        record["identity"] != identity
                        or len(values) != 2
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError(f"Checkpoint differs: {output}")
                    continue
                if client is None:
                    client = DescribedImageReranker()
                    client.adapter.config = {
                        **client.adapter.config,
                        "instruction": protocol["instruction"],
                    }
                image = Path(job["query_image"])
                previous = len(client.adapter.timings)
                values = (
                    client.score_described(
                        job["documents"],
                        query_image=image,
                        description=job["description"],
                    )
                    if job["description"]
                    else client.score(job["documents"], query_image=image)
                )
                if values.shape != (2,) or not np.isfinite(values).all():
                    raise ValueError("Two finite boundary logits are required")
                write_json(
                    output,
                    dict(
                        identity=identity,
                        logits=values.tolist(),
                        timings=client.adapter.timings[previous:],
                    ),
                )
                print(
                    f"{job['cohort']}/{job['qid']}: scored both boundary candidates",
                    flush=True,
                )
            if client is not None:
                write_json(ROOT / "runtime.json", client.diagnostics())
        finally:
            if client is not None:
                client.close()


if __name__ == "__main__":
    main()
