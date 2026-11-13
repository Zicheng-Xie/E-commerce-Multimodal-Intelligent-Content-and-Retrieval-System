"""Rescore contested Top10 slots with an explicit sale-item instruction."""

from __future__ import annotations

from pathlib import Path

import run  # noqa: F401
from module2_expand_pool import ROOT as EXPANDED
from module2_expand_pool import TARGET, WORKER
from multimodal_ecommerce.business_retrieval import BusinessReranker
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import SIMPLE, load
from simple_baseline_cycle import guard

ROOT = TARGET / "subject_challengers"
BASE = "top30_corroborated_subject"
INSTRUCTION = (
    "Judge whether the candidate is relevant to the product actually offered in the query image. "
    + "Read visible offer text when it identifies the sold item. A device, bag, furniture piece "
    + "or other object shown to demonstrate an insert, cover, holder or spare part is the host, "
    + "and its prominence does not make it the offered item. "
    + "Use the candidate original title and picture to identify what the buyer receives. "
    + "An accessory matches a query for an accessory of the same function, but does not match "
    + "a query for the host product. Conversely, the host product is not a substitute for "
    + "the accessory. Favor compatible primary product type and practical use. "
    + "Allow differences in brand, color and style unless explicitly required. "
    + "Do not infer a mismatch from an uncertain detail or reward a shared prop or background."
)


def subject_reranker() -> BusinessReranker:
    """Create the existing full local reranker with a fixed subject instruction.

    Returns:
        Exact complete-model adapter configured for this experiment.
    """
    client = BusinessReranker()
    client.adapter.config = {
        **client.adapter.config,
        "instruction": INSTRUCTION,
        "task_scope": "sale_item_boundary_development",
    }
    return client


def plan(registry: dict) -> dict:
    """Freeze four incumbent and four challenger inputs per image query.

    Args:
        registry: Active data-role rules.

    Returns:
        Label-free jobs and hashed inputs for all new real pair calls.
    """
    guard()
    names = ("sid_protocol.json", "subject_evidence.json", "rankings.json")
    paths = [EXPANDED / name for name in (*names, "summary.json")]
    paths.extend([TARGET / "rankings.json", SIMPLE / "inputs.json"])
    require_data_access(paths, "development", registry)
    receipt = load(EXPANDED / "summary.json")
    for name in names:
        if sha256_file(EXPANDED / name) != receipt["files"][name]:
            raise ValueError(f"Previous experiment changed: {name}")
    target_receipt_path = TARGET / "target_status.json"
    require_data_access([target_receipt_path], "development", registry)
    if (
        sha256_file(TARGET / "rankings.json")
        != load(target_receipt_path)["files"]["rankings.json"]
    ):
        raise ValueError("Earlier ranking support changed")
    paths.append(target_receipt_path)
    source = load(SIMPLE / "inputs.json")
    previous = load(EXPANDED / "rankings.json")
    auxiliary = load(TARGET / "rankings.json")
    evidence = load(EXPANDED / "subject_evidence.json")["cases"]
    jobs = []
    for job in load(EXPANDED / "sid_protocol.json")["jobs"]:
        cohort, query = job["cohort"], job["query"]
        qid = query["id"]
        order = previous[cohort][qid][BASE]
        head = job["head"] + job["extra"]
        if len(head) != 30 or set(head) != set(order[:30]):
            raise ValueError("Boundary selection requires the verified complete Top30")
        offsets = evidence[f"{cohort}/{qid}"]["candidates"]
        positions = {cid: i for i, cid in enumerate(order[:30])}
        incumbents = sorted(
            order[:10], key=lambda cid: (-offsets[cid]["value"], -positions[cid])
        )[:4]
        immediate = order[10:12]
        remaining = set(order[12:30])
        support = [
            cid for cid in auxiliary[cohort][qid]["cross20"] if cid in remaining
        ][:2]
        selected = sorted(incumbents + immediate + support, key=positions.__getitem__)
        if len(selected) != 8 or len(set(selected)) != 8:
            raise ValueError("Expected eight distinct boundary candidates")
        jobs.append(
            dict(
                cohort=cohort,
                query=query,
                extra=selected,
                original_slots=[positions[cid] for cid in selected],
                documents=[source["documents"][cid] for cid in selected],
            )
        )
    if len(jobs) != 24:
        raise ValueError("Expected all 24 exposed image queries")
    paths.extend([Path(__file__).resolve(), WORKER])
    return dict(
        role="exposed_development",
        base_method=BASE,
        instruction=INSTRUCTION,
        jobs=jobs,
        new_pairs=sum(len(job["extra"]) for job in jobs),
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths
        },
        formal_metrics=None,
    )


def main() -> None:
    """Run resumable real Qwen scoring for every frozen boundary pair."""
    ROOT.mkdir(parents=True, exist_ok=True)
    worker = load_project_module("module2_subject_challenger_worker", WORKER)
    worker.ROOT = ROOT
    worker.BusinessReranker = subject_reranker
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = plan(registry)
        path = ROOT / "sid_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen challenger protocol changed")
        write_json(path, protocol)
        print(
            f"Frozen subject-specific real pairs: {protocol['new_pairs']}", flush=True
        )
        runtime = worker.inference(protocol["jobs"], registry)
        if not (ROOT / "runtime.json").exists() or runtime.get("timings"):
            write_json(ROOT / "runtime.json", runtime)


if __name__ == "__main__":
    main()
