"""Freeze and score complementary image candidates from beyond the existing Top30."""

from __future__ import annotations

from pathlib import Path
from statistics import median

import run  # noqa: F401
from module2_expand_pool import ROOT as EXPANDED
from module2_expand_pool import TARGET, WORKER
from module2_poster_text_replay import ROOT as POSTER
from module2_poster_text_replay import TITLE
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import COHORTS, SIMPLE, load
from simple_baseline_cycle import guard

ROOT = TARGET / "recall_boundary"
ADDITIONS = 3


def consensus_order(views: tuple[list[str], list[str], list[str]]) -> list[str]:
    """Rank a full gallery by agreement across whole, crop, and region-title views.

    Args:
        views: Complete gallery permutations from the three retrieval views.

    Returns:
        Median-rank order with deterministic ties by best rank, then gallery ID.
    """
    gallery = set(views[0])
    if not gallery or any(
        len(order) != len(gallery) or set(order) != gallery for order in views
    ):
        raise ValueError("Each retrieval view must cover the same unique gallery")
    positions = [{cid: rank for rank, cid in enumerate(order)} for order in views]

    def order_key(cid: str) -> tuple[float, int, str]:
        """Return the label-free consensus sorting key.

        Args:
            cid: Candidate whose retrieval ranks are compared.

        Returns:
            Median rank, best rank, and deterministic identifier.
        """
        ranks = [view[cid] for view in positions]
        return median(ranks), min(ranks), cid

    return sorted(gallery, key=order_key)


def freeze_plan(registry: dict) -> dict:
    """Select three additions per image query using only fixed retrieval rankings.

    Args:
        registry: Project data-role registry.

    Returns:
        Hash-pinned image jobs for the existing complete-model scoring worker.
    """
    guard()
    protocol = EXPANDED / "sid_protocol.json"
    paths = [
        protocol,
        EXPANDED / "summary.json",
        POSTER / "summary.json",
        POSTER / "rankings.json",
        POSTER / "results.json",
        TARGET / "rankings.json",
        TARGET / "target_status.json",
        SIMPLE / "inputs.json",
        *(SIMPLE / cohort / "predictions.json" for cohort in COHORTS),
    ]
    require_data_access(paths, "development", registry)
    expected = {
        protocol: load(EXPANDED / "summary.json")["files"][protocol.name],
        POSTER
        / "rankings.json": load(POSTER / "summary.json")["files"][
            "poster_text_replay/rankings.json"
        ],
        POSTER
        / "results.json": load(POSTER / "summary.json")["files"][
            "poster_text_replay/results.json"
        ],
        TARGET
        / "rankings.json": load(TARGET / "target_status.json")["files"][
            "rankings.json"
        ],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Previous replay input changed: {path}")
    ranks = load(POSTER / "rankings.json")
    controls = load(TARGET / "rankings.json")
    inputs = load(SIMPLE / "inputs.json")
    cropped = {c: load(SIMPLE / c / "predictions.json") for c in COHORTS}
    jobs = []
    for old in load(protocol)["jobs"]:
        cohort, query = old["cohort"], old["query"]
        qid = query["id"]
        base = ranks[f"{cohort}/{qid}"][TITLE]
        support = controls[cohort][qid]
        consensus = consensus_order(
            (
                cropped[cohort][qid]["whole_cosine"],
                cropped[cohort][qid]["crop_cosine"],
                support["region_title"],
            )
        )
        head = base[:30]
        if (
            set(consensus) != set(base)
            or len(set(base)) != len(base)
            or set(head) != set(old["head"] + old["extra"])
        ):
            raise ValueError("Consensus must preserve the full gallery and scored pool")
        outside = [cid for cid in consensus if cid not in head][:ADDITIONS]
        if len(outside) != ADDITIONS:
            raise ValueError("Too few previously unscored candidates")
        jobs.append(
            dict(
                cohort=cohort,
                query=query,
                head=head,
                consensus=consensus,
                extra=outside,
                documents=[inputs["documents"][cid] for cid in outside],
            )
        )
    if len(jobs) != 24:
        raise ValueError("The expansion must include all 24 exposed image queries")
    return dict(
        role="exposed_development",
        base_method=TITLE,
        added_pairs=sum(len(job["extra"]) for job in jobs),
        pool_policy="Add the three highest median-rank candidates outside the scored Top30.",
        jobs=jobs,
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in [*paths, Path(__file__).resolve(), WORKER]
        },
        formal_metrics=None,
    )


def main() -> None:
    """Freeze the protocol and resume exact local Qwen scores when possible."""
    ROOT.mkdir(parents=True, exist_ok=True)
    worker = load_project_module("module2_recall_boundary_worker", WORKER)
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = freeze_plan(registry)
        path = ROOT / "sid_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen recall-expansion protocol changed")
        if not path.exists():
            write_json(path, protocol)
        print(f"Frozen {protocol['added_pairs']} new Qwen pairs.", flush=True)
        runtime = worker.inference(protocol["jobs"], registry)
        if not (ROOT / "runtime.json").exists() or runtime.get("timings"):
            write_json(ROOT / "runtime.json", runtime)


if __name__ == "__main__":
    main()
