"""Score the complete Top30 development expansion with the existing Qwen worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import run  # noqa: F401
from module2_p80_r90 import ROOT as TARGET
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from paper_fusion_experiment import ROOT as PARENT
from paper_fusion_experiment import SIMPLE, load
from simple_baseline_cycle import guard

ROOT = TARGET / "expanded_pool"
WORKER = PROJECT_ROOT / "scripts/paper_sid_rerank.py"


def plan(registry: dict) -> dict:
    """Freeze the pool expansion and authenticate reused complete-model scores.

    Args:
        registry: Active project rules governing development data access.

    Returns:
        Score jobs with fixed original inputs, prior scores and additional candidates.
    """
    guard()
    files = [
        TARGET / "selection.json",
        TARGET / "rankings.json",
        TARGET / "target_status.json",
        PARENT / "sid_protocol.json",
        PARENT / "summary.json",
        SIMPLE / "inputs.json",
    ]
    require_data_access(files, "development", registry)
    status = load(TARGET / "target_status.json")
    for name in ("selection.json", "rankings.json"):
        if sha256_file(TARGET / name) != status["files"][name]:
            raise ValueError(f"Selected development input changed: {name}")
    parent_protocol = PARENT / "sid_protocol.json"
    parent_digest = sha256_file(parent_protocol)
    if parent_digest != load(PARENT / "summary.json")["files"]["sid_protocol.json"]:
        raise ValueError("Previous pool protocol changed")
    original = load(SIMPLE / "inputs.json")
    ranks = load(TARGET / "rankings.json")
    selected = load(TARGET / "selection.json")["selected"]
    jobs = []
    for old in load(parent_protocol)["jobs"]:
        cohort, query = old["cohort"], old["query"]
        qid = query["id"]
        cache = PARENT / "sid_pairs" / cohort / f"{qid}.json"
        require_data_access([cache], "development", registry)
        files.append(cache)
        saved = load(cache)
        if saved["identity"] != dict(
            protocol_sha256=parent_digest,
            query_sha256=query["source_sha256"],
            extra=old["extra"],
        ):
            raise ValueError("Reused expansion score identity differs")
        head = old["head"] + old["extra"]
        scores = {**old["original_scores"], **saved["scores"]}
        if set(scores) != set(head) or not np.isfinite(list(scores.values())).all():
            raise ValueError("Reused model scores are incomplete")
        base = ranks[cohort][qid][selected]
        if base[: len(head)] != [cid for cid in base if cid in scores]:
            raise ValueError("Base ranking head differs from the real scored pool")
        extra = [cid for cid in base[:30] if cid not in scores]
        jobs.append(
            dict(
                cohort=cohort,
                query=query,
                head=head,
                original_scores=scores,
                extra=extra,
                documents=[original["documents"][cid] for cid in extra],
            )
        )
    pairs = sum(len(job["extra"]) for job in jobs)
    if len(jobs) != 24 or pairs != 129:
        raise ValueError("Planned scope differs from the confirmed 24-query replay")
    files.extend([Path(__file__).resolve(), WORKER])
    return dict(
        role="exposed_development",
        selected_baseline=selected,
        pool_policy="Keep the prior scored head and add original rank Top30.",
        original_pairs=sum(len(job["head"]) for job in jobs),
        new_pairs=pairs,
        jobs=jobs,
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files
        },
        formal_metrics=None,
    )


def main() -> None:
    """Run the resumable exact scoring adapter on the new local experiment folder."""
    ROOT.mkdir(parents=True, exist_ok=True)
    worker = load_project_module("module2_expand_pool_worker", WORKER)
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        protocol = plan(registry)
        path = ROOT / "sid_protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen Top30 protocol changed")
        write_json(path, protocol)
        print(f"Verified {protocol['new_pairs']} new full-model pairs", flush=True)
        runtime = worker.inference(protocol["jobs"], registry)
        if not (ROOT / "runtime.json").exists() or runtime.get("timings"):
            write_json(ROOT / "runtime.json", runtime)


if __name__ == "__main__":
    main()
