"""Freeze, fit, and evaluate a lightweight metric on development cohorts only."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.measurement import cluster_interval  # noqa: E402
from multimodal_ecommerce.metric_training import (  # noqa: E402
    METRIC_POLICY,
    metric_transform,
    train_metric,
)
from multimodal_ecommerce.precision_reranking import (  # noqa: E402
    pareto_gate,
    refine_pool,
)
from multimodal_ecommerce.retrieval_strategies import (  # noqa: E402
    extended_metrics,
    ordered_indices,
)
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)
from prepare_metric_training import DATA, OUT  # noqa: E402

OLD = PROJECT_ROOT / "artifacts/module2_regions_benchmark_v1"
PREVIOUS = PROJECT_ROOT / "artifacts/module2_strategy_exploration"
METRICS = ("precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1")
CODE = [
    "scripts/train_retrieval_metric.py",
    "src/multimodal_ecommerce/metric_training.py",
]
PARENTS = [
    "artifacts/module2_training/protocol.json",
    "artifacts/module2_regions_benchmark_v1/index/vectors.npz",
    "artifacts/module2_regions_benchmark_v1/query_vectors.npz",
    "artifacts/module2_regions_benchmark_v1/index/records.json",
    "artifacts/module2_regions_benchmark_v1/protocol.json",
    "artifacts/module2_strategy_exploration/rankings.json",
    "artifacts/module2_blind/completion_receipt.json",
    "artifacts/module2_precision/completion_receipt.json",
]


def freeze() -> None:
    """Freeze the inputs and policy used by this workflow.

    Raises:
        FileExistsError: Freeze the model policy before training-feature inference.

    Notes:
        Writes derived artifacts through write_json.
    """
    path = OUT / "training_protocol.json"
    if path.exists() or (OUT / "features.npz").exists():
        raise FileExistsError(
            "Freeze the model policy before training-feature inference"
        )
    write_json(
        path,
        {
            "policy": METRIC_POLICY,
            "implementation": {p: sha256_file(PROJECT_ROOT / p) for p in CODE},
            "parents": {p: sha256_file(PROJECT_ROOT / p) for p in PARENTS},
            "new_development_baseline": "full-image original cosine within its own development gallery",
            "old_development_baseline": "existing reciprocal_attention_fusion on all original 334 candidates",
            "caveat": "Different gallery sizes/positive counts; compare deltas within a cohort only",
            "selection": "Report every fixed method; require P+2pp and no mean regression in other five metrics on BOTH development cohorts",
            "formal_acceptance": False,
            "consumed_blind_used_for_training_or_scoring": False,
        },
    )
    print("Training policy frozen before new image encoding")


@dataclass(frozen=True)
class TrainRetrievalMetricEvaluateCohortInputs:
    """Input values for evaluate cohort.

    Attributes:
        name: Name used by this operation.
        queries: Queries used by this operation.
        gallery: Gallery used by this operation.
        qrows: Qrows used by this operation.
        grows: Grows used by this operation.
        baselines: Baselines used by this operation.
        positives: Positives used by this operation.
        weights: Weights used by this operation.
    """

    name: str
    queries: np.ndarray
    gallery: np.ndarray
    qrows: list[dict]
    grows: list[dict]
    baselines: list[list[str]]
    positives: list[set[str]]
    weights: np.ndarray


def evaluate_cohort(options: TrainRetrievalMetricEvaluateCohortInputs) -> dict:
    """Evaluate cohort.

    Args:
        options: Validated input fields described by
            TrainRetrievalMetricEvaluateCohortInputs.

    Returns:
        Mapping with fields: cohort, queries, gallery, groups,
        mean_precision_at_10_ceiling, summary, paired_group_bootstrap_exploratory,
        gates, formal_acceptance.

    Raises:
        ValueError: Every ranking must contain the complete unchanged gallery.

    Notes:
        Writes derived artifacts through write_json.
    """
    name = options.name
    queries = options.queries
    gallery = options.gallery
    qrows = options.qrows
    grows = options.grows
    baselines = options.baselines
    positives = options.positives
    weights = options.weights
    learned = metric_transform(queries, weights) @ metric_transform(gallery, weights).T
    original = queries @ gallery.T
    ids = [r["id"] for r in grows]
    positions = {key: i for i, key in enumerate(ids)}
    rankings, measurements = {}, []
    ceilings = []
    for i, row in enumerate(qrows):
        base = baselines[i]
        candidate_ids = base[:30]
        candidate_scores = learned[i, [positions[c] for c in candidate_ids]]
        current = {
            "baseline": base,
            "original_global": [ids[j] for j in ordered_indices(original[i], ids)],
            "learned_global": [ids[j] for j in ordered_indices(learned[i], ids)],
            "fusion_metric25": refine_pool(
                base, candidate_ids, candidate_scores, blend=0.25
            ),
            "fusion_metric50": refine_pool(
                base, candidate_ids, candidate_scores, blend=0.5
            ),
            "fusion_metric50_anchor2": refine_pool(
                base, candidate_ids, candidate_scores, blend=0.5, anchors=2
            ),
        }
        rankings[row["id"]] = current
        ceilings.append(min(len(positives[i]), 10) / 10)
        for method, ranked in current.items():
            if len(ranked) != len(ids) or set(ranked) != set(ids):
                raise ValueError(
                    "Every ranking must contain the complete unchanged gallery"
                )
            measurements.append(
                {
                    "query_id": row["id"],
                    "group": row["pair_id"],
                    "method": method,
                    **extended_metrics(ranked, positives[i]),
                }
            )
    summary = {}
    for method in rankings[qrows[0]["id"]]:
        rows = [r for r in measurements if r["method"] == method]
        summary[method] = {
            key: float(np.mean([r[key] for r in rows])) for key in METRICS
        }
    paired = {}
    for method in summary:
        if method == "baseline":
            continue
        rows = [r for r in measurements if r["method"] == method]
        refs = [r for r in measurements if r["method"] == "baseline"]
        paired[method] = {
            key: cluster_interval(
                [r[key] - b[key] for r, b in zip(rows, refs)],
                [r["group"] for r in rows],
            )
            for key in METRICS
        }
    result = {
        "cohort": name,
        "queries": len(qrows),
        "gallery": len(grows),
        "groups": len({r["pair_id"] for r in qrows}),
        "mean_precision_at_10_ceiling": float(np.mean(ceilings)),
        "summary": summary,
        "paired_group_bootstrap_exploratory": paired,
        "gates": {
            m: pareto_gate(
                s, summary["baseline"], METRIC_POLICY["minimum_precision_gain"]
            )
            for m, s in summary.items()
            if m != "baseline"
        },
        "formal_acceptance": False,
    }
    write_json(OUT / "analysis" / f"{name}_rankings.json", rankings)
    write_json(OUT / "analysis" / f"{name}_query_metrics.json", measurements)
    return result


def fit() -> None:
    """Apply the fit workflow step.

    Raises:
        FileExistsError: Preserve the existing complete training analysis.
        ValueError: Training configuration changed; Dataset lineage changed.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    target = OUT / "analysis"
    if target.exists():
        raise FileExistsError("Preserve the existing complete training analysis")
    frozen = json.loads((OUT / "training_protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, frozen["implementation"])
    verify_files(PROJECT_ROOT, frozen["parents"])
    if frozen["policy"] != METRIC_POLICY:
        raise ValueError("Training configuration changed")
    feature_manifest = json.loads(
        (OUT / "feature_manifest.json").read_text(encoding="utf-8")
    )
    verify_files(OUT, feature_manifest["files"])
    if feature_manifest["dataset_manifest_sha256"] != sha256_file(
        DATA / "manifest.json"
    ):
        raise ValueError("Dataset lineage changed")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    verify_files(DATA, manifest["files"])
    records = {r["opaque_id"]: r for r in read_rows(DATA / "records.jsonl")}
    with np.load(OUT / "features.npz", allow_pickle=False) as a:
        rows = [records[key] for key in a["ids"].tolist()]
        vectors = a["vectors"].copy()
    started = time.perf_counter()
    model = train_metric(vectors, rows, METRIC_POLICY)
    target.mkdir()
    write_json(target / "model.json", model)
    weights = np.asarray(model["weights"])
    qindices = [
        i
        for i, r in enumerate(rows)
        if r["project_role"] == "development_validation" and r["source"] == "user"
    ]
    gindices = [
        i
        for i, r in enumerate(rows)
        if r["project_role"] == "development_validation" and r["source"] == "shop"
    ]
    qrows, grows = [rows[i] for i in qindices], [rows[i] for i in gindices]
    qs, gs = vectors[qindices], vectors[gindices]
    ids = [r["id"] for r in grows]
    baselines = [[ids[j] for j in ordered_indices(q @ gs.T, ids)] for q in qs]
    positives = [
        {g["id"] for g in grows if g["identity"] == q["identity"]} for q in qrows
    ]
    new = evaluate_cohort(
        TrainRetrievalMetricEvaluateCohortInputs(
            "new_group_development", qs, gs, qrows, grows, baselines, positives, weights
        )
    )
    protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    old_queries = [r["record"] for r in protocol["queries"]]
    for row in old_queries:
        row["pair_id"] = str(row["pair_group"])
    old_gallery = json.loads((OLD / "index/records.json").read_text(encoding="utf-8"))
    previous = json.loads((PREVIOUS / "rankings.json").read_text(encoding="utf-8"))
    with (
        np.load(OLD / "query_vectors.npz", allow_pickle=False) as q,
        np.load(OLD / "index/vectors.npz", allow_pickle=False) as g,
    ):
        old = evaluate_cohort(
            TrainRetrievalMetricEvaluateCohortInputs(
                "old_development",
                q["vectors"][:, 0],
                g["regions"][:, 0],
                old_queries,
                old_gallery,
                [previous[r["id"]]["reciprocal_attention_fusion"] for r in old_queries],
                [set(r["relevant_ids"]) for r in protocol["queries"]],
                weights,
            )
        )
    passing = [
        m
        for m in new["gates"]
        if new["gates"][m]["passes_development_metric_gate"]
        and old["gates"][m]["passes_development_metric_gate"]
    ]
    timings = []
    for q in qs:
        tick = time.perf_counter()
        metric_transform(q[None], weights) @ metric_transform(gs, weights).T
        timings.append(time.perf_counter() - tick)
    report = {
        "new": new,
        "old": old,
        "passing_both_development_gates": passing,
        "default_strategy_changed": False,
        "formal_acceptance": False,
        "formal_metrics": None,
        "training_and_analysis_seconds": time.perf_counter() - started,
        "metric_only_p95_seconds": float(np.percentile(timings, 95)),
        "timing_scope": "CPU feature transformation and gallery scoring only; not image encoding or end-to-end service",
        "new_independent_blind_test_required": True,
    }
    write_json(target / "report.json", report)
    print(
        json.dumps(
            {
                "groups": model["training_queries"],
                "passing": passing,
                "old": old["summary"],
                "new": new["summary"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "fit"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else fit()
