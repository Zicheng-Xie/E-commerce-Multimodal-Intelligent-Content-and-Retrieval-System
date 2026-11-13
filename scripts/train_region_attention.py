"""Encode selected pixel regions, then fit and compare actual trainable attention."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import file_map, verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.embedding import (  # noqa: E402
    QwenEmbedder,
    model_configuration,
)
from multimodal_ecommerce.measurement import cluster_interval  # noqa: E402
from multimodal_ecommerce.precision_reranking import (  # noqa: E402
    pareto_gate,
    refine_pool,
)
from multimodal_ecommerce.regional_retrieval import (  # noqa: E402
    propose_views,
    region_configuration,
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
from multimodal_ecommerce.trained_attention import (  # noqa: E402
    ATTENTION_POLICY,
    adapt_queries,
    attention_scores,
    fit_attention,
)
from prepare_metric_training import DATA, OUT  # noqa: E402
from train_retrieval_metric import METRICS, OLD, PREVIOUS  # noqa: E402

TARGET = OUT / "region_attention"


def load_new() -> tuple:
    """Return load new for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    verify_files(DATA, manifest["files"])
    by_id = {r["opaque_id"]: r for r in read_rows(DATA / "records.jsonl")}
    features = json.loads((OUT / "feature_manifest.json").read_text(encoding="utf-8"))
    verify_files(OUT, features["files"])
    with np.load(OUT / "features.npz", allow_pickle=False) as bank:
        ids, vectors = bank["ids"].tolist(), bank["vectors"].copy()
    return [by_id[i] for i in ids], vectors


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve this attention experiment.
        ValueError: Source training pixels changed.

    Notes:
        Writes derived artifacts through TARGET.mkdir, crop.save, folder.mkdir,
        np.savez_compressed, write_json.
    """
    if TARGET.exists():
        raise FileExistsError("Preserve this attention experiment")
    TARGET.mkdir()
    parents = [
        DATA / "manifest.json",
        OUT / "protocol.json",
        OUT / "feature_manifest.json",
        OUT / "analysis/report.json",
        OLD / "index/vectors.npz",
        OLD / "query_vectors.npz",
        PREVIOUS / "rankings.json",
    ]
    code = [
        "scripts/train_region_attention.py",
        "src/multimodal_ecommerce/trained_attention.py",
        "src/multimodal_ecommerce/regional_retrieval.py",
    ]
    write_json(
        TARGET / "protocol.json",
        {
            "policy": ATTENTION_POLICY,
            "embedding": model_configuration(),
            "regions": region_configuration(),
            "parents": {
                str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p)
                for p in parents
            },
            "implementation": {p: sha256_file(PROJECT_ROOT / p) for p in code},
            "scope": "Internal training/development only; official same-item identity supervision",
            "mechanism": "Learned downstream channel metric, view temperature and global/local gate; 2B encoder frozen; ridge user-to-shop query adapter is a separate ablation",
            "sampling_or_crop_uses_source_boxes": False,
            "consumed_blind_used": False,
            "formal_acceptance": False,
        },
    )
    records, global_vectors = load_new()
    inputs = json.loads((DATA / "model_inputs.json").read_text(encoding="utf-8"))
    index = {r["opaque_id"]: i for i, r in enumerate(records)}
    regions = np.zeros((len(records), 3, 2048), dtype=np.float32)
    mask = np.zeros((len(records), 3), dtype=bool)
    regions[:, 0] = global_vectors
    mask[:, 0] = True
    views_by_id = {}
    embedder = QwenEmbedder()
    started = time.perf_counter()
    try:
        count = 0
        for row in inputs:
            if row["query"]:
                continue
            i = index[row["id"]]
            image = Path(row["image_path"])
            if sha256_file(image) != records[i]["image_sha256"]:
                raise ValueError("Source training pixels changed")
            with Image.open(image) as source:
                picture = ImageOps.exif_transpose(source).convert("RGB")
            views = [
                v
                for v in propose_views(picture, region_configuration())
                if v["kind"] in ATTENTION_POLICY["view_kinds"]
            ]
            folder = TARGET / "crops" / row["id"]
            folder.mkdir(parents=True)
            for number, view in enumerate(views):
                if number == 0:
                    continue
                crop = picture.crop(tuple(view["box_xyxy"]))
                scale = 768 / max(crop.size)
                crop = crop.resize(
                    (
                        max(2, round(crop.width * scale)),
                        max(2, round(crop.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                path = folder / f"{number:02}_{view['kind']}.png"
                crop.save(path)
                regions[i, number] = embedder.encode(image=path, query=False)
                mask[i, number] = True
            views_by_id[row["id"]] = views
            count += 1
            if count % 50 == 0:
                print(f"Regional training candidates {count}/380", flush=True)
        diagnostics = embedder.diagnostics()
    finally:
        embedder.close()
    np.savez_compressed(
        TARGET / "features.npz",
        ids=np.array([r["opaque_id"] for r in records]),
        regions=regions,
        mask=mask,
    )
    write_json(TARGET / "views.json", views_by_id)
    write_json(
        TARGET / "feature_manifest.json",
        {
            "files": file_map(TARGET),
            "seconds": time.perf_counter() - started,
            "diagnostics": diagnostics,
            "encoding_scope": "Global query vectors reused; only ROI/center document crops newly encoded",
        },
    )
    print(
        json.dumps(
            {
                "regional_candidates": count,
                "additional_encodings": len(diagnostics["calls"]),
            }
        )
    )


@dataclass(frozen=True)
class TrainRegionAttentionEvaluateInputs:
    """Input values for evaluate.

    Attributes:
        name: Name used by this operation.
        q: Q used by this operation.
        g: G used by this operation.
        mask: Mask used by this operation.
        qrows: Qrows used by this operation.
        grows: Grows used by this operation.
        baseline: Baseline used by this operation.
        positives: Positives used by this operation.
        model: Model used by this operation.
    """

    name: str
    q: np.ndarray
    g: np.ndarray
    mask: np.ndarray
    qrows: list[dict]
    grows: list[dict]
    baseline: list[list[str]]
    positives: list[set[str]]
    model: dict


def evaluate(options: TrainRegionAttentionEvaluateInputs) -> dict:
    """Return evaluate for this workflow.

    Args:
        options: Validated input fields described by TrainRegionAttentionEvaluateInputs.

    Returns:
        Mapping with fields: summary, paired_group_bootstrap_exploratory, queries,
        gallery, formal_acceptance.

    Raises:
        ValueError: Candidate gallery changed during reranking.

    Notes:
        Writes derived artifacts through write_json.
    """
    name = options.name
    q = options.q
    g = options.g
    mask = options.mask
    qrows = options.qrows
    grows = options.grows
    baseline = options.baseline
    positives = options.positives
    model = options.model
    ids = [r["id"] for r in grows]
    positions = {v: i for i, v in enumerate(ids)}
    frozen_scores = attention_scores(q, g, mask)
    learned = attention_scores(q, g, mask, model)
    ridge = attention_scores(adapt_queries(q, model), g, mask)
    rankings, measurements = {}, []
    for i, row in enumerate(qrows):
        pool = baseline[i][:30]
        indices = [positions[k] for k in pool]
        current = {
            "baseline": baseline[i],
            "frozen_three_view_attention": [
                ids[j] for j in ordered_indices(frozen_scores[i], ids)
            ],
            "learned_three_view_attention": [
                ids[j] for j in ordered_indices(learned[i], ids)
            ],
            "learned_attention_blend25_anchor2": refine_pool(
                baseline[i], pool, learned[i, indices], blend=0.25, anchors=2
            ),
            "ridge_query_attention": [ids[j] for j in ordered_indices(ridge[i], ids)],
            "ridge_attention_blend25_anchor2": refine_pool(
                baseline[i], pool, ridge[i, indices], blend=0.25, anchors=2
            ),
        }
        rankings[row["id"]] = current
        for method, ranked in current.items():
            if len(ranked) != len(ids) or set(ranked) != set(ids):
                raise ValueError("Candidate gallery changed during reranking")
            measurements.append(
                {
                    "query_id": row["id"],
                    "group": row["pair_id"],
                    "method": method,
                    **extended_metrics(ranked, positives[i]),
                }
            )
    summary = {
        m: {
            k: float(np.mean([r[k] for r in measurements if r["method"] == m]))
            for k in METRICS
        }
        for m in current
    }
    comparisons = {}
    refs = [r for r in measurements if r["method"] == "baseline"]
    for method in summary:
        if method == "baseline":
            continue
        rows = [r for r in measurements if r["method"] == method]
        comparisons[method] = {
            k: cluster_interval(
                [r[k] - b[k] for r, b in zip(rows, refs)], [r["group"] for r in rows]
            )
            for k in METRICS
        }
    write_json(TARGET / f"{name}_rankings.json", rankings)
    write_json(TARGET / f"{name}_query_metrics.json", measurements)
    return {
        "summary": summary,
        "paired_group_bootstrap_exploratory": comparisons,
        "queries": len(qrows),
        "gallery": len(grows),
        "formal_acceptance": False,
    }


def fit() -> None:
    """Apply the fit workflow step.

    Raises:
        FileExistsError: Preserve the completed attention comparison.
        ValueError: Attention policy changed after feature inference; Regional feature
            IDs changed.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (TARGET / "report.json").exists():
        raise FileExistsError("Preserve the completed attention comparison")
    protocol = json.loads((TARGET / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["implementation"])
    verify_files(PROJECT_ROOT, protocol["parents"])
    if protocol["policy"] != ATTENTION_POLICY:
        raise ValueError("Attention policy changed after feature inference")
    feature_manifest = json.loads(
        (TARGET / "feature_manifest.json").read_text(encoding="utf-8")
    )
    verify_files(TARGET, feature_manifest["files"])
    records, _ = load_new()
    with np.load(TARGET / "features.npz", allow_pickle=False) as bank:
        if bank["ids"].tolist() != [r["opaque_id"] for r in records]:
            raise ValueError("Regional feature IDs changed")
        regions, mask = bank["regions"].copy(), bank["mask"].copy()
    started = time.perf_counter()
    model = fit_attention(regions, mask, records, ATTENTION_POLICY)
    write_json(TARGET / "model.json", model)
    qi = [
        i
        for i, r in enumerate(records)
        if r["project_role"] == "development_validation" and r["source"] == "user"
    ]
    gi = [
        i
        for i, r in enumerate(records)
        if r["project_role"] == "development_validation" and r["source"] == "shop"
    ]
    qrows, grows = [records[i] for i in qi], [records[i] for i in gi]
    ids = [r["id"] for r in grows]
    baseline = [
        [ids[j] for j in ordered_indices(q @ regions[gi, 0].T, ids)]
        for q in regions[qi, 0]
    ]
    positives = [
        {r["id"] for r in grows if r["identity"] == q["identity"]} for q in qrows
    ]
    new = evaluate(
        TrainRegionAttentionEvaluateInputs(
            "new_group_development",
            regions[qi, 0],
            regions[gi],
            mask[gi],
            qrows,
            grows,
            baseline,
            positives,
            model,
        )
    )
    old_protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    old_queries = [r["record"] for r in old_protocol["queries"]]
    for row in old_queries:
        row["pair_id"] = str(row["pair_group"])
    old_gallery = json.loads((OLD / "index/records.json").read_text(encoding="utf-8"))
    old_rankings = json.loads((PREVIOUS / "rankings.json").read_text(encoding="utf-8"))
    old_regions = np.zeros((len(old_gallery), 3, 2048), dtype=np.float32)
    old_mask = np.zeros((len(old_gallery), 3), dtype=bool)
    with np.load(OLD / "index/vectors.npz", allow_pickle=False) as bank:
        for i, row in enumerate(old_gallery):
            selected = [
                j
                for j, v in enumerate(row["views"])
                if v["kind"] in ATTENTION_POLICY["view_kinds"]
            ]
            old_regions[i, : len(selected)] = bank["regions"][i, selected]
            old_mask[i, : len(selected)] = True
    with np.load(OLD / "query_vectors.npz", allow_pickle=False) as bank:
        old = evaluate(
            TrainRegionAttentionEvaluateInputs(
                "old_development",
                bank["vectors"][:, 0],
                old_regions,
                old_mask,
                old_queries,
                old_gallery,
                [
                    old_rankings[q["id"]]["reciprocal_attention_fusion"]
                    for q in old_queries
                ],
                [set(q["relevant_ids"]) for q in old_protocol["queries"]],
                model,
            )
        )
    gates = {}
    for method in ATTENTION_POLICY["methods"]:
        old_gate = pareto_gate(old["summary"][method], old["summary"]["baseline"])
        new_regressions = [
            k
            for k in METRICS
            if new["summary"][method][k] < new["summary"]["baseline"][k] - 1e-12
        ]
        gates[method] = {
            "old": old_gate,
            "new_regressed_metrics": new_regressions,
            "passes_both_development_gates": old_gate["passes_development_metric_gate"]
            and not new_regressions,
        }
    report = {
        "old": old,
        "new": new,
        "gates": gates,
        "passing_methods": [
            m for m, g in gates.items() if g["passes_both_development_gates"]
        ],
        "training_and_analysis_seconds": time.perf_counter() - started,
        "default_strategy_changed": False,
        "formal_acceptance": False,
        "formal_metrics": None,
        "new_independent_validation_required": True,
    }
    write_json(TARGET / "report.json", report)
    print(
        json.dumps(
            {
                "old": old["summary"],
                "new": new["summary"],
                "passing": report["passing_methods"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["encode", "fit"])
    args = parser.parse_args()
    encode() if args.action == "encode" else fit()
