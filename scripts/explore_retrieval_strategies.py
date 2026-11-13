"""Explore fixed label-free retrieval strategies on saved real Qwen features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
import verify_regions
from multimodal_ecommerce.measurement import cluster_interval, distribution
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalSearchRequest,
    write_regional_gallery,
)
from multimodal_ecommerce.retrieval_strategies import (
    EXPLORATION_POLICY,
    STRATEGY_NAMES,
    StrategyExplorer,
    extended_metrics,
    ordered_indices,
    subset_region_index,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

METHOD_NOTES = {
    "global": "Original full-image cosine baseline",
    "gallery_attention": "Existing baseline: full-image query with gallery view attention",
    "alpha_qe": "Top5 positive-cosine pseudo-feedback, alpha=3, original query weight=1; no labels",
    "alpha_qe_attention": "Same alphaQE, followed by existing gallery-region attention",
    "database_augmentation": "Gallery-only top3 cosine^3 weighted neighbors added to original gallery vector",
    "gallery_centering": "Subtract unlabelled allowed-gallery mean and L2 normalize both sides; no query-batch statistics",
    "density_correction": "2*cosine-query - mean top10 gallery-peer cosine; CSLS-inspired single-gallery adaptation, not full bilingual CSLS",
    "reciprocal_global": "k-reciprocal encoding, 2/3 expansion, k1=20, k2=6 LQE, weighted Jaccard, original distance weight=0.3; one query only",
    "reciprocal_attention_fusion": "0.5 reciprocal distance + 0.5 (1-attention score)/2; engineering combination",
    "rrf_global_attention": "RRF(global,attention), constant=60, equal ranking weights",
    "rrf_reciprocal_attention": "RRF(reciprocal,attention), constant=60, equal ranking weights",
    "symmetric_local": "Bidirectional regional MaxSim mean, global weight=0.6; engineering local matching, not trained ColBERT",
    "graph_diffusion": "20 personalized random-walk iterations on mutual top10 gallery graph, restart=0.15; RRF with global; not region-manifold paper reproduction",
}


def run_exploration(experiment: Path, output: Path) -> dict:
    """Freeze configuration first, preserve every ranking, then evaluate qrels.

    Args:
        experiment: Experiment used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: scope, gallery_count, queries, groups, strategy_count,
        summary, per_category, all_strategies_cpu_seconds, gallery_preparation_seconds,
        precision10_ceiling, formal_acceptance, new_blind_test, learned_network_trained,
        new_model_inference, selection_note.

    Raises:
        FileExistsError: Use a new exploration directory.
        ValueError: This same-item visual protocol has no product-title evidence.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new exploration directory")
    verify_regions.verify(experiment)
    saved_protocol = json.loads(
        (experiment / "protocol.json").read_text(encoding="utf-8")
    )
    protocol = saved_protocol["source_protocol"]
    index, texts, _ = RegionalIndex.load(experiment / "index")
    if texts is not None:
        raise ValueError("This same-item visual protocol has no product-title evidence")
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "date": "2026-10-15",
            "parent_manifest_sha256": sha256_file(experiment / "manifest.json"),
            "source_protocol": protocol,
            "policy": EXPLORATION_POLICY,
            "methods": METHOD_NOTES,
            "scope": "Exploratory existing development queries; no hyperparameter grid or blind-test claim",
            "sorting_uses_labels": False,
            "query_batch_statistics_used": False,
            "exclusions": "Self or identical-image gallery rows removed BEFORE feedback, centering, graph construction and sorting",
            "metric_definition": "Macro averages of complete-list AP/MRR, binary NDCG@10, fixed-denominator Precision@k, all-positive Recall@k; zero positives rejected",
        },
    )
    preparation = time.perf_counter()
    explorer = StrategyExplorer(index)
    preparation_seconds = time.perf_counter() - preparation
    rankings: dict
    rows, rankings, timing_rows, feedback = [], {}, [], {}
    cases, sections = [], []
    arrays_to_save = {
        name: np.full((len(protocol["queries"]), len(index.records)), np.nan)
        for name in STRATEGY_NAMES
    }
    with np.load(experiment / "query_vectors.npz", allow_pickle=False) as archive:
        for number, query in enumerate(protocol["queries"]):
            row = query["record"]
            vectors = archive["vectors"][number, : int(archive["counts"][number])]
            include = np.array(
                [
                    i
                    for i, r in enumerate(index.records)
                    if r["image_sha256"] != row["image_sha256"]
                ],
                dtype=int,
            )
            active = explorer
            if len(include) != len(index.records):
                active = StrategyExplorer(subset_region_index(index, include))
            started = time.perf_counter()
            scores, evidence = active.evaluate(vectors)
            elapsed = time.perf_counter() - started
            timing_rows.append(
                {
                    "query_id": row["id"],
                    "all_13_strategies_seconds": elapsed,
                    "candidate_count": len(include),
                    "scope": "CPU scoring all policies jointly; excludes prior GPU encoding and index load",
                }
            )
            feedback[row["id"]] = evidence
            rankings[row["id"]] = {}
            base_order = ordered_indices(scores["gallery_attention"], active.ids)
            base_ids = [active.ids[i] for i in base_order]
            base_metrics = extended_metrics(base_ids, set(query["relevant_ids"]))
            reference = active.index.search(
                RegionalSearchRequest(
                    vectors[:1], len(active.ids), strategy="attention"
                )
            )
            assert base_ids == [r["id"] for r in reference]
            explanation = {r["id"]: r for r in reference}
            for name in STRATEGY_NAMES:
                order = ordered_indices(scores[name], active.ids)
                ids = [active.ids[i] for i in order]
                arrays_to_save[name][number, include] = scores[name]
                rankings[row["id"]][name] = ids
                metrics = extended_metrics(ids, set(query["relevant_ids"]))
                rows.append(
                    {
                        "query_id": row["id"],
                        "pair_group": row["pair_group"],
                        "category": row["category"],
                        "strategy": name,
                        "positive_count": len(query["relevant_ids"]),
                        **metrics,
                    }
                )
                cases.append(
                    {
                        "query_id": row["id"],
                        "strategy": name,
                        "recall10_delta": metrics["recall@10"]
                        - base_metrics["recall@10"],
                        "ap_delta": metrics["ap"] - base_metrics["ap"],
                        "ndcg10_delta": metrics["ndcg@10"] - base_metrics["ndcg@10"],
                        "base_top10": base_ids[:10],
                        "top10": ids[:10],
                    }
                )
                if number < 2 and name in {
                    "gallery_attention",
                    "alpha_qe_attention",
                    "reciprocal_global",
                    "reciprocal_attention_fusion",
                }:
                    cards = [
                        {
                            **explanation[key],
                            "rank": rank,
                            "score": float(scores[name][active.ids.index(key)]),
                            "strategy": name,
                        }
                        for rank, key in enumerate(ids[:10], 1)
                    ]
                    sections.append(
                        {
                            "query": row["id"],
                            "query_image": row["image_path"],
                            "strategy": name,
                            "results": cards,
                        }
                    )
            print(
                f"Strategy exploration {number + 1}/{len(protocol['queries'])}",
                flush=True,
            )
    summary = {}
    for name in STRATEGY_NAMES:
        selected = [r for r in rows if r["strategy"] == name]
        summary[name] = {
            metric: cluster_interval(
                [r[metric] for r in selected], [r["pair_group"] for r in selected]
            )
            for metric in (
                "precision@10",
                "recall@10",
                "ap",
                "ndcg@10",
                "mrr",
                "hit@1",
                "recall@50",
            )
        }
        related = [c for c in cases if c["strategy"] == name]
        base = [r for r in rows if r["strategy"] == "gallery_attention"]
        summary[name]["paired_delta"] = {
            metric: cluster_interval(
                [a[metric] - b[metric] for a, b in zip(selected, base)],
                [r["pair_group"] for r in selected],
            )
            for metric in ("recall@10", "precision@10", "ap", "ndcg@10")
        }
        summary[name]["recall_case_counts"] = {
            "improved": sum(c["recall10_delta"] > 1e-9 for c in related),
            "degraded": sum(c["recall10_delta"] < -1e-9 for c in related),
            "unchanged": sum(abs(c["recall10_delta"]) <= 1e-9 for c in related),
        }
    per_category: dict = {}
    for category in sorted({r["category"] for r in rows}):
        per_category[category] = {}
        for name in STRATEGY_NAMES:
            chosen = [
                r for r in rows if r["category"] == category and r["strategy"] == name
            ]
            per_category[category][name] = {
                "queries": len(chosen),
                **{
                    metric: float(np.mean([r[metric] for r in chosen]))
                    for metric in ("recall@10", "precision@10", "ap", "ndcg@10")
                },
            }
    np.savez(output / "scores.npz", allow_pickle=False, **arrays_to_save)
    write_json(output / "rankings.json", rankings)
    write_json(output / "query_metrics.json", rows)
    write_json(output / "cases.json", cases)
    write_json(output / "feedback.json", feedback)
    write_json(output / "timings.json", timing_rows)
    write_regional_gallery(output / "comparison.html", sections)
    report = {
        "scope": "Existing source-label same-item development set; exploratory multi-method comparison",
        "gallery_count": len(index.records),
        "queries": len(protocol["queries"]),
        "groups": len({r["pair_group"] for r in rows}),
        "strategy_count": len(STRATEGY_NAMES),
        "summary": summary,
        "per_category": per_category,
        "all_strategies_cpu_seconds": distribution(
            [r["all_13_strategies_seconds"] for r in timing_rows]
        ),
        "gallery_preparation_seconds": preparation_seconds,
        "precision10_ceiling": float(
            np.mean([min(len(q["relevant_ids"]), 10) / 10 for q in protocol["queries"]])
        ),
        "formal_acceptance": False,
        "new_blind_test": False,
        "learned_network_trained": False,
        "new_model_inference": False,
        "selection_note": "All fixed policies disclosed. Highest observed result among multiple exploratory methods is not an unbiased test estimate.",
    }
    prior = json.loads(
        (PROJECT_ROOT / "artifacts/module2_cached_ablations_v1/report.json").read_text(
            encoding="utf-8"
        )
    )
    for metric in ("recall@10", "precision@10"):
        assert np.isclose(
            summary["gallery_attention"][metric]["estimate"],
            prior["summary"]["gallery_attention"][metric]["estimate"],
        )
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/retrieval_strategies.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
            "files": {
                p.relative_to(output).as_posix(): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        },
    )
    return report


def main() -> None:
    """Run the command-line workflow defined by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/module2_regions_benchmark_v1"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/module2_strategy_exploration")
    )
    args = parser.parse_args()
    report = run_exploration(PROJECT_ROOT / args.experiment, PROJECT_ROOT / args.output)
    for name, values in report["summary"].items():
        print(
            name,
            {
                metric: round(values[metric]["estimate"] * 100, 2)
                for metric in (
                    "recall@10",
                    "precision@10",
                    "ap",
                    "ndcg@10",
                    "mrr",
                    "hit@1",
                )
            },
        )


if __name__ == "__main__":
    main()
