"""Cached ROI, center, tile and attention ablations without further GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
import verify_regions
from multimodal_ecommerce.measurement import cluster_interval
from multimodal_ecommerce.regional_retrieval import RegionalIndex, RegionalSearchRequest
from multimodal_ecommerce.retrieval_eval import ranking_metrics
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

VIEW_SETS: dict[str, set[str]] = {
    "roi": {"global", "contrast_roi"},
    "center": {"global", "center"},
    "tiles": {
        "global",
        "center",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    },
    "all": {
        "global",
        "contrast_roi",
        "center",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    },
}


def subset_bank(index: RegionalIndex, kinds: set[str]) -> RegionalIndex:
    """Return subset bank for this workflow.

    Args:
        index: Index used by this operation.
        kinds: Kinds used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    records, vectors = [], np.zeros_like(index.vectors)
    for i, row in enumerate(index.records):
        selected = [j for j, view in enumerate(row["views"]) if view["kind"] in kinds]
        records.append({**row, "views": [row["views"][j] for j in selected]})
        vectors[i, : len(selected)] = index.vectors[i, selected]
    return RegionalIndex(records, vectors, index.config)


def analyze(experiment: Path, output: Path) -> dict:
    """Return analyze for this workflow.

    Args:
        experiment: Experiment used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: scope, summary, gallery_roi_available, gallery_count,
        query_count, formal_acceptance.

    Raises:
        FileExistsError: Use a new cached ablation directory.

    Notes:
        Writes derived artifacts through output.mkdir, write_json.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new cached ablation directory")
    verify_regions.verify(experiment)
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    full, _, _ = RegionalIndex.load(experiment / "index")
    indexes = {name: subset_bank(full, kinds) for name, kinds in VIEW_SETS.items()}
    # Fixed strategy definitions, no fitting temperatures or using qrels in search.
    variants = [
        ("global", "all", "global", False),
        ("gallery_attention", "all", "attention", True),
        ("roi_attention", "roi", "attention", False),
        ("center_attention", "center", "attention", False),
        ("tiles_attention", "tiles", "attention", False),
        ("all_mean", "all", "mean", False),
        ("all_attention", "all", "attention", False),
    ]
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "parent_manifest_sha256": sha256_file(experiment / "manifest.json"),
            "variants": variants,
            "view_sets": {name: sorted(kinds) for name, kinds in VIEW_SETS.items()},
            "no_parameter_fitting": True,
        },
    )
    rankings: dict
    metrics, rankings = [], {}
    with np.load(experiment / "query_vectors.npz", allow_pickle=False) as archive:
        for number, query in enumerate(protocol["queries"], 1):
            row = query["record"]
            query_views = json.loads(
                (experiment / "queries" / f"{number:02}" / "views.json").read_text(
                    encoding="utf-8"
                )
            )
            vectors = archive["vectors"][
                number - 1, : int(archive["counts"][number - 1])
            ]
            excluded = {
                r["id"]
                for r in full.records
                if r["image_sha256"] == row["image_sha256"]
            }
            rankings[row["id"]] = {}
            for name, kind_set, strategy, global_query_only in variants:
                selected = [
                    i
                    for i, view in enumerate(query_views)
                    if view["kind"] in VIEW_SETS[kind_set]
                ]
                if global_query_only:
                    selected = [0]
                ranked = indexes[kind_set].search(
                    RegionalSearchRequest(
                        vectors[selected],
                        len(full.records),
                        strategy=strategy,
                        exclude_ids=excluded,
                    )
                )
                ids = [r["id"] for r in ranked]
                rankings[row["id"]][name] = ids
                metrics.append(
                    {
                        "query_id": row["id"],
                        "pair_group": row["pair_group"],
                        "variant": name,
                        "query_view_count": len(selected),
                        **ranking_metrics(ids, set(query["relevant_ids"]), 10),
                    }
                )
    summary = {}
    for name, _, _, _ in variants:
        rows = [r for r in metrics if r["variant"] == name]
        summary[name] = {
            metric: cluster_interval(
                [r[metric] for r in rows], [r["pair_group"] for r in rows]
            )
            for metric in ("recall@10", "precision@10")
        }
    report = {
        "scope": "Cached development ablations; no blind-test or formal acceptance claim",
        "summary": summary,
        "gallery_roi_available": sum(
            any(v["kind"] == "contrast_roi" for v in r["views"]) for r in full.records
        ),
        "gallery_count": len(full.records),
        "query_count": len(protocol["queries"]),
        "formal_acceptance": False,
    }
    write_json(output / "rankings.json", rankings)
    write_json(output / "query_metrics.json", metrics)
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "script_sha256": sha256_file(Path(__file__)),
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
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
        "--output", type=Path, default=Path("artifacts/module2_cached_ablations_v1")
    )
    args = parser.parse_args()
    report = analyze(PROJECT_ROOT / args.experiment, PROJECT_ROOT / args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
