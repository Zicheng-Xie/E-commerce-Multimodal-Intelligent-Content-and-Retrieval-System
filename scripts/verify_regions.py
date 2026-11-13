"""Recompute the complete frozen same-item ablation from saved region vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.regional_retrieval import RegionalIndex, RegionalSearchRequest
from multimodal_ecommerce.retrieval_eval import ranking_metrics
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def verify(directory: Path) -> dict:
    """Verify every output hash, every ranking and each summary metric independently.

    Args:
        directory: Directory used by this operation.

    Returns:
        Mapping with fields: passed, queries, gallery, rankings_recomputed,
        truth_boxes_used_for_crops, source_classes_used_for_ranking,
        human_quality_scores_produced.

    Raises:
        ValueError: Benchmark implementation changed; Frozen protocol changed.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or sha256_file(path) != checksum
        ):
            raise ValueError(f"Changed benchmark asset: {name}")
    for path, key in [
        (Path(__file__).with_name("benchmark_regions.py"), "script_sha256"),
        (
            PROJECT_ROOT / "src/multimodal_ecommerce/regional_retrieval.py",
            "attention_implementation_sha256",
        ),
    ]:
        if sha256_file(path) != manifest[key]:
            raise ValueError("Benchmark implementation changed")
    saved = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if (
        sha256_file(PROJECT_ROOT / "artifacts/mentor_retrieval_v1/protocol.json")
        != saved["source_sha256"]
    ):
        raise ValueError("Frozen protocol changed")
    protocol = saved["source_protocol"]
    index, texts, _ = RegionalIndex.load(directory / "index")
    assert texts is None
    assert [r["id"] for r in index.records] == protocol["gallery_ids"]
    assert all(r["category"] == "" and r["attributes"] == {} for r in index.records)
    rankings = json.loads((directory / "rankings.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    metric_rows = json.loads(
        (directory / "query_metrics.json").read_text(encoding="utf-8")
    )
    totals: dict = {key: [] for key in ("global", "mean", "attention")}
    with np.load(directory / "query_vectors.npz", allow_pickle=False) as archive:
        for i, query in enumerate(protocol["queries"]):
            vectors = archive["vectors"][i, : int(archive["counts"][i])]
            excluded = {
                r["id"]
                for r in index.records
                if r["image_sha256"] == query["record"]["image_sha256"]
            }
            for strategy in totals:
                results = index.search(
                    RegionalSearchRequest(
                        vectors,
                        len(index.records),
                        strategy=strategy,
                        exclude_ids=excluded,
                    )
                )
                ids = [r["id"] for r in results]
                assert ids == rankings[query["record"]["id"]][strategy]
                scores = ranking_metrics(ids, set(query["relevant_ids"]), 10)
                expected = next(
                    row
                    for row in metric_rows
                    if row["query_id"] == query["record"]["id"]
                    and row["strategy"] == strategy
                )
                for name, value in scores.items():
                    assert np.isclose(value, expected[name])
                totals[strategy].append(scores)
    for strategy, values in totals.items():
        for name in ("recall@10", "precision@10"):
            assert np.isclose(
                np.mean([row[name] for row in values]),
                report["summary"][strategy][name]["estimate"],
            )
    assert report["formal_acceptance"] is False
    return {
        "passed": True,
        "queries": len(protocol["queries"]),
        "gallery": len(index.records),
        "rankings_recomputed": len(protocol["queries"]) * len(totals),
        "truth_boxes_used_for_crops": False,
        "source_classes_used_for_ranking": False,
        "human_quality_scores_produced": False,
    }


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new verification path.

    Notes:
        Writes derived artifacts through write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/module2_regions_benchmark_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/module2_regions_verification_v1.json"),
    )
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new verification path")
    result = verify(PROJECT_ROOT / args.experiment)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
