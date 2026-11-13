"""Verify saved phase-two measurements without new model calls or human labels."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.retrieval_benchmark import prepare_records, same_item_protocol
from multimodal_ecommerce.retrieval_eval import ranking_metrics
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def load(path: Path) -> Any:
    """Read an existing UTF-8 JSON artifact.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Recompute metrics from saved top-ten IDs and all source-defined positives."""
    retrieval = PROJECT_ROOT / "artifacts/mentor_retrieval_v1"
    boundary = PROJECT_ROOT / "artifacts/mentor_copy_boundary_v1"
    dataset = PROJECT_ROOT / "data/mentor_fashion_1000_v1/samples.jsonl"
    recipe = load(PROJECT_ROOT / "configs/copy_recipe_v3.json")
    verify_files(PROJECT_ROOT, recipe)
    for directory in (retrieval, boundary):
        verify_files(directory, load(directory / "manifest.json"))
    report = load(retrieval / "report.json")
    assert report["dataset_sha256"] == sha256_file(dataset)
    records = prepare_records(dataset)
    protocol = load(retrieval / "protocol.json")
    assert protocol == same_item_protocol(records)
    assert load(retrieval / "records.json") == records
    with np.load(retrieval / "vectors.npz", allow_pickle=False) as vectors:
        for key in ("images", "texts"):
            assert vectors[key].shape == (1000, 2048)
            assert np.isfinite(vectors[key]).all()
            assert np.allclose(np.linalg.norm(vectors[key], axis=1), 1, atol=1e-5)
    # v1 saved rankings in its escaped HTML review artifact. Read that exact
    # immutable artifact for independent scoring; newer runs also save JSON.
    markup = (retrieval / "same_item_results.html").read_text(encoding="utf-8")
    headings = [html.unescape(value) for value in re.findall(r"<h2>(.*?)</h2>", markup)]
    ids = [
        html.unescape(value) for value in re.findall(r"<small>(.*?)</small>", markup)
    ]
    expected_metrics = load(retrieval / "query_metrics.json")
    assert len(ids) == 10 * len(protocol["queries"])
    assert headings == [
        "同款找回 · " + row["record"]["id"] for row in protocol["queries"]
    ]
    computed = []
    by_id = {row["id"]: row for row in records}
    for number, query in enumerate(protocol["queries"]):
        ranked = ids[number * 10 : (number + 1) * 10]
        assert set(ranked) <= set(protocol["gallery_ids"])
        assert all(
            by_id[key]["image_sha256"] != query["record"]["image_sha256"]
            for key in ranked
        )
        metrics = ranking_metrics(ranked, set(query["relevant_ids"]), 10)
        assert expected_metrics[number]["id"] == query["record"]["id"]
        for key, value in metrics.items():
            assert np.isclose(value, expected_metrics[number][key])
        computed.append(
            {"query_id": query["record"]["id"], "top10_ids": ranked, **metrics}
        )
    for metric in ("precision@10", "recall@10"):
        assert np.isclose(
            np.mean([row[metric] for row in computed]),
            report["same_item"]["metrics"][metric]["estimate"],
        )
    timings = load(retrieval / "timings.json")
    assert len(timings) == 300
    for modality in ("text", "image"):
        for size in (100, 300, 1000):
            selected = [
                row
                for row in timings
                if row["modality"] == modality and row["gallery_size"] == size
            ]
            assert len(selected) == 50
            for row in selected:
                assert row["search_seconds_preloaded"] > 0
                assert np.isclose(
                    row["combined_seconds"],
                    row["embedding_seconds"] + row["search_seconds_preloaded"],
                )
    results = read_rows(boundary / "results.jsonl")
    assert len(results) == 30
    assert all(
        row["status"] == row["expected_behavior"]
        for row in results
        if row["expected_behavior"] != "generation_attempt"
    )
    assert all(
        row["marketing_line"] is None
        for row in results
        if row["status"] != "draft_requires_human_review"
    )
    assert load(boundary / "report.json")["human_fluency_rate"] is None
    source = load(PROJECT_ROOT / "artifacts/source_integrity.json")
    assert source["passed"]
    code = [
        "src/multimodal_ecommerce/content_contract.py",
        "src/multimodal_ecommerce/measurement.py",
        "src/multimodal_ecommerce/retrieval_benchmark.py",
        "scripts/benchmark_copy_boundary.py",
        "scripts/benchmark_retrieval.py",
        "scripts/prepare_retrieval_benchmark.py",
        "scripts/generate_content.py",
        "scripts/verify_mentor.py",
    ]
    evidence = {
        "passed": True,
        "checks": [
            "frozen_v3_code",
            "both_run_manifests",
            "2000_copied_assets",
            "source_pair_protocol",
            "2048_dimension_unit_vectors",
            "top10_metric_recalculation",
            "300_timing_records",
            "30_boundary_results",
            "no_invented_human_scores",
        ],
        "current_reproduction_code_sha256": {
            name: sha256_file(PROJECT_ROOT / name) for name in code
        },
        "code_note": "Current version additionally exports full ranking JSON and preserves colliding whitespace-cleaned field names. Saved v1 inputs are unaffected; metrics were independently recomputed from the original hashed HTML ranking artifact.",
        "source_integrity": {
            key: value for key, value in source.items() if key != "differences"
        },
        "observed_top10_and_recomputed_metrics": computed,
        "browser_visual_verification": False,
    }
    write_json(PROJECT_ROOT / "artifacts/mentor_verification.json", evidence)
    print(
        json.dumps({"passed": True, "checks": evidence["checks"]}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
