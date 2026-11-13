"""Measure selected label-free post-ranking paths with actual frozen queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.measurement import distribution
from multimodal_ecommerce.postranking import POSTRANK_METHODS, selected_scores
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.retrieval_strategies import subset_region_index
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new latency directory.
        ValueError: Exploration scores changed.

    Notes:
        Writes derived artifacts through output.mkdir, write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/module2_regions_benchmark_v1"),
    )
    parser.add_argument(
        "--exploration",
        type=Path,
        default=Path("artifacts/module2_strategy_exploration"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/module2_postrank_latency")
    )
    args = parser.parse_args()
    parent = PROJECT_ROOT / args.experiment
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new latency directory")
    saved = PROJECT_ROOT / args.exploration
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(saved / "scores.npz") != manifest["files"]["scores.npz"]:
        raise ValueError("Exploration scores changed")
    index, _, _ = RegionalIndex.load(parent / "index")
    protocol = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    rows = []
    with (
        np.load(parent / "query_vectors.npz", allow_pickle=False) as queries,
        np.load(saved / "scores.npz", allow_pickle=False) as expected,
    ):
        prepared = []
        for number, query in enumerate(protocol["queries"]):
            include = np.array(
                [
                    i
                    for i, row in enumerate(index.records)
                    if row["image_sha256"] != query["record"]["image_sha256"]
                ],
                dtype=int,
            )
            prepared.append(
                (
                    query["record"]["id"],
                    (
                        index
                        if len(include) == len(index.records)
                        else subset_region_index(index, include)
                    ),
                    queries["vectors"][number, : int(queries["counts"][number])],
                    include,
                )
            )
        for method in POSTRANK_METHODS:
            selected_scores(prepared[0][1], prepared[0][2], method)
        for repetition in range(2):
            for number, (key, active, vector, include) in enumerate(prepared):
                for method in POSTRANK_METHODS:
                    started = time.perf_counter()
                    actual, _ = selected_scores(active, vector, method)
                    seconds = time.perf_counter() - started
                    assert np.allclose(
                        actual, expected[method][number, include], atol=1e-6
                    )
                    rows.append(
                        {
                            "query_id": key,
                            "strategy": method,
                            "repetition": repetition,
                            "seconds": seconds,
                        }
                    )
            print(f"Selected strategy timing repetition {repetition + 1}/2", flush=True)
    report = {
        "scope": "Preloaded CPU score calculation only, 26 actual queries x 2 repetitions per policy. Excludes GPU encoding, model/index load, result explanation formatting and IO. No end-to-end <=5s claim.",
        "rankings_equivalent_to_exploration": True,
        "summary": {
            method: distribution(
                [row["seconds"] for row in rows if row["strategy"] == method]
            )
            for method in POSTRANK_METHODS
        },
    }
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    write_json(output / "timings.json", rows)
    write_json(
        output / "manifest.json",
        {
            "exploration_manifest_sha256": sha256_file(saved / "manifest.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "selected_implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/postranking.py"
            ),
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
