"""Audit strategy rankings, exact metric arithmetic and a replayed scoring path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.retrieval_strategies import (
    EXPLORATION_POLICY,
    STRATEGY_NAMES,
    StrategyExplorer,
    ordered_indices,
    subset_region_index,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def independent_metrics(ids: list[str], relevant: set[str]) -> dict:
    """Compute metric definitions independently using Python rank positions.

    Args:
        ids: Identifiers in the order expected by the associated arrays.
        relevant: Relevant used by this operation.

    Returns:
        Mapping with fields: ap, mrr, hit@1.

    Raises:
        ValueError: Invalid ranking or incomplete relevance denominator.
    """
    positions = [i for i, key in enumerate(ids, 1) if key in relevant]
    if not positions or len(positions) != len(relevant) or len(set(ids)) != len(ids):
        raise ValueError("Invalid ranking or incomplete relevance denominator")
    result = {
        "ap": sum(hit / rank for hit, rank in enumerate(positions, 1)) / len(relevant),
        "mrr": 1 / positions[0],
        "hit@1": float(positions[0] == 1),
    }
    for k in (5, 10, 20, 50):
        found = sum(rank <= k for rank in positions)
        result[f"precision@{k}"] = found / k
        result[f"recall@{k}"] = found / len(relevant)
        result[f"hit@{k}"] = float(found > 0)
    actual = sum(1 / np.log2(rank + 1) for rank in positions if rank <= 10)
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    result["ndcg@10"] = float(actual / ideal)
    return result


def verify(output: Path) -> dict:
    """Audit every persisted score ranking; replay one full vector-only query.

    Args:
        output: Output used by this operation.

    Returns:
        Mapping with fields: passed, ranking_rows_verified,
        scores_replayed_one_query, metric_implementation, no_label_input_to_ranking,
        formal_acceptance.

    Raises:
        ValueError: Strategy artifact hash mismatch; Strategy implementation changed;
            Experiment script changed.
    """
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha256_file(path) != checksum:
            raise ValueError("Strategy artifact hash mismatch")
    if (
        sha256_file(PROJECT_ROOT / "src/multimodal_ecommerce/retrieval_strategies.py")
        != manifest["implementation_sha256"]
    ):
        raise ValueError("Strategy implementation changed")
    if (
        sha256_file(PROJECT_ROOT / "scripts/explore_retrieval_strategies.py")
        != manifest["script_sha256"]
    ):
        raise ValueError("Experiment script changed")
    saved = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert saved["policy"] == EXPLORATION_POLICY
    assert saved["query_batch_statistics_used"] is False
    parent = PROJECT_ROOT / "artifacts/module2_regions_benchmark_v1"
    assert sha256_file(parent / "manifest.json") == saved["parent_manifest_sha256"]
    index, _, _ = RegionalIndex.load(parent / "index")
    protocol = saved["source_protocol"]
    ids = [r["id"] for r in index.records]
    rankings = json.loads((output / "rankings.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "query_metrics.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    computed: dict[str, list] = {name: [] for name in STRATEGY_NAMES}
    with np.load(output / "scores.npz", allow_pickle=False) as archive:
        for number, query in enumerate(protocol["queries"]):
            include = np.array(
                [
                    i
                    for i, r in enumerate(index.records)
                    if r["image_sha256"] != query["record"]["image_sha256"]
                ],
                dtype=int,
            )
            excluded = sorted(set(range(len(ids))) - set(include.tolist()))
            allowed_ids = [ids[i] for i in include]
            for name in STRATEGY_NAMES:
                scores = archive[name][number]
                assert np.isnan(scores[excluded]).all()
                ordered = ordered_indices(scores[include], allowed_ids)
                ranked = [allowed_ids[i] for i in ordered]
                assert ranked == rankings[query["record"]["id"]][name]
                metrics = independent_metrics(ranked, set(query["relevant_ids"]))
                expected = next(
                    row
                    for row in rows
                    if row["query_id"] == query["record"]["id"]
                    and row["strategy"] == name
                )
                for key, value in metrics.items():
                    assert np.isclose(value, expected[key])
                computed[name].append(metrics)
        # One replay catches dataflow/implementation mismatch without repeating all
        # label-free policies for each query after scores already audit every row.
        first = protocol["queries"][0]
        include = np.array(
            [
                i
                for i, r in enumerate(index.records)
                if r["image_sha256"] != first["record"]["image_sha256"]
            ],
            dtype=int,
        )
        active = subset_region_index(index, include)
        with np.load(parent / "query_vectors.npz", allow_pickle=False) as queries:
            vectors = queries["vectors"][0, : int(queries["counts"][0])]
        scores, _ = StrategyExplorer(active).evaluate(vectors)
        for name in STRATEGY_NAMES:
            assert np.allclose(scores[name], archive[name][0, include], atol=1e-6)
    for name, values in computed.items():
        for metric in (
            "recall@10",
            "precision@10",
            "ap",
            "ndcg@10",
            "mrr",
            "hit@1",
            "recall@50",
        ):
            assert np.isclose(
                np.mean([r[metric] for r in values]),
                report["summary"][name][metric]["estimate"],
            )
    return {
        "passed": True,
        "ranking_rows_verified": len(protocol["queries"]) * len(STRATEGY_NAMES),
        "scores_replayed_one_query": len(STRATEGY_NAMES),
        "metric_implementation": "Independent Python rank-position arithmetic",
        "no_label_input_to_ranking": True,
        "formal_acceptance": False,
    }


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a fresh verification path.

    Notes:
        Writes derived artifacts through write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("artifacts/module2_strategy_exploration"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/module2_strategy_verification.json"),
    )
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a fresh verification path")
    result = verify(PROJECT_ROOT / args.experiment)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
