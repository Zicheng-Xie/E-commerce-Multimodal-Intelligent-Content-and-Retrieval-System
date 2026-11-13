"""Fixed residual follow-up after the absolute rank blends underperformed."""

from __future__ import annotations

import json

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.measurement import cluster_interval
from multimodal_ecommerce.precision_reranking import pareto_gate
from multimodal_ecommerce.rank_correction import (
    RankCorrectionRankCorrectionInputs,
    rank_correction,
)
from multimodal_ecommerce.retrieval_strategies import extended_metrics
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_metric_training import OUT
from train_retrieval_metric import METRICS


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve all residual comparison evidence.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    target = OUT / "rank_correction"
    if target.exists():
        raise FileExistsError("Preserve all residual comparison evidence")
    target.mkdir()
    parents = [OUT / "analysis/report.json", OUT / "region_attention/report.json"]
    for root in [OUT / "analysis", OUT / "region_attention"]:
        for cohort in ["old_development", "new_group_development"]:
            parents.extend(
                [
                    root / f"{cohort}_rankings.json",
                    root / f"{cohort}_query_metrics.json",
                ]
            )
    write_json(
        target / "protocol.json",
        {
            "scope": "Post-result exploratory follow-up; same internal development cohorts, no heldout reuse",
            "reason": "Absolute blends imported a weaker ranker; transfer only the learned change relative to its untrained comparator",
            "weight": 0.5,
            "anchors": 2,
            "pool": 30,
            "hyperparameter_grid": False,
            "methods": ["metric_delta", "attention_delta", "query_adapter_delta"],
            "parents": {
                str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p)
                for p in parents
            },
            "implementation": {
                p: sha256_file(PROJECT_ROOT / p)
                for p in [
                    "scripts/analyze_rank_correction.py",
                    "src/multimodal_ecommerce/rank_correction.py",
                ]
            },
            "formal_acceptance": False,
            "independent_validation_required": True,
        },
    )
    # Full qrels are derived from the already scored development lists, not a new test.
    from multimodal_ecommerce.dataset_release import read_rows
    from train_retrieval_metric import DATA, OLD

    source = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    records = read_rows(DATA / "records.jsonl")
    truth = {
        "old_development": {
            q["record"]["id"]: set(q["relevant_ids"]) for q in source["queries"]
        },
        "new_group_development": {
            q["id"]: {
                g["id"]
                for g in records
                if g["source"] == "shop"
                and g["project_role"] == "development_validation"
                and g["identity"] == q["identity"]
            }
            for q in records
            if q["source"] == "user" and q["project_role"] == "development_validation"
        },
    }
    results: dict = {}
    for cohort in ["old_development", "new_group_development"]:
        metric = json.loads(
            (OUT / f"analysis/{cohort}_rankings.json").read_text(encoding="utf-8")
        )
        regional = json.loads(
            (OUT / f"region_attention/{cohort}_rankings.json").read_text(
                encoding="utf-8"
            )
        )
        old_rows = json.loads(
            (OUT / f"analysis/{cohort}_query_metrics.json").read_text(encoding="utf-8")
        )
        groups = {r["query_id"]: r["group"] for r in old_rows}
        rankings, measurements = {}, []
        for q, m in metric.items():
            a = regional[q]
            rankings[q] = {
                "baseline": m["baseline"],
                "metric_delta": rank_correction(
                    RankCorrectionRankCorrectionInputs(
                        m["baseline"], m["original_global"], m["learned_global"]
                    )
                ),
                "attention_delta": rank_correction(
                    RankCorrectionRankCorrectionInputs(
                        m["baseline"],
                        a["frozen_three_view_attention"],
                        a["learned_three_view_attention"],
                    )
                ),
                "query_adapter_delta": rank_correction(
                    RankCorrectionRankCorrectionInputs(
                        m["baseline"],
                        a["frozen_three_view_attention"],
                        a["ridge_query_attention"],
                    )
                ),
            }
            for method, ranked in rankings[q].items():
                measurements.append(
                    {
                        "query_id": q,
                        "group": groups[q],
                        "method": method,
                        **extended_metrics(ranked, truth[cohort][q]),
                    }
                )
        summary = {
            m: {
                k: float(np.mean([r[k] for r in measurements if r["method"] == m]))
                for k in METRICS
            }
            for m in rankings[q]
        }
        paired = {}
        refs = [r for r in measurements if r["method"] == "baseline"]
        for method in ["metric_delta", "attention_delta", "query_adapter_delta"]:
            rows = [r for r in measurements if r["method"] == method]
            paired[method] = {
                k: cluster_interval(
                    [r[k] - b[k] for r, b in zip(rows, refs)],
                    [r["group"] for r in rows],
                )
                for k in METRICS
            }
        results[cohort] = {
            "summary": summary,
            "paired_group_bootstrap_exploratory": paired,
        }
        write_json(target / f"{cohort}_rankings.json", rankings)
        write_json(target / f"{cohort}_query_metrics.json", measurements)
    gates = {}
    for method in ["metric_delta", "attention_delta", "query_adapter_delta"]:
        old = results["old_development"]["summary"]
        new = results["new_group_development"]["summary"]
        old_gate = pareto_gate(old[method], old["baseline"])
        regressions = [
            k for k in METRICS if new[method][k] < new["baseline"][k] - 1e-12
        ]
        gates[method] = {
            "old": old_gate,
            "new_regressed_metrics": regressions,
            "passes_both_development_gates": old_gate["passes_development_metric_gate"]
            and not regressions,
        }
    write_json(
        target / "report.json",
        {
            **results,
            "gates": gates,
            "default_strategy_changed": False,
            "formal_acceptance": False,
            "formal_metrics": None,
        },
    )
    print(json.dumps({c: r["summary"] for c, r in results.items()}))


if __name__ == "__main__":
    main()
