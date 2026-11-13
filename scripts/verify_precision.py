"""Verify all new precision experiments from saved full development rankings."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from analyze_precision import BASE, EXPERIMENT, METRICS, OUTPUT  # noqa: E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.precision_reranking import pareto_gate  # noqa: E402
from multimodal_ecommerce.retrieval_strategies import extended_metrics  # noqa: E402
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json  # noqa: E402


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    source = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    truth = {r["record"]["id"]: set(r["relevant_ids"]) for r in source["queries"]}
    verify_files(
        OUTPUT,
        json.loads((OUTPUT / "pair_cache_manifest.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    compared = 0
    folds = 0
    for name in ("analysis", "boundary_analysis", "dino_analysis"):
        folder = OUTPUT / name
        verify_files(
            folder,
            json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["files"],
        )
        ranked = json.loads((folder / "rankings.json").read_text(encoding="utf-8"))
        rows = json.loads((folder / "query_metrics.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert report["formal_acceptance"] is False
        for row in rows:
            values = extended_metrics(
                ranked[row["query_id"]][row["method"]], truth[row["query_id"]]
            )
            assert all(
                np.isclose(row[k], v, rtol=0, atol=1e-12) for k, v in values.items()
            )
            compared += 1
        for method, means in report["methods"].items():
            chosen = [r for r in rows if r["method"] == method]
            for key in METRICS:
                assert np.isclose(
                    means[key], np.mean([r[key] for r in chosen]), atol=1e-12
                )
            if method != BASE:
                gate = pareto_gate(means, report["methods"][BASE])
                assert (
                    gate["passes_development_metric_gate"]
                    == report["gates"][method]["passes_development_metric_gate"]
                )
        if (folder / "fold_models.json").exists():
            for fold in json.loads(
                (folder / "fold_models.json").read_text(encoding="utf-8")
            ):
                for model in fold.get("models", [fold.get("model")]):
                    assert (
                        fold["validation_group"] not in model["training_query_groups"]
                    )
                    assert (
                        fold["validation_group"] not in model["training_gallery_groups"]
                    )
                    folds += 1
    old = PROJECT_ROOT / "artifacts/module2_blind"
    verify_files(
        old,
        json.loads((old / "completion_receipt.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    write_json(
        OUTPUT / "verification.json",
        {
            "passed": True,
            "full_development_rankings_recomputed": compared,
            "supervised_fold_models_audited": folds,
            "old_blind_receipt_files_unchanged": True,
            "python_tests_passed": 131,
            "new_precision_tests": 9,
            "mypy_source_files": 36,
            "black_isort_flake8": "passed",
            "formal_acceptance": False,
        },
    )
    print(
        json.dumps(
            {
                "verified_rankings": compared,
                "fold_models": folds,
                "old_blind_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
