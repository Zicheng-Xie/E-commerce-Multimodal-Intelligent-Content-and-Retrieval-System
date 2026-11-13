"""Development follow-up motivated by the measured 30-pair latency failure."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from analyze_precision import BASE, METRICS, OUTPUT, POLICY, load_entries  # noqa: E402
from multimodal_ecommerce.blind_evaluation import file_map, verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.measurement import (  # noqa: E402
    cluster_interval,
    distribution,
)
from multimodal_ecommerce.precision_boundary import boundary_refine  # noqa: E402
from multimodal_ecommerce.precision_reranking import (  # noqa: E402
    fit_pairwise,
    pairwise_scores,
    pareto_gate,
    refine_pool,
)
from multimodal_ecommerce.retrieval_strategies import extended_metrics  # noqa: E402
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve boundary exploration evidence.
        ValueError: Need a complete pair timing trace, not a guessed latency; Timing
            trace mismatch.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    target = OUTPUT / "boundary_analysis"
    if target.exists():
        raise FileExistsError("Preserve boundary exploration evidence")
    verify_files(
        OUTPUT / "analysis",
        json.loads((OUTPUT / "analysis/manifest.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    target.mkdir()
    policy: dict = {
        "role": "internal_development_followup_only",
        "reason": "30-pair reranker too slow; pure learned reorder regressed",
        "editable_windows": [[2, 10], [5, 10], [8, 10]],
        "boundary_rank_blends": [0.5, 1.0],
        "student_residual_rank_blend": 0.25,
        "student_anchors": 2,
        "selection_metric_gate": "Same original +2pp P@10 and no regressions in other five means",
        "no_new_heldout_used": True,
        "parent_report_sha256": sha256_file(OUTPUT / "analysis/report.json"),
    }
    write_json(target / "protocol.json", policy)
    entries = load_entries()
    folds = json.loads(
        (OUTPUT / "analysis/fold_models.json").read_text(encoding="utf-8")
    )
    models = {
        f["validation_group"]: {
            (m["include_teacher"], m["distill"]): m for m in f["models"]
        }
        for f in folds
    }
    diagnostics = json.loads(
        (OUTPUT / "reranker_diagnostics.json").read_text(encoding="utf-8")
    )["timings"]
    cache = read_rows(OUTPUT / "pair_scores.jsonl")
    if len(diagnostics) != sum(len(r["candidate_ids"]) for r in cache):
        raise ValueError("Need a complete pair timing trace, not a guessed latency")
    timings = {}
    offset = 0
    for row in cache:
        trace = diagnostics[offset : offset + len(row["candidate_ids"])]
        offset += len(trace)
        if [r["document_id"] for r in trace] != row["candidate_ids"]:
            raise ValueError("Timing trace mismatch")
        timings[row["query_id"]] = [r["seconds"] for r in trace]
    rankings = {}
    per_query = []
    latency: dict = {}
    for entry in entries:
        key = entry["query_id"]
        base = entry["base_ids"]
        ranked = {BASE: base}
        teacher = np.asarray(entry["teacher"])
        for anchors, count in policy["editable_windows"]:
            for blend in policy["boundary_rank_blends"]:
                method = f"boundary_a{anchors}_n{count}_mix{int(blend*100)}"
                ranked[method] = boundary_refine(
                    base,
                    teacher[anchors : anchors + count],
                    anchors=anchors,
                    count=count,
                    blend=blend,
                )
                latency.setdefault(method, []).append(
                    sum(timings[key][anchors : anchors + count])
                )
        for distill in (False, True):
            method = (
                "residual_" + ("distilled" if distill else "student") + "_anchor2_mix25"
            )
            values = pairwise_scores(entry, models[entry["group"]][(False, distill)])
            ranked[method] = refine_pool(
                base, entry["candidate_ids"], values, blend=0.25, anchors=2
            )
        rankings[key] = ranked
        for method, ids in ranked.items():
            per_query.append(
                {
                    "query_id": key,
                    "pair_group": entry["group"],
                    "method": method,
                    **extended_metrics(ids, set(entry["positive_ids"])),
                }
            )
    methods = list(rankings[entries[0]["query_id"]])
    chosen = {m: [r for r in per_query if r["method"] == m] for m in methods}
    means = {
        m: {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        for m, rows in chosen.items()
    }
    gates = {m: pareto_gate(means[m], means[BASE]) for m in methods if m != BASE}
    for method, gate in gates.items():
        if method.startswith("boundary"):
            gate["cached_pair_component_p95_seconds"] = distribution(latency[method])[
                "p95"
            ]
            gate["pair_component_below_5s"] = (
                gate["cached_pair_component_p95_seconds"] <= 5
            )
        else:
            gate["requires_online_cross_encoder"] = False
    intervals = {
        m: {
            k: cluster_interval(
                [r[k] - b[k] for r, b in zip(rows, chosen[BASE])],
                [r["pair_group"] for r in rows],
            )
            for k in METRICS
        }
        for m, rows in chosen.items()
        if m != BASE
    }
    passing = [
        m
        for m, g in gates.items()
        if g["passes_development_metric_gate"]
        and g.get("pair_component_below_5s", True)
    ]
    passing.sort(key=lambda m: (-means[m]["precision@10"], -means[m]["ap"], m))
    selected = passing[0] if passing else None
    report = {
        "role": "internal_development_followup_only",
        "formal_acceptance": False,
        "methods": means,
        "paired_delta_intervals": intervals,
        "gates": gates,
        "selected_for_new_independent_validation": selected,
        "latency_scope": "Measured cached pair components only; excludes image decoding, embedding, model switching, loading, network and UI; not end-to-end acceptance",
        "latency": {m: distribution(v) for m, v in latency.items()},
        "policy": policy,
    }
    write_json(target / "report.json", report)
    write_json(target / "query_metrics.json", per_query)
    write_json(target / "rankings.json", rankings)
    if selected is not None:
        if selected.startswith("residual"):
            model = fit_pairwise(
                entries, POLICY, excluded_groups=set(), distill="distilled" in selected
            )
            candidate = {
                "type": "residual_student",
                "model": model,
                "anchors": 2,
                "blend": 0.25,
                "pool": 30,
            }
        else:
            bits = selected.split("_")
            candidate = {
                "type": "boundary_cross_encoder",
                "anchors": int(bits[1][1:]),
                "count": int(bits[2][1:]),
                "blend": int(bits[3][3:]) / 100,
            }
        write_json(
            target / "candidate_policy.json",
            {
                "role": "development_candidate_requires_new_validation",
                "method": selected,
                **candidate,
            },
        )
    write_json(
        target / "manifest.json",
        {
            "files": file_map(target),
            "implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/precision_boundary.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {"development_means": means, "selected": selected, "gates": gates},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
