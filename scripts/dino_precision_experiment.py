"""Develop attention-selected visual matching without touching consumed tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from analyze_precision import (  # noqa: E402
    BASE,
    EXPERIMENT,
    METRICS,
    OUTPUT,
    POLICY,
    load_entries,
)
from multimodal_ecommerce.blind_evaluation import file_map  # noqa: E402
from multimodal_ecommerce.local_visual import (  # noqa: E402
    LocalVisualEncoder,
    dense_similarity,
)
from multimodal_ecommerce.measurement import (  # noqa: E402
    cluster_interval,
    distribution,
)
from multimodal_ecommerce.precision_reranking import (  # noqa: E402
    FEATURE_NAMES,
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
        FileExistsError: Keep original visual development evidence.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    target = OUTPUT / "dino_analysis"
    if target.exists():
        raise FileExistsError("Keep original visual development evidence")
    target.mkdir()
    write_json(
        target / "protocol.json",
        {
            "role": "internal_development_only",
            "model": json.loads(
                (PROJECT_ROOT / "configs/dino_model.json").read_text(encoding="utf-8")
            ),
            "model_config_sha256": sha256_file(
                PROJECT_ROOT / "configs/dino_model.json"
            ),
            "global_and_local_blends": [0.25, 0.5],
            "anchors": 2,
            "pool": 30,
            "joint": "half global cosine plus half symmetric local MaxSim",
            "student": "previous group-disjoint hard-negative distillation with two added visual features; rank blend .25 or 1, anchors2",
            "gate": "+2pp Precision@10 and no regression in other five development means",
            "heldout_used": False,
            "feature_implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/local_visual.py"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
    )
    entries = load_entries()
    records = json.loads(
        (EXPERIMENT / "index/records.json").read_text(encoding="utf-8")
    )
    source = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    encoder = LocalVisualEncoder()
    gallery_vectors = []
    gallery_patches = []
    gallery_indices = []
    query_vectors = []
    query_patches = []
    query_indices = []
    extraction = []
    try:
        for number, row in enumerate(records, 1):
            started = time.perf_counter()
            vector, patches, indices = encoder.encode(Path(row["image_path"]))
            extraction.append(
                {
                    "id": row["id"],
                    "role": "gallery",
                    "seconds": time.perf_counter() - started,
                }
            )
            gallery_vectors.append(vector)
            gallery_patches.append(patches)
            gallery_indices.append(indices)
            if number % 50 == 0:
                print(f"Compact visual gallery {number}/{len(records)}", flush=True)
        for item in source["queries"]:
            started = time.perf_counter()
            vector, patches, indices = encoder.encode(
                Path(item["record"]["image_path"])
            )
            extraction.append(
                {
                    "id": item["record"]["id"],
                    "role": "query",
                    "seconds": time.perf_counter() - started,
                }
            )
            query_vectors.append(vector)
            query_patches.append(patches)
            query_indices.append(indices)
    finally:
        encoder.close()
    gallery_bank = np.stack(gallery_vectors)
    patch_bank = np.stack(gallery_patches)
    np.savez(
        target / "features.npz",
        gallery_global=gallery_bank,
        gallery_local=patch_bank,
        gallery_patch_indices=np.stack(gallery_indices),
        query_global=np.stack(query_vectors),
        query_local=np.stack(query_patches),
        query_patch_indices=np.stack(query_indices),
    )
    positions = {r["id"]: i for i, r in enumerate(records)}
    rankings = {}
    dense_timings = []
    extended_entries = []
    for number, entry in enumerate(entries):
        before = time.perf_counter()
        local = dense_similarity(query_patches[number], patch_bank)
        global_scores = gallery_bank @ query_vectors[number]
        dense_timings.append(time.perf_counter() - before)
        ids = entry["candidate_ids"]
        candidate_indices = [positions[key] for key in ids]
        ranked = {BASE: entry["base_ids"]}
        for branch, values in (
            ("global", global_scores),
            ("local", local),
            ("joint", (global_scores + local) / 2),
        ):
            for blend in (0.25, 0.5):
                name = f"dino_{branch}_anchor2_mix{int(blend*100)}"
                ranked[name] = refine_pool(
                    entry["base_ids"],
                    ids,
                    values[candidate_indices],
                    blend=blend,
                    anchors=2,
                )
        rankings[entry["query_id"]] = ranked
        extended_entries.append(
            {
                **entry,
                "features": np.column_stack(
                    [
                        entry["features"],
                        global_scores[candidate_indices],
                        local[candidate_indices],
                    ]
                ).tolist(),
            }
        )
    folds = []
    for group in sorted({e["group"] for e in entries}):
        model = fit_pairwise(
            extended_entries, POLICY, excluded_groups={group}, distill=True
        )
        model["feature_names"] = FEATURE_NAMES + ["dino_global", "dino_local"]
        folds.append({"validation_group": group, "model": model})
        for entry in extended_entries:
            if entry["group"] != group:
                continue
            values = pairwise_scores(entry, model)
            for blend in (0.25, 1.0):
                name = f"dino_distilled_anchor2_mix{int(blend*100)}"
                rankings[entry["query_id"]][name] = refine_pool(
                    entry["base_ids"],
                    entry["candidate_ids"],
                    values,
                    blend=blend,
                    anchors=2,
                )
    methods = list(rankings[entries[0]["query_id"]])
    per_query = []
    for entry in entries:
        for method in methods:
            per_query.append(
                {
                    "query_id": entry["query_id"],
                    "pair_group": entry["group"],
                    "method": method,
                    **extended_metrics(
                        rankings[entry["query_id"]][method], set(entry["positive_ids"])
                    ),
                }
            )
    method_rows = {m: [r for r in per_query if r["method"] == m] for m in methods}
    means = {
        m: {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        for m, rows in method_rows.items()
    }
    deltas = {
        m: {
            k: cluster_interval(
                [r[k] - b[k] for r, b in zip(rows, method_rows[BASE])],
                [r["pair_group"] for r in rows],
            )
            for k in METRICS
        }
        for m, rows in method_rows.items()
        if m != BASE
    }
    gates = {m: pareto_gate(means[m], means[BASE]) for m in methods if m != BASE}
    passing = [m for m, g in gates.items() if g["passes_development_metric_gate"]]
    passing.sort(key=lambda m: (-means[m]["precision@10"], -means[m]["ap"], m))
    selected = passing[0] if passing else None
    report = {
        "role": "internal_development_only",
        "formal_acceptance": False,
        "queries": len(entries),
        "methods": means,
        "paired_delta_intervals": deltas,
        "gates": gates,
        "selected_for_new_independent_validation": selected,
        "query_visual_encoding_seconds": distribution(
            [r["seconds"] for r in extraction if r["role"] == "query"]
        ),
        "dense_comparison_seconds": distribution(dense_timings),
        "source_boxes_or_masks_used": False,
        "limits": "All are development comparisons; student uses leave-one-pair-group-out. No independent acceptance claim.",
    }
    write_json(target / "report.json", report)
    write_json(target / "query_metrics.json", per_query)
    write_json(target / "rankings.json", rankings)
    write_json(target / "fold_models.json", folds)
    write_json(target / "extraction_timings.json", extraction)
    if selected is not None:
        if "distilled" in selected:
            model = fit_pairwise(
                extended_entries, POLICY, excluded_groups=set(), distill=True
            )
            model["feature_names"] = FEATURE_NAMES + ["dino_global", "dino_local"]
            parameters = {
                "model": model,
                "blend": 0.25 if selected.endswith("mix25") else 1.0,
            }
        else:
            parameters = {
                "branch": selected.split("_")[1],
                "blend": 0.25 if selected.endswith("mix25") else 0.5,
            }
        write_json(
            target / "candidate_policy.json",
            {
                "role": "development_candidate_requires_new_validation",
                "method": selected,
                "anchors": 2,
                "pool": 30,
                **parameters,
            },
        )
    write_json(
        target / "manifest.json",
        {"files": file_map(target), "formal_acceptance": False},
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
