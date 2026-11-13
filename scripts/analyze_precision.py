"""Analyze all preregistered precision candidates using development-only groups."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: F401,E402
from multimodal_ecommerce.blind_evaluation import file_map, verify_files  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.measurement import (  # noqa: E402
    cluster_interval,
    distribution,
)
from multimodal_ecommerce.precision_reranking import (  # noqa: E402
    FEATURE_NAMES,
    color_descriptor,
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
from precision_experiment import EXPERIMENT, OUTPUT, POLICY, PREVIOUS  # noqa: E402

METRICS = ("precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1")
BASE = "reciprocal_attention_fusion"


LEARNED: dict[str, tuple[bool, bool, int]] = {
    "ranknet_teacher_oof": (True, False, 0),
    "ranknet_teacher_anchor2_oof": (True, False, 2),
    "ranknet_student_oof": (False, False, 0),
    "ranknet_student_anchor2_oof": (False, False, 2),
    "ranknet_distilled_oof": (False, True, 0),
    "ranknet_distilled_anchor2_oof": (False, True, 2),
}


def load_entries() -> list[dict]:
    """Return load entries for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Preregistered development policy changed; Complete every development
            pair before analysis; Candidate cache mismatch; Nonfinite cached ranking
            features.
    """
    verify_files(
        OUTPUT,
        json.loads((OUTPUT / "pair_cache_manifest.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    frozen = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    if frozen["policy"] != POLICY or FEATURE_NAMES != POLICY["features"]:
        raise ValueError("Preregistered development policy changed")
    source = json.loads((EXPERIMENT / "protocol.json").read_text(encoding="utf-8"))[
        "source_protocol"
    ]
    records = json.loads(
        (EXPERIMENT / "index/records.json").read_text(encoding="utf-8")
    )
    positions = {r["id"]: i for i, r in enumerate(records)}
    record_map = {r["id"]: r for r in records}
    groups = {
        r["id"]: f"{r['split']}/{r['pair_id']}"
        for r in read_rows(PROJECT_ROOT / "data/mentor_fashion_1000_v1/samples.jsonl")
    }
    previous = json.loads((PREVIOUS / "rankings.json").read_text(encoding="utf-8"))
    cache = {r["query_id"]: r for r in read_rows(OUTPUT / "pair_scores.jsonl")}
    if len(cache) != len(source["queries"]):
        raise ValueError("Complete every development pair before analysis")
    colors = {}
    entries = []
    with np.load(PREVIOUS / "scores.npz", allow_pickle=False) as scores:
        for number, query in enumerate(source["queries"]):
            row = query["record"]
            key = row["id"]
            cached = cache[key]
            base = previous[key][BASE]
            pool = base[: POLICY["candidate_pool"]]
            if pool != cached["candidate_ids"]:
                raise ValueError("Candidate cache mismatch")
            query_color = color_descriptor(Path(row["image_path"]))
            for candidate in pool:
                if candidate not in colors:
                    colors[candidate] = color_descriptor(
                        Path(record_map[candidate]["image_path"])
                    )
            indices = [positions[c] for c in pool]
            features = np.column_stack(
                [
                    scores["global"][number, indices],
                    scores["gallery_attention"][number, indices],
                    scores[BASE][number, indices],
                    [float(query_color @ colors[c]) for c in pool],
                    np.linspace(1.0, 0.0, len(pool)),
                ]
            )
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite cached ranking features")
            positives = set(query["relevant_ids"])
            entries.append(
                {
                    "query_id": key,
                    "group": row["pair_group"],
                    "candidate_groups": [groups[c] for c in pool],
                    "features": features.tolist(),
                    "teacher": cached["logits"],
                    "labels": [c in positives for c in pool],
                    "positive_ids": sorted(positives),
                    "base_ids": base,
                    "candidate_ids": pool,
                    "category": row["category"],
                }
            )
    return entries


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the existing precision analysis.

    Notes:
        Writes derived artifacts through target.mkdir, write_json.
    """
    target = OUTPUT / "analysis"
    if target.exists():
        raise FileExistsError("Preserve the existing precision analysis")
    started = time.perf_counter()
    entries = load_entries()
    ranking_by_query = {}
    for entry in entries:
        base, pool, teacher_scores = (
            entry["base_ids"],
            entry["candidate_ids"],
            np.asarray(entry["teacher"]),
        )
        rankings = {
            BASE: base,
            "cross_encoder": refine_pool(base, pool, teacher_scores),
        }
        for weight in POLICY["rank_blend_weights"]:
            rankings[f"cross_encoder_blend{int(weight*100)}"] = refine_pool(
                base, pool, teacher_scores, blend=weight
            )
        rankings["cross_encoder_anchor2_blend50"] = refine_pool(
            base, pool, teacher_scores, blend=0.5, anchors=2
        )
        ranking_by_query[entry["query_id"]] = rankings
    groups = sorted({e["group"] for e in entries})
    fold_evidence = []
    for group in groups:
        models = {}
        for teacher, distill in ((True, False), (False, False), (False, True)):
            models[(teacher, distill)] = fit_pairwise(
                entries,
                POLICY,
                excluded_groups={group},
                include_teacher=teacher,
                distill=distill,
            )
        fold_evidence.append(
            {"validation_group": group, "models": list(models.values())}
        )
        for entry in entries:
            if entry["group"] != group:
                continue
            for name, (teacher, distill, anchors) in LEARNED.items():
                values = pairwise_scores(entry, models[(teacher, distill)])
                ranking_by_query[entry["query_id"]][name] = refine_pool(
                    entry["base_ids"], entry["candidate_ids"], values, anchors=anchors
                )
    query_metrics = []
    methods = list(ranking_by_query[entries[0]["query_id"]])
    for entry in entries:
        for method in methods:
            ranked = ranking_by_query[entry["query_id"]][method]
            query_metrics.append(
                {
                    "query_id": entry["query_id"],
                    "pair_group": entry["group"],
                    "method": method,
                    **extended_metrics(ranked, set(entry["positive_ids"])),
                }
            )
    by_method = {m: [r for r in query_metrics if r["method"] == m] for m in methods}
    means = {
        m: {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        for m, rows in by_method.items()
    }
    intervals = {
        m: {
            k: cluster_interval([r[k] for r in rows], [r["pair_group"] for r in rows])
            for k in METRICS
        }
        for m, rows in by_method.items()
    }
    gates = {m: pareto_gate(means[m], means[BASE]) for m in methods if m != BASE}
    fast_candidates = [
        m
        for m in LEARNED
        if not LEARNED[m][0] and gates[m]["passes_development_metric_gate"]
    ]
    fast_candidates.sort(
        key=lambda m: (
            -means[m]["precision@10"],
            -means[m]["ap"],
            -means[m]["recall@10"],
            m,
        )
    )
    selected = fast_candidates[0] if fast_candidates else None
    report = {
        "role": "internal_development_cross_validation_only",
        "formal_acceptance": False,
        "queries": len(entries),
        "query_groups": len(groups),
        "gallery": len(entries[0]["base_ids"]),
        "methods": means,
        "intervals": intervals,
        "gates": gates,
        "selected_fast_candidate_for_new_validation": selected,
        "precision_at_10_ceiling": float(
            np.mean([min(10, len(e["positive_ids"])) / 10 for e in entries])
        ),
        "top30_precision_at_10_ceiling": float(
            np.mean([min(10, sum(e["labels"])) / 10 for e in entries])
        ),
        "cross_encoder_rerank_only_seconds": distribution(
            [r["rerank_seconds"] for r in read_rows(OUTPUT / "pair_scores.jsonl")]
        ),
        "no_consumed_heldout_used": True,
        "policy": POLICY,
        "limits": "The old development set already informed method design; group CV does not restore independent acceptance. All configurations retained, including regressions.",
        "analysis_seconds": time.perf_counter() - started,
    }
    target.mkdir()
    write_json(target / "report.json", report)
    write_json(target / "query_metrics.json", query_metrics)
    write_json(target / "rankings.json", ranking_by_query)
    write_json(target / "fold_models.json", fold_evidence)
    if selected is not None:
        include_teacher, distill, anchors = LEARNED[selected]
        model = fit_pairwise(
            entries,
            POLICY,
            excluded_groups=set(),
            include_teacher=include_teacher,
            distill=distill,
        )
        write_json(
            target / "candidate_model.json",
            {
                "role": "development_candidate_requires_new_independent_validation",
                "method": selected,
                "candidate_pool": POLICY["candidate_pool"],
                "anchors": anchors,
                "model": model,
                "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
                "task": "clothing_image_same_item",
            },
        )
    write_json(
        target / "manifest.json",
        {
            "files": file_map(target),
            "implementation_sha256": sha256_file(
                PROJECT_ROOT / "src/multimodal_ecommerce/precision_reranking.py"
            ),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        },
    )
    print(
        json.dumps(
            {
                "development_means": means,
                "selected_fast_candidate": selected,
                "formal_acceptance": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
