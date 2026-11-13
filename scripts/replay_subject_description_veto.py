"""Replay a public subject/prop constraint on the exact frozen development logits."""

from __future__ import annotations

from pathlib import Path

import run as _bootstrap  # noqa: F401
from module2_calibrated_development import development_inputs
from module2_development_benchmark import Evidence, load
from module2_shared_evaluation import precision_delta
from module2_shared_gallery_retrieval import _score
from module2_subject_description_development import OUTPUT, change_evidence
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.selected_runtime_io import digest
from multimodal_ecommerce.subject_description import subject_swap


def main() -> None:
    """Evaluate the predeclared type-priority selection without another model call."""
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    receipt = evidence.json(OUTPUT / "inference_receipt.json")
    for name, expected in receipt["files_sha256"].items():
        evidence.digest(OUTPUT / name, expected)
    protocol = evidence.json(OUTPUT / "protocol.json")
    pools = evidence.json(OUTPUT / "pools.json")["subject_caption"]
    scores = evidence.json(OUTPUT / "scores.json")["subject_caption"]
    manifest, baseline, selected = development_inputs(evidence)
    titles = {r["id"]: r["title"] for r in manifest["catalog"]}
    ranking, traces = dict(baseline), {}
    for key, job in protocol["jobs"].items():
        if baseline[key] != job["order"]:
            raise ValueError("Replay requires the unchanged baseline query/gallery")
        ranking[key], traces[key] = subject_swap(
            dict(
                order=baseline[key],
                pool=pools[key]["pool"],
                titles=titles,
                scores=scores[key],
                observation=job["observation"],
                type_veto=True,
            )
        )
    queries = manifest["query_manifest"]
    human = {
        k: dict(
            q, judgments={c: r for c, r in q["judgments"].items() if r.get("human")}
        )
        for k, q in queries.items()
    }
    target = load(PROJECT_ROOT / "configs/module2_target_p80_r90.json")
    result = dict(
        role="exposed_development_cached_subject_constraint_ablation",
        parent=selected["candidate_id"],
        products=845,
        queries=55,
        changed_queries=sum(r["changed"] for r in traces.values()),
        policy="Same actual description/model/pool/logits; explicit foreign props may leave Top10 only for primary-type candidates without explicit attribute conflicts; keep first two and at most one swap.",
        human_metrics=evaluate_human_retrieval(ranking, human, target),
        human_only_six_metric_bounds=_score(ranking, human),
        paired_precision=precision_delta(baseline, ranking, human),
        change_evidence=change_evidence(baseline, ranking, queries),
        traces=traces,
        new_model_calls=0,
        reused_actual_pair_scores=40,
        new_human_tasks=0,
        default_changed=False,
        formal_acceptance=False,
        formal_metrics=None,
        inputs=evidence.files,
        source_sha256={
            name: sha256_file(PROJECT_ROOT / name)
            for name in (
                "src/multimodal_ecommerce/subject_description.py",
                "scripts/replay_subject_description_veto.py",
            )
        },
    )
    output: Path = OUTPUT / "type_veto" / digest(result["source_sha256"])[:16]
    if (output / "results.json").exists() and load(output / "results.json") != result:
        raise ValueError("Preserve the previous postprocessing experiment")
    write_json(output / "rankings.json", ranking)
    write_json(output / "results.json", result)
    write_json(
        OUTPUT / "type_veto/current.json",
        dict(
            result_path=str((output / "results.json").relative_to(PROJECT_ROOT)),
            rankings_path=str((output / "rankings.json").relative_to(PROJECT_ROOT)),
        ),
    )
    print(result["human_metrics"]["scopes"]["all"]["precision_at_10"])
    print(result["change_evidence"])


if __name__ == "__main__":
    main()
