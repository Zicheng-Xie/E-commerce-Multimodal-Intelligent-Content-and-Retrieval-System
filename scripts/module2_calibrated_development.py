"""Compare grouped human pairwise calibration against the frozen development baseline."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import run as _bootstrap  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_human_review_inputs import import_precision_answers
from module2_mainline import CONFIG, REVISION, _inputs
from module2_mainline_reports import label_fingerprints
from module2_shared_evaluation import precision_delta
from module2_shared_gallery_retrieval import _score
from multimodal_ecommerce.calibrated_ranking import (
    FEATURES,
    CalibrationContext,
    calibrated_order,
    fit_pairwise,
    preference_scores,
    public_features,
)
from multimodal_ecommerce.human_retrieval_evaluation import evaluate_human_retrieval
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.selected_runtime_io import RuntimeAccess, digest

OUTPUT = (
    PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/development_calibration"
)
SOURCE_NAMES = (
    "src/multimodal_ecommerce/calibrated_ranking.py",
    "scripts/module2_calibrated_development.py",
)


def development_inputs(evidence: Evidence) -> tuple[dict, dict, dict]:
    """Reuse all verified existing human judgments without rewriting old snapshots.

    Args:
        evidence: Registered development-only reader.

    Returns:
        Shared manifest, complete selected baseline and its frozen configuration.
    """
    manifest, _, _, tasks = _inputs(evidence, load(CONFIG))
    queries = manifest["query_manifest"]
    progress = import_precision_answers(
        evidence, REVISION / "human_review", tasks, queries
    )
    if progress["binary_completed"] != 341:
        raise ValueError("The completed existing human review must remain intact")
    folder = REVISION.parent / "public_conflicts_20261110/human_review"
    tasks = evidence.json(folder / "tasks.private.json")
    progress = import_precision_answers(evidence, folder, tasks, queries)
    if progress["binary_completed"] != 4:
        raise ValueError("All existing public-conflict review decisions required")
    selected = load(
        PROJECT_ROOT / "configs/module2_selected_development_candidate.json"
    )
    baseline = evidence.json(
        PROJECT_ROOT / selected["rankings_file"], selected["rankings_sha256"]
    )[selected["variant"]]
    if label_fingerprints(baseline, queries) != selected["human_labels_sha256"]:
        raise ValueError("Compare against the same fixed human label snapshot")
    return manifest, baseline, selected


def observations(
    evidence: Evidence, manifest: dict, baseline: dict
) -> tuple[dict, dict]:
    """Build public features from the already scored model pool, never from answers.

    Args:
        evidence: Verified development cache reader.
        manifest: Public development gallery/query context.
        baseline: Selected complete orders.

    Returns:
        Public features and explicit fallback reasons for unscored contexts.
    """
    parent = REVISION.parent
    protocols = {
        name: evidence.json(parent / name / "protocol.json")
        for name in ("subject_recall", "primary_recall")
    }
    captions = evidence.json(parent / "subject_recall/query_captions.json")["rows"]
    replay = evidence.json(
        PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/replay/report.json"
    )
    pins = replay["inputs_sha256"]
    titles = {r["id"]: r["title"] for r in manifest["catalog"]}
    result, skipped = {}, {}
    for key, query in manifest["query_manifest"].items():
        name = "subject_recall" if query["mode"] == "text" else "primary_recall"
        job = protocols[name]["jobs"][key]
        if query["mode"] == "image" and not job["active"]:
            skipped[key] = "original_policy_semantic_abstention"
            continue
        path = (
            parent
            / name
            / "scores"
            / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        )
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative not in pins:
            raise ValueError(
                "Only existing source-bound actual model logits may be reused"
            )
        saved = evidence.json(path, pins[relative])
        if saved["identity"] != sha256_file(parent / name / "protocol.json"):
            raise ValueError("The actual Top20 logits belong to another model protocol")
        if set(saved["scores"]) != set(baseline[key][:20]):
            skipped[key] = "protected_original_pool_not_fully_cross_scored"
            continue
        context = CalibrationContext(
            order=baseline[key],
            original=job["baseline"] if name == "subject_recall" else job["original"],
            scores=saved["scores"],
            titles={c: titles[c] for c in baseline[key]},
            text=query.get("text"),
            caption=captions[key]["value"] if key in captions else None,
        )
        matrix, group = public_features(context)
        result[key] = dict(context=context, matrix=matrix, group=group)
    return result, skipped


def crossfit(visible: dict, queries: dict, baseline: dict) -> tuple[dict, dict, dict]:
    """Train outside the scored query's entire semantic family for every comparison.

    Args:
        visible: Public feature matrices and semantic families.
        queries: Explicit existing development judgments, used by training only.
        baseline: Full selected orders to preserve when scoring is unavailable.

    Returns:
        Complete variants, numeric model bank and train/test family audit.
    """
    grades = {
        key: [
            (
                q["judgments"].get(cid, {}).get("grade")
                if q["judgments"].get(cid, {}).get("human")
                else None
            )
            for cid in baseline[key][:20]
        ]
        for key, q in queries.items()
        if key in visible
    }
    variants = {
        "baseline": baseline,
        "pairwise_top20_control": dict(baseline),
        "pairwise_one_swap": dict(baseline),
    }
    bank, audit = {}, {}
    for group in sorted({r["group"] for r in visible.values()}):
        train = [key for key, r in visible.items() if r["group"] != group]
        test = [key for key, r in visible.items() if r["group"] == group]
        if set(train) & set(test):
            raise AssertionError("Training and scored queries must be disjoint")
        bank[group] = fit_pairwise(
            [(visible[key]["matrix"], grades[key]) for key in train]
        )
        audit[group] = dict(
            train_queries=train,
            test_queries=test,
            train_families=sorted({visible[k]["group"] for k in train}),
            excluded_family=group,
            pairs=bank[group]["pairs"],
        )
        for key in test:
            scores = preference_scores(visible[key]["matrix"], bank[group])
            variants["pairwise_top20_control"][key] = calibrated_order(
                baseline[key], scores, False
            )
            variants["pairwise_one_swap"][key] = calibrated_order(
                baseline[key], scores, True
            )
    bank["fallback_all_training"] = fit_pairwise(
        [(r["matrix"], grades[k]) for k, r in visible.items()]
    )
    return variants, bank, audit


def swap_proof(before: dict, after: dict, queries: dict) -> dict:
    """Verify same-label nonregression for the conservative position-swap candidate.

    Args:
        before: Complete baseline orders.
        after: Complete learned ranking orders.
        queries: Evaluation-only human judgments.

    Returns:
        Actual swap evidence, unresolved count and a strict adoption condition.
    """
    rows = []
    for key, old in before.items():
        new = after[key]
        changed = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        if not changed:
            continue
        if (
            len(changed) != 2
            or changed[0] < 2
            or changed[0] >= 10
            or changed[1] < 10
            or changed[1] >= 20
        ):
            raise ValueError(
                "Conservative comparison must make only one protected Top20 swap"
            )
        i, j = changed
        if old[i] != new[j] or old[j] != new[i]:
            raise ValueError(
                "All other positions and the full candidate set must stay fixed"
            )
        labels = queries[key]["judgments"]
        values = [
            labels.get(c, {}).get("grade") if labels.get(c, {}).get("human") else None
            for c in (old[i], old[j])
        ]
        rows.append(
            dict(
                query_id=key,
                outgoing=old[i],
                incoming=old[j],
                from_rank=j + 1,
                to_rank=i + 1,
                outgoing_human=values[0],
                incoming_human=values[1],
            )
        )
    complete = all(
        r["outgoing_human"] in (0, 1) and r["incoming_human"] in (0, 1) for r in rows
    )
    nonregression = complete and all(
        r["incoming_human"] >= r["outgoing_human"] for r in rows
    )
    gain = (
        sum((r["incoming_human"] or 0) - (r["outgoing_human"] or 0) for r in rows)
        if complete
        else None
    )
    return dict(
        rows=rows,
        all_changed_pairs_human_binary=complete,
        unresolved_pairs=sum(
            r[side] is None
            for r in rows
            for side in ("incoming_human", "outgoing_human")
        ),
        same_label_all_six_nondecreasing=nonregression,
        exact_net_hits=gain,
        eligible_for_further_development=bool(nonregression and gain and gain > 0),
        default_changed=False,
    )


def run(output: Path) -> dict:
    """Run one fixed grouped comparison without new labels, model inference or holdout access.

    Args:
        output: Separate registered exposed-development experiment directory.

    Returns:
        Actual results and the unchanged-default decision.
    """
    output = RuntimeAccess().output(output)
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    manifest, baseline, selected = development_inputs(evidence)
    queries = manifest["query_manifest"]
    visible, skipped = observations(evidence, manifest, baseline)
    protocol = dict(
        role="exposed_development_semantic_family_exclusion_crossfit",
        products=len(manifest["catalog"]),
        queries=len(queries),
        scored_queries=len(visible),
        baseline_fallback_queries=skipped,
        parent=selected["candidate_id"],
        human_training_only=True,
        unknown_is_negative=False,
        validation="Leave the entire public semantic family out; same-family image/text queries never train their scored model. These queries have prior development exposure, so this is not independent acceptance.",
        features=list(FEATURES),
        l2=0.03,
        pairwise_margin=1.3862943611198906,
        guarded_swaps_per_query=1,
        guarded_prefix=2,
        optimizer="deterministic damped Newton, at most 60 iterations",
        teacher_models_changed=False,
        new_human_tasks=0,
        protected_data_used=False,
        label_fingerprints=label_fingerprints(baseline, queries),
        inputs=dict(evidence.files),
        source_sha256={n: sha256_file(PROJECT_ROOT / n) for n in SOURCE_NAMES},
    )
    experiment_root = output
    output = experiment_root / "runs" / digest(protocol)[:16]
    if (output / "protocol.json").exists() and load(
        output / "protocol.json"
    ) != protocol:
        raise ValueError(
            "Preserve completed experiment; use a new output for a changed protocol"
        )
    write_json(output / "protocol.json", protocol)
    variants, bank, audit = crossfit(visible, queries, baseline)
    target = load(PROJECT_ROOT / "configs/module2_target_p80_r90.json")
    human_queries = {
        key: dict(
            q, judgments={c: r for c, r in q["judgments"].items() if r.get("human")}
        )
        for key, q in queries.items()
    }
    human = {
        name: evaluate_human_retrieval(order, human_queries, target)
        for name, order in variants.items()
    }
    six = {name: _score(order, human_queries) for name, order in variants.items()}
    proof = swap_proof(baseline, variants["pairwise_one_swap"], queries)
    result = dict(
        role=protocol["role"],
        products=len(manifest["catalog"]),
        queries=len(queries),
        human_labels=sum(len(q["judgments"]) for q in human_queries.values()),
        human_positive=sum(
            r["grade"] == 1
            for q in human_queries.values()
            for r in q["judgments"].values()
        ),
        human_negative=sum(
            r["grade"] == 0
            for q in human_queries.values()
            for r in q["judgments"].values()
        ),
        families=dict(Counter(r["group"] for r in visible.values())),
        human_metrics=human,
        human_only_six_metric_bounds=six,
        paired_precision={
            name: precision_delta(baseline, order, human_queries)
            for name, order in variants.items()
        },
        swap_evidence=proof,
        new_model_calls=0,
        new_human_tasks=0,
        selected_default=selected["candidate_id"],
        default_changed=False,
        decision="retain_frozen_baseline; proposed calibration not promoted without positive same-label evidence",
        formal_acceptance=False,
        formal_metrics=None,
    )
    for name, payload in (
        ("rankings.json", variants),
        ("model_bank.json", bank),
        ("group_audit.json", audit),
        ("results.json", result),
    ):
        write_json(output / name, payload)
    write_json(
        experiment_root / "current.json",
        dict(
            result_path=str((output / "results.json").relative_to(PROJECT_ROOT)),
            rankings_path=str((output / "rankings.json").relative_to(PROJECT_ROOT)),
            model_bank_path=str((output / "model_bank.json").relative_to(PROJECT_ROOT)),
        ),
    )
    print(
        f"Grouped comparison complete: {len(visible)} scored queries, {len(audit)} families, {len(proof['rows'])} conservative swaps."
    )
    return result


def main() -> None:
    """Run the isolated development calibration experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = run(parser.parse_args().output)
    for name, metric in result["human_metrics"].items():
        print(name, metric["scopes"]["all"]["precision_at_10"])
    print("Guarded change evidence:", result["swap_evidence"])


if __name__ == "__main__":
    main()
