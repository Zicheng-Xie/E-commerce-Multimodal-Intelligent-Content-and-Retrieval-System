"""Replay the bounded method, new-query AI review and development exposure guard."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import ijson

import run  # noqa: F401
from advance_rank_stability import CANDIDATE, NAME, PARENT, ROOT, check
from multimodal_ecommerce.ai_review_metrics import metric_ranges
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.rank_stability import stable_tail
from multimodal_ecommerce.retrieval_strategies import extended_metrics
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_fresh_query_check import BUCKETS, POLICY, QUEUE, REGISTRY, bucket
from report_focus_coverage import paired_differences
from run_fresh_query_check import ROOT as FRESH
from run_fresh_query_check import reconstruct


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def persist_or_verify(path: Path, value: object) -> None:
    """Apply the persist or verify workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        assert load(path) == value, f"Replay differs: {path}"
    else:
        write_json(path, value)


def replay_metadata() -> tuple[list[dict], list[dict], Any]:
    """Return replay metadata for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    protocol = load(FRESH / "protocol.json")
    source = PROJECT_ROOT / protocol["source"]
    assert sha256_file(source) == protocol["source_sha256"]
    prior = load(PARENT / "post_development_exclusion_snapshot.json")
    excluded = set(prior["family_keys"])
    groups: dict = {b: [] for b in BUCKETS}
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > POLICY["scan_after"]:
                name = bucket(str(raw.get("title") or ""))
                if name and raw.get("url"):
                    row = dict(
                        id=f"product/train/{sid}",
                        source_id=str(sid),
                        dataset="product",
                        split="train",
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=name,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                        project_role="exposed_new_query_development_metadata_not_acceptance",
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{POLICY['seed']}:{row['id']}".encode()
                            ).hexdigest(),
                        )
                        groups[name].append(row)
            if number == POLICY["scan_limit"]:
                break
    selected, seen = [], set(excluded)
    for name in BUCKETS:
        count = 0
        for row in sorted(groups[name], key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            selected.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    assert selected == read_rows(QUEUE) == read_rows(FRESH / "source_candidates.jsonl")
    receipt = load(FRESH / "metadata_receipt.json")
    assert receipt["scanned"] == number
    assert receipt["eligible_before_family_dedup"] == {
        b: len(v) for b, v in groups.items()
    }
    queries = read_rows(FRESH / "queries.evaluator.jsonl")
    signatures = list(prior["image_fingerprints"])
    for q in queries:
        expected = next(r for r in selected if r["bucket"] == q["acquisition_bucket"])
        assert expected["id"] == q["source_product_id"]
        actual = fingerprints(Path(q["image_path"]))
        assert actual == q["image_fingerprints"]
        assert not is_exposed(actual, signatures, 4)
        signatures.append(actual)
    assert len(queries) == 6
    return selected, queries, prior


def replay_reviews() -> None:
    """Apply the replay reviews workflow step."""
    observations = load(FRESH / "native_review/observations.json")
    assert observations["all_gallery_sheets_viewed"] == list(range(1, 13))
    mapping = read_rows(FRESH / "native_review/gallery_mapping.jsonl")
    expected = []
    for qid, review in observations["queries"].items():
        for doc in mapping:
            pos = doc["position"]
            grade = (
                1
                if pos in review["positive_positions"]
                else None if str(pos) in review["unknown_positions"] else 0
            )
            reason = (
                review["positive_reason"]
                if grade == 1
                else (
                    review["unknown_positions"][str(pos)]
                    if grade is None
                    else review["negative_reason"]
                )
            )
            expected.append(
                dict(
                    query_id=qid,
                    candidate_id=doc["id"],
                    grade=grade,
                    reason=reason,
                    native_position=pos,
                    evidence="native_review/observations.json",
                    review_origin="model",
                    human_label=False,
                )
            )
    assert expected == read_rows(FRESH / "judgments.ai.jsonl")
    assert len(expected) == 2508 and sum(r["grade"] is None for r in expected) == 8


def tail_effect(rankings: dict, labels: list[dict], modes: dict) -> dict:
    """Return tail effect for this workflow.

    Args:
        rankings: Candidate rankings keyed by query and retrieval method.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Mapping with fields: query_rows, same_unknown_completions, formal_acceptance.
    """
    lookup = {(r["query_id"], r["candidate_id"]): r["grade"] for r in labels}
    result = []
    for qid, methods in rankings.items():
        positives = {c for q, c in lookup if q == qid and lookup[q, c] == 1}
        unknown = sorted(c for q, c in lookup if q == qid and lookup[q, c] is None)
        changes: dict = {m: [] for m in ["ap", "mrr"]}
        for bits in itertools.product([False, True], repeat=len(unknown)):
            relevant = positives | {c for c, keep in zip(unknown, bits) if keep}
            before, after = [
                (
                    extended_metrics(methods[n], relevant)
                    if relevant
                    else dict(ap=0.0, mrr=0.0)
                )
                for n in [CANDIDATE, NAME]
            ]
            for m in changes:
                changes[m].append(after[m] - before[m])
        result.append(
            dict(
                query_id=qid,
                mode=modes[qid],
                delta_ranges={m: [min(v), max(v)] for m, v in changes.items()},
            )
        )
    return dict(
        query_rows=result, same_unknown_completions=True, formal_acceptance=False
    )


def verify_history() -> list:
    """Verify history.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    roots = [
        ("module2_blind", "completion"),
        ("module2_precision", "completion"),
        ("module2_training", "completion"),
        ("module2_precision60", "completion"),
        ("module2_business_review", "preparation"),
        ("module2_business_acceptance", "preparation"),
        ("module2_machine_assist", "completion"),
        ("module2_continuation", "completion"),
        ("module2_target_p60_r85", "completion"),
        ("module2_precision_recall", "completion"),
        ("module2_catalog_review", "preparation"),
        ("module2_ai_review", "completion"),
        ("module2_expanded_development", "completion"),
        ("module2_visual_coverage", "completion"),
        ("module2_focus_coverage", "completion"),
    ]
    checked = []
    for name, prefix in roots:
        folder = PROJECT_ROOT / "artifacts" / name
        path = folder / (prefix + "_receipt.json")
        receipt = load(path)
        verify_files(folder, receipt["files"])
        if receipt.get("project_files"):
            verify_files(PROJECT_ROOT, receipt["project_files"])
        checked.append(
            dict(
                stage=name,
                receipt_sha256=sha256_file(path),
                files=len(receipt["files"]),
            )
        )
    return checked


def main() -> None:
    """Run the command-line workflow defined by this module."""
    check()
    verify_files(ROOT, load(ROOT / "comparison_receipt.json")["files"])
    verify_files(PROJECT_ROOT, load(FRESH / "inference_protocol.json")["files"])
    verify_files(PROJECT_ROOT, load(FRESH / "acquisition_receipt.json")["files"])
    verify_files(PROJECT_ROOT, load(ROOT / "runtime_protocol.json")["files"])
    selected, queries, prior = replay_metadata()
    replay_reviews()
    inputs = load(PARENT / "inputs.public.json")
    modes = {q["id"]: "image" if q["image_path"] else "text" for q in inputs["queries"]}
    old = load(PARENT / "region_guard/rankings.json")
    rankings = load(ROOT / "rankings.json")
    for qid, methods in rankings.items():
        assert methods[NAME] == (
            stable_tail(old[qid]["region_title"], old[qid][CANDIDATE])
            if modes[qid] == "image"
            else old[qid]["cross_evidence_pool60"]
        )
        assert methods[NAME][:10] == old[qid][CANDIDATE][:10]
        assert all(methods[n] == old[qid][n] for n in methods if n != NAME)
    labels = read_rows(PARENT / "judgments.ai.jsonl")
    assert metric_ranges(rankings, labels, modes) == load(
        ROOT / "internal_metrics.ai.development.json"
    )
    assert paired_differences(rankings, labels, modes) == load(
        ROOT / "internal_paired_differences.json"
    )
    persist_or_verify(
        ROOT / "internal_tail_effect.json", tail_effect(rankings, labels, modes)
    )
    fresh_rankings = reconstruct()
    assert fresh_rankings == load(FRESH / "rankings.json")
    fresh_labels = read_rows(FRESH / "judgments.ai.jsonl")
    fresh_modes = {q: "image" for q in fresh_rankings}
    metrics = metric_ranges(fresh_rankings, fresh_labels, fresh_modes)
    assert metrics == load(FRESH / "internal_metrics.ai.development.json")
    assert paired_differences(fresh_rankings, fresh_labels, fresh_modes) == load(
        FRESH / "internal_paired_differences.json"
    )
    verify_files(FRESH, load(FRESH / "evaluation_receipt.json")["files"])
    embeds, pairs = 0, 0
    for q in queries:
        folder = FRESH / "runs" / q["id"]
        description = load(folder / "query_description.json")
        assert description["gallery_or_judgments_supplied"] is False
        diagnostics = load(folder / "results.json")["diagnostics"]
        assert (
            diagnostics["device"] == "cuda:0"
            and diagnostics["dtype"] == "torch.bfloat16"
        )
        embeds += len(diagnostics["calls"])
        cross = load(folder / "cross_encoder.json")
        assert len(cross["diagnostics"]["timings"]) == len(cross["pool"]) == 20
        assert all(
            r["query_modalities"] == ["image", "text"]
            for r in cross["diagnostics"]["timings"]
        )
        pairs += len(cross["pool"])
    assert pairs == 120
    demo = load(ROOT / "demo_stable_fresh_request/results.json")
    verify_files(PROJECT_ROOT, demo["source_files"])
    assert demo["full_order"] == fresh_rankings["fresh_headphones"][NAME]
    assert load(ROOT / "query_guard_probe.json")["output_created"] is False
    snapshot = dict(prior)
    snapshot["ids"] = sorted(set(prior["ids"]) | {r["id"] for r in selected})
    snapshot["product_source_ids"] = sorted(
        set(prior["product_source_ids"]) | {r["source_id"] for r in selected}
    )
    snapshot["family_keys"] = sorted(
        set(prior["family_keys"]) | {k for r in selected for k in r["family_keys"]}
    )
    snapshot["image_fingerprints"] = prior["image_fingerprints"] + [
        q["image_fingerprints"] for q in queries
    ]
    snapshot["files"] = {
        **prior["files"],
        **{
            p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
            for p in [QUEUE, REGISTRY]
        },
    }
    snapshot["parent_snapshot_sha256"] = sha256_file(
        PARENT / "post_development_exclusion_snapshot.json"
    )
    snapshot.update(
        role="all_exposed_development_and_reserved_exclusion_not_acceptance",
        stage_new_metadata_candidates=18,
        stage_new_query_images=6,
        stage_new_query_predictions=6,
    )
    snapshot["counts"] = dict(
        records=len(snapshot["ids"]),
        product_ids=len(snapshot["product_source_ids"]),
        deepfashion_pair_ids=len(
            snapshot["deepfashion_pair_ids_conservative_cross_split"]
        ),
        unique_image_bytes=len({s["sha256"] for s in snapshot["image_fingerprints"]}),
        family_keys=len(snapshot["family_keys"]),
    )
    persist_or_verify(ROOT / "post_development_exclusion_snapshot.json", snapshot)
    capacities = dict(
        query_capacity=metrics["query_capacity"],
        capacity_by_scope=metrics["capacity_by_scope"],
        zero_positive_queries=[
            r["query_id"]
            for r in metrics["query_capacity"]
            if r["recall_status"] == "zero_positive"
        ],
        uncertain_only_queries=[
            r["query_id"]
            for r in metrics["query_capacity"]
            if r["recall_status"] == "possibly_zero_positive"
        ],
        fixed_k=10,
        all_six_queries_retained=True,
        labels_unchanged=True,
        formal_metrics=None,
        goal_met=False,
    )
    persist_or_verify(FRESH / "internal_feasibility.json", capacities)
    history = verify_history()
    result = dict(
        verified=True,
        old_query_count=18,
        old_new_method_full_rankings=18,
        fresh_query_count=6,
        fresh_full_rankings=30,
        fresh_AI_relations=2508,
        unknown_relations=8,
        full_embedding_calls=embeds,
        full_cross_pairs=pairs,
        query_only_VLM_calls=6,
        metadata_replayed=True,
        all_AI_relations_replayed=True,
        real_query_orders_replayed=True,
        top10_invariant=True,
        prior_sealed_roots=history,
        human_labels=0,
        acceptance_predictions=0,
        heldout_pixels_opened=0,
        formal_acceptance=False,
    )
    persist_or_verify(ROOT / "verification.json", result)
    if (ROOT / "completion_receipt.json").exists():
        receipt = load(ROOT / "completion_receipt.json")
        verify_files(ROOT, receipt["files"])
        verify_files(PROJECT_ROOT, receipt["project_files"])
    print(json.dumps({k: v for k, v in result.items() if k != "prior_sealed_roots"}))


if __name__ == "__main__":
    main()
