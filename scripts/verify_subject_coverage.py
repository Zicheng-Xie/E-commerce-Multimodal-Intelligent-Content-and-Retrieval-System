"""Replay full comparisons with an explicit machine-roundoff feature tolerance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import run  # noqa: F401
from evaluate_primary_form_coverage import paired
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import sha256_file
from multimodal_ecommerce.semantic_ranker import ranked_ids
from subject_coverage_acquisition import PARENT, QUEUE, ROOT
from subject_coverage_experiment import (
    COHORTS,
    COVERAGE,
    calculate_features,
    calculate_orders,
    feature_inputs,
    guard,
    pair_identity,
)
from subject_roi_experiment import load, once, parent_check


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    once(
        ROOT / "verification_protocol.json",
        dict(
            source_sha256=sha256_file(Path(__file__)),
            feature_absolute_tolerance=1e-12,
            feature_relative_tolerance=0,
            reason="Strict dictionary equality failed on tune/text_06: one feature differed by 1.1102230246251565e-16. Preserve saved features and predictions; tolerate only numerical roundoff. Candidate pools and every full ranking still must match exactly.",
            score_and_metric_tolerance=0,
            labels_changed=False,
            model_or_ranking_policy_changed=False,
        ),
    )
    assert load(ROOT / "verification_protocol.json")["source_sha256"] == sha256_file(
        Path(__file__)
    )
    for name in [
        "label_receipt.json",
        "detection_receipt.json",
        "fg_encoding_receipt.json",
        "reranker_receipt.json",
        "prediction_receipt.json",
    ]:
        verify_files(ROOT, load(ROOT / name)["files"])
    verify_files(PARENT, load(PARENT / "completion_receipt.json")["files"])
    parent_check()
    before = load(PARENT / "post_experiment_exclusion_snapshot.json")
    candidates = read_rows(QUEUE)
    assert len(candidates) == 80
    assert all(
        not set(r["family_keys"]) & set(before["family_keys"]) for r in candidates
    )
    signatures = list(before["image_fingerprints"])
    for row in read_rows(ROOT / "retry/selected.final.jsonl"):
        assert not is_exposed(row["fingerprints"], signatures, 4)
        signatures.append(row["fingerprints"])
    discrepancies = []
    for cohort in COHORTS:
        ranks, details = calculate_orders(cohort)
        assert ranks == load(ROOT / cohort / "predictions.json")
        assert details == load(ROOT / cohort / "runtime_scores.json")
        banks = feature_inputs(cohort)
        docs = {d["id"]: d for d in banks[0].records}
        for query in load(ROOT / cohort / "queries.public.json"):
            qid = query["id"]
            actual = calculate_features(cohort, query, banks)
            saved = load(ROOT / cohort / "runs" / qid / "features.json")
            assert {
                k: v for k, v in actual.items() if k not in {"base", "features"}
            } == {k: v for k, v in saved.items() if k not in {"base", "features"}}
            for key in ["base", "features"]:
                a, b = np.array(actual[key]), np.array(saved[key])
                assert np.allclose(a, b, rtol=0, atol=1e-12)
                error = float(np.max(np.abs(a - b)))
                if error:
                    discrepancies.append(
                        dict(
                            cohort=cohort,
                            query=qid,
                            field=key,
                            max_absolute_difference=error,
                        )
                    )
            raw = load(ROOT / cohort / "runs" / qid / "cross.json")
            prior = load(COVERAGE / cohort / "runs" / qid / "cross.json")
            assert raw["cache_sha256"] == sha256_file(
                COVERAGE / cohort / "runs" / qid / "cross.json"
            )
            cached = dict(zip(prior["pool"], prior["logits"]))
            assert set(raw["new_pair_ids"]) | set(raw["reused_pair_ids"]) == set(
                raw["pool"]
            )
            assert not set(raw["new_pair_ids"]) & set(raw["reused_pair_ids"])
            for cid, score in zip(raw["pool"], raw["logits"]):
                assert (
                    pair_identity(query, actual["description"], docs[cid])
                    == raw["pair_identities"][cid]
                )
                if cid in raw["reused_pair_ids"]:
                    assert score == cached[cid]
            offsets = (
                np.array(actual["offsets"]) if actual["offsets"] is not None else None
            )
            assert (
                rerank_complete(
                    actual["ids"],
                    np.array(actual["base"]),
                    raw["pool"],
                    np.array(raw["logits"]),
                    offsets,
                )
                == ranks[qid]["cross20"]
            )
            matrix = np.array(actual["features"])
            assert ranked_ids(actual["ids"], matrix[:, 0]) == ranks[qid]["global"]
            assert (
                ranked_ids(actual["ids"], 0.8 * matrix[:, 1] + 0.2 * matrix[:, 3])
                == ranks[qid]["region_title"]
            )
            if qid.startswith("text_"):
                assert (
                    ranks[qid]["cross20"]
                    == ranks[qid]["image_fusion20"]
                    == ranks[qid]["image_roi_fusion20"]
                )
        labels = read_rows(ROOT / cohort / "judgments.ai.jsonl")
        original = read_rows(COVERAGE / cohort / "judgments.ai.jsonl")
        assert labels[: len(original)] == original and len(labels) == 193 * len(ranks)
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired(ranks, labels, modes, metrics) == load(
            ROOT / cohort / "internal_paired_differences.json"
        )
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            source_sha256=sha256_file(Path(__file__)),
            same_gallery_and_queries=True,
            complete_full_rankings_replayed=240,
            Qwen_recall_features_replayed=True,
            feature_roundoff_discrepancies=discrepancies,
            original_6720_labels_unchanged=True,
            AI_labels=7720,
            human_labels=0,
            exact_reused_pair_identities_and_logits=True,
            original_text_route_exact=True,
            all_predecessor_chain_seals_unchanged=True,
            new_raw_family_and_image_overlap=0,
            metrics_and_shared_label_differences_replayed=True,
            formal_metrics=None,
            independent_acceptance_queries=0,
            default_changed=False,
        ),
    )
    print(dict(verified=True, roundoff_discrepancies=discrepancies), flush=True)


if __name__ == "__main__":
    main()
