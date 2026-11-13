"""Fixed image-only / semantic-ROI comparisons on exposed development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from compare_fgclip2_retrieval import ROOT as FGROOT
from compare_fgclip2_retrieval import check as check_fg
from compare_fgclip2_retrieval import image_file, read_array, relative, text_file
from evaluate_primary_form_coverage import paired
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.fgclip_retrieval import (
    FGClipEncoder,
    FgclipRetrievalAuxiliaryScoresInputs,
    auxiliary_scores,
)
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.subject_roi import (
    POLICY,
    SUBJECTS,
    SubjectDetector,
    SubjectRoiImageOnlyOrderInputs,
    image_only_order,
    save_subject_crop,
    subject_prompt,
)
from report_focus_coverage import paired_differences

ROOT = PROJECT_ROOT / "artifacts/module2_subject_roi"
COVERAGE = PROJECT_ROOT / "artifacts/module2_primary_form_coverage"
COHORTS = ("tune", "regression", "fresh")
METHODS = ("image_fusion20", "image_roi_fusion20")


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def once(path: Path, value: object) -> None:
    """Persist a new result without replacing an existing artifact.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        if load(path) != value:
            raise ValueError(f"Preserve frozen artifact: {path.name}")
    else:
        write_json(path, value)


def parent_check() -> None:
    """Apply the parent check workflow step."""
    receipt = load(FGROOT / "completion_receipt.json")
    verify_files(FGROOT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["source_files"])
    check_fg()


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen ROI protocol.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Preserve frozen ROI protocol")
    parent_check()
    inputs = load(FGROOT / "inputs.json")
    files = {
        relative(p): sha256_file(p)
        for p in [
            Path(__file__),
            PROJECT_ROOT / "scripts/prepare_grounding_model.py",
            ROOT / "detector_manifest.json",
            FGROOT / "completion_receipt.json",
            FGROOT / "inputs.json",
            *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
        ]
    }
    profiles: dict = {}
    for cohort in ["tune", "regression"]:
        path = COVERAGE / cohort / "profiles.public.jsonl"
        rows = read_rows(path)
        for row in rows:
            previous = profiles.get(row["id"])
            if previous is not None:
                assert previous == row["product_type"]
            profiles[row["id"]] = row["product_type"]
        files[relative(path)] = sha256_file(path)
    requests: dict = {}

    def request(image_key: object, hint: str) -> str:
        """Return request for this workflow.

        Args:
            image_key: Image key used by this operation.
            hint: Hint used by this operation.

        Returns:
            Text produced by this operation.
        """
        prompt = subject_prompt(hint)
        identity = dict(image_key=image_key, prompt=prompt)
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        requests.setdefault(key, dict(**identity, hint=hint))
        return key

    for cohort, data in inputs["cohorts"].items():
        for doc in data["gallery"]:
            doc["roi_request"] = request(doc["image_key"], profiles[doc["id"]])
        for query in data["queries"]:
            query["roi_request"] = (
                request(query["image_key"], inputs["texts"].get(query["text_key"], ""))
                if query["image_key"]
                else None
            )
        for name in ["runtime_scores.json", "predictions.json"]:
            path = FGROOT / cohort / name
            files[relative(path)] = sha256_file(path)
    inputs["roi_requests"] = requests
    once(ROOT / "inputs.json", inputs)
    files[relative(ROOT / "inputs.json")] = sha256_file(ROOT / "inputs.json")
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            subject_dictionary=SUBJECTS,
            files=files,
            primary_method=POLICY["primary_method"],
            ablation="image_fusion20: identical image-only fusion with ROI removed",
            same_gallery_queries_labels=True,
            fixed_k=10,
            original_judgments=6720,
            default_strategy_changed=False,
            all_queries_already_exposed=True,
            independent_AI_blind_predictions=0,
            formal_acceptance=False,
            formal_metrics=None,
            selection="Main method fixed before detection; no hyperparameter search. Compare all six paired metrics, P@10>=.60 hard goal, Recall@10>=.85 preference; default unchanged unless a later independently validated release is authorized.",
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    print(
        json.dumps(
            dict(
                prepared=True,
                requests=len(requests),
                unsupported=sum(r["prompt"] is None for r in requests.values()),
            )
        ),
        flush=True,
    )


def check() -> Any:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "protocol.json")
    assert protocol["policy"] == POLICY
    assert protocol["subject_dictionary"] == [[a, b] for a, b in SUBJECTS]
    verify_files(PROJECT_ROOT, protocol["files"])
    parent_check()
    return load(ROOT / "inputs.json")


def detection_path(key: str) -> Path:
    """Return detection path for this workflow.

    Args:
        key: Key used by this operation.

    Returns:
        Numeric result computed from the supplied values.
    """
    return ROOT / "detections" / key / "result.json"


def detect() -> None:
    """Apply the detect workflow step.

    Raises:
        FileExistsError: Preserve completed detections.

    Notes:
        Writes derived artifacts through once.
    """
    inputs = check()
    if (ROOT / "detection_receipt.json").exists():
        raise FileExistsError("Preserve completed detections")
    detector = SubjectDetector(ROOT / "detector_manifest.json")
    try:
        for i, (key, row) in enumerate(inputs["roi_requests"].items(), 1):
            target = detection_path(key)
            if target.exists():
                continue
            source = PROJECT_ROOT / inputs["images"][row["image_key"]]
            assert sha256_file(source) == row["image_key"]
            if row["prompt"]:
                result = detector.detect(source, row["prompt"])
            else:
                result = dict(
                    prompt=None,
                    decision=dict(
                        status="global_fallback",
                        reason="unsupported or ambiguous public subject; query/product retained",
                        selected=None,
                    ),
                    detections=[],
                    seconds=0,
                )
            result.update(request_id=key, source_sha256=row["image_key"])
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    source, result["decision"], target.parent
                )
            once(target, result)
            if i % 24 == 0:
                print(
                    json.dumps(dict(detected=i, total=len(inputs["roi_requests"]))),
                    flush=True,
                )
        runtime = detector.diagnostics()
    finally:
        detector.close()
    once(ROOT / "detector_runtime.json", runtime)
    files = {
        relative(p): sha256_file(p)
        for p in [
            ROOT / "detector_runtime.json",
            *sorted((ROOT / "detections").rglob("*")),
        ]
        if p.is_file()
    }
    decisions = [load(detection_path(k)) for k in inputs["roi_requests"]]
    once(
        ROOT / "detection_receipt.json",
        dict(
            files=files,
            model_forward_calls=len(runtime["calls"]),
            accepted_roi=sum(r["decision"]["status"] == "roi" for r in decisions),
            global_fallback=sum(r["decision"]["status"] != "roi" for r in decisions),
            all_images_and_queries_retained=True,
        ),
    )


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve completed ROI features.

    Notes:
        Writes derived artifacts through np.savez_compressed, once, target.parent.mkdir.
    """
    inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "detection_receipt.json")["files"])
    if (ROOT / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed ROI features")
    encoder = FGClipEncoder(FGROOT / "model_manifest.json")
    mapping: dict[str, dict] = {}
    try:
        for i, key in enumerate(inputs["roi_requests"], 1):
            result = load(detection_path(key))
            if result["decision"]["status"] == "roi":
                source = detection_path(key).with_name("zoom.png")
                checksum = sha256_file(source)
                assert checksum == result["crop"]["zoom_sha256"]
                target = ROOT / "features" / f"{checksum}.npz"
                if not target.exists():
                    global_vector, dense, info = encoder.image(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        target, global_vector=global_vector, dense=dense
                    )
                    once(
                        target.with_suffix(".json"),
                        dict(
                            crop_sha256=checksum,
                            feature_sha256=sha256_file(target),
                            detail=info,
                        ),
                    )
                mapping[key] = dict(file=relative(target), crop_sha256=checksum)
            else:
                mapping[key] = dict(
                    file=relative(image_file(result["source_sha256"])), crop_sha256=None
                )
            if i % 48 == 0:
                print(
                    json.dumps(dict(encoded=i, total=len(inputs["roi_requests"]))),
                    flush=True,
                )
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    once(ROOT / "roi_feature_index.json", mapping)
    once(ROOT / "encoding_runtime.json", runtime)
    files = {
        relative(p): sha256_file(p)
        for p in [
            ROOT / "roi_feature_index.json",
            ROOT / "encoding_runtime.json",
            *sorted((ROOT / "features").rglob("*")),
        ]
        if p.is_file()
    }
    once(
        ROOT / "encoding_receipt.json",
        dict(
            files=files,
            new_image_requests=len(runtime["calls"]),
            actual_full_FG_vision_forward_calls=2 * len(runtime["calls"]),
            text_features_reused_exactly=True,
        ),
    )


def calculate(cohort: str, inputs: dict) -> tuple[dict, dict]:
    """Return calculate for this workflow.

    Args:
        cohort: Cohort used by this operation.
        inputs: Inputs used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    data = inputs["cohorts"][cohort]
    ids = [doc["id"] for doc in data["gallery"]]
    fmap = load(ROOT / "roi_feature_index.json")
    original = load(FGROOT / cohort / "predictions.json")
    oldscores = load(FGROOT / cohort / "runtime_scores.json")
    titles = np.stack(
        [read_array(text_file(d["title_key"]), "vector") for d in data["gallery"]]
    )
    images = np.stack(
        [
            read_array(PROJECT_ROOT / fmap[d["roi_request"]]["file"], "global_vector")
            for d in data["gallery"]
        ]
    )
    dense = [
        read_array(PROJECT_ROOT / fmap[d["roi_request"]]["file"], "dense")
        for d in data["gallery"]
    ]
    predictions, details = {}, {}
    for query in data["queries"]:
        start = time.perf_counter()
        qid = query["id"]
        baseline = original[qid]["cross20"]
        current = {
            name: original[qid][name]
            for name in ["global", "region_title", "cross20", "fgclip2_local_fusion20"]
        }
        if not query["image_key"]:
            current.update({name: list(baseline) for name in METHODS})
            details[qid] = dict(
                route="original_text_exact", new_detector_calls=0, new_FG_calls=0
            )
        else:
            features = load(PROJECT_ROOT / query["original_features"])
            raw = load(PROJECT_ROOT / query["original_cross"])
            assert features["ids"] == ids == oldscores[qid]["ids"]
            offsets = (
                np.array(features["offsets"])
                if features["offsets"] is not None
                else None
            )
            scores = auxiliary_scores(
                FgclipRetrievalAuxiliaryScoresInputs(
                    ids,
                    images,
                    titles,
                    dense,
                    query_image=read_array(
                        PROJECT_ROOT / fmap[query["roi_request"]]["file"],
                        "global_vector",
                    ),
                    query_text=(
                        read_array(text_file(query["text_key"]), "vector")
                        if query["text_key"]
                        else None
                    ),
                    query_box=(
                        read_array(text_file(query["text_key"], True), "vector")
                        if query["text_key"]
                        else None
                    ),
                )
            )
            whole = np.array(oldscores[qid]["scores"]["fused_scores"])
            kwargs = dict(
                is_image=True,
                ids=ids,
                base=np.array(features["base"]),
                pool=raw["pool"][:20],
                logits=np.array(raw["logits"][:20]),
                whole_scores=whole,
                offsets=offsets,
            )
            current["image_fusion20"] = image_only_order(
                SubjectRoiImageOnlyOrderInputs(baseline, roi_scores=None, **kwargs)
            )
            current["image_roi_fusion20"] = image_only_order(
                SubjectRoiImageOnlyOrderInputs(
                    baseline, roi_scores=np.array(scores["fused_scores"]), **kwargs
                )
            )
            assert current["image_fusion20"] == original[qid]["fgclip2_local_fusion20"]
            assert all(
                current[m][20:] == baseline[20:]
                and set(current[m][:20]) == set(baseline[:20])
                for m in METHODS
            )
            details[qid] = dict(
                route="image_whole_and_semantic_ROI",
                ids=ids,
                roi_scores=scores,
                seconds=time.perf_counter() - start,
            )
        assert all(
            len(order) == len(ids) and set(order) == set(ids)
            for order in current.values()
        )
        predictions[qid] = current
    return predictions, details


def predict() -> None:
    """Apply the predict workflow step.

    Raises:
        FileExistsError: Preserve completed predictions.

    Notes:
        Writes derived artifacts through once.
    """
    inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "encoding_receipt.json")["files"])
    if (ROOT / "prediction_receipt.json").exists():
        raise FileExistsError("Preserve completed predictions")
    for cohort in COHORTS:
        predictions, details = calculate(cohort, inputs)
        once(ROOT / cohort / "predictions.json", predictions)
        once(ROOT / cohort / "runtime_scores.json", details)
        print(json.dumps(dict(predicted=cohort, queries=len(predictions))), flush=True)
    once(
        ROOT / "prediction_receipt.json",
        dict(
            files={
                relative(p): sha256_file(p)
                for c in COHORTS
                for p in [
                    ROOT / c / "predictions.json",
                    ROOT / c / "runtime_scores.json",
                ]
            },
            text_outputs_equal_original=True,
            original_Qwen_full_pairs_reused=800,
            new_Qwen_pairs=0,
            qrels_used_in_runtime=False,
        ),
    )


def incremental_comparison(ranks: dict, labels: list[dict], modes: dict) -> dict:
    """Return incremental comparison for this workflow.

    Args:
        ranks: Ranks used by this operation.
        labels: Judgments used for the stated development or evaluation protocol.
        modes: Modes used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    result = paired_differences(
        {
            q: {
                "cross_evidence_pool60": orders["image_fusion20"],
                "image_roi_fusion20": orders["image_roi_fusion20"],
            }
            for q, orders in ranks.items()
        },
        labels,
        modes,
    )
    result["baseline_alias"] = {"cross_evidence_pool60": "image_fusion20"}
    return result


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "prediction_receipt.json")["files"])
    for cohort in COHORTS:
        ranks = load(ROOT / cohort / "predictions.json")
        labels = read_rows(PROJECT_ROOT / inputs["cohorts"][cohort]["labels"])
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        differences = paired(ranks, labels, modes, metrics)
        once(ROOT / cohort / "internal_metrics.ai.development.json", metrics)
        once(ROOT / cohort / "internal_paired_differences.json", differences)
        once(
            ROOT / cohort / "internal_roi_incremental_paired.json",
            incremental_comparison(ranks, labels, modes),
        )
        print(json.dumps(dict(evaluated=cohort, relations=len(labels))), flush=True)


def verify() -> None:
    """Verify the recorded artifacts and their consistency.

    Notes:
        Writes derived artifacts through once.
    """
    inputs = check()
    for name in [
        "detection_receipt.json",
        "encoding_receipt.json",
        "prediction_receipt.json",
    ]:
        verify_files(PROJECT_ROOT, load(ROOT / name)["files"])
    for key, row in inputs["roi_requests"].items():
        result = load(detection_path(key))
        assert (
            result["source_sha256"] == row["image_key"]
            and result["prompt"] == row["prompt"]
        )
    for cohort in COHORTS:
        ranks, details = calculate(cohort, inputs)
        assert ranks == load(ROOT / cohort / "predictions.json")
        prior = load(ROOT / cohort / "runtime_scores.json")
        for qid in details:
            assert {k: v for k, v in details[qid].items() if k != "seconds"} == {
                k: v for k, v in prior[qid].items() if k != "seconds"
            }
        labels = read_rows(PROJECT_ROOT / inputs["cohorts"][cohort]["labels"])
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired(ranks, labels, modes, metrics) == load(
            ROOT / cohort / "internal_paired_differences.json"
        )
        assert incremental_comparison(ranks, labels, modes) == load(
            ROOT / cohort / "internal_roi_incremental_paired.json"
        )
        for q, mode in modes.items():
            if mode == "text":
                assert all(ranks[q][m] == ranks[q]["cross20"] for m in METHODS)
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            original_text_exact=True,
            original_twenty_candidate_sets_and_tails_preserved=True,
            same_gallery_queries_6720_AI_labels=True,
            unknown_not_imputed=True,
            rankings_and_metrics_replayed=True,
            earlier_seals_unchanged=True,
            formal_metrics=None,
            new_independent_AI_blind_predictions=0,
        ),
    )
    print(json.dumps(dict(verified=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["prepare", "detect", "encode", "predict", "evaluate", "verify"],
    )
    args = parser.parse_args()
    globals()[args.action]()
