"""Separate, reproducible FG-CLIP 2 versus original Qwen development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from evaluate_primary_form_coverage import paired
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.fgclip_retrieval import (
    POLICY,
    FGClipEncoder,
    FgclipRetrievalAuxiliaryScoresInputs,
    FgclipRetrievalFusePoolInputs,
    auxiliary_scores,
    fuse_pool,
)
from multimodal_ecommerce.intent_retrieval import rerank_complete
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import ranked_ids

ROOT = PROJECT_ROOT / "artifacts/module2_fgclip2_comparison"
PARENT = PROJECT_ROOT / "artifacts/module2_primary_form_coverage"
STRUCTURE = PROJECT_ROOT / "artifacts/module2_structure_development"
COHORTS = ("tune", "regression", "fresh")
METHODS = ("fgclip2_standalone", "fgclip2_global_fusion20", "fgclip2_local_fusion20")


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_once(path: Path, value: object) -> None:
    """Persist once.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through write_json.
    """
    if path.exists():
        if load(path) != value:
            raise RuntimeError(f"Preserve completed artifact: {path.name}")
    else:
        write_json(path, value)


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def text_key(value: str) -> str:
    """Return text key for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_parent() -> None:
    """Verify parent."""
    receipt = load(PARENT / "completion_receipt.json")
    verify_files(PARENT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["project_files"])


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve frozen experiment protocol.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("Preserve frozen experiment protocol")
    verify_parent()
    model = load(ROOT / "model_manifest.json")
    files = {
        relative(p): sha256_file(p)
        for p in [
            Path(__file__),
            PROJECT_ROOT / "scripts/prepare_fgclip2_model.py",
            PROJECT_ROOT / "pyproject.toml",
            ROOT / "model_manifest.json",
            PARENT / "completion_receipt.json",
            PARENT / "post_development_exclusion_snapshot.json",
            *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
        ]
    }
    images: dict
    images, texts, box_texts, cohorts = {}, {}, set(), {}
    for cohort in COHORTS:
        folder = PARENT / cohort
        index = PARENT / ("tune" if cohort == "fresh" else cohort) / "index"
        public = load(folder / "queries.public.json")
        gallery = []
        queries = []
        paths = [
            index / "records.json",
            folder / "queries.public.json",
            folder / "predictions.json",
            folder / "judgments.ai.jsonl",
        ]

        def register_image(path: Path) -> str:
            """Return register image for this workflow.

            Args:
                path: Path of the input or output artifact used by this operation.

            Returns:
                Text produced by this operation.
            """
            file = Path(path)
            key = sha256_file(file)
            images.setdefault(key, relative(file))
            files[relative(file)] = key
            return key

        def register_text(value: str) -> str:
            """Return register text for this workflow.

            Args:
                value: Input value to validate or transform.

            Returns:
                Text produced by this operation.
            """
            key = text_key(value)
            texts[key] = value
            return key

        for doc in load(index / "records.json"):
            gallery.append(
                dict(
                    id=doc["id"],
                    image_key=register_image(doc["image_path"]),
                    title_key=register_text(doc["title"]),
                )
            )
        for query in public:
            image_key = (
                register_image(query["image_path"]) if query["image_path"] else None
            )
            feature = folder / "runs" / query["id"] / "features.json"
            cross = feature.with_name("cross.json")
            paths.extend([feature, cross])
            description = load(feature)["description"]
            value = description if image_key else query["text"]
            key = register_text(value) if value else None
            if key:
                box_texts.add(key)
            queries.append(
                dict(
                    id=query["id"],
                    image_key=image_key,
                    text_key=key,
                    original_features=relative(feature),
                    original_cross=relative(cross),
                )
            )
        files.update({relative(p): sha256_file(p) for p in paths})
        cohorts[cohort] = dict(
            gallery=gallery,
            queries=queries,
            labels=relative(folder / "judgments.ai.jsonl"),
            original_predictions=relative(folder / "predictions.json"),
            role="previously_exposed_development_not_blind",
        )
    inputs = dict(
        images=images, texts=texts, box_texts=sorted(box_texts), cohorts=cohorts
    )
    save_once(ROOT / "inputs.json", inputs)
    files[relative(ROOT / "inputs.json")] = sha256_file(ROOT / "inputs.json")
    write_json(
        ROOT / "protocol.json",
        dict(
            policy=POLICY,
            model=model,
            primary_method="fgclip2_local_fusion20",
            ablations=list(METHODS[:2]),
            baseline="cross20, reconstructed from identical pinned full Qwen outputs",
            methods_frozen_before_FG_predictions=True,
            hyperparameter_search=False,
            cohort_roles={c: "exposed_AI_development_only" for c in COHORTS},
            selection="No post-result variant selection; main method fixed to local_fusion20. All methods retained, default unchanged.",
            judgment_count=6720,
            original_labels_unchanged=True,
            human_labels=0,
            formal_metrics=None,
            independent_AI_blind_predictions=0,
            files=files,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                images=len(images),
                global_texts=len(texts),
                regional_texts=len(box_texts),
                queries=sum(len(v["queries"]) for v in cohorts.values()),
            )
        ),
        flush=True,
    )


def check() -> tuple[Any, Any]:
    """Verify the prerequisites and integrity of the current workflow.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        RuntimeError: Frozen FG policy changed.
    """
    protocol = load(ROOT / "protocol.json")
    if protocol["policy"] != POLICY:
        raise RuntimeError("Frozen FG policy changed")
    verify_files(PROJECT_ROOT, protocol["files"])
    verify_parent()
    return protocol, load(ROOT / "inputs.json")


def image_file(key: str) -> Path:
    """Return image file for this workflow.

    Args:
        key: Key used by this operation.

    Returns:
        Resolved artifact path.
    """
    return ROOT / "features/images" / f"{key}.npz"


def text_file(key: str, box: bool = False) -> Path:
    """Return text file for this workflow.

    Args:
        key: Key used by this operation.
        box: Box used by this operation.

    Returns:
        Resolved artifact path.
    """
    return ROOT / "features/text" / f"{key}_{'box' if box else 'long'}.npz"


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        FileExistsError: Preserve completed FG features.

    Notes:
        Writes derived artifacts through np.savez_compressed, path.parent.mkdir,
        write_json.
    """
    _, inputs = check()
    if (ROOT / "encoding_receipt.json").exists():
        raise FileExistsError("Preserve completed FG features")
    encoder = FGClipEncoder(ROOT / "model_manifest.json")
    try:
        total = len(inputs["images"]) + len(inputs["texts"]) + len(inputs["box_texts"])
        done = 0
        for key, source in inputs["images"].items():
            path = image_file(key)
            if not path.exists():
                global_vector, dense, info = encoder.image(PROJECT_ROOT / source)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, global_vector=global_vector, dense=dense)
                write_json(
                    path.with_suffix(".json"),
                    dict(
                        source_sha256=key, detail=info, feature_sha256=sha256_file(path)
                    ),
                )
            else:
                assert load(path.with_suffix(".json"))["feature_sha256"] == sha256_file(
                    path
                )
            done += 1
            if done % 24 == 0:
                print(
                    json.dumps(dict(encoded=done, total=total, phase="images")),
                    flush=True,
                )
        for box in [False, True]:
            keys = inputs["box_texts"] if box else inputs["texts"]
            for key in keys:
                path = text_file(key, box)
                if not path.exists():
                    vector, info = encoder.text(inputs["texts"][key], regional=box)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, vector=vector)
                    write_json(
                        path.with_suffix(".json"),
                        dict(
                            source_sha256=key,
                            detail=info,
                            feature_sha256=sha256_file(path),
                        ),
                    )
                else:
                    assert load(path.with_suffix(".json"))[
                        "feature_sha256"
                    ] == sha256_file(path)
                done += 1
                if done % 48 == 0:
                    print(
                        json.dumps(dict(encoded=done, total=total, phase="text")),
                        flush=True,
                    )
        runtime = encoder.diagnostics()
    finally:
        encoder.close()
    save_once(ROOT / "encoding_runtime.json", runtime)
    files = {
        relative(p): sha256_file(p)
        for p in [
            ROOT / "encoding_runtime.json",
            *sorted((ROOT / "features").rglob("*")),
        ]
        if p.is_file()
    }
    save_once(
        ROOT / "encoding_receipt.json",
        dict(
            files=files,
            actual_new_requests=len(runtime["calls"]),
            requests=sum(
                r["detail"]["kind"] == "image_global_and_dense"
                for r in [load(p) for p in (ROOT / "features/images").glob("*.json")]
            ),
            full_model=True,
        ),
    )


def read_array(path: Path, field: str) -> np.ndarray:
    """Read array.

    Args:
        path: Path of the input or output artifact used by this operation.
        field: Field used by this operation.

    Returns:
        Computed array in the order of the supplied records.
    """
    with np.load(path, allow_pickle=False) as bank:
        return bank[field].copy()


def rankings_for(cohort: str, inputs: dict) -> tuple[dict, dict]:
    """Return rankings for for this workflow.

    Args:
        cohort: Cohort used by this operation.
        inputs: Inputs used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    data = inputs["cohorts"][cohort]
    gallery = data["gallery"]
    ids = [row["id"] for row in gallery]
    images = np.stack(
        [read_array(image_file(row["image_key"]), "global_vector") for row in gallery]
    )
    titles = np.stack(
        [read_array(text_file(row["title_key"]), "vector") for row in gallery]
    )
    dense = [read_array(image_file(row["image_key"]), "dense") for row in gallery]
    original = load(PROJECT_ROOT / data["original_predictions"])
    rankings, details = {}, {}
    for query in data["queries"]:
        start = time.perf_counter()
        qid = query["id"]
        features = load(PROJECT_ROOT / query["original_features"])
        cross = load(PROJECT_ROOT / query["original_cross"])
        assert features["ids"] == ids and cross["pool"] == features["pool"]
        base = np.array(features["base"])
        pool, logits = cross["pool"][:20], np.array(cross["logits"][:20])
        offsets = (
            np.array(features["offsets"]) if features["offsets"] is not None else None
        )
        baseline = rerank_complete(ids, base, pool, logits, offsets)
        assert baseline == original[qid]["cross20"]
        scores = auxiliary_scores(
            FgclipRetrievalAuxiliaryScoresInputs(
                ids,
                images,
                titles,
                dense,
                query_image=(
                    read_array(image_file(query["image_key"]), "global_vector")
                    if query["image_key"]
                    else None
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
        ranks = {k: original[qid][k] for k in ["global", "region_title", "cross20"]}
        ranks.update(
            fgclip2_standalone=ranked_ids(ids, np.array(scores["fused_scores"])),
            fgclip2_global_fusion20=fuse_pool(
                FgclipRetrievalFusePoolInputs(
                    ids, base, pool, logits, np.array(scores["global_scores"]), offsets
                )
            ),
            fgclip2_local_fusion20=fuse_pool(
                FgclipRetrievalFusePoolInputs(
                    ids, base, pool, logits, np.array(scores["fused_scores"]), offsets
                )
            ),
        )
        assert all(
            len(order) == len(ids) and set(order) == set(ids)
            for order in ranks.values()
        )
        assert all(
            ranks[m][20:] == baseline[20:] and set(ranks[m][:20]) == set(pool)
            for m in METHODS[1:]
        )
        rankings[qid] = ranks
        details[qid] = dict(
            ids=ids,
            pool=pool,
            scores=scores,
            original_cross_reused=20,
            new_Qwen_pairs=0,
            seconds=time.perf_counter() - start,
        )
    return rankings, details


def predict() -> None:
    """Apply the predict workflow step.

    Raises:
        FileExistsError: Preserve completed predictions.
    """
    _, inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "encoding_receipt.json")["files"])
    if (ROOT / "prediction_receipt.json").exists():
        raise FileExistsError("Preserve completed predictions")
    for cohort in COHORTS:
        rankings, details = rankings_for(cohort, inputs)
        save_once(ROOT / cohort / "predictions.json", rankings)
        save_once(ROOT / cohort / "runtime_scores.json", details)
        print(
            json.dumps(
                dict(
                    cohort=cohort,
                    queries=len(rankings),
                    methods=len(next(iter(rankings.values()))),
                )
            ),
            flush=True,
        )
    save_once(
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
            qrels_used_in_runtime=False,
            original_Qwen_full_pairs_reused=800,
            new_Qwen_full_pairs=0,
            methods_frozen=True,
        ),
    )


def evaluate() -> None:
    """Apply the evaluate workflow step."""
    _, inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "prediction_receipt.json")["files"])
    for cohort in COHORTS:
        ranks = load(ROOT / cohort / "predictions.json")
        labels = read_rows(PROJECT_ROOT / inputs["cohorts"][cohort]["labels"])
        modes = {qid: qid.split("_")[0] for qid in ranks}
        metrics = evaluate_intervals(ranks, labels, modes)
        differences = paired(ranks, labels, modes, metrics)
        save_once(ROOT / cohort / "internal_metrics.ai.development.json", metrics)
        save_once(ROOT / cohort / "internal_paired_differences.json", differences)
        print(json.dumps(dict(evaluated=cohort, label_count=len(labels))), flush=True)


def verify() -> None:
    """Verify the recorded artifacts and their consistency.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol, inputs = check()
    verify_files(PROJECT_ROOT, load(ROOT / "encoding_receipt.json")["files"])
    verify_files(PROJECT_ROOT, load(ROOT / "prediction_receipt.json")["files"])
    for cohort in COHORTS:
        rankings, details = rankings_for(cohort, inputs)
        assert rankings == load(ROOT / cohort / "predictions.json")
        stored = load(ROOT / cohort / "runtime_scores.json")
        for qid in details:
            assert {k: v for k, v in details[qid].items() if k != "seconds"} == {
                k: v for k, v in stored[qid].items() if k != "seconds"
            }
        labels = read_rows(PROJECT_ROOT / inputs["cohorts"][cohort]["labels"])
        modes = {q: q.split("_")[0] for q in rankings}
        metrics = evaluate_intervals(rankings, labels, modes)
        assert metrics == load(ROOT / cohort / "internal_metrics.ai.development.json")
        assert paired(rankings, labels, modes, metrics) == load(
            ROOT / cohort / "internal_paired_differences.json"
        )
    write_json(
        ROOT / "verification.json",
        dict(
            passed=True,
            baseline_replayed_exactly=True,
            predictions_replayed_from_real_features=True,
            same_gallery_queries_labels=True,
            full_rankings=240,
            original_judgments=6720,
            original_judgments_unchanged=True,
            fixed_K=10,
            unjudged_not_imputed=True,
            parent_seal_unchanged=True,
            formal_metrics=None,
            independent_AI_blind_predictions=0,
            protocol_sha256=sha256_file(ROOT / "protocol.json"),
        ),
    )
    print(json.dumps(dict(verified=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "encode", "predict", "evaluate", "verify"]
    )
    args = parser.parse_args()
    globals()[args.action]()
