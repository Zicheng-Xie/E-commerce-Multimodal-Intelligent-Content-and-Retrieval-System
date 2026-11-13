"""Audit review packets, saved predictions and cleaned clothing assets offline."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.frozen_retrieval import load_kit, score_predictions
from multimodal_ecommerce.retrieval import VectorIndex
from multimodal_ecommerce.review_workflow import validate_batch
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        AssertionError: Blank labels incorrectly produced formal metrics.

    Notes:
        Writes derived artifacts through write_json.
    """
    predictions = PROJECT_ROOT / "artifacts/stage2_predictions_v1"
    packet = PROJECT_ROOT / "artifacts/review_stage2_v1"
    release = PROJECT_ROOT / "data/stage2_fashion_clean_v1"
    verify_files(PROJECT_ROOT, load(PROJECT_ROOT / "configs/copy_recipe_v3.json"))
    verify_files(predictions, load(predictions / "manifest.json"))
    run_info = load(predictions / "run.json")
    kit = Path(run_info["kit"])
    kit_info, queries, _ = load_kit(kit)
    index = VectorIndex.load(Path(kit_info["index"]))
    saved = load(predictions / "rankings.json")
    with np.load(predictions / "query_vectors.npz", allow_pickle=False) as archive:
        vectors = archive["vectors"]
    assert vectors.shape == (18, 2048)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert run_info["query_order"] == [query["id"] for query in queries]
    for query, vector in zip(queries, vectors):
        replayed = index.search(
            vector, top_k=len(index.records), exclude_id=query["exclude_id"]
        )
        original = saved[query["id"]]
        assert [row["id"] for row in replayed] == [row["id"] for row in original]
        assert np.allclose(
            [row["score"] for row in replayed],
            [row["score"] for row in original],
            atol=1e-6,
        )
    assert sum(map(len, saved.values())) == 5394
    try:
        score_predictions(predictions, kit / "qrels.template.jsonl")
    except ValueError as error:
        logging.getLogger(__name__).debug("Handled exception in main", exc_info=True)
        assert "Unjudged" in str(error)
    else:
        raise AssertionError("Blank labels incorrectly produced formal metrics")
    info = load(packet / "packet.json")
    verify_files(packet, info)
    counts: Counter = Counter()
    balanced = []
    assets = 0
    for batch in info["batches"]:
        rows = read_rows(packet / batch["template"])
        assert validate_batch(rows, rows)["completed"] == 0
        counts[batch["task"]] += len(rows)
        markup = (packet / batch["page"]).read_text(encoding="utf-8")
        match = re.search(r"<script>const DATA = (.*);</script>", markup)
        assert match and "__REVIEW_" not in markup
        data = json.loads(match.group(1))
        assert [item["label"] for item in data["items"]] == rows
        if batch["task"] == "copy":
            categories = Counter(item["record"]["category"] for item in data["items"])
            assert len(categories) == 3 and set(categories.values()) == {10}
            balanced.append(dict(categories))
        else:
            assert "rankings" not in data
            for item in data["items"]:
                for obj in (item["product"], item["query"]):
                    if obj["image_path"]:
                        image = (packet / obj["image_path"]).resolve()
                        assert image.is_relative_to(PROJECT_ROOT) and image.is_file()
                        assets += 1
    assert counts == {"retrieval": 5394, "copy": 90}
    verify_files(release, load(release / "release_manifest.json"))
    original = {
        row["id"]: row
        for row in read_rows(PROJECT_ROOT / "data/stage2_fashion_v1/samples.jsonl")
    }
    clean = read_rows(release / "samples.jsonl")
    assert len(clean) == 179
    assert "deepfashion/train/000006" not in {row["id"] for row in clean}
    for row in clean:
        assert all(row[key] == value for key, value in original[row["id"]].items())
    for split in ("train", "validation"):
        coco = load(release / "annotations" / f"deepfashion_{split}.json")
        selected = {
            row["id"]: row
            for row in clean
            if row["dataset"] == "deepfashion" and row["split"] == split
        }
        assert {row["source_record_id"] for row in coco["images"]} == set(selected)
        ids = {row["id"] for row in coco["images"]}
        assert all(row["image_id"] in ids for row in coco["annotations"])
        assert len(coco["annotations"]) == sum(
            len(row["instances"]) for row in selected.values()
        )
    assert (
        load(PROJECT_ROOT / "artifacts/stage2_fashion_clean_audit_v1/report.json")[
            "issues"
        ]
        == []
    )
    report = {
        "passed": True,
        "prediction_queries": 18,
        "replayed_candidate_rankings": 5394,
        "gpu_required_for_replay": False,
        "blank_labels_rejected": True,
        "review_batches": 21,
        "review_counts": dict(counts),
        "balanced_copy_batches": balanced,
        "local_image_references_checked": assets,
        "eligible_fashion_records": 179,
        "original_fashion_values_preserved": True,
        "coco_joins_verified": True,
        "frozen_copy_v3_verified": True,
        "browser_visual_verification": False,
    }
    write_json(PROJECT_ROOT / "artifacts/stage2_review_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
