"""Prepare reserved three-category acceptance media without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import (
    ACCEPTANCE_POLICY,
    prepare_acceptance_metadata,
)
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

OUT = PROJECT_ROOT / "artifacts/module2_business_acceptance"


def fetch_one(row: dict) -> dict:
    """Return fetch one for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, status, path, url, fingerprints, error, error_type.

    Raises:
        ValueError: Cached image checksum mismatch.

    Notes:
        Writes derived artifacts through path.parent.mkdir, temporary.write_bytes.
    """
    try:
        with httpx.Client(timeout=15, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        path = OUT / "cache/images" / (info["sha256"] + info["suffix"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        if sha256_file(path) != info["sha256"]:
            raise ValueError("Cached image checksum mismatch")
        return {
            "id": row["id"],
            "status": "downloaded",
            "path": path.relative_to(OUT).as_posix(),
            "url": url,
            "fingerprints": fingerprints(path),
        }
    except (httpx.HTTPError, OSError, ValueError) as error:
        logging.getLogger(__name__).debug(
            "Handled exception in fetch_one", exc_info=True
        )
        return {
            "id": row["id"],
            "status": "failed",
            "error": str(error),
            "error_type": type(error).__name__,
        }


def download() -> None:
    """Apply the download workflow step.

    Raises:
        FileExistsError: Preserve this reserved candidate dataset; Preserve the existing
            reserved acceptance registry.
        ValueError: Acceptance sampling policy changed; Source candidate queue changed;
            Cached source bytes changed.

    Notes:
        Writes derived artifacts through (dataset / 'evaluator').mkdir, (dataset /
        'gallery/images').mkdir, (dataset / 'queries/images').mkdir, write_json,
        write_rows.
    """
    if (OUT / "dataset/manifest.json").exists():
        raise FileExistsError("Preserve this reserved candidate dataset")
    if (PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl").exists():
        raise FileExistsError("Preserve the existing reserved acceptance registry")
    protocol = json.loads(
        (OUT / "preparation_protocol.json").read_text(encoding="utf-8")
    )
    verify_files(PROJECT_ROOT, protocol["implementation"])
    if protocol["policy"] != ACCEPTANCE_POLICY:
        raise ValueError("Acceptance sampling policy changed")
    receipt = json.loads((OUT / "metadata_receipt.json").read_text(encoding="utf-8"))
    if sha256_file(OUT / "source_candidates.jsonl") != receipt["candidate_file_sha256"]:
        raise ValueError("Source candidate queue changed")
    previous = json.loads((OUT / "prior_exposure.json").read_text(encoding="utf-8"))[
        "image_fingerprints"
    ]
    rows = read_rows(OUT / "source_candidates.jsonl")
    labels = [
        label for labels in ACCEPTANCE_POLICY["fine_types"].values() for label in labels
    ]
    by_label = {label: [r for r in rows if r["label"] == label] for label in labels}
    cache_path = OUT / "download_cache.json"
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    if not cache:
        first = fetch_one(rows[0])
        if first["status"] == "failed" and first["error_type"] == "ConnectError":
            raise ConnectionError(first["error"])
        cache[rows[0]["id"]] = first
        write_json(cache_path, cache)
    used_keys = set(protocol["old_product_family_keys"])
    new_fingerprints: list = []
    reasons: Counter[str] = Counter()
    position = {label: 0 for label in labels}
    chosen: dict = {label: [] for label in labels}
    needed = (
        ACCEPTANCE_POLICY["catalog_per_fine_type"]
        + ACCEPTANCE_POLICY["image_queries_per_fine_type"]
    )
    while any(len(chosen[label]) < needed for label in labels):
        pending = []
        for label in labels:
            if len(chosen[label]) >= needed:
                continue
            start = position[label]
            stop = min(
                start + ACCEPTANCE_POLICY["download_batch_per_fine_type"],
                len(by_label[label]),
            )
            pending.extend(by_label[label][start:stop])
            position[label] = stop
        if not pending:
            available_counts = {key: len(values) for key, values in chosen.items()}
            raise ValueError(
                "Not enough unique valid images: " + json.dumps(available_counts)
            )
        missing = [r for r in pending if r["id"] not in cache]
        with ThreadPoolExecutor(
            max_workers=ACCEPTANCE_POLICY["download_workers"]
        ) as workers:
            for result in workers.map(fetch_one, missing):
                cache[result["id"]] = result
        write_json(cache_path, cache)
        # Evaluate in frozen queue order; completion timing cannot choose the sample.
        for row in pending:
            if len(chosen[row["label"]]) >= needed:
                continue
            result = cache[row["id"]]
            if result["status"] != "downloaded":
                reasons["download_or_decode_failure"] += 1
                continue
            image = OUT / result["path"]
            if sha256_file(image) != result["fingerprints"]["sha256"]:
                raise ValueError("Cached source bytes changed")
            signature = result["fingerprints"]
            keys = set(row["family_keys"])
            if keys & used_keys:
                reasons["source_family_overlap"] += 1
                continue
            if is_exposed(signature, previous, ACCEPTANCE_POLICY["dhash_threshold"]):
                reasons["old_image_duplicate_risk"] += 1
                continue
            if is_exposed(
                signature, new_fingerprints, ACCEPTANCE_POLICY["dhash_threshold"]
            ):
                reasons["new_cross_family_duplicate_risk"] += 1
                continue
            chosen[row["label"]].append({**row, "download": result})
            used_keys.update(keys)
            new_fingerprints.append(signature)
        print(
            f"Reserved acceptance images {sum(len(v) for v in chosen.values())}/330; HTTP attempts {len(cache)}",
            flush=True,
        )
    dataset = OUT / "dataset"
    (dataset / "gallery/images").mkdir(parents=True)
    (dataset / "queries/images").mkdir(parents=True)
    (dataset / "evaluator").mkdir()
    gallery = []
    lineage = []
    drafts = []
    registry = []
    for type_number, label in enumerate(labels, 1):
        for number, row in enumerate(chosen[label]):
            role = "query" if number == 0 else "gallery"
            opaque = ("q-" if role == "query" else "g-") + hashlib.sha256(
                row["id"].encode()
            ).hexdigest()[:24]
            source = OUT / row["download"]["path"]
            folder = "queries" if role == "query" else "gallery"
            relative = f"{folder}/images/{opaque}{source.suffix}"
            shutil.copyfile(source, dataset / relative)
            signature = row["download"]["fingerprints"]
            lineage.append(
                {
                    **{k: v for k, v in row.items() if k != "download"},
                    "opaque_id": opaque,
                    "role": role,
                    "image_path": relative,
                    "fingerprints": signature,
                }
            )
            registry.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "dataset": "product",
                    "split": "test",
                    "family_keys": row["family_keys"],
                    "image_path": os.path.relpath(
                        dataset / relative,
                        PROJECT_ROOT / "data/module2_reserved_acceptance",
                    ).replace("\\", "/"),
                    "project_role": "reserved_business_acceptance_not_for_development",
                    "image_sha256": signature["sha256"],
                }
            )
            if role == "gallery":
                gallery.append(
                    {
                        "id": opaque,
                        "title": row["text"],
                        "attributes": row["attributes"],
                        "category": "",
                        "image_path": "images/" + (dataset / relative).name,
                        "image_sha256": signature["sha256"],
                    }
                )
            else:
                drafts.append(
                    {
                        "id": f"image_{type_number:02}",
                        "kind": "image",
                        "fine_type_scope": label,
                        "category_for_reporting": row["category"],
                        "image_path": relative,
                        "text": None,
                        "relevance_rule": "商品主要类型和用途相符，且可见外观相似；背景或人物相似不算相关；看不清主体时记录原因，不猜不可见材质或规格。",
                        "approved": False,
                        "reviewer": "",
                        "review_origin": None,
                        "note": "",
                    }
                )
        drafts.append(
            {
                "id": f"text_{type_number:02}",
                "kind": "text",
                "fine_type_scope": label,
                "category_for_reporting": chosen[label][0]["category"],
                "image_path": None,
                "text": None,
                "relevance_rule": "",
                "approved": False,
                "reviewer": "",
                "review_origin": None,
                "note": "",
            }
        )
    write_rows(dataset / "gallery/products.jsonl", gallery)
    write_rows(dataset / "evaluator/lineage.jsonl", lineage)
    write_rows(dataset / "evaluator/queries.draft.jsonl", drafts)
    write_rows(
        PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl", registry
    )
    manifest = {
        "role": "reserved_acceptance_candidate_not_ready",
        "catalog": len(gallery),
        "image_queries": 30,
        "text_query_slots": 30,
        "human_approved_queries": 0,
        "human_relevance_labels": 0,
        "model_predictions": 0,
        "model_inference_allowed": False,
        "formal_acceptance": False,
        "formal_metrics": None,
        "scope": ACCEPTANCE_POLICY["scope"],
        "human_taxonomy_and_family_quality_review_pending": True,
        "exclusion_reasons": dict(reasons),
        "source_groups_have_no_recorded_prior_or_cross_role_family_key_overlap": True,
        "near_image_check": "Prior and new cross-product dHash<=4/byte/pixel matches rejected; not a proof of all semantic duplicates",
        "files": {
            p.relative_to(dataset).as_posix(): sha256_file(p)
            for p in dataset.rglob("*")
            if p.is_file()
        },
        "reserved_registry_sha256": sha256_file(
            PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
        ),
    }
    write_json(dataset / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "catalog": len(gallery),
                "query_slots": len(drafts),
                "formal_ready": False,
                "model_inference": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["metadata", "download"])
    args = parser.parse_args()
    if args.action == "metadata":
        print(json.dumps(prepare_acceptance_metadata(OUT), ensure_ascii=False))
    else:
        download()
