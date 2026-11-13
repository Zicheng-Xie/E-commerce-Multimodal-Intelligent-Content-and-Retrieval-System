"""Fresh official-train identity groups for metric learning, never acceptance data."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from multimodal_ecommerce.dataset_release import order_key, write_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.safety import (
    PROTECTED_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

DATA_POLICY: dict[str, Any] = {
    "seed": 2026102201,
    "source_split": "train",
    "scan_limit": 40000,
    "category_ids": [1, 2, 7, 8, 12],
    "training_groups_per_category": 30,
    "development_groups_per_category": 10,
    "max_shop_views": 3,
    "dhash_threshold": 4,
    "query_rule": "single-item user photo; style>0; exact split/pair/style/category",
    "selection_uses_model_scores": False,
    "formal_acceptance": False,
}


def prepare_training_data(output: Path, exposure: dict) -> dict:
    """Choose metadata-defined groups, screen exposure, then copy selected pixels.

    Args:
        output: Output used by this operation.
        exposure: Exposure used by this operation.

    Returns:
        Mapping with fields: role, formal_acceptance, policy, source_scan_sha256,
        counts, groups, metadata_eligible_by_category, exclusion_reasons,
        excluded_previous_pair_groups, cross_group_dhash_screening, files.

    Raises:
        FileExistsError: Preserve the prepared training dataset.
        ValueError: Metadata scan count differs from the protocol.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Preserve the prepared training dataset")
    policy = DATA_POLICY
    source = PROTECTED_ROOT / "服饰分割" / policy["source_split"]
    old_ids = set(exposure["ids"])
    old_pairs = set(exposure["deepfashion_pair_ids_conservative_cross_split"])
    previous = exposure["image_fingerprints"]
    paths = sorted((source / "annos").glob("*.json"))[: policy["scan_limit"]]
    if len(paths) != policy["scan_limit"]:
        raise ValueError("Metadata scan count differs from the protocol")
    identities: dict[str, dict] = {}
    records = {}
    digest = hashlib.sha256()
    skipped: Counter[str] = Counter()
    for number, path in enumerate(paths, 1):
        raw = path.read_bytes()
        digest.update(path.name.encode() + hashlib.sha256(raw).digest())
        annotation = json.loads(raw)
        pair = str(annotation["pair_id"])
        identifier = f"deepfashion/train/{path.stem}"
        if pair in old_pairs or identifier in old_ids:
            skipped["previous_id_or_pair"] += 1
            continue
        items = [v for k, v in annotation.items() if k.startswith("item")]
        # Multi-garment images are excluded in both domains to keep supervision clear.
        if len(items) != 1 or annotation.get("source") not in {"user", "shop"}:
            continue
        item = items[0]
        category, style = item["category_id"], item.get("style", 0)
        if category not in policy["category_ids"] or style <= 0:
            continue
        image = source / "image" / f"{path.stem}.jpg"
        if not image.is_file():
            skipped["missing_image"] += 1
            continue
        key = f"train/{pair}/{style}/{category}"
        records[identifier] = {
            "id": identifier,
            "pair_id": pair,
            "identity": key,
            "category_id": category,
            "source": annotation["source"],
            "source_image": image,
            "annotation_path": path,
            "annotation_sha256": hashlib.sha256(raw).hexdigest(),
        }
        identities.setdefault(key, {"user": [], "shop": []})[
            annotation["source"]
        ].append(identifier)
        if number % 5000 == 0:
            print(f"Training metadata {number}/{len(paths)}", flush=True)
    eligible = [k for k, v in identities.items() if v["user"] and v["shop"]]
    available = Counter(int(k.rsplit("/", 1)[1]) for k in eligible)
    print(f"Eligible metadata identities: {dict(available)}", flush=True)
    cache: dict[str, Any] = {}
    selected_signatures: list[dict] = []
    selected_pairs: set[str] = set()
    selected = []
    counts: Counter[int] = Counter()
    quota = (
        policy["training_groups_per_category"]
        + policy["development_groups_per_category"]
    )

    def valid(identifier: str) -> bool:
        """Return valid for this workflow.

        Args:
            identifier: Identifier used by this operation.

        Returns:
            Whether the condition checked by this operation holds.
        """
        if identifier not in cache:
            signature = fingerprints(records[identifier]["source_image"])
            cache[identifier] = (
                None
                if is_exposed(signature, previous, policy["dhash_threshold"])
                else signature
            )
            if cache[identifier] is None:
                skipped["prior_image_duplicate_risk"] += 1
        return cache[identifier] is not None

    for key in sorted(eligible, key=lambda k: order_key(k, policy["seed"])):
        entry = identities[key]
        pair, category = key.split("/")[1], int(key.split("/")[-1])
        if pair in selected_pairs or counts[category] >= quota:
            continue
        users = sorted(entry["user"], key=lambda k: order_key(k, policy["seed"]))
        shops = sorted(entry["shop"], key=lambda k: order_key(k, policy["seed"]))
        for query in users:
            if not valid(query):
                continue
            views = []
            seen_bytes = {cache[query]["sha256"]}
            seen_pixels = {cache[query]["pixels"]}
            for shop in shops:
                if not valid(shop):
                    continue
                signature = cache[shop]
                if (
                    signature["sha256"] in seen_bytes
                    or signature["pixels"] in seen_pixels
                ):
                    continue
                views.append(shop)
                seen_bytes.add(signature["sha256"])
                seen_pixels.add(signature["pixels"])
                if len(views) >= policy["max_shop_views"]:
                    break
            chosen = [query, *views]
            if not views:
                continue
            if any(
                is_exposed(cache[k], selected_signatures, policy["dhash_threshold"])
                for k in chosen
            ):
                skipped["new_cross_group_duplicate_risk"] += 1
                continue
            # Quota allocation uses hashed identity order, before any inference.
            role = (
                "training"
                if counts[category] < policy["training_groups_per_category"]
                else "development_validation"
            )
            selected.append({"identity": key, "role": role, "ids": chosen})
            selected_pairs.add(pair)
            selected_signatures.extend(cache[k] for k in chosen)
            counts[category] += 1
            break
        if all(counts[c] == quota for c in policy["category_ids"]):
            break
    if any(counts[c] != quota for c in policy["category_ids"]):
        raise ValueError(f"Insufficient fresh groups for fixed quotas: {dict(counts)}")
    output.mkdir(parents=True)
    rows, model_inputs = [], []
    for group in selected:
        for identifier in group["ids"]:
            original = records[identifier]
            opaque = hashlib.sha256(identifier.encode()).hexdigest()[:24]
            relative = f"images/{opaque}.jpg"
            (output / "images").mkdir(exist_ok=True)
            shutil.copyfile(original["source_image"], output / relative)
            row = {
                **{
                    k: str(v) if isinstance(v, Path) else v for k, v in original.items()
                },
                "image_path": relative,
                "image_sha256": cache[identifier]["sha256"],
                "fingerprints": cache[identifier],
                "opaque_id": opaque,
                "project_role": group["role"],
                "label_origin": "official_source_identity_not_human_business_relevance",
            }
            rows.append(row)
            model_inputs.append(
                {
                    "id": opaque,
                    "image_path": str((output / relative).resolve()),
                    "query": original["source"] == "user",
                }
            )
    write_rows(output / "records.jsonl", rows)
    write_json(output / "model_inputs.json", model_inputs)
    manifest = {
        "role": "training_and_development_only",
        "formal_acceptance": False,
        "policy": policy,
        "source_scan_sha256": digest.hexdigest(),
        "counts": dict(Counter(r["project_role"] for r in rows)),
        "groups": dict(Counter(g["role"] for g in selected)),
        "metadata_eligible_by_category": dict(available),
        "exclusion_reasons": dict(skipped),
        "excluded_previous_pair_groups": len(old_pairs),
        "cross_group_dhash_screening": "distance<=4 rejected; not a proof of all near duplicates",
        "files": {
            "records.jsonl": sha256_file(output / "records.jsonl"),
            "model_inputs.json": sha256_file(output / "model_inputs.json"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
