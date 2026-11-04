"""Known exposure audit and fresh group-disjoint same-item holdout preparation."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from multimodal_ecommerce.dataset_release import (
    difference_hash,
    order_key,
    read_rows,
    write_rows,
)
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def fingerprints(path: Path) -> dict:
    """Compare bytes, oriented decoded pixels, and conservative dHash candidates.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Mapping with fields: sha256, pixels, dhash.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return {
        "sha256": sha256_file(path),
        "pixels": digest,
        "dhash": difference_hash(path),
    }


def audit_exposure(output: Path) -> dict:
    """Treat all existing prepared datasets, even formerly called heldout, as
    exposed.

    This inventories recorded local use, not undocumented human access. Local
    prepared media are conservative exclusions whether or not metrics used them.

    Args:
        output: Output used by this operation.

    Returns:
        Mapping with fields: scope, date, files, ids,
        deepfashion_pair_ids_conservative_cross_split, product_source_ids,
        image_fingerprints, counts.

    Raises:
        FileExistsError: Exposure snapshots are immutable.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Exposure snapshots are immutable")
    source_files = sorted((PROJECT_ROOT / "data").rglob("*.jsonl"))
    source_files = [p for p in source_files if "module2_blind" not in p.parts]
    ids, pair_ids, product_ids, images, evidence = set(), set(), set(), {}, {}
    for path in source_files:
        rows = read_rows(path)
        evidence[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
        for row in rows:
            key = str(row.get("id", ""))
            if not key:
                continue
            ids.add(key)
            if key.startswith("product/"):
                product_ids.add(str(row.get("source_id") or key.rsplit("/", 1)[-1]))
            if key.startswith("deepfashion/"):
                pair = row.get("pair_id")
                if pair is None and row.get("source_path"):
                    original = Path(row["source_path"])
                    if original.is_file():
                        pair = json.loads(original.read_text(encoding="utf-8")).get(
                            "pair_id"
                        )
                if pair is not None:
                    pair_ids.add(str(pair))
            if row.get("image_path"):
                image = (path.parent / row["image_path"]).resolve()
                if image.is_file():
                    images[str(image)] = image
    signatures: dict[str, dict] = {}
    for number, image in enumerate(images.values(), 1):
        signature = fingerprints(image)
        signatures.setdefault(signature["sha256"], signature)
        if number % 300 == 0:
            print(f"Exposure image audit {number}/{len(images)}", flush=True)
    result = {
        "scope": "All recorded prepared datasets excluded; undocumented prior access cannot be certified",
        "date": "2026-10-22",
        "files": evidence,
        "ids": sorted(ids),
        "deepfashion_pair_ids_conservative_cross_split": sorted(pair_ids),
        "product_source_ids": sorted(product_ids),
        "image_fingerprints": list(signatures.values()),
        "counts": {
            "records": len(ids),
            "deepfashion_pair_ids": len(pair_ids),
            "product_ids": len(product_ids),
            "unique_image_bytes": len(signatures),
        },
    }
    write_json(output, result)
    return result


def is_exposed(signature: dict, previous: list[dict], threshold: int = 4) -> bool:
    """Reject byte/pixel matches or close dHash risks, not a proof of all duplicates.

    Args:
        signature: Signature used by this operation.
        previous: Previous used by this operation.
        threshold: Threshold used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    return any(
        signature["sha256"] == old["sha256"]
        or signature["pixels"] == old["pixels"]
        or (signature["dhash"] ^ old["dhash"]).bit_count() <= threshold
        for old in previous
    )


def prepare_same_item(bundle: Path, recipe: dict, exposure: dict) -> dict:
    """Sample label-defined identities before inference, excluding every old group.

    Args:
        bundle: Bundle used by this operation.
        recipe: Recipe used by this operation.
        exposure: Exposure used by this operation.

    Returns:
        Mapping with fields: role, source_split, query_count, gallery_count, groups,
        query_categories, recorded_exposure_audit_passed, prior_id_overlap,
        prior_pair_overlap_conservative_cross_split, prior_byte_or_pixel_overlap,
        prior_dhash_candidates_at_or_below_threshold, dhash_threshold, duplicate_limit,
        unrecorded_human_access, source_metadata_files_scanned,
        source_metadata_scan_sha256, pre_inference_exclusions, sampling_scope,
        qrels_origin, recipe_sha256, exposure_sha256.

    Raises:
        FileExistsError: Holdout data cannot be regenerated in place.
        ValueError: Frozen metadata pool count changed; Insufficient disjoint heldout
            gallery; Copied image mismatch; Query has no positive within the frozen
            gallery.
    """
    policy = recipe["data_policy"]
    output = ensure_output_path(bundle / "dataset")
    if output.exists():
        raise FileExistsError("Holdout data cannot be regenerated in place")
    source = PROTECTED_ROOT / "服饰分割" / policy["source_split"]
    old_ids = set(exposure["ids"])
    old_pairs = set(exposure["deepfashion_pair_ids_conservative_cross_split"])
    previous = exposure["image_fingerprints"]
    identities: dict[str, dict] = {}
    records: dict[str, dict] = {}
    scan_digest = hashlib.sha256()
    skipped: Counter[str] = Counter()
    paths = sorted((source / "annos").glob("*.json"))[: policy["scan_limit"]]
    if len(paths) != policy["scan_limit"]:
        raise ValueError("Frozen metadata pool count changed")
    for number, path in enumerate(paths, 1):
        data = path.read_bytes()
        scan_digest.update(path.name.encode() + hashlib.sha256(data).digest())
        row = json.loads(data)
        identifier = f"deepfashion/{policy['source_split']}/{path.stem}"
        group = str(row["pair_id"])
        if identifier in old_ids or group in old_pairs:
            skipped["known_id_or_pair_group"] += 1
            continue
        items = [
            v for k, v in row.items() if k.startswith("item") and isinstance(v, dict)
        ]
        image = source / "image" / (path.stem + ".jpg")
        if not image.is_file() or row.get("source") not in {"shop", "user"}:
            skipped["missing_image_or_source"] += 1
            continue
        keys = [
            f"{group}/{item['style']}/{item['category_id']}"
            for item in items
            if item.get("style", 0) > 0
        ]
        record = {
            "source_id": identifier,
            "pair_id": group,
            "image": image,
            "annotation": path,
            "annotation_sha256": hashlib.sha256(data).hexdigest(),
            "source": row["source"],
            "keys": keys,
            "single": len(items) == 1,
            "category_id": items[0]["category_id"] if len(items) == 1 else None,
        }
        records[identifier] = record
        for key in keys:
            entry = identities.setdefault(key, {"user": [], "shop": []})
            if row["source"] == "shop" or record["single"]:
                entry[row["source"]].append(identifier)
        if number % 2000 == 0:
            print(f"Heldout metadata scan {number}/{len(paths)}", flush=True)
    cache: dict[str, Any] = {}

    def valid(identifier: str) -> bool:
        """Return valid for this workflow.

        Args:
            identifier: Identifier used by this operation.

        Returns:
            Whether the condition checked by this operation holds.
        """
        if identifier not in cache:
            try:
                signature = fingerprints(records[identifier]["image"])
            except (OSError, ValueError):
                logging.getLogger(__name__).debug(
                    "Handled exception in valid", exc_info=True
                )
                cache[identifier] = None
                skipped["decode_failure"] += 1
            else:
                cache[identifier] = (
                    None
                    if is_exposed(signature, previous, policy["dhash_threshold"])
                    else signature
                )
                if cache[identifier] is None:
                    skipped["prior_image_duplicate_risk"] += 1
        return cache[identifier] is not None

    chosen_queries, chosen_gallery, selected_groups = [], {}, set()
    selected_fingerprints: list[tuple[str, dict]] = []
    counts: Counter[int] = Counter()
    ordered = sorted(identities, key=lambda key: order_key(key, policy["seed"]))
    for key in ordered:
        entry = identities[key]
        group, _, category_text = key.split("/")
        category = int(category_text)
        if (
            category not in policy["category_ids"]
            or counts[category] >= policy["queries_per_category"]
            or group in selected_groups
        ):
            continue
        users = sorted(
            entry["user"], key=lambda value: order_key(value, policy["seed"])
        )
        shops = sorted(
            entry["shop"], key=lambda value: order_key(value, policy["seed"])
        )
        for user in users:
            if not valid(user):
                continue
            valid_shops = []
            seen_bytes, seen_pixels = {cache[user]["sha256"]}, {cache[user]["pixels"]}
            for shop in shops:
                if not valid(shop):
                    continue
                signature = cache[shop]
                if (
                    signature["sha256"] in seen_bytes
                    or signature["pixels"] in seen_pixels
                ):
                    continue
                valid_shops.append(shop)
                seen_bytes.add(signature["sha256"])
                seen_pixels.add(signature["pixels"])
            candidates = [user, *valid_shops[: policy["max_positive_shop_views"]]]
            if len(candidates) < 2:
                continue
            if any(
                is_exposed(
                    cache[k],
                    [f for g, f in selected_fingerprints if g != group],
                    policy["dhash_threshold"],
                )
                for k in candidates
            ):
                skipped["heldout_cross_group_duplicate_risk"] += 1
                continue
            chosen_queries.append(user)
            chosen_gallery.update({s: records[s] for s in candidates[1:]})
            selected_groups.add(group)
            selected_fingerprints.extend((group, cache[s]) for s in candidates)
            counts[category] += 1
            break
        if all(
            counts[c] == policy["queries_per_category"] for c in policy["category_ids"]
        ):
            break
    if any(counts[c] != policy["queries_per_category"] for c in policy["category_ids"]):
        raise ValueError(
            f"Insufficient independent query groups: {dict(counts)}; do not change policy using model scores"
        )
    for identifier in sorted(
        records, key=lambda value: order_key(value, policy["seed"] + 1)
    ):
        if len(chosen_gallery) >= policy["gallery_size"]:
            break
        record = records[identifier]
        if not any(
            int(key.rsplit("/", 1)[-1]) in policy["category_ids"]
            for key in record["keys"]
        ):
            continue
        if (
            record["source"] != "shop"
            or record["pair_id"] in selected_groups
            or identifier in chosen_gallery
            or not valid(identifier)
        ):
            continue
        if any(
            cache[identifier]["sha256"] == f["sha256"]
            or cache[identifier]["pixels"] == f["pixels"]
            for _, f in selected_fingerprints
        ):
            continue
        if is_exposed(
            cache[identifier],
            [f for g, f in selected_fingerprints if g != record["pair_id"]],
            policy["dhash_threshold"],
        ):
            skipped["heldout_cross_group_duplicate_risk"] += 1
            continue
        chosen_gallery[identifier] = record
        selected_fingerprints.append((record["pair_id"], cache[identifier]))
    if len(chosen_gallery) != policy["gallery_size"]:
        raise ValueError("Insufficient disjoint heldout gallery")
    output.mkdir(parents=True)
    public = output / "public"
    private = output / "evaluator"
    private.mkdir()
    sanitized, lineage, mapping = {}, [], {}
    for role, identifiers in (
        ("gallery", sorted(chosen_gallery)),
        ("queries", chosen_queries),
    ):
        folder = public / role
        (folder / "images").mkdir(parents=True)
        rows = []
        for identifier in identifiers:
            record = records[identifier]
            opaque = (
                role[0] + "-" + hashlib.sha256(identifier.encode()).hexdigest()[:24]
            )
            mapping[identifier] = opaque
            image = folder / "images" / (opaque + ".jpg")
            shutil.copyfile(record["image"], image)
            signature = cache[identifier]
            if sha256_file(image) != signature["sha256"]:
                raise ValueError("Copied image mismatch")
            row = {
                "id": opaque,
                "image_path": "images/" + image.name,
                "image_sha256": signature["sha256"],
                "evaluation_eligible": True,
                "text": None,
                "category": "",
                "attributes": {},
            }
            rows.append(row)
            lineage.append(
                {
                    "id": opaque,
                    "source_id": identifier,
                    "pair_group": record["pair_id"],
                    "role": role,
                    "category_id": record["category_id"],
                    "identities": record["keys"],
                    "annotation_sha256": record["annotation_sha256"],
                    "fingerprints": signature,
                }
            )
        write_rows(folder / "products.jsonl", rows)
        sanitized[role] = rows
    truth = []
    for identifier in chosen_queries:
        query = records[identifier]
        positive = [
            mapping[key]
            for key, row in chosen_gallery.items()
            if set(query["keys"]) & set(row["keys"])
            and cache[key]["sha256"] != cache[identifier]["sha256"]
            and cache[key]["pixels"] != cache[identifier]["pixels"]
        ]
        if not positive:
            raise ValueError("Query has no positive within the frozen gallery")
        truth.append(
            {
                "query_id": mapping[identifier],
                "pair_group": query["pair_id"],
                "category_id": query["category_id"],
                "relevant_ids": positive,
            }
        )
    write_json(private / "qrels.json", truth)
    write_json(private / "lineage.json", lineage)
    all_fresh = [cache[key] for key in [*chosen_queries, *chosen_gallery]]
    assert all(
        not is_exposed(f, previous, policy["dhash_threshold"]) for f in all_fresh
    )
    fresh_groups = {records[k]["pair_id"] for k in [*chosen_queries, *chosen_gallery]}
    assert not fresh_groups & old_pairs
    report = {
        "role": "independent_same_item_holdout",
        "source_split": policy["source_split"],
        "query_count": len(chosen_queries),
        "gallery_count": len(chosen_gallery),
        "groups": len(fresh_groups),
        "query_categories": dict(counts),
        "recorded_exposure_audit_passed": True,
        "prior_id_overlap": 0,
        "prior_pair_overlap_conservative_cross_split": 0,
        "prior_byte_or_pixel_overlap": 0,
        "prior_dhash_candidates_at_or_below_threshold": 0,
        "dhash_threshold": policy["dhash_threshold"],
        "duplicate_limit": "dHash screens candidates; not a proof against all semantic duplicates",
        "unrecorded_human_access": "not certified",
        "source_metadata_files_scanned": len(paths),
        "source_metadata_scan_sha256": scan_digest.hexdigest(),
        "pre_inference_exclusions": dict(skipped),
        "sampling_scope": "Hash-prioritized metadata prefix; single-item user queries with paired shop views; not random population representation",
        "qrels_origin": "Source annotation identity; not human broad-product relevance",
        "recipe_sha256": sha256_file(bundle / "recipe.json"),
        "exposure_sha256": sha256_file(bundle / "exposure.json"),
    }
    write_json(output / "audit.json", report)
    manifest = {
        "files": {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in output.rglob("*")
            if p.is_file()
        },
        "role": "independent_same_item_holdout",
        "receipt": "Evaluator qrels kept separate from model input; same local account can read them; logical separation, not security isolation",
    }
    write_json(output / "manifest.json", manifest)
    return report
