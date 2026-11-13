"""Development-only index extension with pre-inference acceptance isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.semantic_ranker import assign_folds

POLICY: dict = dict(
    role="AI_assessed_exposed_business_development_only",
    fixed_k=10,
    precision_required=0.6,
    recall_preferred=0.85,
    exact_original_queries=18,
    original_products=286,
    new_products=58,
    total_products=344,
    family_folds=3,
    fold_seed=2026102205,
    duplicate_dhash_distance=4,
    inference_uses_AI_judgment_fields=False,
    acceptance_inference_allowed=False,
    acceptance_labels_allowed=False,
    default_promotion_allowed=False,
    development_scores_allowed_in_public_status=False,
)


def clean_document(row: dict) -> dict:
    """The inference schema deliberately omits buckets, qrels and native reasons.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, title, image_path.

    Raises:
        ValueError: Need product ID, original title and existing image; Development
            image must be a real project-local file; Reserved acceptance paths are
            blocked.
    """
    if not all(
        isinstance(row.get(k), str) and row[k] for k in ["id", "title", "image_path"]
    ):
        raise ValueError("Need product ID, original title and existing image")
    path = Path(row["image_path"]).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise ValueError("Development image must be a real project-local file")
    if (
        "module2_business_acceptance" in path.parts
        or "module2_reserved_acceptance" in path.parts
    ):
        raise ValueError("Reserved acceptance paths are blocked")
    return {"id": row["id"], "title": row["title"], "image_path": str(path)}


def acceptance_guard(
    documents: list[dict], raw: dict[str, dict], reserved: list[dict]
) -> dict:
    """Read reserved metadata/fingerprints only; never open heldout pixels.

    Args:
        documents: Documents used by this operation.
        raw: Raw used by this operation.
        reserved: Reserved used by this operation.

    Returns:
        Mapping with fields: role, checked_products, reserved_products,
        reserved_images_opened, acceptance_predictions, overlap_product_ids,
        overlap_family_keys, overlap_image_fingerprints, signatures, family_keys.

    Raises:
        ValueError: Reserved acceptance product ID blocked; Reserved acceptance product
            family blocked; Reserved acceptance exact/near-duplicate image blocked.
    """
    reserved_ids = {r["id"] for r in reserved}
    reserved_keys = {k for r in reserved for k in r["family_keys"]}
    reserved_images = [r["fingerprints"] for r in reserved]
    signatures = {}
    keys = {}
    for item in documents:
        clean = clean_document(item)
        if item["id"] in reserved_ids:
            raise ValueError("Reserved acceptance product ID blocked")
        family = product_family_keys(raw[item["id"]])
        if family & reserved_keys:
            raise ValueError("Reserved acceptance product family blocked")
        signature = fingerprints(Path(clean["image_path"]))
        if is_exposed(signature, reserved_images, POLICY["duplicate_dhash_distance"]):
            raise ValueError("Reserved acceptance exact/near-duplicate image blocked")
        signatures[item["id"]] = signature
        keys[item["id"]] = sorted(family)
    return dict(
        role="development_pre_inference_isolation_check",
        checked_products=len(documents),
        reserved_products=len(reserved),
        reserved_images_opened=0,
        acceptance_predictions=0,
        overlap_product_ids=0,
        overlap_family_keys=0,
        overlap_image_fingerprints=0,
        signatures=signatures,
        family_keys=keys,
    )


def fold_metadata(documents: list[dict], queries: list[dict], guard: dict) -> dict:
    """Join exact families and near-image duplicates before assigning fixed folds.

    Args:
        documents: Documents used by this operation.
        queries: Queries used by this operation.
        guard: Guard used by this operation.

    Returns:
        Mapping with fields: gallery_ids, query_ids, gallery_families,
        query_families, gallery_folds, query_folds, family_count.
    """
    ids = [r["id"] for r in documents]
    parent = dict(zip(ids, ids))

    def find(value: str) -> str:
        """Return find for this workflow.

        Args:
            value: Input value to validate or transform.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        """Apply the union workflow step.

        Args:
            left: Left used by this operation.
            right: Right used by this operation.
        """
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)

    owners: dict[str, str] = {}
    for cid in ids:
        signature = guard["signatures"][cid]
        for key in [
            *guard["family_keys"][cid],
            "byte:" + signature["sha256"],
            "pixel:" + signature["pixels"],
        ]:
            if key in owners:
                union(cid, owners[key])
            owners[key] = cid
    for i, cid in enumerate(ids):
        left = guard["signatures"][cid]["dhash"]
        for other in ids[:i]:
            if (left ^ guard["signatures"][other]["dhash"]).bit_count() <= POLICY[
                "duplicate_dhash_distance"
            ]:
                union(cid, other)
    qmatches = {}
    for query in queries:
        if query["image_path"]:
            signature = fingerprints(Path(query["image_path"]))
            matched = [
                cid
                for cid in ids
                if is_exposed(
                    signature,
                    [guard["signatures"][cid]],
                    POLICY["duplicate_dhash_distance"],
                )
            ]
            for cid in matched[1:]:
                union(matched[0], cid)
            qmatches[query["id"]] = matched
    families = [find(cid) for cid in ids]
    gallery_assignment = assign_folds(families, 3, POLICY["fold_seed"])
    query_families = [
        (
            find(qmatches[q["id"]][0])
            if qmatches.get(q["id"])
            else (
                "outside-query:" + sha256_file(Path(q["image_path"]))
                if q["image_path"]
                else None
            )
        )
        for q in queries
    ]
    qfolds = np.zeros(len(queries), dtype=int)
    for mode in ["text", "image"]:
        positions = [
            i for i, q in enumerate(queries) if bool(q["text"]) == (mode == "text")
        ]
        keys = [query_families[i] or "text:" + queries[i]["text"] for i in positions]
        assignment = assign_folds(keys, 3, POLICY["fold_seed"])
        for i, key in zip(positions, keys):
            qfolds[i] = assignment[key]
    return dict(
        gallery_ids=ids,
        query_ids=[q["id"] for q in queries],
        gallery_families=families,
        query_families=query_families,
        gallery_folds=[gallery_assignment[f] for f in families],
        query_folds=qfolds.tolist(),
        family_count=len(set(families)),
    )


def load_expanded_index(path: Path) -> tuple[RegionalIndex, np.ndarray]:
    """Reuse frozen original crops by reference, verifying every referenced crop.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Expanded index asset changed; Frozen upstream asset changed; Changed
            referenced region pixels; Expanded title vectors mismatch.
    """
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or sha256_file(target) != checksum:
            raise ValueError("Expanded index asset changed")
    for relative, checksum in manifest["upstream_files"].items():
        target = (PROJECT_ROOT / relative).resolve()
        if not target.is_relative_to(PROJECT_ROOT) or sha256_file(target) != checksum:
            raise ValueError("Frozen upstream asset changed")
    records = json.loads((path / "records.json").read_text(encoding="utf-8"))
    for record in records:
        for view in record["views"]:
            target = Path(view["image_path"]).resolve()
            if (
                not target.is_relative_to(PROJECT_ROOT)
                or sha256_file(target) != view["sha256"]
            ):
                raise ValueError("Changed referenced region pixels")
    with np.load(path / "vectors.npz", allow_pickle=False) as bank:
        index = RegionalIndex(records, bank["regions"], manifest["regions"])
        titles = bank["texts"].copy()
    if titles.shape != (len(records), 2048) or not np.isfinite(titles).all():
        raise ValueError("Expanded title vectors mismatch")
    return index, titles
