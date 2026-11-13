"""Freeze a new unreviewed acceptance set; technical preparation never runs models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import run  # noqa: F401
import simple_cycle_holdout as technical
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_final_acceptance_20261110"
CONTROL = PROJECT_ROOT / "artifacts/module2_acceptance_isolation_20261110"
PARENT = (
    PROJECT_ROOT
    / "artifacts/module2_final_acceptance_20261105/future_selection_exclusions.json"
)
SHARED = (
    PROJECT_ROOT
    / "artifacts/module2_p80_r90_20261106/development_benchmark/shared_gallery.json"
)
SOURCES = (
    "scripts/module2_final_reservation.py",
    "scripts/simple_cycle_holdout.py",
    "src/multimodal_ecommerce/reservation_sampling.py",
    "src/multimodal_ecommerce/evaluation_access.py",
    "scripts/run.py",
)
POLICY = dict(
    technical.POLICY,
    seed=2026111003,
    prefix_scan_start=400000,
    prefix_scan_limit=800000,
    queue_per_fine_type=200,
    reservation_date="2026-11-10",
    source_range_frozen_before_download=True,
    project_human_relevance_labels=0,
    project_AI_relevance_labels=0,
    source_category_metadata_is_not_relevance_gold=True,
    no_development_query_or_score_based_selection=True,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusions(registry: dict) -> Path:
    path = CONTROL / "all_prior_exclusions.json"
    if path.exists():
        saved = _load(path)
        if any(
            sha256_file(PROJECT_ROOT / p) != digest
            for p, digest in saved["current_input_sha256"].items()
        ):
            raise ValueError(
                "Exposure inputs changed after reservation; preserve this snapshot"
            )
        return path
    require_data_access([PARENT], "exclusion_metadata", registry)
    prior = _load(PARENT)
    ids = set(prior["ids"])
    source_ids = set(prior["product_source_ids"])
    family = set(prior["family_keys"])
    signatures = {row["sha256"]: row for row in prior["image_fingerprints"]}
    inputs = {PARENT.relative_to(PROJECT_ROOT).as_posix(): sha256_file(PARENT)}
    images: set[Path] = set()
    skipped = []

    def collect(row: dict, folder: Path) -> None:
        """Collect exclusion identities without relevance judgments.

        Args:
            row: Existing development metadata record.
            folder: Directory resolving its recorded source image path.
        """
        key = str(row.get("id") or "")
        if key:
            ids.add(key)
        if key.startswith("product/") or row.get("dataset") == "product":
            source_ids.add(str(row.get("source_id") or key.rsplit("/", 1)[-1]))
            family.update(product_family_keys(row))
        value = row.get("image_path")
        if value:
            candidates = [(folder / value).resolve(), (PROJECT_ROOT / value).resolve()]
            for image in candidates:
                if image.is_file() and image.is_relative_to(PROJECT_ROOT):
                    images.add(image)
                    break

    for metadata in sorted((PROJECT_ROOT / "data").rglob("*.jsonl")):
        try:
            require_data_access([metadata], "development", registry)
        except PermissionError:
            logging.getLogger(__name__).debug(
                "Skip protected registry covered by prior exclusions", exc_info=True
            )
            skipped.append(metadata.relative_to(PROJECT_ROOT).as_posix())
            continue
        inputs[metadata.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(metadata)
        with metadata.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        collect(row, metadata.parent)
    require_data_access([SHARED], "development", registry)
    inputs[SHARED.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(SHARED)
    shared = _load(SHARED)
    for row in shared["catalog"]:
        collect(row, PROJECT_ROOT)
    for query in shared["query_manifest"].values():
        collect(query, PROJECT_ROOT)
    for i, image in enumerate(sorted(images), 1):
        try:
            require_data_access([image], "development", registry)
        except PermissionError:
            logging.getLogger(__name__).debug(
                "Skip protected image; retain prior exclusion fingerprints",
                exc_info=True,
            )
            continue
        digest = sha256_file(image)
        if digest not in signatures:
            signatures[digest] = fingerprints(image)
        if i % 1000 == 0:
            print(f"Exclusion identity audit {i}/{len(images)}", flush=True)
    write_json(
        path,
        dict(
            role="exclusion_metadata_only_not_relevance_truth",
            ids=sorted(ids),
            product_source_ids=sorted(source_ids),
            family_keys=sorted(family),
            image_fingerprints=list(signatures.values()),
            current_input_sha256=inputs,
            protected_registries_not_reopened=skipped,
            protected_registries_covered_by_parent_exclusion_overlay=True,
            scope="Conservative union of recorded prior use and older reservations; no certification of undocumented human access or foundation-model pretraining",
            counts=dict(
                ids=len(ids),
                product_source_ids=len(source_ids),
                family_keys=len(family),
                image_fingerprints=len(signatures),
            ),
        ),
    )
    return path


def _seal(exposure: Path, code: dict) -> None:
    frozen = ROOT / "frozen_media_manifest.json"
    if not frozen.exists():
        raise ValueError(
            "Technical reservation incomplete; inspect the shortage report"
        )
    if any(sha256_file(PROJECT_ROOT / p) != digest for p, digest in code.items()):
        raise ValueError("Preparation source changed during media acquisition")
    manifest = _load(frozen)
    if manifest["gallery"] != 300 or manifest["image_queries"] != 30:
        raise ValueError("Preserve the original acceptance sampling counts")
    if any(
        sha256_file(ROOT / name) != digest for name, digest in manifest["files"].items()
    ):
        raise ValueError("Reserved media differs from its frozen identity")
    prior = _load(exposure)
    gallery = json.loads((ROOT / "gallery.reserved.json").read_text())
    image_queries = json.loads((ROOT / "image_queries.reserved.json").read_text())
    selected = gallery + image_queries
    if (
        len(gallery) != 300
        or len(image_queries) != 30
        or len({r["id"] for r in selected}) != 330
        or len(manifest["files"]) != 330
    ):
        raise ValueError("Distinct complete 300-product and 30-query media required")
    selected_families = set(prior["family_keys"])
    selected_signatures = list(prior["image_fingerprints"])
    for row in selected:
        if (
            set(row["family_keys"]) & selected_families
            or is_exposed(row["fingerprints"], selected_signatures, 4)
            or row["source_id"] in set(prior["product_source_ids"])
            or not POLICY["prefix_scan_start"]
            < row["source_position"]
            <= POLICY["prefix_scan_limit"]
            or manifest["files"].get(row["image_path"]) != row["fingerprints"]["sha256"]
        ):
            raise ValueError(
                "Prior exposure, duplicate or source-range mismatch in reservation"
            )
        selected_families.update(row["family_keys"])
        selected_signatures.append(row["fingerprints"])
    queue = json.loads((ROOT / "candidates.reserved.json").read_text())
    keys = set(prior["family_keys"])
    product_ids = set(prior["product_source_ids"])
    for row in queue:
        keys.update(row["family_keys"])
        product_ids.add(row["source_id"])
    technical_rows = _load(ROOT / "technical_downloads.json")
    signatures = {row["sha256"]: row for row in prior["image_fingerprints"]}
    for row in technical_rows.values():
        if row["status"] == "downloaded":
            signatures[row["fingerprints"]["sha256"]] = row["fingerprints"]
    export = dict(
        role="exclusion_only_prior_development_and_all_reserved_candidates",
        ids=sorted(set(prior["ids"]) | {r["id"] for r in queue}),
        product_source_ids=sorted(product_ids),
        family_keys=sorted(keys),
        image_fingerprints=list(signatures.values()),
        reservation_root=ROOT.relative_to(PROJECT_ROOT).as_posix(),
        reservation_policy_sha256=sha256_file(ROOT / "reservation_protocol.json"),
        no_labels_predictions_or_quality_scores=True,
    )
    export_path = ROOT / "future_selection_exclusions.json"
    if export_path.exists() and _load(export_path) != export:
        raise ValueError("Preserve the sealed future exclusion export")
    write_json(export_path, export)
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "seal.json"
    }
    seal = dict(
        role="reserved_unexposed_final_acceptance_not_ready_for_inference",
        reservation_date="2026-11-10",
        policy=POLICY,
        files=files,
        exclusion_snapshot_sha256=sha256_file(exposure),
        implementation_sha256=code,
        gallery=300,
        image_queries=30,
        independently_authored_text_queries_pending=30,
        human_relevance_labels=0,
        AI_relevance_labels=0,
        model_predictions=0,
        semantic_image_views=0,
        source_native_categories_used_for_stratification_only=True,
        consumed=False,
        development_or_training_use_allowed=False,
        tuning_use_allowed=False,
        model_inference_allowed=False,
        final_model_freeze_required=True,
        independent_complete_human_gold_required=True,
        formal_acceptance=False,
        formal_metrics=None,
        duplicate_limit="Source IDs, family keys, bytes, decoded pixels and dHash<=4 screened; not proof against all semantic duplicates",
        prior_and_internal_family_byte_pixel_near_overlap=0,
    )
    if (ROOT / "seal.json").exists() and _load(ROOT / "seal.json") != seal:
        raise ValueError("Preserve the sealed acceptance manifest")
    write_json(ROOT / "seal.json", seal)
    write_json(
        CONTROL / "reservation_status.json",
        dict(
            role=seal["role"],
            gallery=300,
            image_queries=30,
            text_queries_pending=30,
            project_relevance_labels=0,
            model_predictions=0,
            semantic_image_views=0,
            consumed=False,
            acceptance_inference_allowed=False,
            reservation_root=ROOT.relative_to(PROJECT_ROOT).as_posix(),
            seal_sha256=sha256_file(ROOT / "seal.json"),
            exclusion_counts=prior["counts"],
            formal_metrics=None,
        ),
    )


def _verify() -> None:
    seal = _load(ROOT / "seal.json")
    expected = seal["files"]
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "seal.json"
    }
    if actual != set(expected) or any(
        sha256_file(ROOT / name) != digest for name, digest in expected.items()
    ):
        raise ValueError("Sealed acceptance contents changed")
    archived = CONTROL / "preparation_source_snapshot"
    for name, digest in seal["implementation_sha256"].items():
        path = archived / name
        if not path.exists():
            path = PROJECT_ROOT / name
        if sha256_file(path) != digest:
            raise ValueError(
                "Original preparation implementation is unavailable or altered"
            )
    for name in ("gallery.reserved.json", "image_queries.reserved.json"):
        for row in json.loads((ROOT / name).read_text()):
            if fingerprints(ROOT / row["image_path"]) != row["fingerprints"]:
                raise ValueError("Reserved image pixel/hash metadata changed")
    if (
        seal["policy"] != POLICY
        or seal["model_predictions"]
        or seal["human_relevance_labels"]
        or seal["AI_relevance_labels"]
    ):
        raise ValueError("Preserve the unreviewed, model-unseen reservation contract")


def main() -> None:
    """Prepare or technically verify the new reservation without exposing its content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "download", "worker", "verify"))
    phase = parser.parse_args().phase
    registry = _load(PROJECT_ROOT / "configs/evaluation_registry.json")
    if phase == "verify":
        _verify()
        print(
            "Verified sealed 300-product / 30-image-query reservation; no semantic access."
        )
        return
    if (ROOT / "seal.json").exists():
        raise FileExistsError(
            "This reservation is sealed; only technical verification is allowed"
        )
    code = {p: sha256_file(PROJECT_ROOT / p) for p in SOURCES}
    exposure = _exclusions(registry)
    technical.ROOT, technical.EXPOSURE = ROOT, exposure
    technical.POLICY = POLICY
    technical.prepare()
    protocol = _load(ROOT / "reservation_protocol.json")
    if protocol["policy"] != POLICY or protocol["exclusion_sha256"] != sha256_file(
        exposure
    ):
        raise ValueError(
            "Preserve the original sampling policy and prior-exposure snapshot"
        )
    if phase in ("download", "worker"):
        technical.reserve_media()
        _seal(exposure, code)
        print(
            "Sealed fresh acceptance media: 300 products, 30 image queries; zero model predictions."
        )
    else:
        print(
            "Frozen metadata queue from source positions 400001–800000; no model predictions."
        )


if __name__ == "__main__":
    main()
