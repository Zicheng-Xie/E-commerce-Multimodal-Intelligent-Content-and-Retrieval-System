"""Record derived development crops so future blind sampling excludes them."""

from __future__ import annotations

import os

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.holdout_data import fingerprints
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from subject_roi_experiment import COVERAGE, ROOT, check, detection_path, load, once

REGISTRY = PROJECT_ROOT / "data/module2_subject_roi_exposure/records.jsonl"


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve completed exposure registration.

    Notes:
        Writes derived artifacts through once, write_rows.
    """
    if REGISTRY.exists() or (ROOT / "post_experiment_exclusion_snapshot.json").exists():
        raise FileExistsError("Preserve completed exposure registration")
    inputs = check()
    prior_path = COVERAGE / "post_development_exclusion_snapshot.json"
    prior = load(prior_path)
    rows = []
    for key in inputs["roi_requests"]:
        result = load(detection_path(key))
        if result["decision"]["status"] != "roi":
            continue
        for name in ["native.png", "zoom.png"]:
            path = detection_path(key).with_name(name)
            rows.append(
                dict(
                    id=f"development_roi/{key}/{name}",
                    source_image_sha256=result["source_sha256"],
                    image_path=os.path.relpath(path, REGISTRY.parent).replace(
                        "\\", "/"
                    ),
                    fingerprints=fingerprints(path),
                    project_role="exposed_derived_development_ROI_not_acceptance",
                    reusable_as_unseen=False,
                )
            )
    write_rows(REGISTRY, rows)
    snapshot = dict(prior)
    snapshot.update(
        scope="All earlier exposures plus derived subject ROIs; none reusable as unseen",
        files=dict(
            prior["files"],
            **{REGISTRY.relative_to(PROJECT_ROOT).as_posix(): sha256_file(REGISTRY)},
        ),
        image_fingerprints=prior["image_fingerprints"]
        + [r["fingerprints"] for r in rows],
        derived_subject_ROI_images=len(rows),
        parent_snapshot_sha256=sha256_file(prior_path),
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    once(
        ROOT / "exposure_receipt.json",
        dict(
            registered_crop_images=len(rows),
            registry_sha256=sha256_file(REGISTRY),
            original_ids_and_family_keys_preserved=True,
            pixels_and_dhash_recorded=True,
            new_raw_product_images=0,
        ),
    )
    print(f"Registered {len(rows)} derived development images", flush=True)


if __name__ == "__main__":
    main()
