"""Reuse audited gallery fingerprints without decoding all products per query."""

from __future__ import annotations

import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def main() -> None:
    """Reuse audited gallery fingerprints without decoding all products per query."""
    root = PROJECT_ROOT / "artifacts/module2_machine_assist"
    target = root / "semantic_ranker/runtime_fingerprints.json"
    if target.exists():
        raise FileExistsError("Preserve runtime fingerprint evidence")
    training = PROJECT_ROOT / "artifacts/module2_training"
    receipt = json.loads(
        (training / "completion_receipt.json").read_text(encoding="utf-8")
    )
    name = "post_training_exposure_snapshot.json"
    verify_files(training, {name: receipt["files"][name]})
    snapshot = json.loads((training / name).read_text(encoding="utf-8"))
    lookup = {r["sha256"]: r for r in snapshot["image_fingerprints"]}
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, protocol["input_files"])
    index, _, _ = RegionalIndex.load(PROJECT_ROOT / "data/index_regions_products_v1")
    records = {}
    for row in index.records:
        image = row["image_path"]
        digest = sha256_file(Path(image))
        if digest not in lookup:
            raise ValueError(
                "Gallery signature was not in the audited exposure snapshot"
            )
        records[row["id"]] = lookup[digest]
    write_json(
        target,
        dict(
            records=records,
            source_files=protocol["input_files"],
            audited_snapshot_sha256=sha256_file(training / name),
            policy="Byte and oriented decoded-pixel identity; snapshot fingerprints reused exactly, no approximate self exclusion",
        ),
    )
    write_json(
        root / "semantic_ranker/runtime_receipt.json",
        dict(
            files={"runtime_fingerprints.json": sha256_file(target)},
            implementation_sha256=sha256_file(Path(__file__).resolve()),
        ),
    )
    print(
        json.dumps(
            dict(
                cached_gallery_signatures=len(records),
                decode_per_query="query image only",
            )
        )
    )


if __name__ == "__main__":
    main()
