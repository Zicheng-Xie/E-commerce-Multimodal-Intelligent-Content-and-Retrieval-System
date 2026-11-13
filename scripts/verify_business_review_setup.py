"""Verify reserved acceptance lineage and portable human review resources."""

from __future__ import annotations

import json
import posixpath
import re
import zipfile
from collections import Counter

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json


def main() -> None:
    """Verify reserved acceptance lineage and portable human review resources."""
    root = PROJECT_ROOT / "artifacts/module2_business_acceptance"
    protocol = json.loads(
        (root / "preparation_protocol.json").read_text(encoding="utf-8")
    )
    verify_files(PROJECT_ROOT, protocol["implementation"])
    old = json.loads((root / "prior_exposure.json").read_text(encoding="utf-8"))
    verify_files(PROJECT_ROOT, old["files"])
    dataset = root / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    verify_files(dataset, manifest["files"])
    lineage = read_rows(dataset / "evaluator/lineage.jsonl")
    assert len(lineage) == 330 and len({r["source_id"] for r in lineage}) == 330
    counts = Counter((r["category"], r["label"], r["role"]) for r in lineage)
    assert len(counts) == 60 and all(
        count == (1 if role == "query" else 10)
        for (_, _, role), count in counts.items()
    )
    used = set(protocol["old_product_family_keys"])
    previous: list = []
    for row in lineage:
        keys = set(row["family_keys"])
        assert keys and not keys & used
        used.update(keys)
        signature = fingerprints(dataset / row["image_path"])
        assert signature == row["fingerprints"]
        assert not is_exposed(signature, old["image_fingerprints"], 4)
        assert not is_exposed(signature, previous, 4)
        previous.append(signature)
    gallery = read_rows(dataset / "gallery/products.jsonl")
    assert len(gallery) == 300
    for row in gallery:
        assert (
            sha256_file(dataset / "gallery" / row["image_path"]) == row["image_sha256"]
        )
    registry = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    assert sha256_file(registry) == manifest["reserved_registry_sha256"]
    reserved = read_rows(registry)
    assert {r["id"] for r in reserved} == {r["id"] for r in lineage}
    for row in reserved:
        assert sha256_file(registry.parent / row["image_path"]) == row["image_sha256"]
    packet = root / "query_author"
    query_manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    verify_files(packet, query_manifest["files"])
    verify_files(PROJECT_ROOT, query_manifest["implementation"])
    review = PROJECT_ROOT / "artifacts/module2_business_review/development"
    receipt = json.loads((review / "release_receipt.json").read_text(encoding="utf-8"))
    verify_files(review, receipt["files"])
    jobs = json.loads((review / "manifest.json").read_text(encoding="utf-8"))
    assert len(jobs["batches"]) == 54
    coverage = [r for batch in jobs["batches"] for r in batch["row_ids"]]
    assert len(coverage) == 5143 and len(set(coverage)) == 5143
    pages = 0
    for slot in ["A", "B"]:
        with zipfile.ZipFile(review / f"rater_{slot}.zip") as archive:
            names = set(archive.namelist())
            assert not any(
                name.startswith("rater_" + ("B" if slot == "A" else "A") + "/")
                for name in names
            )
            for name in names:
                if "/batches/" not in name or not name.endswith(".html"):
                    continue
                content = archive.read(name).decode("utf-8-sig")
                match = re.search(r"window.JOB=(.*?);</script>", content)
                assert match
                payload = json.loads(match.group(1))
                assert payload["review_slot"] == slot
                assert len(payload["rows"]) <= 96
                for fields in payload["presentation"].values():
                    for value in fields.values():
                        if value:
                            assert (
                                posixpath.normpath(
                                    posixpath.join(posixpath.dirname(name), value)
                                )
                                in names
                            )
                script_match = re.search(r'<script src="([^"]+)"', content)
                if script_match is None:
                    raise ValueError("Review page is missing its JavaScript resource")
                script = script_match.group(1)
                assert (
                    posixpath.normpath(posixpath.join(posixpath.dirname(name), script))
                    in names
                )
                pages += 1
    for name in [
        "module2_blind",
        "module2_precision",
        "module2_training",
        "module2_precision60",
    ]:
        previous_root = PROJECT_ROOT / "artifacts" / name
        previous_receipt = json.loads(
            (previous_root / "completion_receipt.json").read_text(encoding="utf-8")
        )
        verify_files(previous_root, previous_receipt["files"])
    result = dict(
        candidate_products=300,
        image_query_candidates=30,
        text_query_slots=30,
        distinct_source_ids=330,
        prior_family_key_overlaps=0,
        cross_role_family_key_overlaps=0,
        prior_byte_pixel_or_dhash_le4_matches=0,
        new_byte_pixel_or_dhash_le4_matches=0,
        reserved_registry_paths_verified=330,
        portable_review_pages_verified=pages,
        old_sealed_artifact_roots_verified=4,
        human_query_approvals=0,
        human_relevance_labels=0,
        model_predictions=0,
        formal_acceptance=False,
        formal_metrics=None,
        limitations=[
            "Automated fingerprints do not certify absence of all semantic duplicate products",
            "Human taxonomy and catalog-family QA still pending",
            "Actual reviewer identity is self-declared, not independently authenticated",
            "Prepared cohort is balanced over selected 30 fine types; not real traffic representative",
        ],
    )
    output = root / "preparation_verification.json"
    if output.exists():
        assert json.loads(output.read_text(encoding="utf-8")) == result
    else:
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
