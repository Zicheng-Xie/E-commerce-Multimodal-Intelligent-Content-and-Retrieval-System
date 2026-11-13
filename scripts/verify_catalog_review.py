"""Verify sampled provenance, duplicate exclusions, portable packets and blank
labels.
"""

from __future__ import annotations

import json
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any

import ijson

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_review_jobs import ROOT as ORIGINAL
from multimodal_ecommerce.business_review_jobs import (
    canonical_hash,
    validate_submission,
)
from multimodal_ecommerce.catalog_preparation import POLICY, metadata_bucket
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_catalog_review"
REGISTRY = PROJECT_ROOT / "data/module2_development_expansion/records.jsonl"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def payload(content: str) -> Any:
    """Return payload for this workflow.

    Args:
        content: Content used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.

    Raises:
        ValueError: Missing actual page payload.
    """
    match = re.search(r"window\.JOB=(.*?);</script>", content, re.S)
    if match is None:
        raise ValueError("Missing actual page payload")
    return json.loads(match[1])


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    protocol = load(ROOT / "preparation_protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    assert protocol["policy"] == POLICY
    assert sha256_file(PROJECT_ROOT / protocol["source"]) == protocol["source_sha256"]
    assert (
        sha256_file(ROOT / "prior_exclusion.json") == protocol["prior_exclusion_sha256"]
    )
    prior = load(ROOT / "prior_exclusion.json")
    verify_files(PROJECT_ROOT, prior["data_files"])
    reserved = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    assert sha256_file(reserved) == prior["reserved_registry_sha256"]
    metadata = load(ROOT / "metadata_receipt.json")
    assert (
        sha256_file(ROOT / "source_candidates.jsonl")
        == metadata["source_candidates_sha256"]
    )
    queue = read_rows(ROOT / "source_candidates.jsonl")
    expected = {r["id"].rsplit("/", 1)[-1]: r for r in queue}
    found = set()
    with (PROJECT_ROOT / protocol["source"]).open("rb") as handle:
        for n, (sid, row) in enumerate(ijson.kvitems(handle, ""), 1):
            if str(sid) in expected:
                value = expected[str(sid)]
                assert canonical_hash(row) == value["source_record_sha256"]
                assert n == value["source_position"]
                assert metadata_bucket(row["label"], row["title"]) == value["bucket"]
                found.add(str(sid))
            if n == POLICY["scan_limit"]:
                break
    assert len(found) == len(queue) == 912
    manifest = load(ROOT / "dataset/manifest.json")
    verify_files(ROOT / "dataset", manifest["files"])
    assert sha256_file(REGISTRY) == manifest["registry_sha256"]
    registry = read_rows(REGISTRY)
    assert {r["id"] for r in registry} == {r["id"] for r in queue}
    products = read_rows(ROOT / "dataset/products.jsonl")
    assert len(products) == len({r["id"] for r in products}) == 64
    assert not {r["id"] for r in products} & {r["id"] for r in read_rows(reserved)}
    keys = set(prior["family_keys"])
    signatures: list = []
    for row in products:
        image = ROOT / "dataset" / row["image_path"]
        signature = fingerprints(image)
        assert signature == row["fingerprints"]
        assert not set(row["family_keys"]) & keys
        assert not is_exposed(signature, prior["image_fingerprints"], 4)
        assert not is_exposed(signature, signatures, 4)
        assert row["catalog_QA_status"] == "pending" and row["relevance_grade"] is None
        keys.update(row["family_keys"])
        signatures.append(signature)
    ready = load(ROOT / "reviews_ready.json")
    assert (
        ready["priority_relations"] == 59
        and ready["batches_containing_priorities"] == 24
    )
    original_manifest = load(ORIGINAL / "manifest.json")
    original_rows = read_rows(ORIGINAL / "judgments.blank.jsonl")
    original_by_id = {r["id"]: r for r in original_rows}
    preferred = set(load(ROOT / "priority_review/mapping.json")["priority_row_ids"])
    page_count = 0
    for slot in ["A", "B"]:
        seen: set = set()
        for file in (ROOT / "priority_review" / ("rater_" + slot) / "batches").glob(
            "*.html"
        ):
            job = payload(file.read_text(encoding="utf-8"))
            rows = job["rows"]
            assert all(original_by_id[r["id"]] == r for r in rows)
            first = sum(r["id"] in preferred for r in rows)
            assert all(r["id"] in preferred for r in rows[:first])
            assert all(r["id"] not in preferred for r in rows[first:])
            # In-memory structural validation only; no export or human judgment created.
            probe = dict(
                release_id=job["release_id"],
                batch_id=job["batch_id"],
                review_slot=slot,
                reviewer="structure probe only; no judgments",
                declared_human=True,
                adjudication_of=None,
                rows=rows,
            )
            assert validate_submission(original_manifest, original_rows, probe) == rows
            assert all(
                r["grade"] is None and r["review_origin"] is None and not r["reviewer"]
                for r in rows
            )
            seen.update(r["id"] for r in rows)
            page_count += 1
        assert seen == set(original_by_id)
    assert page_count == 108
    qa = ROOT / "catalog_QA"
    verify_files(qa, load(qa / "release_receipt.json")["files"])
    templates = read_rows(qa / "items.blank.jsonl")
    assert len(templates) == 64 and all(
        r["verdict"] is None and r["review_origin"] is None and "grade" not in r
        for r in templates
    )
    for folder in (qa / "merges").iterdir():
        if folder.is_dir():
            verify_files(folder, load(folder / "receipt.json")["files"])
    # Replay the immutable empty-inbox rehearsal, not later real QA returns.
    status = load(qa / "merges" / canonical_hash([])[:24] / "status.json")
    assert (
        status["pending"] == 64
        and status["declared_reviewers"] == 0
        and not status["accepted_catalog_ids"]
    )
    archive_html = 0
    for archive in [*(ROOT / "priority_review").glob("*.zip"), *qa.glob("*.zip")]:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names = set(z.namelist())
            for name in names:
                assert not name.startswith("/") and ".." not in Path(name).parts
                if not name.endswith(".html"):
                    continue
                content = z.read(name).decode("utf-8")
                for ref in re.findall(r'(?:src|href)="([^"]+)"', content):
                    if not ref.startswith(("http:", "https:", "#")):
                        assert (
                            posixpath.normpath(
                                posixpath.join(posixpath.dirname(name), ref)
                            )
                            in names
                        ), (archive, name, ref)
                if "window.JOB=" in content:
                    job = payload(content)
                    media = [
                        v
                        for entry in job.get("presentation", {}).values()
                        for v in entry.values()
                        if v
                    ]
                    media.extend(r["image"] for r in job["rows"] if "image" in r)
                    for ref in media:
                        assert (
                            posixpath.normpath(
                                posixpath.join(posixpath.dirname(name), ref)
                            )
                            in names
                        )
                    archive_html += 1
    assert archive_html == 110
    previous = load(
        PROJECT_ROOT / "artifacts/module2_business_acceptance/prior_exposure.json"
    )
    exposure_fingerprints = {r["sha256"]: r for r in previous["image_fingerprints"]}
    reserved_lineage = read_rows(
        PROJECT_ROOT
        / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
    )
    for row in reserved_lineage:
        exposure_fingerprints[row["fingerprints"]["sha256"]] = row["fingerprints"]
    cache = load(ROOT / "download_cache.json")
    for row in cache.values():
        if row["status"] == "downloaded":
            assert fingerprints(ROOT / row["path"]) == row["fingerprints"]
            exposure_fingerprints[row["fingerprints"]["sha256"]] = row["fingerprints"]
    exclusion_ids = sorted(
        set(previous["ids"])
        | {r["id"] for r in read_rows(reserved)}
        | {r["id"] for r in queue}
    )
    exposure = dict(
        scope="Combined exclusion snapshot for later dataset selection; reserved acceptance role remains unchanged and is not development data",
        files={
            **prior["data_files"],
            REGISTRY.relative_to(PROJECT_ROOT).as_posix(): sha256_file(REGISTRY),
        },
        ids=exclusion_ids,
        deepfashion_pair_ids_conservative_cross_split=previous[
            "deepfashion_pair_ids_conservative_cross_split"
        ],
        product_source_ids=sorted(
            set(previous["product_source_ids"])
            | {r["source_id"] for r in read_rows(reserved)}
            | {r["source_id"] for r in queue}
        ),
        image_fingerprints=list(exposure_fingerprints.values()),
        family_keys=sorted(
            set(prior["family_keys"]) | {k for r in queue for k in r["family_keys"]}
        ),
        new_metadata_candidates=len(queue),
        new_selected_catalog_candidates=64,
        new_model_predictions=0,
    )
    if (ROOT / "post_preparation_exclusion_snapshot.json").exists():
        assert load(ROOT / "post_preparation_exclusion_snapshot.json") == exposure
    else:
        write_json(ROOT / "post_preparation_exclusion_snapshot.json", exposure)
    old_roots = [
        "module2_blind",
        "module2_precision",
        "module2_training",
        "module2_precision60",
        "module2_business_review",
        "module2_business_acceptance",
        "module2_machine_assist",
        "module2_continuation",
        "module2_target_p60_r85",
        "module2_precision_recall",
    ]
    for name in old_roots:
        folder = PROJECT_ROOT / "artifacts" / name
        receipt = (
            "preparation_receipt.json"
            if name in ["module2_business_review", "module2_business_acceptance"]
            else "completion_receipt.json"
        )
        verify_files(folder, load(folder / receipt)["files"])
    result = dict(
        metadata_rows_replayed=912,
        selected_images_fingerprinted=64,
        original_review_pages_checked=108,
        portable_job_pages_checked=110,
        priority_relations=59,
        original_pairs_retained=5143,
        new_catalog_QA_items=64,
        human_catalog_approvals=0,
        human_relevance_labels=0,
        old_sealed_roots_unchanged=10,
        reserved_registry_unchanged=True,
        model_predictions=0,
        formal_metrics=None,
        goal_met=False,
    )
    if (ROOT / "verification.json").exists():
        assert load(ROOT / "verification.json") == result
    else:
        write_json(ROOT / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
