"""Check native review materialization, fresh-query execution and next exposure."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson

import run  # noqa: F401
from advance_visual_coverage import ROOT, load
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_focus_coverage_candidates import POLICY as NEXT_POLICY
from prepare_focus_coverage_candidates import REGISTRY as NEXT_REGISTRY
from prepare_focus_coverage_candidates import bucket


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    core = load(ROOT / "verification.json")
    assert (
        core["development_products"] == 383 and core["old_sealed_roots_unchanged"] == 13
    )
    products = read_rows(ROOT / "dataset/products.jsonl")
    with (ROOT / "native_audit/observations.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        observations = list(csv.DictReader(handle, delimiter="\t"))
    queries = load(ROOT / "inputs.public.json")["queries"]
    qrels = {
        (r["query_id"], r["candidate_id"]): r
        for r in read_rows(ROOT / "judgments.new.ai.jsonl")
    }
    profiles = {r["id"]: r for r in read_rows(ROOT / "profiles.public.jsonl")}
    qa = {r["id"]: r for r in read_rows(ROOT / "catalog_QA.ai.jsonl")}
    for product, observation in zip(products, observations, strict=True):
        positive = set(filter(None, observation["positive_queries"].split(",")))
        unknown = set(filter(None, observation["unknown_queries"].split(",")))
        cid = product["id"]
        assert (
            qa[cid]["status"] == observation["qa"]
            and qa[cid]["reason"] == observation["reason"]
        )
        for query in queries:
            expected = (
                1 if query["id"] in positive else None if query["id"] in unknown else 0
            )
            actual = qrels[query["id"], cid]
            assert (
                actual["grade"] == expected
                and actual["human_label"] is False
                and actual["review_origin"] == "model"
            )
        if observation["qa"] == "usable":
            assert (
                profiles[cid]["product_type"] == observation["product_type"]
                and profiles[cid]["appearance"] == observation["appearance"]
            )
            assert profiles[cid]["image_sha256"] == sha256_file(
                Path(product["image_path"])
            )
        else:
            assert cid not in profiles
    verify_files(PROJECT_ROOT, load(ROOT / "query_runtime_protocol.json")["files"])
    source_checksum = sha256_file(PROJECT_ROOT / "scripts/retrieve_visual_coverage.py")
    for name in ["demo_fresh_text", "demo_fresh_image_roi"]:
        demo = load(ROOT / name / "results.json")
        assert demo["catalog_size"] == 383 and len(demo["results"]) == 10
        assert demo["implementation_sha256"] == source_checksum
        assert (
            demo["historical_query_rankings_used"] is False
            and demo["relevance_table_used_as_runtime_feature"] is False
        )
        assert demo["formal_metrics"] is None and demo["acceptance_predictions"] == 0
        assert (
            demo["diagnostics"]["device"] == "cuda:0"
            and len(demo["diagnostics"]["calls"]) == 1
        )
    text = load(ROOT / "demo_fresh_text/results.json")
    assert text["method"] == "cross_evidence_pool60" and text["candidate_count"] == 383
    cross = load(ROOT / "demo_fresh_text/cross_encoder.json")
    assert (
        len(cross["pool"])
        == len(cross["logits"])
        == len(cross["diagnostics"]["timings"])
        == 60
    )
    assert all(
        r["score_kind"] == "ordinal_position_after_cross_reranking"
        for r in text["results"]
    )
    image = load(ROOT / "demo_fresh_image_roi/results.json")
    assert image["candidate_count"] == 382 and image["roi"] == [0.1, 0.05, 0.9, 0.95]
    assert (ROOT / "demo_fresh_image_roi/query_roi.png").is_file()
    probe = load(ROOT / "query_guard_probe.json")
    assert (
        probe["exit_code"] == 1
        and probe["heldout_pixels_opened"] is False
        and probe["models_loaded"] is False
        and probe["output_directory_created"] is False
    )
    next_root = ROOT / "coverage_next"
    next_protocol = load(next_root / "protocol.json")
    receipt = load(next_root / "receipt.json")
    verify_files(next_root, receipt["files"])
    assert next_protocol["policy"] == NEXT_POLICY
    assert (
        sha256_file(PROJECT_ROOT / "scripts/prepare_focus_coverage_candidates.py")
        == next_protocol["implementation_sha256"]
    )
    assert sha256_file(NEXT_REGISTRY) == receipt["exposure_registry_sha256"]
    future = read_rows(next_root / "source_candidates.jsonl")
    assert future == read_rows(NEXT_REGISTRY)
    prior = load(ROOT / "post_development_exclusion_snapshot.json")
    assert (
        sha256_file(ROOT / "post_development_exclusion_snapshot.json")
        == next_protocol["prior_snapshot_sha256"]
    )
    excluded = set(prior["family_keys"])
    raw_source = PROJECT_ROOT / next_protocol["source"]
    assert sha256_file(raw_source) == next_protocol["source_sha256"]
    groups: dict = {b: [] for b in ["wall_bottle_racks", "classical_storage_cabinets"]}
    with raw_source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            next_bucket = bucket(str(raw.get("title") or ""))
            if next_bucket and raw.get("url"):
                row = dict(
                    id=f"product/train/{sid}",
                    source_id=str(sid),
                    dataset="product",
                    split="train",
                    label=raw.get("label"),
                    text=raw["title"],
                    image_url=raw["url"],
                    attributes=parse_pv(raw.get("pv") or ""),
                    bucket=next_bucket,
                    source_position=number,
                    source_record_sha256=canonical_hash(raw),
                )
                keys = product_family_keys(row)
                if not keys & excluded:
                    row.update(
                        family_keys=sorted(keys),
                        priority=hashlib.sha256(
                            f"{NEXT_POLICY['seed']}:{row['id']}".encode()
                        ).hexdigest(),
                        project_role="exposed_development_metadata_not_relevance_truth",
                        image_path=None,
                    )
                    groups[next_bucket].append(row)
            if number == NEXT_POLICY["scan_limit"]:
                break
    replayed, seen = [], set(excluded)
    for name, values in groups.items():
        count = 0
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            replayed.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == NEXT_POLICY["metadata_per_bucket"]:
                break
    assert replayed == future and len(future) == 41
    assert receipt["per_bucket"] == dict(Counter(r["bucket"] for r in future))
    snapshot = load(ROOT / "post_preparation_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, snapshot["files"])
    assert set(snapshot["family_keys"]) == seen
    assert set(snapshot["ids"]) == set(prior["ids"]) | {r["id"] for r in future}
    assert snapshot["image_fingerprints"] == prior["image_fingerprints"]
    capacity = load(next_root / "capacity_note.json")
    assert (
        capacity["queue_capacity_before_download_dedup"]
        == sum(
            min(NEXT_POLICY["future_images_per_bucket"], n)
            for n in receipt["per_bucket"].values()
        )
        == 21
    )
    assert (
        capacity["metadata_shortfall"] == 3
        and capacity["queue_not_ready_for_24_images"] is True
    )
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    assert (
        registry["current_module2_progress_file"]
        == "artifacts/module2_visual_coverage/public_status.json"
    )
    status = load(ROOT / "public_status.json")
    assert status["formal_metrics"] is None and status["quality_metrics"] is None
    assert (
        status["formal_acceptance"] is False
        and status["goal_met"] is False
        and status["default_strategy_changed"] is False
    )
    result = dict(
        native_AI_products_checked=40,
        new_relations_replayed=720,
        new_query_independent_caption_fields_checked=39,
        fresh_query_modes_verified=2,
        actual_fresh_query_embeddings=2,
        actual_fresh_text_cross_pairs=60,
        reserved_path_blocked_before_decode=True,
        next_metadata_replayed=41,
        next_queue_nominal_image_capacity=21,
        next_wall_rack_metadata_shortfall=3,
        heldout_pixels_opened=0,
        acceptance_predictions=0,
        formal_metrics=None,
        goal_met=False,
    )
    destination = ROOT / "delivery_verification.json"
    if destination.exists():
        assert load(destination) == result
    else:
        write_json(destination, result)
    if (ROOT / "completion_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "completion_receipt.json")["files"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
