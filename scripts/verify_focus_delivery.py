"""Verify review materialization, train-only supplement and fresh-query execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson

import run  # noqa: F401
from advance_focus_coverage import ROOT, load
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_form_coverage import POLICY, QUEUE, bucket
from report_focus_coverage import candidate_coverage, paired_differences


def verify_prepared() -> dict:
    """Verify prepared.

    Returns:
        Result mapping produced by the workflow.

    Notes:
        Writes derived artifacts through write_json.
    """
    destination = ROOT / "preparation_verification.json"
    if destination.exists():
        result = load(destination)
        verify_files(PROJECT_ROOT, result["files"])
        return result
    qa = {r["id"]: r for r in read_rows(ROOT / "catalog_QA.ai.jsonl")}
    qrels = {
        (r["query_id"], r["candidate_id"]): r
        for r in read_rows(ROOT / "judgments.new.ai.jsonl")
    }
    profiles = {r["id"]: r for r in read_rows(ROOT / "profiles.public.jsonl")}
    queries = load(ROOT / "inputs.public.json")["queries"]
    total = 0
    for relative in ["", "supplement"]:
        folder = ROOT / relative
        products = read_rows(folder / "dataset/products.jsonl")
        with (folder / "native_audit/observations.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            table = list(csv.DictReader(handle, delimiter="\t"))
        for product, observed in zip(products, table, strict=True):
            total += 1
            cid = product["id"]
            assert (
                qa[cid]["status"] == observed["qa"]
                and qa[cid]["reason"] == observed["reason"]
            )
            positive = set(filter(None, observed["positive_queries"].split(",")))
            unknown = set(filter(None, observed["unknown_queries"].split(",")))
            for q in queries:
                expected = (
                    1 if q["id"] in positive else None if q["id"] in unknown else 0
                )
                actual = qrels[q["id"], cid]
                assert (
                    actual["grade"] == expected
                    and actual["review_origin"] == "model"
                    and actual["human_label"] is False
                )
                assert (
                    actual["evidence"]
                    == (folder / "native_audit/observations.tsv")
                    .relative_to(ROOT)
                    .as_posix()
                )
            if observed["qa"] == "usable":
                assert (
                    profiles[cid]["product_type"] == observed["product_type"]
                    and profiles[cid]["appearance"] == observed["appearance"]
                )
                assert profiles[cid]["image_sha256"] == sha256_file(
                    Path(product["image_path"])
                )
            else:
                assert cid not in profiles
    assert total == 35 and len(qrels) == 630
    folder = ROOT / "supplement"
    protocol = load(folder / "protocol.json")
    receipt = load(folder / "metadata_receipt.json")
    assert protocol["policy"] == POLICY
    source = PROJECT_ROOT / protocol["source"]
    assert sha256_file(source) == protocol["source_sha256"]
    exclusion = load(folder / "prior_exclusion.json")
    excluded = set(exclusion["family_keys"])
    groups: dict = {
        b: [] for b in ["tiered_wall_spice_racks", "classical_low_cabinets"]
    }
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > POLICY["scan_after"]:
                name = bucket(str(raw.get("title") or ""))
                if name and raw.get("url"):
                    row = dict(
                        id=f"product/train/{sid}",
                        source_id=str(sid),
                        dataset="product",
                        split="train",
                        label=raw.get("label"),
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=name,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{POLICY['seed']}:{row['id']}".encode()
                            ).hexdigest(),
                            project_role="exposed_development_metadata_not_relevance_truth",
                            image_path=None,
                        )
                        groups[name].append(row)
            if number == POLICY["scan_limit"]:
                break
    expected_rows, seen = [], set(excluded)
    for name, values in groups.items():
        count = 0
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            expected_rows.append(row)
            seen.update(row["family_keys"])
            count += 1
            if count == POLICY["metadata_per_bucket"]:
                break
    assert (
        expected_rows
        == read_rows(QUEUE)
        == read_rows(folder / "source_candidates.jsonl")
    )
    assert number == receipt["scanned"] == 1000000 and len(expected_rows) == 30
    assert Counter(r["bucket"] for r in expected_rows) == receipt["per_bucket"]
    assert receipt["per_bucket"] == dict(
        tiered_wall_spice_racks=6, classical_low_cabinets=24
    )
    assert all(
        r["source_position"] > 200000 and r["split"] == "train" for r in expected_rows
    )
    assets = [
        Path(__file__),
        source,
        QUEUE,
        ROOT / "catalog_QA.ai.jsonl",
        ROOT / "judgments.new.ai.jsonl",
        ROOT / "profiles.public.jsonl",
        ROOT / "inputs.public.json",
        ROOT / "supplement/protocol.json",
        ROOT / "supplement/prior_exclusion.json",
        ROOT / "supplement/metadata_receipt.json",
        ROOT / "supplement/source_candidates.jsonl",
    ]
    for relative in ["", "supplement"]:
        folder = ROOT / relative
        assets.extend(
            [
                folder / "dataset/products.jsonl",
                folder / "native_audit/observations.tsv",
                folder / "native_audit/review_protocol.json",
                folder / "native_audit/mapping.json",
            ]
        )
    result = dict(
        native_products_replayed=total,
        relations_replayed=len(qrels),
        metadata_replayed=len(expected_rows),
        train_rows_scanned=number,
        files={p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in assets},
        heldout_pixels_opened=0,
        acceptance_predictions=0,
    )
    write_json(destination, result)
    return result


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    preparation = verify_prepared()
    assert (
        preparation["native_products_replayed"] == 35
        and preparation["relations_replayed"] == 630
    )
    core = load(ROOT / "verification.json")
    assert (
        core["development_products"] == 418 and core["old_sealed_roots_unchanged"] == 14
    )
    queries = load(ROOT / "inputs.public.json")["queries"]
    rankings = load(ROOT / "rankings.json")
    labels = read_rows(ROOT / "judgments.ai.jsonl")
    modes = {q["id"]: "text" if q["text"] else "image" for q in queries}
    assert paired_differences(rankings, labels, modes) == load(
        ROOT / "internal_paired_differences.json"
    )
    assert candidate_coverage(
        rankings,
        labels,
        modes,
        load(ROOT / "pools.json"),
        load(ROOT / "query_enrichment/ranking_features.json"),
    ) == load(ROOT / "internal_candidate_coverage.json")
    verify_files(PROJECT_ROOT, load(ROOT / "query_runtime_protocol.json")["files"])
    checksum = sha256_file(PROJECT_ROOT / "scripts/retrieve_focus_development.py")
    demos = {}
    for name in ["demo_fresh_text", "demo_auto_image_roi"]:
        demo = load(ROOT / name / "results.json")
        assert demo["catalog_size"] == 418 and len(demo["results"]) == 10
        assert demo["implementation_sha256"] == checksum
        assert (
            demo["historical_query_rankings_used"] is False
            and demo["relevance_table_used_as_runtime_feature"] is False
        )
        assert (
            demo["formal_metrics"] is None
            and demo["acceptance_predictions"] == 0
            and demo["diagnostics"]["device"] == "cuda:0"
        )
        demos[name] = demo
    assert len(demos["demo_fresh_text"]["diagnostics"]["calls"]) == 1
    image = demos["demo_auto_image_roi"]
    assert (
        image["method"] == "described_cross_pool60" and image["candidate_count"] == 417
    )
    assert image["roi"] == [0.48, 0.13, 0.98, 0.80]
    description = load(ROOT / "demo_auto_image_roi/query_description.json")
    assert (
        description["gallery_or_judgments_supplied"] is False
        and description["human_label"] is False
    )
    cross = load(ROOT / "demo_auto_image_roi/cross_encoder.json")
    assert (
        len(cross["pool"])
        == len(cross["logits"])
        == len(cross["diagnostics"]["timings"])
        == 60
    )
    view_metadata = load(ROOT / "demo_auto_image_roi/query_views.json")
    assert len(image["diagnostics"]["calls"]) == len(view_metadata) + int(
        bool(description["text"])
    )
    guard = load(ROOT / "query_guard_probe.json")
    assert (
        guard["exit_code"] == 1
        and guard["heldout_pixels_opened"] is False
        and guard["models_loaded"] is False
    )
    snapshot = load(ROOT / "post_development_exclusion_snapshot.json")
    verify_files(PROJECT_ROOT, snapshot["files"])
    expected = read_rows(QUEUE)
    assert {r["id"] for r in expected} <= set(snapshot["ids"])
    assert {k for r in expected for k in r["family_keys"]} <= set(
        snapshot["family_keys"]
    )
    status = load(ROOT / "public_status.json")
    assert (
        status["quality_metrics"] is None
        and status["formal_metrics"] is None
        and status["formal_acceptance"] is False
        and status["goal_met"] is False
    )
    assert (
        status["acceptance_predictions"] == 0
        and status["default_strategy_changed"] is False
    )
    result = dict(
        native_products_replayed=35,
        new_AI_relations_replayed=630,
        train_metadata_replayed=30,
        train_rows_scanned=1000000,
        supplement_images=14,
        supplement_shortfall=2,
        paired_unknown_completion_differences_replayed=True,
        candidate_pool_diagnostics_replayed=True,
        fresh_query_modes_verified=2,
        actual_automatic_fresh_image_description=1,
        actual_fresh_image_cross_pairs=60,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-only", action="store_true")
    args = parser.parse_args()
    if args.prepared_only:
        result = verify_prepared()
        print(json.dumps({k: v for k, v in result.items() if k != "files"}))
    else:
        main()
