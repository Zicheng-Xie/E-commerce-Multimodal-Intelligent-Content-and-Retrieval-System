"""Verify real corrected artifacts, frozen baselines and review provenance."""

from __future__ import annotations

import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.content_guard import (
    check_claim_support,
    check_source_consistency,
)
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.taxonomy import load_policy, resolve_category


def load(path: Path) -> dict:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through write_json.
    """
    root = PROJECT_ROOT
    verify_files(root, load(root / "configs/copy_recipe_v3.json"))
    old = root / "data/stage2_v1/release"
    new = root / "data/stage2_taxonomy_v2"
    verify_files(old, load(old / "manifest.json"))
    verify_files(new, load(new / "manifest.json"))
    original = {r["id"]: r for r in read_rows(old / "products.jsonl")}
    kept = read_rows(new / "products.jsonl")
    decisions = read_rows(new / "decisions.jsonl")
    quarantine = read_rows(new / "quarantine.jsonl")
    policy = load_policy(root / "configs/taxonomy_v2.json")
    report = load(new / "report.json")
    assert report["source_dataset_sha256"] == sha256_file(old / "products.jsonl")
    assert report["policy_sha256"] == sha256_file(root / "configs/taxonomy_v2.json")
    assert decisions == [resolve_category(r, policy) for r in original.values()]
    assert len(kept) == 286 and len(quarantine) == 14
    assert {r["id"] for r in kept}.isdisjoint(
        {r["source_record"]["id"] for r in quarantine}
    )
    assert {r["id"] for r in kept} | {
        r["source_record"]["id"] for r in quarantine
    } == set(original)
    for row in kept:
        previous = original[row["id"]]
        for key, value in previous.items():
            if key not in {"category", "category_rule", "category_review_status"}:
                assert row[key] == value, (row["id"], key)
        assert sha256_file(new / row["image_path"]) == previous["image_sha256"]
    audit = root / "artifacts/copy_guard_audit_v2"
    verify_files(audit, load(audit / "manifest.json"))
    for path, expected in load(audit / "manifest.json")["code_sha256"].items():
        assert sha256_file(root / path) == expected
    copy_root = root / "artifacts/copy_v3_heldout_001"
    verify_files(copy_root, load(copy_root / "manifest.json"))
    copies = {r["id"]: r for r in read_rows(copy_root / "results.jsonl")}
    audited = read_rows(audit / "results.jsonl")
    assert len(audited) == len(copies) == 90
    for row in audited:
        source = copies[row["id"]]
        assert row["historical_copy"] == source.get("copy")
        assert row["source_guard"] == check_source_consistency(source["source"])
        if source.get("copy"):
            assert row["claim_guard"] == check_claim_support(
                source["source"], source["copy"]
            )
        assert row["human_reviewed"] is False
    findings = read_rows(
        root / "artifacts/stage2_semantic_review_v1/copy_findings.jsonl"
    )
    for finding in findings:
        row = copies[finding["id"]]
        field = finding["output_field"].split("/")
        text = row["copy"][field[0]]
        if len(field) == 2:
            text = text[int(field[1])]
        assert finding["output_quote"] in text
        assert finding["output_full_field"] == text
        assert finding["source_evidence"] == {
            k: row["source"]["attributes"][k] for k in finding["source_evidence"]
        }
        assert finding["source_results_sha256"] == sha256_file(
            copy_root / "results.jsonl"
        )
        assert (
            finding["review_origin"] == "model_assisted"
            and not finding["human_reviewed"]
        )
    visual_root = root / "artifacts/fashion_spotcheck_v1"
    verify_files(visual_root, load(visual_root / "manifest.json"))
    visual = read_rows(visual_root / "ai_visual_review.jsonl")
    assert len(visual) == 9
    assert len({r["id"] for r in visual}) == 9
    assert all(
        r["review_origin"] == "model_assisted" and not r["human_reviewed"]
        for r in visual
    )
    assert all(
        sha256_file(visual_root / r["render"]) == r["render_sha256"] for r in visual
    )
    for artifact in ("artifacts/stage2_predictions_v1", "artifacts/review_stage2_v1"):
        path = root / artifact
        manifest = "packet.json" if "review_stage2" in artifact else "manifest.json"
        verify_files(path, load(path / manifest))
    result = {
        "passed": True,
        "corrected_products": len(kept),
        "quarantined_products": len(quarantine),
        "reclassified_products": len(report["changed_category_ids"]),
        "replayed_copy_guards": len(audited),
        "ai_copy_cases_evidence_verified": len(findings),
        "ai_visual_spotchecks": len(visual),
        "original_ids_roles_attributes_images_preserved": True,
        "frozen_v1_data_predictions_reviews_and_v3_generator_verified": True,
        "human_quality_metrics_produced": False,
        "stage3_work": False,
    }
    write_json(root / "artifacts/stage2_corrections_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
