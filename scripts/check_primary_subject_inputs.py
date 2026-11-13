"""CPU-only input and exposure checks before any new retrieval predictions."""

from __future__ import annotations

from pathlib import Path

import run  # noqa: F401
from acquire_primary_subjects import MEDIA, PARENT, QUEUE, ROOT, check
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from subject_roi_experiment import load, once
from verify_rank_stability import verify_history


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through once.
    """
    check()
    verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
    verify_files(
        PROJECT_ROOT, load(ROOT / "post_acquisition_exclusion_snapshot.json")["files"]
    )
    docs = load(ROOT / "additions/catalog.public.json")
    profiles = read_rows(ROOT / "additions/profiles.public.jsonl")
    assert len(docs) == len(profiles) == 103
    assert [r["id"] for r in docs] == [r["id"] for r in profiles]
    assert all(set(r) == {"id", "product_type", "appearance"} for r in profiles)
    assert all(
        not any(
            word in r["product_type"] + r["appearance"]
            for word in ["查询", "相关性", "正例", "负例", "匹配度", "推荐"]
        )
        for r in profiles
    )
    media, candidates = read_rows(MEDIA), read_rows(QUEUE)
    assert len(media) == len(candidates) == 298
    assert sum(r["selected_for_gallery"] for r in media) == 103
    prior = load(PARENT / "post_experiment_exclusion_snapshot.json")
    assert all(
        not set(r["family_keys"]) & set(prior["family_keys"]) for r in candidates
    )
    signatures = list(prior["image_fingerprints"])
    for row in read_rows(ROOT / "selected.evaluator.jsonl"):
        assert not is_exposed(row["fingerprints"], signatures, 4)
        assert row["fingerprints"]["sha256"] == sha256_file(Path(row["image_path"]))
        signatures.append(row["fingerprints"])
    viewed = set()
    for name in ["a", "b"]:
        review = load(ROOT / "native" / f"review_part_{name}.json")
        for key in [
            "viewed_sheets",
            "viewed_additional_originals",
            "viewed_originals",
            "query_definitions",
        ]:
            for entry in review.get(key, []):
                if not isinstance(entry, dict) or not {"file", "sha256"} <= set(entry):
                    continue
                path = (PROJECT_ROOT / entry["file"]).resolve()
                assert (
                    path.is_relative_to(PROJECT_ROOT)
                    and sha256_file(path) == entry["sha256"]
                )
                viewed.add(path)
    labels_total = 0
    for cohort in ["tune", "regression", "fresh"]:
        original = read_rows(PARENT / cohort / "judgments.ai.jsonl")
        labels = read_rows(ROOT / cohort / "judgments.ai.jsonl")
        assert labels[: len(original)] == original
        queries = load(ROOT / cohort / "queries.public.json")
        ids = {d["id"] for d in load(ROOT / cohort / "catalog.public.json")}
        assert len(ids) == 296
        expected = {(q["id"], cid) for q in queries for cid in ids}
        assert (
            len(labels) == len(expected)
            and {(r["query_id"], r["candidate_id"]) for r in labels} == expected
        )
        assert all(
            r["review_origin"] == "model"
            and r["human_label"] is False
            and (r["grade"] is None or type(r["grade"]) is int and r["grade"] in [0, 1])
            for r in labels
        )
        labels_total += len(labels)
    history = verify_history()
    stages = [
        "module2_rank_stability",
        "module2_independent_business_blind",
        "module2_structure_development",
        "module2_primary_form_coverage",
        "module2_fgclip2_comparison",
        "module2_subject_roi",
        "module2_subject_coverage_expansion",
    ]
    for name in stages:
        stage = PROJECT_ROOT / "artifacts" / name
        sealed = load(stage / "completion_receipt.json")
        verify_files(stage, sealed["files"])
        for key in ["source_files", "project_files"]:
            if sealed.get(key):
                verify_files(PROJECT_ROOT, sealed[key])
    once(
        ROOT / "pre_inference_checks.json",
        dict(
            passed=True,
            source_sha256=sha256_file(Path(__file__)),
            candidates=298,
            native_images=103,
            complete_AI_relations=labels_total,
            old_7720_relations_unchanged=True,
            public_profiles_without_judgment_fields=True,
            independent_family_and_image_overlap=0,
            unavailable_images_not_labeled=True,
            native_viewed_files_hashes_verified=len(viewed),
            previous_sealed_stages_verified=len(history) + len(stages),
            new_model_inference_required=True,
            quality_or_formal_metrics=None,
        ),
    )
    print(
        dict(
            inputs_verified=True,
            AI_relations=labels_total,
            previous_seals=len(history) + len(stages),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
