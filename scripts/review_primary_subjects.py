"""Integrate explicit native AI reviews; original relations remain byte-for-byte
values.
"""

from __future__ import annotations

from pathlib import Path

import run  # noqa: F401
from acquire_primary_subjects import PARENT, ROOT, check
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import sha256_file
from primary_form_coverage import receipt
from subject_roi_experiment import load, once


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve pre-ranking native labels.

    Notes:
        Writes derived artifacts through once, receipt, write_rows.
    """
    check()
    verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
    if (ROOT / "label_receipt.json").exists():
        raise FileExistsError("Preserve pre-ranking native labels")
    assert not any(
        (ROOT / c / "predictions.json").exists()
        for c in ["tune", "regression", "fresh"]
    )
    docs = load(ROOT / "additions/catalog.public.json")
    files = [ROOT / "native/review_part_a.json", ROOT / "native/review_part_b.json"]
    reviews = [load(p) for p in files]
    assert all(
        r["review_origin"] == "model"
        and r["human_label"] is False
        and r["new_rankings_seen"] is False
        for r in reviews
    )
    rows = sorted(
        [row for review in reviews for row in review["rows"]],
        key=lambda r: r["position"],
    )
    assert [r["position"] for r in rows] == list(range(1, len(docs) + 1))
    for row, doc in zip(rows, docs):
        assert row["id"] == doc["id"] and row["product_type"] and row["appearance"]
        assert set(row["relations"]) == {"tune", "regression", "fresh"}
    profiles = [
        dict(id=r["id"], product_type=r["product_type"], appearance=r["appearance"])
        for r in rows
    ]
    write_rows(ROOT / "additions/profiles.public.jsonl", profiles)
    files += [ROOT / "additions/profiles.public.jsonl"]
    counts = dict(positive=0, negative=0, unknown=0)
    original_count = 0
    for cohort in ["tune", "regression", "fresh"]:
        queries = load(PARENT / cohort / "queries.public.json")
        qids = {q["id"] for q in queries}
        original = read_rows(PARENT / cohort / "judgments.ai.jsonl")
        original_count += len(original)
        added = []
        for query in queries:
            for row in rows:
                rel = row["relations"][cohort]
                assert set(rel) == {"positive", "unknown"}
                assert set(rel["positive"]) <= qids and set(rel["unknown"]) <= qids
                assert not set(rel["positive"]) & set(rel["unknown"])
                grade = (
                    1
                    if query["id"] in rel["positive"]
                    else None if query["id"] in rel["unknown"] else 0
                )
                counts[
                    (
                        "positive"
                        if grade == 1
                        else "unknown" if grade is None else "negative"
                    )
                ] += 1
                reason = (
                    "已逐图比较：主要商品类型、用途及关键可见结构符合该查询，来源文字支持；不要求同SKU。"
                    if grade == 1
                    else (
                        rel["unknown"][query["id"]]
                        if grade is None
                        else row["negative_reason"]
                    )
                )
                added.append(
                    dict(
                        query_id=query["id"],
                        candidate_id=row["id"],
                        grade=grade,
                        reason=reason,
                        review_origin="model",
                        human_label=False,
                        evidence=(
                            "native/review_part_a.json"
                            if row["position"] <= 56
                            else "native/review_part_b.json"
                        ),
                        addition_position=row["position"],
                    )
                )
        folder = ROOT / cohort
        write_rows(folder / "judgments.ai.jsonl", original + added)
        once(folder / "queries.public.json", queries)
        once(
            folder / "catalog.public.json",
            load(PARENT / cohort / "catalog.public.json") + docs,
        )
        write_rows(
            folder / "profiles.public.jsonl",
            read_rows(PARENT / cohort / "profiles.public.jsonl") + profiles,
        )
        files += [
            folder / n
            for n in [
                "judgments.ai.jsonl",
                "queries.public.json",
                "catalog.public.json",
                "profiles.public.jsonl",
            ]
        ]
    once(
        ROOT / "native/review_provenance.json",
        dict(
            review_origin="model",
            human_labels=0,
            native_product_images=len(rows),
            source_titles_used=True,
            new_rankings_seen=False,
            old_development_background_known=True,
            independent_human_or_double_blind_reviewer=False,
            existing_age_and_dresser_boundary_uncertainty_retained=True,
            unavailable_images_not_labeled=True,
            original_labels_unchanged=original_count,
            complete_new_relation_counts=counts,
        ),
    )
    files += [ROOT / "native/review_provenance.json"]
    receipt(
        ROOT,
        "label_receipt.json",
        files,
        source_sha256=sha256_file(Path(__file__)),
        native_product_images=len(rows),
        original_relations=original_count,
        new_relations=40 * len(rows),
        total_relations=original_count + 40 * len(rows),
        new_grade_counts=counts,
        human_labels=0,
        frozen_before_predictions=True,
    )
    print(
        dict(native_images=len(rows), new_relations=40 * len(rows), grades=counts),
        flush=True,
    )


if __name__ == "__main__":
    main()
