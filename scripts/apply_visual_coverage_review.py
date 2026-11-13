"""Materialize explicit native AI observations; keep evaluator fields off runtime."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from prepare_visual_coverage import PARENT, ROOT, check, load


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve actual direct visual judgments.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    check()
    if (ROOT / "review_receipt.json").exists():
        raise FileExistsError("Preserve actual direct visual judgments")
    products = read_rows(ROOT / "dataset/products.jsonl")
    source = ROOT / "native_audit/observations.tsv"
    with source.open(encoding="utf-8", newline="") as handle:
        observations = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(r["number"]) for r in observations] == list(range(1, 41))
    queries = load(PARENT / "inputs.public.json")["queries"]
    query_ids = {q["id"] for q in queries}
    qa, judgments, captions, catalog = [], [], [], []
    for product, observation in zip(products, observations):
        positive = set(filter(None, observation["positive_queries"].split(",")))
        unknown = set(filter(None, observation["unknown_queries"].split(",")))
        assert not positive & unknown and (positive | unknown) <= query_ids
        assert observation["qa"] in {"usable", "uncertain", "reject"}
        assert all(observation[k] for k in ["product_type", "appearance", "reason"])
        qa.append(
            dict(
                id=product["id"],
                status=observation["qa"],
                reason=observation["reason"],
                native_position=int(observation["number"]),
                evidence="native_audit/observations.tsv",
                review_origin="model",
                human_label=False,
            )
        )
        for query in queries:
            judgments.append(
                dict(
                    query_id=query["id"],
                    candidate_id=product["id"],
                    grade=(
                        1
                        if query["id"] in positive
                        else None if query["id"] in unknown else 0
                    ),
                    reason=(
                        observation["reason"]
                        if query["id"] in positive | unknown
                        else "已核查售卖主体为"
                        + observation["product_type"]
                        + "；原图及原标题不满足此查询的主体类型、用途或相似外形要求，场景道具不计。"
                    ),
                    native_position=int(observation["number"]),
                    evidence="native_audit/observations.tsv",
                    review_origin="model",
                    human_label=False,
                )
            )
        if observation["qa"] == "usable":
            catalog.append(
                dict(
                    id=product["id"],
                    title=product["text"],
                    image_path=product["image_path"],
                )
            )
            captions.append(
                dict(
                    id=product["id"],
                    product_type=observation["product_type"],
                    appearance=observation["appearance"],
                    caption_origin="native_AI_query_independent_visible_product_description",
                    image_sha256=sha256_file(Path(product["image_path"])),
                )
            )
    old_catalog = load(PARENT / "inputs.public.json")["catalog"]
    all_catalog = old_catalog + catalog
    ids = {r["id"] for r in all_catalog}
    assert len(ids) == len(all_catalog)
    old_labels = read_rows(
        PROJECT_ROOT / "artifacts/module2_ai_review/judgments.ai.jsonl"
    )
    combined = [r for r in old_labels + judgments if r["candidate_id"] in ids]
    pairs = {(r["query_id"], r["candidate_id"]) for r in combined}
    expected = {
        (q["id"], r["id"])
        for q in queries
        for r in all_catalog
        if q["exclude_id"] != r["id"]
    }
    assert pairs == expected and len(pairs) == len(combined)
    write_rows(ROOT / "catalog_QA.ai.jsonl", qa)
    write_rows(ROOT / "judgments.new.ai.jsonl", judgments)
    write_rows(ROOT / "judgments.ai.jsonl", combined)
    write_rows(ROOT / "profiles.public.jsonl", captions)
    write_json(ROOT / "inputs.public.json", dict(queries=queries, catalog=all_catalog))
    write_json(
        ROOT / "review_receipt.json",
        dict(
            products_reviewed=40,
            new_usable_products=len(catalog),
            quality_control=dict(Counter(r["status"] for r in qa)),
            new_relations=len(judgments),
            indexed_relations=len(combined),
            development_products=len(all_catalog),
            review_origin="model",
            human_labels=0,
            independent_human_validation=False,
            scores_observed_before_review=False,
            relevance_based_candidate_replacement=False,
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p)
                for p in [
                    source,
                    ROOT / "native_audit/review_protocol.json",
                    ROOT / "native_audit/mapping.json",
                    *sorted((ROOT / "native_audit").glob("catalog_*.jpg")),
                    ROOT / "catalog_QA.ai.jsonl",
                    ROOT / "judgments.new.ai.jsonl",
                    ROOT / "judgments.ai.jsonl",
                    ROOT / "profiles.public.jsonl",
                    ROOT / "inputs.public.json",
                ]
            },
            heldout_pixels_opened=0,
            formal_metrics=None,
            goal_met=False,
        ),
    )
    print(
        json.dumps(
            dict(
                reviewed=40,
                usable=len(catalog),
                gallery=len(all_catalog),
                relations=len(combined),
                human_labels=0,
            )
        )
    )


if __name__ == "__main__":
    main()
