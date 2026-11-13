"""Source-labeled image retrieval and real 100/300/1000-image timing tests."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from multimodal_ecommerce.dataset_release import order_key, read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.measurement import cluster_interval, distribution
from multimodal_ecommerce.retrieval import VectorIndex
from multimodal_ecommerce.retrieval_eval import ranking_metrics, write_gallery
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json

TIMING_TEXTS = [
    "黑色短裤",
    "白色短袖上衣",
    "蓝色牛仔长裤",
    "红色连衣裙",
    "运动外套",
    "长袖衬衫",
    "女士半身裙",
    "无袖背心",
    "针织长袖上衣",
    "夏季休闲服饰",
]


def item_key(row: dict, item: dict) -> str:
    """Scope pair identity to source split, then style and clothing category.

    Args:
        row: Row used by this operation.
        item: Item used by this operation.

    Returns:
        Text produced by this operation.
    """
    return f"{row['split']}/{row['pair_id']}/{item['style']}/{item['category_id']}"


def foreground_darkness(row: dict, path: Path) -> float | None:
    """Measure low-luminance fraction inside supplied polygons, not model masks.

    Args:
        row: Row used by this operation.
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if len(row["instances"]) != 1:
        return None
    with Image.open(path) as picture:
        gray = np.asarray(picture.convert("L"))
        mask = Image.new("L", picture.size)
    drawer = ImageDraw.Draw(mask)
    for polygon in row["instances"][0]["segmentation"]:
        drawer.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
    pixels = gray[np.asarray(mask) > 0]
    return float(np.mean(pixels < 70)) if len(pixels) else None


def prepare_records(dataset: Path) -> list[dict]:
    """Validate copied assets and retain metadata for source-defined positives.

    Args:
        dataset: Dataset used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Changed or escaped copied image; Copied source annotation changed;
            Duplicate source IDs.
    """
    manifest = json.loads(
        (dataset.parent / "manifest.json").read_text(encoding="utf-8")
    )
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    records = []
    for row in read_rows(dataset):
        if row["dataset"] != "deepfashion":
            continue
        path = (dataset.parent / row["image_path"]).resolve()
        if (
            not path.is_relative_to(dataset.parent.resolve())
            or sha256_file(path) != expected[row["image_path"]]
        ):
            raise ValueError("Changed or escaped copied image")
        annotation = (dataset.parent / row["annotation_path"]).resolve()
        if (
            not annotation.is_relative_to(dataset.parent.resolve())
            or sha256_file(annotation) != expected[row["annotation_path"]]
        ):
            raise ValueError("Copied source annotation changed")
        categories = sorted({item["category_name"] for item in row["instances"]})
        records.append(
            {
                "id": row["id"],
                "split": row["split"],
                "pair_group": f"{row['split']}/{row['pair_id']}",
                "source": row["image_source"],
                "category": " + ".join(categories),
                "category_ids": sorted(
                    {item["category_id"] for item in row["instances"]}
                ),
                "title": " / ".join(categories) + " · " + row["id"],
                "image_path": str(path),
                "image_sha256": expected[row["image_path"]],
                "item_keys": [
                    item_key(row, item)
                    for item in row["instances"]
                    if item.get("style", 0) > 0
                ],
                "instance_count": len(row["instances"]),
                "foreground_dark_fraction": foreground_darkness(row, path),
            }
        )
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    return records


def same_item_protocol(
    records: list[dict], per_category: int = 6, max_categories: int = 5
) -> dict:
    """Fully label user-to-shop same-item retrieval from original annotations.

    Related means same source split/pair_id/style/category. A gallery contains
    shop images only. Similar-but-different items are negatives for THIS task,
    not automatically negatives for broad similar-style relevance evaluation.

    Args:
        records: Records used by this operation.
        per_category: Per category used by this operation.
        max_categories: Max categories used by this operation.

    Returns:
        Mapping with fields: gallery_ids, queries, category_counts,
        available_identity_counts, label_source, scope.
    """
    gallery = [row for row in records if row["source"] == "shop"]
    groups: dict[str, list[dict]] = defaultdict(list)
    identities: set[str] = set()
    for row in sorted(records, key=lambda item: order_key(item["id"], 109)):
        if (
            row["source"] != "user"
            or row["instance_count"] != 1
            or len(row["item_keys"]) != 1
        ):
            continue
        identity = row["item_keys"][0]
        if identity in identities:
            continue
        relevant = [
            item["id"]
            for item in gallery
            if identity in item["item_keys"]
            and item["image_sha256"] != row["image_sha256"]
        ]
        if not relevant:
            continue
        identities.add(identity)
        groups[row["category"]].append({"record": row, "relevant_ids": relevant})
    categories = sorted(
        groups, key=lambda category: (-len(groups[category]), category)
    )[:max_categories]
    queries = [
        query for category in categories for query in groups[category][:per_category]
    ]
    return {
        "gallery_ids": [row["id"] for row in gallery],
        "queries": queries,
        "category_counts": {
            category: min(per_category, len(groups[category]))
            for category in categories
        },
        "available_identity_counts": {
            category: len(group) for category, group in groups.items()
        },
        "label_source": "Original DeepFashion2 source split + pair_id + style>0 + category; query=user, gallery=shop. No model-derived or invented human judgments.",
        "scope": "Development same-item benchmark with full source-defined positives within this gallery. Not the 300-product similar-style relevance benchmark.",
    }


def benchmark(embedder: QwenEmbedder, dataset: Path, output: Path) -> dict:
    """Build real image vectors, measure separated paths and score paired items.

    Args:
        embedder: Embedder used by this operation.
        dataset: Dataset used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: model, dataset_sha256, corpus_records,
        unique_image_sha256, corpus_by_source, timing_unique_inputs_per_modality,
        timing_repetitions, model_load_seconds, in_memory_index_construction_seconds,
        latency, same_item, hard_negatives, cosine_distributions, threshold,
        diagnostics, environment_threads, sampling, timing_corpus_texts, scope.

    Raises:
        FileExistsError: Use a new benchmark directory.
        ValueError: Need at least 1000 real image records; no replication allowed; Need
            at least three categories with source-labeled positives.
    """
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new benchmark directory")
    records = prepare_records(dataset)
    if len(records) < 1000:
        raise ValueError(
            "Need at least 1000 real image records; no replication allowed"
        )
    output.mkdir(parents=True)
    protocol = same_item_protocol(records)
    if len(protocol["category_counts"]) < 3:
        raise ValueError("Need at least three categories with source-labeled positives")
    write_json(output / "protocol.json", protocol)
    vectors = []
    text_cache = {}
    for number, row in enumerate(records, 1):
        vectors.append(embedder.encode(image=Path(row["image_path"])))
        if row["category"] not in text_cache:
            text_cache[row["category"]] = embedder.encode(text=row["category"])
        if number % 100 == 0:
            print(f"Benchmark images {number}/{len(records)}", flush=True)
    images = np.stack(vectors)
    texts = np.stack([text_cache[row["category"]] for row in records])
    np.savez(output / "vectors.npz", images=images, texts=texts)
    write_json(output / "records.json", records)
    by_id = {row["id"]: index for index, row in enumerate(records)}
    # Same real vectors and fixed nesting for scale comparisons; no synthetic rows.
    order = sorted(range(len(records)), key=lambda i: order_key(records[i]["id"], 110))
    loaded = {}
    load_timings = {}
    for size in (100, 300, 1000):
        started = time.perf_counter()
        ids = order[:size]
        loaded[size] = VectorIndex([records[i] for i in ids], images[ids], texts[ids])
        load_timings[str(size)] = time.perf_counter() - started
    query_images = [Path(records[i]["image_path"]) for i in order[:10]]
    embedder.encode(text=TIMING_TEXTS[0], query=True)
    embedder.encode(image=query_images[0], query=True)
    timings: list[dict] = []
    scores: dict[str, list[float]] = {"text": [], "image": []}
    for repetition in range(5):
        for modality in ("text", "image"):
            for number in range(10):
                started = time.perf_counter()
                vector = (
                    embedder.encode(text=TIMING_TEXTS[number], query=True)
                    if modality == "text"
                    else embedder.encode(image=query_images[number], query=True)
                )
                embedding_seconds = time.perf_counter() - started
                for size, index in loaded.items():
                    started = time.perf_counter()
                    index.search(vector, top_k=10, target="image")
                    search_seconds = time.perf_counter() - started
                    timings.append(
                        {
                            "modality": modality,
                            "repetition": repetition,
                            "query_number": number,
                            "gallery_size": size,
                            "embedding_seconds": embedding_seconds,
                            "search_seconds_preloaded": search_seconds,
                            "combined_seconds": embedding_seconds + search_seconds,
                        }
                    )
                if repetition == 0:
                    scores[modality].extend(
                        float(value) for value in loaded[1000].images @ vector
                    )
    write_json(output / "timings.json", timings)
    shop_ids = [by_id[key] for key in protocol["gallery_ids"]]
    shop = VectorIndex(
        [records[i] for i in shop_ids], images[shop_ids], texts[shop_ids]
    )
    quality: list[dict] = []
    sections: list[dict] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    hard_negatives: list[dict] = []
    rankings: list[dict] = []
    for query in protocol["queries"]:
        row = query["record"]
        vector = embedder.encode(image=Path(row["image_path"]), query=True)
        ranked = shop.search(vector, top_k=len(shop.records))
        # Remove any byte-identical copy before computing the bounded-pool metric.
        ranked = [
            item for item in ranked if item["image_sha256"] != row["image_sha256"]
        ]
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        relevant = set(query["relevant_ids"])
        metrics = ranking_metrics([item["id"] for item in ranked], relevant, 10)
        rankings.append(
            {
                "query_id": row["id"],
                "results": [
                    {key: item[key] for key in ("id", "rank", "score")}
                    for item in ranked
                ],
            }
        )
        quality.append(
            {
                "id": row["id"],
                "group": row["pair_group"],
                "category": row["category"],
                "relevant_count": len(relevant),
                "precision_ceiling@10": min(10, len(relevant)) / 10,
                **metrics,
            }
        )
        positive_scores.extend(
            item["score"] for item in ranked if item["id"] in relevant
        )
        negative_scores.extend(
            item["score"] for item in ranked if item["id"] not in relevant
        )
        negatives = [
            item
            for item in ranked
            if not set(row["category_ids"]) & set(item["category_ids"])
        ]
        dark = (
            row["foreground_dark_fraction"] is not None
            and row["foreground_dark_fraction"] >= 0.5
        )
        dark_negatives = [
            item
            for item in negatives
            if item["foreground_dark_fraction"] is not None
            and item["foreground_dark_fraction"] >= 0.5
        ]
        first_positive = next(item["rank"] for item in ranked if item["id"] in relevant)
        hard_negatives.append(
            {
                "query_id": row["id"],
                "query_category": row["category"],
                "query_dark": dark,
                "first_positive_rank": first_positive,
                "cross_category_in_top10": sum(
                    not set(row["category_ids"]) & set(item["category_ids"])
                    for item in ranked[:10]
                ),
                "highest_cross_category_negative": negatives[:1],
                "same_dark_cross_category_negatives": (
                    dark_negatives[:3] if dark else []
                ),
                "dark_negative_before_positive": dark
                and any(item["rank"] < first_positive for item in dark_negatives),
            }
        )
        sections.append(
            {
                "query": "同款找回 · " + row["id"],
                "query_image": row["image_path"],
                "results": ranked[:10],
            }
        )
    write_json(output / "query_metrics.json", quality)
    write_json(output / "rankings.json", rankings)
    write_json(output / "hard_negatives.json", hard_negatives)
    write_gallery(output / "same_item_results.html", sections)
    aggregates = {
        name: cluster_interval(
            [row[name] for row in quality], [row["group"] for row in quality]
        )
        for name in ("precision@10", "recall@10")
    }
    dark_cases = [
        row
        for row in hard_negatives
        if row["query_dark"] and row["same_dark_cross_category_negatives"]
    ]
    report = {
        "model": embedder.config,
        "dataset_sha256": sha256_file(dataset),
        "corpus_records": len(records),
        "unique_image_sha256": len({row["image_sha256"] for row in records}),
        "corpus_by_source": dict(Counter(row["source"] for row in records)),
        "timing_unique_inputs_per_modality": 10,
        "timing_repetitions": 5,
        "model_load_seconds": embedder.load_seconds,
        "in_memory_index_construction_seconds": load_timings,
        "latency": {
            f"{modality}/{size}": {
                key: distribution(
                    [
                        row[key]
                        for row in timings
                        if row["modality"] == modality and row["gallery_size"] == size
                    ]
                )
                for key in (
                    "embedding_seconds",
                    "search_seconds_preloaded",
                    "combined_seconds",
                )
            }
            for modality in ("text", "image")
            for size in (100, 300, 1000)
        },
        "same_item": {
            "queries": len(quality),
            "gallery_size": len(shop.records),
            "category_counts": protocol["category_counts"],
            "metrics": aggregates,
            "by_category": {
                category: {
                    "queries": sum(row["category"] == category for row in quality),
                    **{
                        metric: float(
                            np.mean(
                                [
                                    row[metric]
                                    for row in quality
                                    if row["category"] == category
                                ]
                            )
                        )
                        for metric in ("precision@10", "recall@10")
                    },
                }
                for category in protocol["category_counts"]
            },
            "mean_precision_ceiling@10": float(
                np.mean([row["precision_ceiling@10"] for row in quality])
            ),
        },
        "hard_negatives": {
            "dark_queries_with_cross_category_dark_candidates": len(dark_cases),
            "dark_negative_before_first_positive": sum(
                row["dark_negative_before_positive"] for row in dark_cases
            ),
            "cross_category_top10_fraction": sum(
                row["cross_category_in_top10"] for row in hard_negatives
            )
            / (10 * len(hard_negatives)),
        },
        "cosine_distributions": {
            "text_to_image_unlabeled": distribution(scores["text"]),
            "image_to_image_unlabeled": distribution(scores["image"]),
            "same_item_positive": distribution(positive_scores),
            "same_item_negative": distribution(negative_scores),
        },
        "threshold": {
            "applied": None,
            "reason": "L2 normalization already exists. Raw cosine is not a probability. No min-max rescaling or threshold fitted on this evaluation set; similar-style calibration requires separate relevance labels.",
        },
        "diagnostics": embedder.diagnostics(),
        "environment_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "sampling": "First 500 source records per official train/validation split; development coverage sample, not a random population sample or untouched holdout. At most 6 distinct item identities per category across 5 eligible categories.",
        "timing_corpus_texts": "Source category names, cached per unique category combination; image retrieval timing uses only image vectors. This is not a product-title text index quality benchmark.",
        "scope": "Full source-labeled user-to-shop same-item task, not broad similar-style relevance. Foreground darkness uses supplied masks only for diagnostics, not retrieval inputs. No model training, segmentation inference, reranking or threshold optimization. Timings are warmed sequential calls, not concurrent SLA.",
    }
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "files": {
                path.name: sha256_file(path)
                for path in output.iterdir()
                if path.is_file()
            }
        },
    )
    return report
