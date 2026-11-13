"""Traceable small multi-category generation runs with explicit split reports."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_ecommerce.fact_check import normalize_product
from multimodal_ecommerce.model_interfaces import BatchGenerator
from multimodal_ecommerce.ollama_client import OllamaError
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def generate_batch(
    client: BatchGenerator, dataset: Path, output: Path, per_category: int = 2
) -> dict[str, Any]:
    """Generate once per selected product; checkpoint successes and failures.

    Select the first N available images per category per original split. This is
    a fixed small engineering readout, not a random or representative benchmark.
    Existing run directories are refused to keep previous evidence intact.

    Args:
        client: Configured local model client used for inference.
        dataset: Dataset used by this operation.
        output: Output used by this operation.
        per_category: Per category used by this operation.

    Returns:
        Mapping with fields: splits, requested_per_category_per_split,
        selected_by_group, completed, selection_shortfalls, all_generated,
        interpretation.

    Raises:
        ValueError: per_category must be positive; No eligible local products; download
            images first.
    """
    if per_category < 1:
        raise ValueError("per_category must be positive")
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    records = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    counts: Counter = Counter()
    selected = []
    for record in records:
        group = (record["split"], record["category"])
        if not record.get("image_path") or not record.get("evaluation_eligible"):
            continue
        if counts[group] < per_category:
            counts[group] += 1
            selected.append(record)
    if not selected:
        raise ValueError("No eligible local products; download images first")
    output.mkdir(parents=True)
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "selected_ids": [r["id"] for r in selected],
        "per_category_per_split": per_category,
        "selection": "first eligible in source order",
        "note": "Frozen prompt for both splits; no automatic retries of flagged copy.",
    }
    write_json(output / "run.json", run)
    results = []
    for record in selected:
        product = normalize_product(record)
        item: dict[str, Any] = {
            "id": record["id"],
            "split": record["split"],
            "category": record["category"],
            "input": product,
        }
        try:
            item["copy"] = client.generate_copy(product)
            item["status"] = "generated"
        except (OllamaError, ValueError) as error:
            logging.getLogger(__name__).debug(
                "Handled exception in generate_batch", exc_info=True
            )
            item.update({"status": "generation_failed", "error": str(error)})
        results.append(item)
        filename = hashlib.sha256(record["id"].encode()).hexdigest()[:24] + ".json"
        write_json(output / "records" / filename, item)
    groups = {}
    for split in sorted({item["split"] for item in results}):
        subset = [item for item in results if item["split"] == split]
        completed = [item for item in subset if item["status"] == "generated"]
        flagged = [
            item
            for item in completed
            if item["copy"]["_meta"]["fact_check"]["status"] == "flagged"
        ]
        groups[split] = {
            "attempted": len(subset),
            "generated": len(completed),
            "rule_flagged": len(flagged),
            "no_rule_flags": len(completed) - len(flagged),
            "flagged_ids": [item["id"] for item in flagged],
            "wall_seconds": [
                item["copy"]["_meta"]["response"]["wall_seconds"] for item in completed
            ],
        }
    summary = {
        "splits": groups,
        "requested_per_category_per_split": per_category,
        "selected_by_group": {
            f"{key[0]}/{key[1]}": value for key, value in counts.items()
        },
        "completed": len(results),
        "selection_shortfalls": {
            f"{split}/{category}": per_category - counts[(split, category)]
            for split, category in sorted(
                {(r["split"], r["category"]) for r in records}
            )
            if counts[(split, category)] < per_category
        },
        "all_generated": all(item["status"] == "generated" for item in results),
        "interpretation": "Rule flag counts are not accuracy or human quality scores.",
    }
    write_json(output / "summary.json", summary)
    return summary
