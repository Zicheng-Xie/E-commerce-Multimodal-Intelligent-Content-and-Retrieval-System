"""Freeze copy recipes, preserve blocked cases and prepare real human scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multimodal_ecommerce.copy_policy import (
    POLICY_VERSION,
    PROMPT,
    PROMPT_VERSION,
    SourceConflictError,
    source_policy,
)
from multimodal_ecommerce.dataset_release import order_key, read_rows, write_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.fact_check import normalize_product
from multimodal_ecommerce.ollama_client import CHAT_OPTIONS, OllamaClient, OllamaError
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)

RECIPE_FILES = [
    "src/multimodal_ecommerce/" + name + ".py"
    for name in ("copy_policy", "fact_check", "ollama_client", "copy_experiment")
]
LEGACY_IDS = {
    "619926302539",
    "585155392011",
    "617154527746",
    "607435815431",
    "623181777103",
    "606717266978",
    "620859226255",
}


def model_identity(client: OllamaClient) -> dict:
    """Read model digest from the actual local server, without downloading.

    Args:
        client: Configured local model client used for inference.

    Returns:
        Mapping with fields: name, digest, ollama_version.

    Raises:
        OllamaError: Requested model is missing or has no digest.
    """
    health = client.health()
    matching = [row for row in health["models"] if row["name"] == client.model]
    if not matching or not matching[0].get("digest"):
        raise OllamaError("Requested model is missing or has no digest")
    return {
        "name": client.model,
        "digest": matching[0]["digest"],
        "ollama_version": health["version"],
    }


def recipe_content(client: OllamaClient, dataset: Path) -> dict:
    """Fingerprint inference settings, model, data and all generation logic.

    Args:
        client: Configured local model client used for inference.
        dataset: Dataset used by this operation.

    Returns:
        Mapping with fields: prompt_version, policy_version, prompt_sha256, options,
        think, model, dataset_sha256, files.
    """
    return {
        "prompt_version": PROMPT_VERSION,
        "policy_version": POLICY_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "options": dict(CHAT_OPTIONS),
        "think": False,
        "model": model_identity(client),
        "dataset_sha256": sha256_file(dataset),
        "files": {name: sha256_file(PROJECT_ROOT / name) for name in RECIPE_FILES},
    }


def freeze_recipe(client: OllamaClient, dataset: Path, path: Path) -> dict:
    """Write an immutable recipe before opening the heldout outputs.

    Args:
        client: Configured local model client used for inference.
        dataset: Dataset used by this operation.
        path: Path of the input or output artifact used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        FileExistsError: Use a new recipe path.
    """
    path = ensure_output_path(path)
    if path.exists():
        raise FileExistsError("Use a new recipe path")
    verify_files(
        dataset.parent,
        json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8")),
    )
    recipe = recipe_content(client, dataset)
    write_json(path, recipe)
    return recipe


def select_records(dataset: Path, role: str, per_category: int) -> list[dict]:
    """Select by fixed hash priority within role/category, before any outcomes.

    Args:
        dataset: Dataset used by this operation.
        role: Role used by this operation.
        per_category: Per category used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Invalid selection arguments.
    """
    if per_category < 1 or role not in {"development", "heldout", "legacy"}:
        raise ValueError("Invalid selection arguments")
    rows = read_rows(dataset)
    if role == "legacy":
        return [row for row in rows if row.get("source_id") in LEGACY_IDS]
    counts: Counter = Counter()
    selected = []
    for row in sorted(rows, key=lambda item: order_key(item["id"], 31)):
        if row.get("project_role") != role or counts[row["category"]] >= per_category:
            continue
        selected.append(row)
        counts[row["category"]] += 1
    return selected


@dataclass(frozen=True)
class CopyExperimentRunCopyExperimentInputs:
    """Input values for run copy experiment.

    Attributes:
        client: Configured local model client used for inference.
        dataset: Dataset used by this operation.
        output: Output used by this operation.
        role: Role used by this operation.
        per_category: Per category used by this operation.
        recipe_path: Recipe path used by this operation.
    """

    client: OllamaClient
    dataset: Path
    output: Path
    _: KW_ONLY
    role: str = "development"
    per_category: int = 5
    recipe_path: Path | None = None


def run_copy_experiment(
    options: CopyExperimentRunCopyExperimentInputs,
) -> dict[str, Any]:
    """Generate each preselected item once, keeping conflicts and errors visible.

    Args:
        options: Validated input fields described by
            CopyExperimentRunCopyExperimentInputs.

    Returns:
        Mapping with fields: role, selected, status_counts, source_issue_counts,
        output_rule_counts, generated, generation_coverage, latency_after_warmup,
        quality_scores, interpretation.

    Raises:
        FileExistsError: Use a new run directory.
        ValueError: No records in requested role; Heldout generation requires a frozen
            recipe; Code, model, data or settings changed after recipe freeze.
    """
    client = options.client
    dataset = options.dataset
    output = options.output
    role = options.role
    per_category = options.per_category
    recipe_path = options.recipe_path
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Use a new run directory")
    selected = select_records(dataset, role, per_category)
    if not selected:
        raise ValueError("No records in requested role")
    if role != "legacy":
        verify_files(
            dataset.parent,
            json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8")),
        )
    current = recipe_content(client, dataset)
    if role == "heldout":
        if recipe_path is None:
            raise ValueError("Heldout generation requires a frozen recipe")
        frozen = json.loads(recipe_path.read_text(encoding="utf-8"))
        if current != frozen:
            raise ValueError(
                "Code, model, data or settings changed after recipe freeze"
            )
    output.mkdir(parents=True)
    run = {
        "role": role,
        "dataset": str(dataset.resolve()),
        "recipe": current,
        "recipe_file_sha256": sha256_file(recipe_path) if recipe_path else None,
        "selected_ids": [row["id"] for row in selected],
        "selection": "seeded SHA256 priority (seed=31) per category; no replacement of blocked/failed items",
        "counts": dict(Counter(row["category"] for row in selected)),
        "per_category": per_category,
        "score_status": "awaiting_independent_human_review",
    }
    write_json(output / "run.json", run)
    # Separate warm-up from workload timings; model coldness is not asserted.
    warmup = client.chat("请仅输出：准备就绪。")
    write_json(output / "warmup.json", warmup)
    results = []
    for number, row in enumerate(selected, 1):
        started = time.perf_counter()
        source = normalize_product(row)
        item: dict[str, Any] = {
            "id": row["id"],
            "role": role,
            "category": row["category"],
            "source": source,
            "source_policy": source_policy(source),
        }
        try:
            item["copy"] = client.generate_copy(source)
            item["status"] = item["copy"]["_meta"]["release_status"]
        except SourceConflictError as error:
            logging.getLogger(__name__).debug(
                "Handled exception in run_copy_experiment", exc_info=True
            )
            item.update(
                {"status": "blocked_source_conflict", "source_policy": error.policy}
            )
        except (OllamaError, ValueError) as error:
            logging.getLogger(__name__).debug(
                "Handled exception in run_copy_experiment", exc_info=True
            )
            item.update({"status": "generation_failed", "error": str(error)})
        item["wall_seconds_including_checks"] = time.perf_counter() - started
        results.append(item)
        name = hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".json"
        write_json(output / "records" / name, item)
        print(f"Copy {number}/{len(selected)}: {item['status']}", flush=True)
    write_rows(output / "results.jsonl", results)
    generated = [item for item in results if "copy" in item]
    timings = [item["wall_seconds_including_checks"] for item in generated]
    report = {
        "role": role,
        "selected": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "source_issue_counts": dict(
            Counter(
                issue["code"]
                for item in results
                for issue in item["source_policy"]["findings"]
            )
        ),
        "output_rule_counts": dict(
            Counter(
                issue["rule"]
                for item in generated
                for issue in item["copy"]["_meta"]["fact_check"]["findings"]
            )
        ),
        "generated": len(generated),
        "generation_coverage": len(generated) / len(results),
        "latency_after_warmup": {
            "count": len(timings),
            "p50_seconds": float(np.percentile(timings, 50)) if timings else None,
            "p95_seconds": float(np.percentile(timings, 95)) if timings else None,
            "max_seconds": max(timings) if timings else None,
            "within_10s": sum(value <= 10 for value in timings),
            "scope": "Sequential generation including checks; excludes warmup, source blocks and failed requests. Raw failure timings remain in records; not a concurrent service SLA.",
        },
        "quality_scores": None,
        "interpretation": "Rule-clean drafts and generation coverage are not factual accuracy, fluency or human relevance scores.",
    }
    write_json(output / "summary.json", report)
    # Keep every selected sample in the rubric, including blocked or failed ones.
    write_rows(
        output / "human_scores.template.jsonl",
        [
            {
                "id": item["id"],
                "status": item["status"],
                "relevance_pass": None,
                "fluency_pass": None,
                "factual_error": None,
                "reviewer": "",
                "note": "",
            }
            for item in results
        ],
    )
    write_json(
        output / "manifest.json",
        {
            "files": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        },
    )
    return report


def score_copy_run(run_root: Path, judgments: Path) -> dict:
    """Compute explicit human pass rates without hiding abstentions/failures.

    Args:
        run_root: Run root used by this operation.
        judgments: Judgments used by this operation.

    Returns:
        Mapping with fields: selected, generated_scored, generated_only_pass_rates,
        all_selected_pass_rates_counting_abstentions_as_not_passed,
        factual_error_rate_generated, judgments_sha256, run_manifest_sha256, scope.

    Raises:
        ValueError: Judgments must contain every selected ID exactly once; Missing
            reviewer; Judgment status does not match recorded generation; Absent output
            cannot receive invented quality labels; All generated outputs require
            boolean human judgments.
    """
    verify_files(
        run_root, json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    )
    results = {row["id"]: row for row in read_rows(run_root / "results.jsonl")}
    labels = read_rows(judgments)
    if len(labels) != len(results) or {row["id"] for row in labels} != set(results):
        raise ValueError("Judgments must contain every selected ID exactly once")
    scored = []
    for label in labels:
        result = results[label["id"]]
        if not str(label.get("reviewer") or "").strip():
            raise ValueError("Missing reviewer")
        if label.get("status") != result["status"]:
            raise ValueError("Judgment status does not match recorded generation")
        if "copy" not in result:
            if any(
                label.get(key) is not None
                for key in ("relevance_pass", "fluency_pass", "factual_error")
            ):
                raise ValueError("Absent output cannot receive invented quality labels")
            continue
        if not all(
            type(label.get(key)) is bool
            for key in ("relevance_pass", "fluency_pass", "factual_error")
        ):
            raise ValueError("All generated outputs require boolean human judgments")
        scored.append(label)

    def rates(denominator: int) -> dict:
        """Return rates for this workflow.

        Args:
            denominator: Denominator used by this operation.

        Returns:
            Result mapping produced by the workflow.
        """
        return {
            name: (
                sum(item[name] for item in scored) / denominator
                if denominator
                else None
            )
            for name in ("relevance_pass", "fluency_pass")
        }

    return {
        "selected": len(results),
        "generated_scored": len(scored),
        "generated_only_pass_rates": rates(len(scored)),
        "all_selected_pass_rates_counting_abstentions_as_not_passed": rates(
            len(results)
        ),
        "factual_error_rate_generated": (
            sum(row["factual_error"] for row in scored) / len(scored)
            if scored
            else None
        ),
        "judgments_sha256": sha256_file(judgments),
        "run_manifest_sha256": sha256_file(run_root / "manifest.json"),
        "scope": "Human binary rubric results; source truth is limited to supplied metadata. Both denominators reported to avoid hiding blocked samples.",
    }
