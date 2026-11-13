"""Preserve semantic self-contradiction as abstention without rerunning the VLM."""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import grounded_subject_experiment as original
import grounded_subject_runtime as core
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

ROOT = original.ROOT
SOURCES = [
    "scripts/grounded_subject_recovery.py",
    "tests/test_grounded_subject_recovery.py",
]


def safe_localize(
    image: Path,
    folder: Path,
    client: OllamaClient,
    save: Callable[[Path, object], None],
    read: Callable[[Path], dict],
) -> dict:
    """Return safe localize for this workflow.

    Args:
        image: Image input used by the operation.
        folder: Directory containing the operation artifacts.
        client: Configured local model client used for inference.
        save: Save used by this operation.
        read: Read used by this operation.

    Returns:
        Mapping with fields: value, pixel_box, review, accepted, source_image,
        selected_image, image_uses_gallery_or_labels, semantic_abstention.
    """
    try:
        return core.localize(image, folder, client, save, read)
    except ValueError as error:
        if str(error) != "Native accept contradicts required checks":
            raise
        # Valid complete native response with logically inconsistent semantic
        # fields is recorded explicitly. No engineering failure becomes a grade.
        value = core.parse_native(
            read(folder / "grounding.native.json")["raw"], core.SCHEMA
        )
        review = core.parse_native(
            read(folder / "review.native.json")["raw"], core.REVIEW_SCHEMA
        )
        assert review["decision"] == "accept" and not (
            review["correct_sale_item"] is True
            and review["complete_visible_item"] is True
        )
        with core.Image.open(image) as raw:
            size = core.ImageOps.exif_transpose(raw).size
        result = dict(
            value=value,
            pixel_box=core.geometry(value, size),
            review=review,
            accepted=False,
            source_image=str(Path(image).resolve()),
            selected_image=str(Path(image).resolve()),
            image_uses_gallery_or_labels=False,
            semantic_abstention="Native accept/check contradiction; retain full query, original raw response unchanged, no retry",
        )
        save(folder / "result.json", result)
        return result


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    original.guard()
    failure = load(ROOT / "failure.json")
    assert "Native accept contradicts required checks" in failure["traceback"]
    assert not (ROOT / "prediction_receipt.json").exists()
    assert load(ROOT / "recovery_engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES] + [
        ROOT / "protocol.json",
        ROOT / "failure.json",
        ROOT / "recovery_engineering_checks.pre.json",
    ]
    once(
        ROOT / "semantic_abstention_execution_protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            old_sources_and_protocol_unchanged=True,
            execution_change="Valid complete native semantic accept contradicting required checks is explicitly uncertain and full-image fallback; all other parse, schema, context or transport failures stop. No response rewritten or retried.",
            applies_to="Every query and future heldout with the same contradiction, no query-ID exception",
            existing_native_results_reused=True,
            new_model_prompt_schema_options=False,
            ranking_rules_unchanged=True,
            quality_results_seen=False,
            formal_metrics=None,
        ),
    )


def execution_guard() -> None:
    """Apply the execution guard workflow step."""
    original.guard()
    verify_files(
        PROJECT_ROOT,
        load(ROOT / "semantic_abstention_execution_protocol.json")["files"],
    )


def runtime() -> ModuleType:
    """Return runtime for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    module = load_project_module(
        "grounded_subject_abstention_runtime",
        PROJECT_ROOT / "scripts/grounded_subject_experiment.py",
    )
    module.localize = safe_localize
    module.guard = execution_guard
    return module


def worker() -> None:
    """Apply the worker workflow step.

    Raises:
        RuntimeError: Another subject worker holds lock.
    """
    module = runtime()
    with module.attributes.locked() as acquired:
        if not acquired:
            raise RuntimeError("Another subject worker holds lock")
        execution_guard()
        for name, stage in [
            ("grounding_receipt.json", module.grounding),
            ("encoding_receipt.json", module.encoding),
            ("plan_receipt.json", module.plan),
            ("reranker_receipt.json", module.infer),
            ("prediction_receipt.json", module.prediction),
            ("evaluation_receipt.json", module.evaluation),
            ("completion_receipt.json", module.finish),
        ]:
            if (ROOT / name).exists():
                module.checked(name)
                continue
            while stage() is False:
                print(
                    dict(waiting_for_natural_GPU_release=True, stage=name), flush=True
                )
                time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "verify"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "verify":
        execution_guard()
        original.verify()
        print(dict(seal_and_semantic_abstention_verified=True))
    else:
        try:
            worker()
        except Exception:
            once(
                ROOT / "recovery_failure.json",
                dict(traceback=traceback.format_exc(), formal_metrics=None),
            )
            raise
