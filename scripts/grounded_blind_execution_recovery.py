"""Mechanical encode_views ABI correction; frozen methods/data stay unchanged."""

from __future__ import annotations

import argparse
from types import ModuleType, SimpleNamespace

import numpy as np

import grounded_blind_runtime as original
import grounded_blind_source_recovery as source_recovery
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.regional_retrieval import (
    RegionalRetrievalEncodeViewsInputs,
    encode_views,
)
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file

ROOT = original.ROOT


def ordered_views(
    options: RegionalRetrievalEncodeViewsInputs,
) -> tuple[np.ndarray, list[dict]]:
    """Validate the view-encoding request before delegating to the full encoder.

    Args:
        options: Encoder, input image, output directory and frozen region policy.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        TypeError: Expected frozen configuration and local views output.
    """
    if not isinstance(options.config, dict) or not hasattr(options.output, "resolve"):
        raise TypeError("Expected frozen configuration and local views output")
    return encode_views(options)


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    original.dataset.guard()
    source_recovery.guard()
    assert (
        not (ROOT / "prediction_receipt.json").exists()
        and not (ROOT / "pair_plan.json").exists()
    )
    assert (ROOT / "native_query_receipt.json").exists() and not (
        ROOT / "encoding_receipt.json"
    ).exists()
    assert load(ROOT / "execution_recovery_checks.pre.json")["passed"]
    once(
        ROOT / "runtime_argument_failure_evidence.json",
        dict(
            observed_exception="AttributeError: 'dict' object has no attribute 'resolve'",
            stage="first encode_views in blind gallery encoding",
            cause="Frozen caller passes configuration before output; callee signature requires output before configuration",
            frozen_native_queries_preserved=True,
            gallery_vector_checkpoints=0,
            prediction_files=0,
            model_scores_or_quality_results_seen=False,
        ),
    )
    paths = [
        PROJECT_ROOT / "scripts/grounded_blind_execution_recovery.py",
        PROJECT_ROOT / "tests/test_grounded_blind_execution_recovery.py",
        ROOT / "execution_recovery_checks.pre.json",
        ROOT / "runtime_argument_failure_evidence.json",
        ROOT / "release_protocol.json",
        ROOT / "source_range_execution_protocol.json",
        ROOT / "native_query_receipt.json",
        ROOT / "label_receipt.json",
    ]
    once(
        ROOT / "argument_order_execution_protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            mechanical_execution_change="Call original complete encode_views with documented output/configuration argument order; no source modification, new preprocessing, thresholds, model budget, vector formula, pooling or ranking change",
            applies_uniformly_to_every_gallery_and_query=True,
            all_AI_labels_and_queries_preserved=True,
            quality_results_seen=False,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    original.dataset.guard()
    source_recovery.guard()
    verify_files(
        PROJECT_ROOT, load(ROOT / "argument_order_execution_protocol.json")["files"]
    )


def runtime() -> ModuleType:
    """Return runtime for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    module = load_project_module(
        "grounded_blind_ordered_views_runtime",
        PROJECT_ROOT / "scripts/grounded_blind_runtime.py",
    )
    module.encode_views = ordered_views
    module.dataset = SimpleNamespace(**vars(original.dataset))
    module.dataset.guard = guard
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "verify"])
    a = parser.parse_args().action
    if a == "prepare":
        prepare()
    elif a == "worker":
        guard()
        runtime().worker()
    else:
        guard()
        print(dict(execution_source_and_arguments_verified=True))
