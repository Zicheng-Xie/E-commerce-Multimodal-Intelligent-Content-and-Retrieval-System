"""Fail-closed execution admission for unchanged frozen caption ROI assets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path

import caption_roi_assets as assets
import run  # noqa: F401
from caption_roi_assets import (
    IDLE_BELOW_MIB,
    PARENT,
    ROI_ROOTS,
    ROOT,
    WHOLE_PARENT,
    load,
    once,
    query_asset,
    relative,
)
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.subject_roi import SubjectDetector, save_subject_crop

COHORTS = assets.COHORTS


def prepare() -> dict:
    """Prepare the inputs and protocol for this development workflow.

    Returns:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through once.
    """
    plan = assets.prepare()
    target = ROOT / "crop_asset_execution_protocol.json"
    if target.exists():
        guard_execution()
        return plan
    original = load(ROOT / "crop_asset_protocol.json")
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/caption_roi_assets.py",
        ROOT / "crop_asset_protocol.json",
        ROOT / "crop_plan.json",
        ROI_ROOTS[0] / "detector_manifest.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    files = dict(original["files"])
    files.update({relative(p): sha256_file(p) for p in paths})
    once(
        target,
        dict(
            files=files,
            original_protocol_sha256=sha256_file(ROOT / "crop_asset_protocol.json"),
            original_plan_sha256=sha256_file(ROOT / "crop_plan.json"),
            coarse_prompts=original["coarse_prompts"],
            detector_policy=original["detector_policy"],
            supported_query_requests=original["supported_query_requests"],
            execution_only_change=True,
            old_source_and_protocol_unchanged=True,
            same_input_plan_prompts_weights_thresholds_crop_and_output_schema=True,
            memory_admission=dict(
                finite_nonnegative=True,
                strictly_below_mib=IDLE_BELOW_MIB,
                nvidia_smi_timeout_seconds=15,
                nonzero_exit_or_parse_error="wait; no Ollama request",
            ),
            ollama_admission=dict(
                timeout_seconds=10,
                required_response="dictionary with explicit empty models list",
                requests_only_after_memory_admission=True,
                invalid_response_or_exception="wait",
                unload_or_keepalive_changes=False,
            ),
            formal_metrics=None,
        ),
    )
    return plan


def guard_execution() -> dict:
    """Return guard execution for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    protocol = load(ROOT / "crop_asset_execution_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    original = load(ROOT / "crop_asset_protocol.json")
    assert protocol["coarse_prompts"] == original["coarse_prompts"]
    assert protocol["detector_policy"] == original["detector_policy"]
    assert protocol["supported_query_requests"] == original["supported_query_requests"]
    return assets.guard()


def guard() -> dict:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return guard_execution()


def gpu_available() -> bool:
    """Return gpu available for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return False
        memory = [
            float(line.strip()) for line in result.stdout.splitlines() if line.strip()
        ]
        if not memory or not all(
            math.isfinite(value) and 0 <= value < IDLE_BELOW_MIB for value in memory
        ):
            return False
    except Exception:
        logging.getLogger(__name__).debug(
            "Handled exception in gpu_available", exc_info=True
        )
        return False
    client, allowed = None, False
    try:
        client = OllamaClient(timeout=10)
        response = client._request("/api/ps")
        models = response.get("models") if isinstance(response, dict) else None
        allowed = isinstance(models, list) and not models
    except Exception:
        logging.getLogger(__name__).debug(
            "Handled exception in gpu_available", exc_info=True
        )
        allowed = False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Handled exception in gpu_available", exc_info=True
                )
                allowed = False
    return allowed


def detect() -> bool:
    """Return detect for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once.
    """
    plan = prepare()
    if (ROOT / "detection_receipt.json").exists():
        verify()
        return True
    pending = []
    for data in plan["cohorts"].values():
        for query in data["queries"]:
            target = PROJECT_ROOT / query["output_dir"] / "result.json"
            if target.exists():
                query_asset(query, target)
            else:
                pending.append(query)
    if any(q["prompt"] is not None for q in pending) and not gpu_available():
        return False
    detector, invocation = None, None
    try:
        for query in pending:
            source = PROJECT_ROOT / query["source_image"]
            assert sha256_file(source) == query["source_sha256"]
            target = PROJECT_ROOT / query["output_dir"] / "result.json"
            if query["prompt"] is None:
                result: dict = dict(
                    prompt=None,
                    detections=[],
                    seconds=0,
                    decision=dict(
                        status="global_fallback",
                        reason="Unresolved public coarse subject; retain whole query",
                        selected=None,
                    ),
                )
            else:
                if detector is None:
                    detector = SubjectDetector(ROI_ROOTS[0] / "detector_manifest.json")
                result = detector.detect(source, query["prompt"])
            result.update(
                request_id=query["roi_request"],
                source_sha256=query["source_sha256"],
                actual_model_call=query["prompt"] is not None,
                caption_sha256=query["caption_sha256"],
            )
            if result["decision"]["status"] == "roi":
                result["crop"] = save_subject_crop(
                    source, result["decision"], target.parent
                )
            once(target, result)
        if detector is not None:
            invocation = detector.diagnostics()
    finally:
        if detector is not None:
            detector.close()
    cohorts, completed = {}, []
    for cohort, data in plan["cohorts"].items():
        queries = []
        for q in data["queries"]:
            result_path = PROJECT_ROOT / q["output_dir"] / "result.json"
            queries.append(query_asset(q, result_path))
            result = load(result_path)
            completed.append(
                dict(
                    cohort=cohort,
                    query_id=q["id"],
                    actual_model_call=result["actual_model_call"],
                    seconds=result["seconds"],
                    result_file=relative(result_path),
                    result_sha256=sha256_file(result_path),
                )
            )
        cohorts[cohort] = dict(queries=queries, gallery=data["gallery"])
    files = dict(load(ROOT / "crop_asset_protocol.json")["files"])
    files[relative(ROOT / "crop_asset_protocol.json")] = sha256_file(
        ROOT / "crop_asset_protocol.json"
    )
    files.update(
        {
            relative(p): sha256_file(p)
            for p in sorted((ROOT / "query_rois").rglob("*"))
            if p.is_file()
        }
    )
    files[relative(ROOT / "crop_asset_execution_protocol.json")] = sha256_file(
        ROOT / "crop_asset_execution_protocol.json"
    )
    files[relative(Path(__file__))] = sha256_file(Path(__file__))
    for data in cohorts.values():
        for asset in data["gallery"] + data["queries"]:
            for field in [
                "source_image",
                "selected_image",
                "native_image",
                "result_file",
            ]:
                if asset.get(field):
                    path = PROJECT_ROOT / asset[field]
                    files[relative(path)] = sha256_file(path)
    once(
        ROOT / "crop_inputs.json",
        dict(
            parent=relative(PARENT),
            whole_parent=relative(WHOLE_PARENT),
            cohorts=cohorts,
            files=files,
        ),
    )
    runtime = ROOT / "detector_runtime.json"
    if runtime.exists():
        assert load(runtime)["all_completed_calls"] == completed
    else:
        once(
            runtime,
            dict(
                invocation=invocation,
                all_completed_calls=completed,
                full_official_model=True,
            ),
        )
    receipt_files = [
        ROOT / "crop_inputs.json",
        runtime,
        *[p for p in sorted((ROOT / "query_rois").rglob("*")) if p.is_file()],
    ]
    queries = [q for c in cohorts.values() for q in c["queries"]]
    once(
        ROOT / "detection_receipt.json",
        dict(
            files={
                p.relative_to(ROOT).as_posix(): sha256_file(p) for p in receipt_files
            },
            query_requests=len(queries),
            actual_full_detector_calls=sum(c["actual_model_call"] for c in completed),
            accepted=sum(q["status"] == "roi" for q in queries),
            fallback=sum(q["status"] == "global_fallback" for q in queries),
            gallery_detector_calls=0,
            all_queries_and_gallery_retained=True,
            old_crops_changed=False,
            formal_metrics=None,
        ),
    )
    verify()
    return True


def verify() -> dict:
    """Verify the recorded artifacts and their consistency.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    guard_execution()
    return assets.verify()


def exposure() -> dict:
    """Return exposure for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    guard_execution()
    return assets.exposure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "detect", "verify", "exposure"])
    action = parser.parse_args().action
    value = globals()[action]()
    if action == "prepare":
        value = dict(
            prepared=True,
            safe_execution=True,
            image_queries=sum(len(c["queries"]) for c in value["cohorts"].values()),
        )
    print(json.dumps(value, ensure_ascii=False), flush=True)
