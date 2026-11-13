"""Resume frozen development stages after the user's GPU job releases naturally."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
PYTHON = PROJECT_ROOT / ".venv312/Scripts/python.exe"
LOG = PROJECT_ROOT / "tmp/module2_primary_subjects_runner.log"
STATUS = ROOT / "continuation_status.json"
GPU_IDLE_BELOW_MIB = 1500
COHORTS = ("tune", "regression", "fresh")


@dataclass(frozen=True)
class Stage:
    """Describe a resumable workflow stage, its command arguments and required
    outputs.
    """

    name: str
    arguments: tuple[str, ...]
    outputs: tuple[str, ...]
    gpu: bool = False
    manifest_base: Path = ROOT


STAGES = (
    Stage(
        "query_captions",
        ("scripts/primary_subject_captions_safe.py", "run_captions"),
        ("query_caption_receipt.json",),
        gpu=True,
        manifest_base=PROJECT_ROOT,
    ),
    Stage(
        "prepare_features",
        ("scripts/encode_primary_subjects.py", "prepare"),
        ("feature_protocol.json",),
        manifest_base=PROJECT_ROOT,
    ),
    Stage(
        "qwen_embeddings",
        ("scripts/encode_primary_subjects.py", "qwen"),
        ("encoding_receipt.json",),
        gpu=True,
    ),
    Stage(
        "subject_detection",
        ("scripts/encode_primary_subjects.py", "detect"),
        ("detection_receipt.json",),
        gpu=True,
    ),
    Stage(
        "fg_embeddings",
        ("scripts/encode_primary_subjects.py", "fg"),
        ("fg_encoding_receipt.json",),
        gpu=True,
    ),
    Stage(
        "prepare_comparison",
        ("scripts/primary_subject_experiment.py", "prepare"),
        ("comparison_protocol.json",),
        manifest_base=PROJECT_ROOT,
    ),
    Stage(
        "reranker_inference",
        ("scripts/primary_subject_experiment.py", "infer"),
        ("reranker_receipt.json",),
        gpu=True,
    ),
    Stage(
        "predictions",
        ("scripts/primary_subject_experiment.py", "predict"),
        ("prediction_receipt.json",),
    ),
    Stage(
        "evaluation",
        ("scripts/primary_subject_experiment.py", "evaluate"),
        tuple(
            f"{cohort}/{name}"
            for cohort in COHORTS
            for name in (
                "internal_metrics.ai.development.json",
                "internal_paired_differences.json",
                "internal_guard_incremental.json",
                "internal_roi_incremental.json",
            )
        ),
    ),
    Stage(
        "verification",
        ("scripts/primary_subject_experiment.py", "verify"),
        ("verification.json",),
    ),
    Stage(
        "freeze_report",
        ("scripts/report_primary_subjects.py", "freeze"),
        ("report_protocol.json",),
    ),
    Stage(
        "report",
        ("scripts/report_primary_subjects.py", "report"),
        (
            "decision.json",
            "internal_comparison.html",
            "internal_comparison.ai.development.md",
            "runtime_audit.json",
        ),
    ),
    Stage(
        "exposure",
        ("scripts/report_primary_subjects.py", "exposure"),
        ("exposure_receipt.json",),
    ),
    Stage(
        "artifact_checks",
        ("scripts/check_primary_subject_results.py",),
        ("checks.json",),
    ),
)
SEAL = Stage(
    "seal",
    ("scripts/report_primary_subjects.py", "seal"),
    ("completion_receipt.json",),
)


def load(path: Path) -> dict:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty object: {path.name}")
    return value


def emit(stage: str, phase: str) -> None:
    """Apply the emit workflow step.

    Args:
        stage: Stage used by this operation.
        phase: Phase used by this operation.
    """
    print(json.dumps(dict(stage=stage, phase=phase)), flush=True)


def execution_guard() -> None:
    """Apply the execution guard workflow step."""
    verify_files(PROJECT_ROOT, load(ROOT / "execution_source_manifest.json")["files"])


def completed() -> bool:
    """Return completed for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    path = ROOT / "completion_receipt.json"
    if not path.is_file():
        return False
    receipt = load(path)
    verify_files(ROOT, receipt["files"])
    verify_files(PROJECT_ROOT, receipt["source_files"])
    verify_files(PROJECT_ROOT, receipt["project_files"])
    return True


@contextmanager
def exclusive_runner() -> Iterator:
    """Use an OS lock; leave the persistent lock file intact on every exit.

    Yields:
        Values produced by this operation, in the stated return order.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = ensure_output_path(PROJECT_ROOT / "tmp/module2_primary_subjects.guard.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x+b")
        handle.write(b"0")
        handle.flush()
    except FileExistsError:
        logging.getLogger(__name__).debug(
            "Handled exception in exclusive_runner", exc_info=True
        )
        handle = path.open("r+b")
    locked = False
    try:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            locked = True
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in exclusive_runner", exc_info=True
            )
            yield False
            return
        yield True
    finally:
        if locked:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def status(phase: str, stage: str, **extra: object) -> None:
    # The completion receipt is the final authority. Never mutate sealed files.
    """Apply the status workflow step.

    Args:
        phase: Phase used by this operation.
        stage: Stage used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Raises:
        RuntimeError: Completed evidence is read-only.

    Notes:
        Writes derived artifacts through write_json.
    """
    if (ROOT / "completion_receipt.json").exists():
        raise RuntimeError("Completed evidence is read-only")
    write_json(
        STATUS,
        dict(
            phase=phase,
            stage=stage,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            formal_metrics=None,
            independent_acceptance_started=False,
            **extra,
        ),
    )


def gpu_available() -> tuple[bool, dict]:
    """Inspect memory first; call local /api/ps once only below 1500 MiB.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        if result.returncode:
            return False, dict(reason="GPU_memory_observation_unavailable")
        memory = float(result.stdout.strip().splitlines()[0].strip())
        if not 0 <= memory < GPU_IDLE_BELOW_MIB:
            return False, dict(reason="GPU_memory_busy", memory_used_mib=memory)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        logging.getLogger(__name__).debug(
            "Handled exception in gpu_available", exc_info=True
        )
        return False, dict(reason="GPU_memory_observation_unavailable")
    client = OllamaClient(timeout=10)
    try:
        models = client._request("/api/ps").get("models")
    except (OSError, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Handled exception in gpu_available", exc_info=True
        )
        return False, dict(
            reason="Cannot_confirm_Ollama_is_idle",
            memory_used_mib=memory,
            error_type=type(error).__name__,
        )
    finally:
        client.close()
    if not isinstance(models, list):
        return False, dict(
            reason="Ollama_residency_response_invalid", memory_used_mib=memory
        )
    return not models, dict(
        reason="GPU_and_Ollama_idle" if not models else "Ollama_model_resident",
        memory_used_mib=memory,
        resident_models=len(models),
    )


def stage_complete(stage: Stage) -> bool:
    """Return stage complete for this workflow.

    Args:
        stage: Stage used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if not all((ROOT / name).is_file() for name in stage.outputs):
        return False
    for name in stage.outputs:
        path = ROOT / name
        if not path.stat().st_size:
            raise ValueError(f"Empty stage output: {name}")
        if path.suffix == ".json":
            value = load(path)
            if "files" in value:
                verify_files(stage.manifest_base, value["files"])
            if "project_files" in value:
                verify_files(PROJECT_ROOT, value["project_files"])
            if "passed" in value and value["passed"] is not True:
                raise ValueError(f"Stage checks did not pass: {name}")
    return True


def run_child(stage: Stage) -> None:
    """Apply the run child workflow step.

    Args:
        stage: Stage used by this operation.

    Notes:
        Writes derived artifacts through ensure_output_path(LOG).parent.mkdir.
    """
    ensure_output_path(LOG).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {stage.name}\n")
        handle.flush()
        result = subprocess.run(
            [str(PYTHON), *stage.arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"\n[{stage.name}] exit_code={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"{stage.name} failed with exit code {result.returncode}")


def log_tail() -> str:
    """Return log tail for this workflow.

    Returns:
        Text produced by this operation.
    """
    if not LOG.is_file():
        return ""
    with LOG.open("rb") as handle:
        handle.seek(max(0, LOG.stat().st_size - 8192))
        return handle.read().decode("utf-8", errors="replace")[-2000:]


def continue_stages() -> int:
    # Do these reads before opening/creating the lock or writing any status.
    """Return continue stages for this workflow.

    Returns:
        Count or index produced by this operation.

    Raises:
        RuntimeError: Seal returned without completion receipt.
    """
    execution_guard()
    if completed():
        emit("seal", "completed")
        return 0
    with exclusive_runner() as acquired:
        if not acquired:
            return 0
        # Another runner may have sealed while this invocation was starting.
        execution_guard()
        if completed():
            emit("seal", "completed")
            return 0
        current = "integrity"
        try:
            for stage in STAGES:
                current = stage.name
                if stage_complete(stage):
                    continue
                execution_guard()
                if stage.gpu:
                    available, observation = gpu_available()
                    if not available:
                        status(
                            "waiting_for_natural_GPU_release",
                            stage.name,
                            observation=observation,
                        )
                        emit(stage.name, "waiting_for_natural_GPU_release")
                        return 0
                status("running", stage.name)
                emit(stage.name, "running")
                run_child(stage)
                if not stage_complete(stage):
                    raise RuntimeError(f"Missing completed output: {stage.name}")
                emit(stage.name, "done")
            current = SEAL.name
            execution_guard()
            status("ready_to_seal", SEAL.name)
            emit(SEAL.name, "running")
            run_child(SEAL)
            # No ROOT writes are allowed after seal, including status updates.
            if not completed():
                raise RuntimeError("Seal returned without completion receipt")
            emit(SEAL.name, "completed")
            return 0
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Handled exception in continue_stages", exc_info=True
            )
            if not (ROOT / "completion_receipt.json").exists():
                status("failed", current, error_type=type(error).__name__)
            emit(current, "failed")
            print(str(error), flush=True)
            tail = log_tail()
            if tail:
                print(tail, flush=True)
            return 1


def main() -> int:
    """Run the command-line workflow defined by this module.

    Returns:
        Count or index produced by this operation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    parser.parse_args()
    try:
        return continue_stages()
    except Exception as error:
        # Source/receipt errors before lock acquisition must not mutate ROOT.
        logging.getLogger(__name__).debug("Handled exception in main", exc_info=True)
        emit("integrity", "failed")
        print(str(error), flush=True)
        tail = log_tail()
        if tail:
            print(tail, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
