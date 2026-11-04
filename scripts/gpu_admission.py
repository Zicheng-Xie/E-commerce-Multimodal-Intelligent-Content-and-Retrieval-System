"""Nonintrusive CUDA admission check for historical experiment workers."""

from __future__ import annotations

import logging
import subprocess

from multimodal_ecommerce.ollama_client import OllamaClient

GPU_IDLE_BELOW_MIB = 1500


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
