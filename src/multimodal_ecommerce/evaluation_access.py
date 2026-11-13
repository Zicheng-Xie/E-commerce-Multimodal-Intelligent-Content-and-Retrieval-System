"""Reject consumed and reserved data from current development entry points."""

from __future__ import annotations

import json
from pathlib import Path

from multimodal_ecommerce.safety import PROJECT_ROOT


def require_data_access(paths: list[Path], purpose: str, registry: dict) -> None:
    """Apply the most specific recorded role before opening evaluation inputs.

    Args:
        paths: Resolved or project-relative inputs to a guarded operation.
        purpose: Development, training, tuning, acceptance inference or exclusion metadata.
        registry: Current evaluation-role registry with explicit protected paths.

    Raises:
        ValueError: The purpose or project-relative input is invalid.
        PermissionError: A consumed or reserved input is forbidden for the purpose.
    """
    allowed = {
        "development",
        "training",
        "tuning",
        "acceptance_inference",
        "exclusion_metadata",
    }
    if purpose not in allowed:
        raise ValueError("Unknown evaluation access purpose")
    for path in paths:
        actual = (PROJECT_ROOT / path).resolve()
        relative = actual.relative_to(PROJECT_ROOT).as_posix()
        matches = [
            rule
            for rule in registry["rules"]
            if relative.startswith(rule["path_prefix"])
        ]
        if not matches:
            raise PermissionError(f"Unregistered evaluation input: {relative}")
        rule = max(matches, key=lambda item: len(item["path_prefix"]))
        if purpose == "exclusion_metadata":
            if relative not in registry.get("exclusion_metadata_allowlist", []):
                raise PermissionError(f"Not exclusion-only metadata: {relative}")
            continue
        if rule.get("consumed") is True:
            raise PermissionError(f"One-shot evaluation already consumed: {relative}")
        if purpose in {"development", "training", "tuning"}:
            if rule.get("development_or_training_use_allowed") is False:
                raise PermissionError(f"Protected evaluation data: {relative}")
            if "consumed" in rule["role"] or "reserved" in rule["role"]:
                raise PermissionError(f"Consumed/reserved data: {relative}")
            if purpose == "tuning" and rule.get("tuning_use_allowed") is False:
                raise PermissionError(f"Tuning forbidden: {relative}")
        if purpose == "acceptance_inference" and (
            rule.get("model_inference_allowed") is not True
            or rule.get("formal_acceptance_ready") is not True
        ):
            raise PermissionError(
                f"Final model and human gold are not frozen: {relative}"
            )


def require_command_access(script: Path, registry: dict) -> None:
    """Stop retired evaluation commands before they read media or load model assets.

    Args:
        script: Invoked command path; imports in ordinary development remain allowed.
        registry: Current policy mapping retired commands to protected dataset roots.

    Raises:
        PermissionError: A retired evaluation command attempts to reopen a sealed set.
    """
    actual = script.resolve()
    if not actual.is_relative_to(PROJECT_ROOT):
        return
    relative = actual.relative_to(PROJECT_ROOT).as_posix()
    for rule in registry.get("sealed_evaluation_commands", []):
        if relative == rule["script"]:
            raise PermissionError(
                f"Retired evaluation command {relative}: {rule['dataset_root']} is consumed or sealed. Use archived reports; do not rerun or tune on these data."
            )


def check_invoked_command(script: Path) -> None:
    """Apply the repository command policy during CLI bootstrap.

    Args:
        script: Command being invoked, before protected files or models are opened.
    """
    path = PROJECT_ROOT / "configs/evaluation_registry.json"
    require_command_access(script, json.loads(path.read_text(encoding="utf-8")))
