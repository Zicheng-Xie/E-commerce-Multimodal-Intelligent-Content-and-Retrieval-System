"""Immutable engineering releases and metadata-only acceptance readiness checks."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.selected_runtime_io import (
    RuntimeAccess,
    atomic_json,
    digest,
    load,
)


def verify_release(path: Path) -> dict:
    """Verify code, actual checkpoint files and the frozen engineering evidence.

    Args:
        path: Engineering release manifest.

    Returns:
        Verified release document.
    """
    release = load(RuntimeAccess().check(path))
    body = {k: v for k, v in release.items() if k != "release_id"}
    if digest(body) != release["release_id"]:
        raise ValueError("Engineering release identity changed")
    for name, expected in release["files_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("Frozen engineering release file changed: " + name)
    for name, expected in release["model_policy"].get("libraries", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Frozen runtime dependency changed: " + name)
    return release


def freeze_release(policy: Path, index: Path, evidence: dict, output: Path) -> dict:
    """Freeze real runtime assets only after replay and cold inference have passed.

    Args:
        policy: Exact runtime policy used for index and inference.
        index: Real cold-start development index.
        evidence: Replay and cold-start report paths.
        output: New immutable release manifest.

    Returns:
        Complete release binding actual weights, model digest and code.
    """
    access = RuntimeAccess()
    for path in (
        policy,
        index / "manifest.json",
        index / "vectors.npz",
        *(Path(p) for p in evidence.values()),
    ):
        access.check(path)
    access.output(output.parent)
    replay = load(Path(evidence["replay"]))
    cold = load(Path(evidence["cold"]))
    if (
        replay.get("exact_queries") != 55
        or not replay.get("all_full_rankings_equal")
        or not cold.get("real_models")
        or not cold.get("passed")
        or cold.get("acceptance_data_used") is not False
    ):
        raise ValueError(
            "A complete 55-query replay and actual development cold run are required"
        )
    manifest = load(index / "manifest.json")
    if (
        manifest["model_policy"]["policy_sha256"] != digest(load(policy))
        or cold.get("index_content_id") != manifest["content_id"]
        or cold.get("model_policy_sha256") != digest(manifest["model_policy"])
    ):
        raise ValueError("Index was generated with another policy")
    for name, expected in manifest["model_policy"]["assets_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("Actual model asset changed before freezing: " + name)
    for name, expected in manifest["model_policy"]["implementation_sha256"].items():
        path = PROJECT_ROOT / "src/multimodal_ecommerce" / (name + ".py")
        if sha256_file(path) != expected:
            raise ValueError("Runtime implementation changed after cold inference")
    files = {
        str(Path(name).resolve()): sha256_file(Path(name))
        for name in [
            str(policy),
            str(index / "manifest.json"),
            str(index / "vectors.npz"),
            evidence["replay"],
            evidence["cold"],
        ]
    }
    files.update(manifest["model_policy"]["assets_sha256"])
    for folder in (PROJECT_ROOT / "src",):
        files.update({str(p.resolve()): sha256_file(p) for p in folder.rglob("*.py")})
    entry = PROJECT_ROOT / "scripts/module2_selected_runtime.py"
    files[str(entry)] = sha256_file(entry)
    entry = PROJECT_ROOT / "scripts/module2_acceptance_gold.py"
    files[str(entry)] = sha256_file(entry)
    for relative in (
        "scripts/run.py",
        "scripts/replay_selected_runtime.py",
        "configs/embedding_model.json",
        "configs/reranker_model.json",
        "configs/retrieval_regions_v1.json",
        load(policy)["detector_manifest"],
    ):
        path = PROJECT_ROOT / relative
        files[str(path)] = sha256_file(path)
    target = PROJECT_ROOT / "configs/module2_target_p80_r90.json"
    files[str(target)] = sha256_file(target)
    release = dict(
        format="selected_engineering_release_v1",
        policy=load(policy),
        model_policy=manifest["model_policy"],
        files_sha256=files,
        development_index_content_id=manifest["content_id"],
        validated_replay_queries=55,
        actual_cold_run=cold,
        target=load(target),
        relevance_instruction=BUSINESS_INSTRUCTION,
        formal_metrics=None,
        formal_acceptance=False,
    )
    release["release_id"] = digest(release)
    if output.exists():
        if load(output) != release:
            raise ValueError("Preserve the original engineering release")
    else:
        atomic_json(output, release)
    return release


def reservation_role(holdout: Path) -> dict:
    """Accept only the current exact registered final reservation.

    Args:
        holdout: Technical reservation directory; only registry metadata are read.

    Returns:
        Exact registered lifecycle rule, including its consumed state.
    """
    registry = RuntimeAccess().registry
    relative = holdout.resolve().relative_to(PROJECT_ROOT).as_posix()
    rules = [r for r in registry["rules"] if r["path_prefix"] == relative + "/"]
    if (
        len(rules) != 1
        or registry.get("current_final_acceptance_root") != relative
        or "reserved" not in rules[0]["role"]
    ):
        raise PermissionError("Only the current registered final reservation is valid")
    return rules[0]


def acceptance_preflight(release_path: Path, holdout: Path, gold: Path) -> dict:
    """Read only seal/status metadata, never reserved images or answer CSVs.

    Args:
        release_path: Frozen running-code and model manifest.
        holdout: Unchanged technical reservation directory.
        gold: Separate private annotation directory; only its public status is read.

    Returns:
        Readiness and explicit reasons while inference remains closed.
    """
    release = verify_release(release_path)
    role = reservation_role(holdout)
    seal_path = holdout / "seal.json"
    seal = load(seal_path)
    reasons = []
    if role.get("consumed") is True:
        reasons.append("dataset_already_consumed")
    status_path = gold / "public_status.json"
    status = load(status_path) if status_path.exists() else {}
    allowed_status = {
        "release_id",
        "dataset_seal_sha256",
        "expected",
        "answered",
        "binary_complete",
        "all_answers_complete",
        "text_queries_complete",
        "packet_sha256",
        "formal_metrics",
        "unresolved",
        "gold_sha256",
    }
    if set(status) - allowed_status or status.get("formal_metrics") is not None:
        raise PermissionError(
            "Readiness may expose only public counts and freeze hashes"
        )
    if not status.get("text_queries_complete"):
        reasons.append("independent_text_queries_not_frozen")
    if not status.get("all_answers_complete"):
        reasons.append("independent_human_gold_incomplete")
    if status.get("all_answers_complete") and not status.get("gold_sha256"):
        reasons.append("independent_human_gold_not_frozen")
    if status and (
        status.get("release_id") != release["release_id"]
        or status.get("dataset_seal_sha256") != sha256_file(seal_path)
    ):
        reasons.append("gold_freeze_identity_mismatch")
    if (
        not seal.get("final_model_freeze_required")
        or seal.get("model_predictions")
        or seal.get("human_relevance_labels")
        or seal.get("AI_relevance_labels")
    ):
        reasons.append("reservation_contract_changed")
    return dict(
        ready=not reasons,
        reasons=reasons,
        release_id=release["release_id"],
        dataset_seal_sha256=sha256_file(seal_path),
        gallery=seal["gallery"],
        image_queries=seal["image_queries"],
        text_queries_pending=seal["independently_authored_text_queries_pending"],
        expected_relations=seal["gallery"]
        * (seal["image_queries"] + seal["independently_authored_text_queries_pending"]),
        gold_status=status,
        inference_started=False,
        formal_metrics=None,
    )
