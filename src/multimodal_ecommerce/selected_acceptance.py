"""One-shot scoped acceptance coordinator; private truth is read after prediction seals."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from multimodal_ecommerce import platform_lock
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file
from multimodal_ecommerce.selected_evaluation import evaluate_sealed_predictions
from multimodal_ecommerce.selected_release import acceptance_preflight, verify_release
from multimodal_ecommerce.selected_runtime_io import atomic_json, digest, load


@contextmanager
def _lock(folder: Path) -> Iterator[None]:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        platform_lock.locking(handle.fileno(), platform_lock.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            platform_lock.locking(handle.fileno(), platform_lock.LK_UNLCK, 1)


def acceptance_run(
    release_path: Path, holdout: Path, gold_root: Path, worker: Any
) -> dict:
    """Claim one dataset once; resume only the exact frozen job and seal before scoring.

    Args:
        release_path: Frozen engineering release.
        holdout: Technical reservation, unchanged throughout the run.
        gold_root: Separate private annotation freeze.
        worker: Launches an isolated index/prediction worker using only its claim.

    Returns:
        Final immutable receipt, or a failed running claim available for exact resume.
    """
    release = verify_release(release_path)
    seal_path = holdout / "seal.json"
    seal_sha = sha256_file(seal_path)
    folder = ensure_output_path(
        PROJECT_ROOT / "artifacts/module2_acceptance_runs" / seal_sha[:24]
    )
    with _lock(folder):
        status = load(gold_root / "public_status.json")
        binding = dict(
            release_id=release["release_id"],
            dataset_seal_sha256=seal_sha,
            gold_sha256=status.get("gold_sha256"),
            packet_sha256=status.get("packet_sha256"),
        )
        claim_path = folder / "claim.json"
        if claim_path.exists():
            claim = load(claim_path)
            if claim["binding"] != binding:
                raise PermissionError(
                    "Consumed dataset cannot run with another model or gold"
                )
            if claim["state"] == "complete":
                receipt = load(folder / "receipt.json")
                if (
                    sha256_file(folder / "metrics.json") != receipt["metrics_sha256"]
                    or sha256_file(
                        Path(claim["work_root"]) / "predictions/predictions.json"
                    )
                    != receipt["predictions_sha256"]
                    or sha256_file(gold_root / "gold.private.json")
                    != binding["gold_sha256"]
                ):
                    raise ValueError("Completed acceptance receipt was altered")
                return receipt
        else:
            readiness = acceptance_preflight(release_path, holdout, gold_root)
            if not readiness["ready"]:
                raise PermissionError(
                    "Acceptance not ready: " + ", ".join(readiness["reasons"])
                )
            if (
                sha256_file(gold_root / "gold.private.json") != binding["gold_sha256"]
                or sha256_file(gold_root / "packet.private.json")
                != binding["packet_sha256"]
            ):
                raise ValueError("Independent gold freeze changed")
            seal = load(seal_path)
            for name, expected in seal["files"].items():
                if sha256_file(holdout / name) != expected:
                    raise ValueError("Reserved data changed before first inference")
            packet = load(gold_root / "packet.private.json")
            work = folder / "worker"
            inputs = work / "public_inputs"
            atomic_json(inputs / "catalog.json", packet["catalog"])
            atomic_json(inputs / "queries.json", packet["queries"])
            atomic_json(inputs / "policy.json", release["policy"])
            authorized = {
                str((holdout / name).resolve()): value
                for name, value in seal["files"].items()
            }
            claim = dict(
                state="running",
                binding=binding,
                token=secrets.token_hex(32),
                work_root=str(work),
                private_gold_root=str(gold_root.resolve()),
                authorized_files=authorized,
                model_policy_sha256=digest(release["model_policy"]),
                inference_attempts=1,
            )
            atomic_json(claim_path, claim)
            claim_path.chmod(0o600)
        if (
            sha256_file(gold_root / "gold.private.json") != binding["gold_sha256"]
            or sha256_file(gold_root / "packet.private.json")
            != binding["packet_sha256"]
        ):
            raise ValueError("Exact acceptance resume requires unchanged human gold")
        registry_path = PROJECT_ROOT / "configs/evaluation_registry.json"
        registry = load(registry_path)
        relative = holdout.resolve().relative_to(PROJECT_ROOT).as_posix() + "/"
        rules = [r for r in registry["rules"] if r["path_prefix"] == relative]
        if len(rules) != 1:
            raise ValueError("One exact reservation rule is required")
        rules[0].update(
            consumed=True,
            development_or_training_use_allowed=False,
            tuning_use_allowed=False,
            model_inference_allowed=False,
            formal_acceptance_ready=False,
        )
        atomic_json(registry_path, registry)
        atomic_json(
            folder / "started.json",
            dict(
                binding=binding, consumed_before_inference=True, exact_resume_only=True
            ),
        )
        # The worker receives no private gold filename or label values.
        worker(
            dict(claim=claim_path, token=claim["token"], work=Path(claim["work_root"]))
        )
        prediction_path = Path(claim["work_root"]) / "predictions/predictions.json"
        predictions = load(prediction_path)
        checksum = predictions.pop("result_sha256")
        if (
            digest(predictions) != checksum
            or digest(predictions["model_policy"]) != claim["model_policy_sha256"]
        ):
            raise ValueError("Predictions or runtime model changed")
        predictions["result_sha256"] = checksum
        atomic_json(
            folder / "prediction_receipt.json",
            dict(
                binding=binding,
                predictions_sha256=sha256_file(prediction_path),
                private_answers_not_loaded_before_this_receipt=True,
            ),
        )
        # Independent truth is opened only after complete predictions are immutable.
        if (
            sha256_file(gold_root / "gold.private.json") != binding["gold_sha256"]
            or sha256_file(gold_root / "packet.private.json")
            != binding["packet_sha256"]
        ):
            raise ValueError("Independent gold changed during model inference")
        packet = load(gold_root / "packet.private.json")
        gold = load(gold_root / "gold.private.json")
        if gold["binding"] != {
            k: binding[k] for k in ("release_id", "dataset_seal_sha256")
        }:
            raise ValueError("Scorer gold binding differs from the run")
        for name, expected in gold["answer_csv_sha256"].items():
            if sha256_file(Path(name)) != expected:
                raise ValueError("Independent answer source changed")
        metrics = evaluate_sealed_predictions(
            predictions, packet, gold, release["target"]
        )
        atomic_json(folder / "metrics.json", metrics)
        receipt = dict(
            binding=binding,
            state="complete",
            predictions_sha256=sha256_file(prediction_path),
            metrics_sha256=sha256_file(folder / "metrics.json"),
            dataset_consumed=True,
            goal_met=metrics["goal_met"],
            immutable_after_completion=True,
        )
        atomic_json(folder / "receipt.json", receipt)
        claim["state"] = "complete"
        atomic_json(claim_path, claim)
        return receipt
