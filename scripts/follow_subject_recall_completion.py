"""Read-only completion verification followed by serialized progress refresh."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "artifacts/module2_subject_recall_learning"
VERIFIER = "scripts/subject_recall_experiment.py"


def follow() -> None:
    """Return follow for this workflow.

    Raises:
        RuntimeError: Completion verification failed; preserved evidence in
            tmp/module2_subject_completion_verification_failure.json.
    """
    attempts = 0
    while True:
        state = json.loads(
            (EXPERIMENT / "public_status.json").read_text(encoding="utf-8")
        )
        if state["phase"] == "engineering_failure":
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/sync_subject_learning_progress.py"),
                ],
                cwd=ROOT,
                check=True,
            )
            print(
                dict(engineering_failure_preserved=True, formal_metrics=None),
                flush=True,
            )
            return
        if state["completed"] and (EXPERIMENT / "completion_receipt.json").is_file():
            # Completion status precedes sealing and process-lock release. A
            # verifier retry must wait for both without touching model work.
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / VERIFIER),
                    "verify",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/sync_subject_learning_progress.py"),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                print(
                    dict(seal_verified_and_progress_updated=True, formal_metrics=None),
                    flush=True,
                )
                return
            attempts += 1
            if attempts >= 6:
                record = dict(
                    verification_returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    formal_metrics=None,
                )
                path = ROOT / "tmp/module2_subject_completion_verification_failure.json"
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                raise RuntimeError(
                    "Completion verification failed; preserved evidence in tmp/module2_subject_completion_verification_failure.json"
                )
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise", action="store_true")
    args = parser.parse_args()
    VERIFIER = (
        "scripts/pointwise_recall_experiment.py"
        if args.pointwise
        else "scripts/subject_recall_experiment.py"
    )
    if args.pointwise:
        EXPERIMENT = ROOT / "artifacts/module2_pointwise_recall_learning"
    follow()
