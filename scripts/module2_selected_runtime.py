"""Index, predict, freeze and prepare one-shot evaluation for the selected runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.selected_acceptance import acceptance_run
from multimodal_ecommerce.selected_engine import SelectedEngine
from multimodal_ecommerce.selected_release import acceptance_preflight, freeze_release
from multimodal_ecommerce.selected_runtime_io import RuntimeAccess, load

DEFAULT = PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110"


def _worker(job: dict) -> None:
    base = [sys.executable, str(Path(__file__).resolve())]
    work = job["work"]
    options = [
        "--policy",
        str(work / "public_inputs/policy.json"),
        "--cache",
        str(work / "cache"),
        "--run-ticket",
        str(job["claim"]),
        "--run-token",
        job["token"],
    ]
    subprocess.run(
        base
        + [
            "index",
            "--catalog",
            str(work / "public_inputs/catalog.json"),
            "--output",
            str(work / "index"),
        ]
        + options,
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        base
        + [
            "predict",
            "--index",
            str(work / "index"),
            "--queries",
            str(work / "public_inputs/queries.json"),
            "--output",
            str(work / "predictions"),
        ]
        + options,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    """Run public inference or verify readiness without automatically opening acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("index", "predict"):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "--policy",
            type=Path,
            default=PROJECT_ROOT / "configs/module2_selected_runtime.json",
        )
        cmd.add_argument("--cache", type=Path, default=DEFAULT / "cache")
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--run-ticket", type=Path, help=argparse.SUPPRESS)
        cmd.add_argument("--run-token", help=argparse.SUPPRESS)
        if name == "index":
            cmd.add_argument("--catalog", type=Path, required=True)
        else:
            cmd.add_argument("--index", type=Path, required=True)
            cmd.add_argument("--queries", type=Path, required=True)
    cmd = sub.add_parser("freeze")
    for name in ("index", "replay", "cold", "output"):
        cmd.add_argument("--" + name, type=Path, required=True)
    cmd.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "configs/module2_selected_runtime.json",
    )
    for name in ("acceptance-preflight", "acceptance-run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--release", type=Path, required=True)
        cmd.add_argument(
            "--holdout",
            type=Path,
            default=PROJECT_ROOT / "artifacts/module2_final_acceptance_20261110",
        )
        cmd.add_argument(
            "--gold",
            type=Path,
            default=PROJECT_ROOT / "artifacts/module2_acceptance_gold_20261110",
        )
        if name == "acceptance-run":
            cmd.add_argument("--execute-one-shot", action="store_true", required=True)
    args = parser.parse_args()
    if args.action in ("index", "predict"):
        access = RuntimeAccess(args.run_ticket, args.run_token)
        access.check(args.policy)
        for folder in (args.output, args.cache):
            access.check(folder / "runtime_output_guard")
        engine = SelectedEngine(load(args.policy), args.cache, access=access)
        try:
            result = (
                engine.index(args.catalog, args.output)
                if args.action == "index"
                else engine.predict(args.index, args.queries, args.output)
            )
        finally:
            engine.models.close()
        summary = dict(
            action=args.action,
            output=str(args.output),
            identity=result.get("content_id", result.get("batch_id")),
            runtime=result.get("runtime"),
            formal_metrics=None,
        )
    elif args.action == "freeze":
        result = freeze_release(
            args.policy,
            args.index,
            dict(replay=str(args.replay), cold=str(args.cold)),
            args.output,
        )
        summary = dict(
            release_id=result["release_id"],
            output=str(args.output),
            formal_metrics=None,
        )
    elif args.action == "acceptance-preflight":
        summary = acceptance_preflight(args.release, args.holdout, args.gold)
    else:
        summary = acceptance_run(args.release, args.holdout, args.gold, _worker)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
