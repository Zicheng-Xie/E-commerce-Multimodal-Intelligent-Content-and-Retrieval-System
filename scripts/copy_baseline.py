"""Run phase-two copy regressions, freeze a recipe and collect heldout outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.copy_experiment import (
    CopyExperimentRunCopyExperimentInputs,
    freeze_recipe,
    run_copy_experiment,
    score_copy_run,
)
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def main() -> None:
    """Expose immutable runs and scoring while keeping default role development.

    Raises:
        FileExistsError: Use a new score output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "freeze", "score"])
    parser.add_argument(
        "--role", choices=["development", "legacy", "heldout"], default="development"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/stage2_v1/release/products.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if args.command == "score":
        if not args.run_dir or not args.judgments:
            parser.error("score requires --run-dir and --judgments")
        if output.exists():
            raise FileExistsError("Use a new score output")
        report = score_copy_run(
            PROJECT_ROOT / args.run_dir, PROJECT_ROOT / args.judgments
        )
        write_json(output, report)
    else:
        client = OllamaClient()
        try:
            client.verify_model_storage(Path("E:/LLM_for_Ollama"))
            dataset = PROJECT_ROOT / args.dataset
            if args.command == "freeze":
                report = freeze_recipe(client, dataset, output)
            else:
                report = run_copy_experiment(
                    CopyExperimentRunCopyExperimentInputs(
                        client,
                        dataset,
                        output,
                        role=args.role,
                        per_category=args.per_category,
                        recipe_path=PROJECT_ROOT / args.recipe if args.recipe else None,
                    )
                )
        finally:
            client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
