"""Freeze all 18 retrieval queries, or score saved rankings without inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.frozen_retrieval import freeze_predictions, score_predictions
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Use a new score report path; Use a new prediction directory.

    Notes:
        Writes derived artifacts through write_json.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["predict", "score"])
    parser.add_argument(
        "--kit", type=Path, default=Path("artifacts/evaluation_stage2_v1")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/stage2_predictions_v1")
    )
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predictions = PROJECT_ROOT / args.predictions
    if args.command == "score":
        if not args.judgments or not args.output:
            parser.error("score requires --judgments and --output")
        output = ensure_output_path(PROJECT_ROOT / args.output)
        if output.exists():
            raise FileExistsError("Use a new score report path")
        report = score_predictions(predictions, PROJECT_ROOT / args.judgments)
        write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    ensure_output_path(predictions)
    if predictions.exists():
        raise FileExistsError("Use a new prediction directory")
    subprocess.run(
        ["ollama", "stop", "qwen3.5:4b-q4_K_M"], check=True, capture_output=True
    )
    embedder = QwenEmbedder()
    try:
        report = freeze_predictions(embedder, PROJECT_ROOT / args.kit, predictions)
    finally:
        embedder.close()
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("queries", "candidate_pairs", "quality_metrics")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
