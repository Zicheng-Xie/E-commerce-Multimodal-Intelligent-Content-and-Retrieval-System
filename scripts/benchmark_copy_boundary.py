"""Run explicit missing-attribute/contradiction probes without changing v3."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.content_contract import boundary_cases, generate_content
from multimodal_ecommerce.dataset_release import write_rows
from multimodal_ecommerce.evaluation import verify_files
from multimodal_ecommerce.measurement import wilson_interval
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    ensure_output_path,
    sha256_file,
    write_json,
)


def main() -> None:
    """Keep automatic contract checks separate from human quality percentages.

    Raises:
        FileExistsError: Use a new boundary run directory.
        ValueError: Model digest differs from frozen v3.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mentor_copy_boundary_v1")
    )
    args = parser.parse_args()
    output = ensure_output_path(PROJECT_ROOT / args.output)
    if output.exists():
        raise FileExistsError("Use a new boundary run directory")
    recipe = json.loads(
        (PROJECT_ROOT / "configs/copy_recipe_v3.json").read_text(encoding="utf-8")
    )
    verify_files(PROJECT_ROOT, recipe)
    cases = boundary_cases()
    output.mkdir(parents=True)
    write_rows(output / "cases.jsonl", cases)
    client = OllamaClient()
    try:
        storage = client.verify_model_storage(Path("E:/LLM_for_Ollama"))
        installed = next(
            row for row in storage["models"] if row["name"] == client.model
        )
        if installed["digest"] != recipe["model"]["digest"]:
            raise ValueError("Model digest differs from frozen v3")
        results = []
        for number, case in enumerate(cases, 1):
            result = generate_content(client, case["product"], case["style"])
            content = result.get("factual_copy")
            text = (
                "\n".join(
                    [
                        content["title"],
                        content["description"],
                        *content["selling_points"],
                    ]
                )
                if content
                else ""
            )
            # Mentions are review signals, not a semantic assertion/truth oracle.
            risks = []
            if case["scenario"] == "missing_material":
                risks = [
                    term
                    for term in (
                        "纯棉",
                        "全棉",
                        "棉质",
                        "塑料",
                        "实木",
                        "不锈钢",
                        "真皮",
                        "涤纶",
                    )
                    if term in text
                ]
            elif case["scenario"] == "missing_dimensions":
                risks = re.findall(
                    r"\d+(?:\.\d+)?\s*(?:厘米|毫米|cm|mm|英寸)", text, re.IGNORECASE
                )
            elif case["scenario"] == "missing_brand":
                risks = [
                    term for term in ("示例甲", "示例乙", "示例丙") if term in text
                ]
            result.update(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "scenario": case["scenario"],
                    "expected_behavior": case["expected_behavior"],
                    "missing_field_mention_review_signals": risks,
                }
            )
            results.append(result)
            write_json(output / f"case_{number:02d}.json", result)
            print(f"Boundary {number}/30: {result['status']}", flush=True)
    finally:
        client.close()
    write_rows(output / "results.jsonl", results)
    structural = [
        row for row in results if row["expected_behavior"] != "generation_attempt"
    ]
    generated = [row for row in results if row["factual_copy"]]
    report = {
        "cases": len(results),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "by_category": dict(Counter(row["category"] for row in results)),
        "by_style": dict(Counter(row["style"] for row in results)),
        "preflight_expected_cases": len(structural),
        "preflight_correct_abstentions": sum(
            row["status"] == row["expected_behavior"] for row in structural
        ),
        "generated": len(generated),
        "missing_field_mention_signal_cases": [
            row["id"] for row in results if row["missing_field_mention_review_signals"]
        ],
        "human_relevance_rate": None,
        "human_fluency_rate": None,
        "sample_size_illustration_assuming_80_percent": {
            str(n): wilson_interval(round(0.8 * n), n) for n in (30, 90, 300)
        },
        "scope": "30 synthetic boundary probes across 3 categories. Automatic rule signals and deterministic preflight checks are not independent human quality scores; no change to frozen v3 generator.",
    }
    write_json(output / "report.json", report)
    write_json(
        output / "manifest.json",
        {
            "files": {
                path.name: sha256_file(path)
                for path in output.iterdir()
                if path.is_file()
            }
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
