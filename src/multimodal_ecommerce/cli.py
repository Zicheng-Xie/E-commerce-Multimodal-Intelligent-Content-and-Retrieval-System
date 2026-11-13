"""Command-line entry points for reproducible phase-one experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from multimodal_ecommerce.batch import generate_batch
from multimodal_ecommerce.content_contract import generate_content
from multimodal_ecommerce.data import prepare_samples
from multimodal_ecommerce.media import localize_products
from multimodal_ecommerce.ollama_client import OllamaClient, OllamaError
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, write_json


def configuration() -> dict[str, Any]:
    """Read the checked-in local configuration.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    config = json.loads(
        (PROJECT_ROOT / "configs/local.json").read_text(encoding="utf-8")
    )
    if platform.system() == "Darwin":
        config["ollama_models_root"] = str(Path.home() / ".ollama/models")
    return config


def doctor(client: OllamaClient) -> dict[str, Any]:
    """Check imports, local API, GPU driver and disk space without model loading.

    Args:
        client: Configured local model client used for inference.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    packages = ("httpx", "ijson", "numpy", "pillow", "pytest")
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in packages},
        "ollama": client.health(),
    }
    report["disk"] = {"free_gib": shutil.disk_usage(PROJECT_ROOT).free / 2**30}
    import torch

    from multimodal_ecommerce.runtime_device import RuntimeDevice

    runtime = RuntimeDevice(torch)
    report["gpu"] = runtime.name
    report["gpu_query_passed"] = True
    write_json(PROJECT_ROOT / "artifacts/environment.json", report)
    return report


def smoke(client: OllamaClient, samples: Path) -> dict[str, Any]:
    """Load copied samples and run real text, structured-copy and vision calls.

    Args:
        client: Configured local model client used for inference.
        samples: Samples used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Sample image escapes the prepared dataset.
    """
    records = [
        json.loads(line)
        for line in (samples / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    visual = next(record for record in records if record["dataset"] == "deepfashion")
    product = next(record for record in records if record["dataset"] == "product")
    image_path = (samples / visual["image_path"]).resolve()
    if not image_path.is_relative_to(samples.resolve()):
        raise ValueError("Sample image escapes the prepared dataset.")
    with Image.open(image_path) as image:
        image.load()
        dimensions = image.size
    report: dict[str, Any] = {
        "model": client.model,
        "sample_count": len(records),
        "image_sample_id": visual["id"],
        "image_path": str(image_path),
        "image_dimensions": dimensions,
        "product_sample_id": product["id"],
        "scope": "Availability smoke test, not a formal quality evaluation",
    }
    report["text"] = client.chat("请仅用简体中文回答：本地模型已成功加载。")
    write_json(PROJECT_ROOT / "artifacts/smoke_text.json", report["text"])
    report["generation"] = client.generate_copy(
        {
            "name": product["text"],
            "category": product["category"],
            "attributes": product["attributes"],
        }
    )
    write_json(PROJECT_ROOT / "artifacts/smoke_copy.json", report["generation"])
    report["vision"] = client.chat(
        "请观察图片中的主要服饰，用简体中文描述类别、颜色、袖长、领型。"
        + "看不清写无法判断，没有袖子或领口则写不适用。"
        + "仅依据图片，不推测品牌、材质成分、价格或功效。",
        image_path=image_path,
    )
    write_json(PROJECT_ROOT / "artifacts/smoke_vision.json", report["vision"])
    process = subprocess.run(
        ["ollama", "ps"], capture_output=True, text=True, check=False
    )
    report["ollama_ps"] = process.stdout.strip()
    report["passed"] = True
    report["quality_review"] = "Manual fact checking required; no quality score claimed"
    write_json(PROJECT_ROOT / "artifacts/stage1_smoke.json", report)
    return report


def main() -> None:
    """Execute environment, storage, data preparation or inference commands.

    Raises:
        OllamaError: Model missing; download it first; Model missing; run
            scripts/pull_model.ps1 first.
        FileExistsError: Choose a new output path to preserve prior drafts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "verify-storage",
            "prepare-samples",
            "smoke",
            "generate",
            "download-images",
            "generate-batch",
        ],
    )
    parser.add_argument("--product", type=Path, help="UTF-8 product JSON for generate")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/generated_content.json")
    )
    parser.add_argument("--run-name", default="batch_v2_001")
    parser.add_argument("--per-category", type=int, default=2)
    args = parser.parse_args()
    config = configuration()
    source = PROJECT_ROOT / config["source_root"]
    samples = PROJECT_ROOT / config["samples_root"]
    client = OllamaClient(config["ollama_base_url"], config["generation_model"])
    try:
        if args.command == "doctor":
            result = doctor(client)
        elif args.command == "verify-storage":
            result = client.verify_model_storage(
                Path(config["ollama_models_root"]),
                reference_model=None if platform.system() == "Darwin" else "llama3:8b",
            )
            write_json(PROJECT_ROOT / "artifacts/model_storage.json", result)
        elif args.command == "prepare-samples":
            result = prepare_samples(source, samples)
        elif args.command == "download-images":
            result = localize_products(
                samples / "samples.jsonl", PROJECT_ROOT / "data/localized"
            )
        elif args.command == "generate-batch":
            if (
                not args.run_name
                or Path(args.run_name).name != args.run_name
                or args.run_name in {".", ".."}
            ):
                parser.error("--run-name must be a single directory name")
            storage = client.verify_model_storage(
                Path(config["ollama_models_root"]),
                reference_model=None if platform.system() == "Darwin" else "llama3:8b",
            )
            if not storage["model_installed"]:
                raise OllamaError("Model missing; download it first.")
            result = generate_batch(
                client,
                PROJECT_ROOT / "data/localized/products.jsonl",
                PROJECT_ROOT / "artifacts" / args.run_name,
                args.per_category,
            )
        elif args.command == "smoke":
            storage = client.verify_model_storage(
                Path(config["ollama_models_root"]),
                reference_model=None if platform.system() == "Darwin" else "llama3:8b",
            )
            if not storage["model_installed"]:
                raise OllamaError("Model missing; run scripts/pull_model.ps1 first.")
            result = smoke(client, samples)
        else:
            if args.product is None:
                parser.error("generate requires --product")
            output = ensure_output_path(PROJECT_ROOT / args.output)
            if output.exists():
                raise FileExistsError(
                    "Choose a new output path to preserve prior drafts"
                )
            product = json.loads(args.product.read_text(encoding="utf-8-sig"))
            result = generate_content(client, product)
            write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        client.close()
