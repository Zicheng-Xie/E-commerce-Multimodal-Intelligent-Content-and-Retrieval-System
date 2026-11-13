"""Read candidate-pool originals with the existing local Apple Vision helper."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import run  # noqa: F401
from module2_poster_text_replay import EXPANDED, SIMPLE, SWIFT, TARGET, load
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = TARGET / "candidate_ocr"
BATCH = 32


def candidate_inputs(registry: dict) -> tuple[list[dict], dict]:
    """Verify provenance and deduplicate the actual candidate image inputs.

    Args:
        registry: Active data-role registry.

    Returns:
        Ordered original-image inputs and their immutable protocol.
    """
    source = SIMPLE / "inputs.json"
    protocol = EXPANDED / "sid_protocol.json"
    summary = EXPANDED / "summary.json"
    require_data_access([source, protocol, summary], "development", registry)
    if sha256_file(protocol) != load(summary)["files"]["sid_protocol.json"]:
        raise ValueError("The previous scoring protocol changed")
    prior = load(protocol)
    if sha256_file(source) != prior["files"][str(source.relative_to(PROJECT_ROOT))]:
        raise ValueError("Original document inputs changed")
    documents = load(source)["documents"]
    ids = sorted({cid for job in prior["jobs"] for cid in job["head"] + job["extra"]})
    items = [documents[cid] for cid in ids]
    images = [Path(item["image_path"]) for item in items]
    require_data_access(images, "development", registry)
    for image, item in zip(images, items, strict=True):
        if sha256_file(image) != item["source_sha256"]:
            raise ValueError("A candidate original changed after feature extraction")
    return items, dict(
        files={
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (source, protocol, summary, SWIFT, Path(__file__).resolve())
        },
        items=[
            dict(id=item["id"], source_sha256=item["source_sha256"]) for item in items
        ],
        batch_size=BATCH,
        role="exposed_development",
        formal_metrics=None,
    )


def recognize(items: list[dict], binary: Path, digest: str) -> dict:
    """Read and checkpoint bounded image batches, retaining explicit failures.

    Args:
        items: Verified candidate image inputs.
        binary: Compiled local text-recognition adapter.
        digest: Immutable candidate OCR protocol checksum.

    Returns:
        OCR results keyed by candidate identity.
    """
    output = {}
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        ids = [item["id"] for item in batch]
        cache = ROOT / f"batch_{start // BATCH + 1:03d}.json"
        if cache.exists():
            saved = load(cache)
            if saved["protocol_sha256"] != digest or list(saved["results"]) != ids:
                raise ValueError(
                    "Candidate OCR batch cache differs from its frozen inputs"
                )
            output.update(saved["results"])
            continue
        result = subprocess.run(
            [str(binary)],
            input=json.dumps([item["image_path"] for item in batch]),
            text=True,
            capture_output=True,
            check=True,
        )
        observations = json.loads(result.stdout)
        values = dict(zip(ids, observations, strict=True))
        write_json(cache, dict(protocol_sha256=digest, results=values))
        output.update(values)
        print(f"Candidate OCR {len(output)}/{len(items)}", flush=True)
    return output


def main() -> None:
    """Run local Vision on the previously scored development pool."""
    from module2_expand_pool import WORKER
    from multimodal_ecommerce.module_loading import load_project_module

    ROOT.mkdir(parents=True, exist_ok=True)
    worker = load_project_module("candidate_ocr_worker_lock", WORKER)
    worker.ROOT = ROOT
    with worker.worker_lock():
        registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
        items, protocol = candidate_inputs(registry)
        path = ROOT / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("Frozen candidate OCR protocol changed")
        write_json(path, protocol)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            binary = directory / "candidate-ocr"
            subprocess.run(
                [
                    "swiftc",
                    "-module-cache-path",
                    str(directory / "cache"),
                    "-o",
                    str(binary),
                    str(SWIFT),
                ],
                check=True,
            )
            observations = recognize(items, binary, sha256_file(path))
        errors = sum(bool(row.get("error")) for row in observations.values())
        if errors == len(observations):
            raise RuntimeError("Apple Vision failed on all candidate inputs")
        write_json(ROOT / "ocr.json", observations)
        write_json(
            ROOT / "receipt.json",
            dict(
                attempted=len(items),
                succeeded=len(items) - errors,
                errors=errors,
                with_detected_text=sum(
                    bool(row["observations"]) for row in observations.values()
                ),
                files={p.name: sha256_file(p) for p in (path, ROOT / "ocr.json")},
                formal_metrics=None,
            ),
        )


if __name__ == "__main__":
    main()
