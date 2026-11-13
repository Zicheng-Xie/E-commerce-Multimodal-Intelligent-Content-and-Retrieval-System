"""Download a pinned official checkpoint into the existing external model root."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download a pinned official checkpoint into the existing external model root."""
    META = json.loads(
        (ROOT / "artifacts/module2_precision/model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    TARGET = Path("E:/LLM_for_Ollama/external/Qwen3-VL-Reranker-2B").resolve()
    TARGET.mkdir(parents=True, exist_ok=True)

    def checksum(path: Path, kind: str = "sha256") -> str:
        """Return checksum for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.
            kind: Kind used by this operation.

        Returns:
            Text produced by this operation.
        """
        digest = hashlib.new(kind)
        with path.open("rb") as handle:
            if kind == "sha1":
                digest.update(f"blob {path.stat().st_size}\0".encode())
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    selected = [
        x
        for x in META["files"]
        if Path(x["rfilename"]).suffix in {".json", ".txt", ".jinja", ".safetensors"}
    ]
    selected.sort(key=lambda x: x["rfilename"].endswith(".safetensors"))
    if shutil.disk_usage(TARGET).free < sum(x["size"] for x in selected) + 2 * 1024**3:
        raise RuntimeError("Insufficient model disk space")
    manifest = {}
    for info in selected:
        name = info["rfilename"]
        path = (TARGET / name).resolve()
        if not path.is_relative_to(TARGET):
            raise ValueError("Escaped checkpoint path")
        path.parent.mkdir(parents=True, exist_ok=True)
        kind = "sha256" if info.get("lfs") else "sha1"
        expected = info["lfs"]["sha256"] if info.get("lfs") else info["blobId"]
        if path.exists():
            if path.stat().st_size != info["size"] or checksum(path, kind) != expected:
                raise ValueError("Existing checkpoint file differs; refusing overwrite")
        else:
            partial = path.with_suffix(path.suffix + ".download")
            url = f"https://huggingface.co/{META['repository']}/resolve/{META['revision']}/{urllib.parse.quote(name, safe='/')}?download=true"
            count = 0
            last = 0
            request = urllib.request.Request(
                url, headers={"User-Agent": "module2-local-evaluation"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as handle,
            ):
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
                    count += len(block)
                    if count - last >= 128 * 1024**2:
                        print(
                            f'{name}: {count//1024**2}/{info["size"]//1024**2} MiB',
                            flush=True,
                        )
                        last = count
            if (
                partial.stat().st_size != info["size"]
                or checksum(partial, kind) != expected
            ):
                raise ValueError("Checkpoint download integrity mismatch")
            partial.replace(path)
            print(f"Verified {name}", flush=True)
        manifest[name] = checksum(path)
    configuration = {
        "repository": META["repository"],
        "revision": META["revision"],
        "code_revision": META["code_revision"],
        "path": str(TARGET),
        "files": manifest,
        "vendor_sha256": {
            p.name: checksum(p)
            for p in (ROOT / "vendor/qwen3_vl_reranker").iterdir()
            if p.is_file()
        },
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "max_length": 10240,
        "min_pixels": 4096,
        "max_pixels": 1843200,
        "batch_size": 1,
        "instruction": "Retrieve images of the exact same clothing item as the query. Match garment type, color, pattern, cut and design details. Ignore the person, pose and background. A similar-looking but different item is not an exact match.",
    }
    (ROOT / "configs/reranker_model.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Pinned reranker checkpoint verified and configuration saved", flush=True)


if __name__ == "__main__":
    main()
