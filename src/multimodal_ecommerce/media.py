"""Localize small product image sets with validation and resumable manifests."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from PIL import Image, UnidentifiedImageError

from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def image_url(value: str) -> str:
    """Allow only HTTPS Alibaba public image hosts, upgrading source HTTP URLs.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (
        parsed.scheme not in {"http", "https"}
        or not (host == "alicdn.com" or host.endswith(".alicdn.com"))
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 80, 443}
    ):
        raise ValueError(f"Unsupported image URL: {value}")
    return urlunsplit(("https", host, parsed.path, parsed.query, ""))


def fetch_image(client: httpx.Client, url: str) -> tuple[bytes, str]:
    """Fetch bounded image bytes and validate every redirect destination.

    Args:
        client: Configured local model client used for inference.
        url: Url used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Image exceeds 20 MiB download limit; Too many image redirects.
    """
    current = image_url(url)
    for _ in range(5):
        with client.stream("GET", current, follow_redirects=False) as response:
            if response.is_redirect:
                current = image_url(urljoin(current, response.headers["location"]))
                continue
            response.raise_for_status()
            length = int(response.headers.get("content-length", "0"))
            if length > MAX_BYTES:
                raise ValueError("Image exceeds 20 MiB download limit")
            buffer = bytearray()
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > MAX_BYTES:
                    raise ValueError("Image exceeds 20 MiB download limit")
            return bytes(buffer), current
    raise ValueError("Too many image redirects")


def inspect_image(content: bytes) -> dict[str, Any]:
    """Decode image bytes and compute byte-level and exact pixel fingerprints.

    Args:
        content: Content used by this operation.

    Returns:
        Mapping with fields: width, height, suffix, sha256, pixel_sha256, bytes.

    Raises:
        ValueError: Image dimensions outside allowed limits; Animated images require a
            separate preprocessing policy.
    """
    with Image.open(io.BytesIO(content)) as picture:
        width, height = picture.size
        if width * height > MAX_PIXELS or min(width, height) < 16:
            raise ValueError("Image dimensions outside allowed limits")
        if getattr(picture, "n_frames", 1) != 1:
            raise ValueError("Animated images require a separate preprocessing policy")
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
            picture.format or ""
        )
        if suffix is None:
            raise ValueError(f"Unsupported image format: {picture.format}")
        picture.load()
        pixels = picture.convert("RGB").tobytes()
        digest = hashlib.sha256(f"{width}x{height}:".encode() + pixels).hexdigest()
        return {
            "width": width,
            "height": height,
            "suffix": suffix,
            "sha256": hashlib.sha256(content).hexdigest(),
            "pixel_sha256": digest,
            "bytes": len(content),
        }


def localize_products(
    samples: Path, output: Path, client: httpx.Client | None = None
) -> dict[str, Any]:
    """Create a product-only localized dataset, retaining every failure and split.

    Args:
        samples: Prepared JSONL input, read-only for this operation.
        output: Project-owned directory for downloads, records and manifests.
        client: Optional HTTP transport for testing.

    Returns:
        Counts, failure details and exact-duplicate leakage audit.

    Raises:
        ValueError: Duplicate record IDs in input; Cached path escapes download
            directory.
    """
    output = ensure_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ensure_output_path(output / "media_manifest.json")
    prior = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    cache = prior.get("entries", {})
    records = [
        json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()
    ]
    products = [dict(record) for record in records if record["dataset"] == "product"]
    if len({record["id"] for record in products}) != len(products):
        raise ValueError("Duplicate record IDs in input")
    owned_client = client is None
    client = client or httpx.Client(timeout=20, trust_env=False)
    entries: dict[str, Any] = {}
    failures = []
    try:
        for progress, record in enumerate(products, 1):
            identifier = record["id"]
            record["image_path"] = None
            original_url = record.get("image_url")
            try:
                url = image_url(original_url or "")
                previous = cache.get(identifier, {})
                cached_path = ensure_output_path(
                    output / previous.get("path", "missing")
                )
                if not cached_path.is_relative_to(output):
                    raise ValueError("Cached path escapes download directory")
                if (
                    previous.get("url") == url
                    and cached_path.is_file()
                    and sha256_file(cached_path) == previous.get("sha256")
                ):
                    content = cached_path.read_bytes()
                    details = inspect_image(content)
                    entry = {**previous, **details, "cache_hit": True}
                else:
                    for attempt in range(2):
                        try:
                            content, final_url = fetch_image(client, url)
                            break
                        except httpx.TransportError:
                            if attempt == 1:
                                raise
                            time.sleep(0.5)
                    details = inspect_image(content)
                    name = hashlib.sha256(identifier.encode()).hexdigest()[:24]
                    relative = f"images/{name}{details['suffix']}"
                    target = ensure_output_path(output / relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = ensure_output_path(
                        target.with_suffix(target.suffix + ".part")
                    )
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    entry = {
                        **details,
                        "path": relative,
                        "url": url,
                        "source_url": original_url,
                        "final_url": final_url,
                        "cache_hit": False,
                    }
                entries[identifier] = entry
                record.update(
                    {
                        "image_path": entry["path"],
                        "media_status": "downloaded",
                        "image_sha256": entry["sha256"],
                        "pixel_sha256": entry["pixel_sha256"],
                        "width": entry["width"],
                        "height": entry["height"],
                    }
                )
            except (
                httpx.HTTPError,
                ValueError,
                OSError,
                UnidentifiedImageError,
            ) as error:
                logging.getLogger(__name__).debug(
                    "Handled exception in localize_products", exc_info=True
                )
                record["media_status"] = "download_failed"
                record["media_error"] = str(error)
                failures.append(
                    {"id": identifier, "url": original_url, "error": str(error)}
                )
            # Persist completed downloads so an interrupted run can safely resume.
            write_json(
                manifest_path, {"schema_version": 1, "entries": {**cache, **entries}}
            )
            if progress % 25 == 0 or progress == len(products):
                print(
                    f"Images {progress}/{len(products)}: {len(entries)} valid, {len(failures)} failed",
                    flush=True,
                )
    finally:
        if owned_client:
            client.close()
    groups: dict[str, list[dict]] = defaultdict(list)
    source_ids: dict[str, list[dict]] = defaultdict(list)
    for record in products:
        source_ids[record["source_id"]].append(record)
        if record["image_path"]:
            groups[record["pixel_sha256"]].append(record)
    duplicates = []
    excluded: set[str] = set()
    for kind, group_map in (("exact_pixels", groups), ("source_id", source_ids)):
        for fingerprint, group in group_map.items():
            if len(group) < 2:
                continue
            cross_split = len({record["split"] for record in group}) > 1
            duplicates.append(
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "ids": [record["id"] for record in group],
                    "cross_split": cross_split,
                }
            )
            if cross_split:
                excluded.update(
                    record["id"] for record in group if record["split"] == "test"
                )
    for record in products:
        record["evaluation_eligible"] = (
            bool(record["image_path"]) and record["id"] not in excluded
        )
    target = ensure_output_path(output / "products.jsonl")
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in products),
        encoding="utf-8",
    )
    summary = {
        "total": len(products),
        "downloaded": len(entries),
        "failures": failures,
        "download_bytes": sum(entry["bytes"] for entry in entries.values()),
        "cache_hits": sum(entry["cache_hit"] for entry in entries.values()),
        "downloaded_by_split_category": dict(
            Counter(
                f"{r['split']}/{r['category']}" for r in products if r["image_path"]
            )
        ),
        "duplicates": duplicates,
        "excluded_test_ids": sorted(excluded),
        "audit_limit": "Exact pixels and source IDs only; near-duplicates and semantic mismatches need review.",
    }
    write_json(output / "media_summary.json", summary)
    write_json(manifest_path, {"schema_version": 1, "entries": entries})
    return summary
