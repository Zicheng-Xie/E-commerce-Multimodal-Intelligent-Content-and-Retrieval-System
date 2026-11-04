"""Deterministic product sampling and immutable, leakage-aware data releases."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

import ijson
from PIL import Image, ImageOps

from multimodal_ecommerce.data import PRODUCT_RULES, classify_product, parse_pv
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def read_rows(path: Path) -> list[dict]:
    """Read UTF-8 JSONL records, ignoring blank lines.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write records only inside a permitted project output path.

    Args:
        path: Path of the input or output artifact used by this operation.
        rows: Records processed by this operation.
    """
    path = ensure_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def order_key(identifier: str, seed: int) -> str:
    """Assign a stable seeded hash priority independent of iteration order.

    Args:
        identifier: Identifier used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Returns:
        Text produced by this operation.
    """
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


@dataclass(frozen=True)
class DatasetReleaseSampleCandidatesInputs:
    """Input values for sample candidates.

    Attributes:
        source: Source used by this operation.
        prior: Prior used by this operation.
        output: Output used by this operation.
        per_group: Per group used by this operation.
        scan_limit: Scan limit used by this operation.
        seed: Seed used to reproduce randomized ordering.
    """

    source: Path
    prior: Path
    output: Path
    _: KW_ONLY
    per_group: int = 100
    scan_limit: int = 50000
    seed: int = 20260923


def sample_candidates(options: DatasetReleaseSampleCandidatesInputs) -> dict:
    """Sample within bounded source prefixes, excluding all old product IDs.

    This is hash sampling of eligible records in each scanned prefix, not a
    representative sample of the entire source dataset. Original splits remain.

    Args:
        options: Validated input fields described by
            DatasetReleaseSampleCandidatesInputs.

    Returns:
        Mapping with fields: seed, per_group, scan_limit_per_split, sampling,
        excluded_prior_ids, prior_sha256, sources, counts, samples_sha256,
        category_rules, limitation.

    Raises:
        ValueError: Counts must be positive.
        FileExistsError: Candidate releases must use a new directory.
    """
    source = options.source
    prior = options.prior
    output = options.output
    per_group = options.per_group
    scan_limit = options.scan_limit
    seed = options.seed
    if per_group < 1 or scan_limit < 1:
        raise ValueError("Counts must be positive")
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Candidate releases must use a new directory")
    known = {
        str(row["source_id"]) for row in read_rows(prior) if row["dataset"] == "product"
    }
    selected: list[dict] = []
    sources = []
    for split in ("train", "test"):
        path = source / "多模态电商数据" / f"product1m_product5m_{split}_id_label.json"
        heaps: dict[str, list] = defaultdict(list)
        counts: Counter = Counter()
        scanned = 0
        with path.open("rb") as handle:
            for source_id, item in ijson.kvitems(handle, ""):
                scanned += 1
                matched = classify_product(item)
                if str(source_id) in known:
                    counts["previously_used"] += 1
                elif (
                    matched
                    and item.get("title")
                    and item.get("url")
                    and parse_pv(item.get("pv") or "")
                ):
                    category, rule = matched
                    counts[category] += 1
                    identifier = f"product/{split}/{source_id}"
                    priority = order_key(identifier, seed)
                    row = {
                        "id": identifier,
                        "source_id": str(source_id),
                        "dataset": "product",
                        "split": split,
                        "source_path": str(path.resolve()),
                        "source_position": scanned,
                        "image_url": item["url"],
                        "image_path": None,
                        "media_status": "pending_download",
                        "text": item["title"],
                        "label": item.get("label", ""),
                        "category": category,
                        "category_rule": rule,
                        "attributes": parse_pv(item.get("pv") or ""),
                        "pv_raw": item.get("pv", ""),
                        "sampling_priority": priority,
                        "provenance": {
                            "source_split": split,
                            "source_record_sha256": hashlib.sha256(
                                json.dumps(
                                    item,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    default=str,
                                ).encode()
                            ).hexdigest(),
                        },
                    }
                    entry = (-int(priority, 16), identifier, row)
                    heap = heaps[category]
                    if len(heap) < per_group:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
                else:
                    counts["unmapped_or_missing_required"] += 1
                if scanned >= scan_limit:
                    break
        selected.extend(entry[2] for heap in heaps.values() for entry in heap)
        stat = path.stat()
        sources.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "scanned": scanned,
                "eligible_counts": dict(counts),
            }
        )
    selected.sort(
        key=lambda row: (row["split"], row["category"], row["sampling_priority"])
    )
    write_rows(output / "samples.jsonl", selected)
    report = {
        "seed": seed,
        "per_group": per_group,
        "scan_limit_per_split": scan_limit,
        "sampling": "lowest seeded SHA256 priorities within scanned prefix, original split/category strata",
        "excluded_prior_ids": len(known),
        "prior_sha256": sha256_file(prior),
        "sources": sources,
        "counts": dict(Counter(f"{r['split']}/{r['category']}" for r in selected)),
        "samples_sha256": sha256_file(output / "samples.jsonl"),
        "category_rules": PRODUCT_RULES,
        "limitation": "Heuristic category mapping, prefix scope and subsequent URL availability cause selection bias.",
    }
    write_json(output / "sampling.json", report)
    return report


def normalized_title(text: str) -> str:
    """Normalize compatibility characters, spacing and punctuation for dedup.

    Args:
        text: Text input used by the operation.

    Returns:
        Text produced by this operation.
    """
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def difference_hash(path: Path) -> int:
    """Compute a 64-bit horizontal dHash; similarity is only a risk signal.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Count or index produced by this operation.
    """
    with Image.open(path) as image:
        small = (
            ImageOps.exif_transpose(image)
            .convert("L")
            .resize((9, 8), Image.Resampling.LANCZOS)
        )
        pixels = [int(value) for value in small.tobytes()]
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def duplicate_groups(
    rows: list[dict], hashes: dict[str, int], threshold: int = 4
) -> tuple[list[list[str]], list[dict]]:
    """Union ID/title/pixel matches and conservative near-image candidates.

    Args:
        rows: Records processed by this operation.
        hashes: Hashes used by this operation.
        threshold: Threshold used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Duplicate record IDs.
    """
    parent = {row["id"]: row["id"] for row in rows}
    if len(parent) != len(rows):
        raise ValueError("Duplicate record IDs")

    def root(key: str) -> str:
        """Return root for this workflow.

        Args:
            key: Key used by this operation.

        Returns:
            Text produced by this operation.
        """
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    edges: list[dict] = []

    def join(a: str, b: str, kind: str, distance: int | None = None) -> None:
        """Apply the join workflow step.

        Args:
            a: A used by this operation.
            b: B used by this operation.
            kind: Kind used by this operation.
            distance: Distance used by this operation.
        """
        parent[root(b)] = root(a)
        edges.append({"a": a, "b": b, "kind": kind, "distance": distance})

    for field in ("source_id", "pixel_sha256", "normalized_title"):
        seen: dict[str, str] = {}
        for row in rows:
            key = row.get(field)
            if key:
                if key in seen:
                    join(seen[key], row["id"], field)
                else:
                    seen[key] = row["id"]
    items = list(hashes.items())
    for index, (a, ahash) in enumerate(items):
        for b, bhash in items[index + 1 :]:
            distance = (ahash ^ bhash).bit_count()
            if distance <= threshold:
                join(a, b, "dhash_candidate", distance)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in parent:
        groups[root(key)].append(key)
    return list(groups.values()), edges


@dataclass(frozen=True)
class DatasetReleaseFreezeReleaseInputs:
    """Input values for freeze release.

    Attributes:
        localized: Localized used by this operation.
        prior_samples: Prior samples used by this operation.
        prior_localized: Prior localized used by this operation.
        output: Output used by this operation.
        per_group: Per group used by this operation.
        seed: Seed used to reproduce randomized ordering.
    """

    localized: Path
    prior_samples: Path
    prior_localized: Path
    output: Path
    _: KW_ONLY
    per_group: int = 50
    seed: int = 20260923


def freeze_release(options: DatasetReleaseFreezeReleaseInputs) -> dict[str, Any]:
    """Freeze copied media, conservative dedup decisions and split manifests.

    Args:
        options: Validated input fields described by DatasetReleaseFreezeReleaseInputs.

    Returns:
        Mapping with fields: schema_version, seed, candidate_count,
        downloaded_candidates, selected, counts, per_split_category_target, shortfalls,
        roles, decisions, quality_flags, duplicate_edges, prior_records_excluded,
        inputs, scope.

    Raises:
        ValueError: per_group must be positive; Candidates contain previously used IDs;
            No valid products available.
        FileExistsError: Frozen releases cannot be overwritten.
        OSError: Copied image checksum mismatch.
    """
    localized = options.localized
    prior_samples = options.prior_samples
    prior_localized = options.prior_localized
    output = options.output
    per_group = options.per_group
    seed = options.seed
    if per_group < 1:
        raise ValueError("per_group must be positive")
    output = ensure_output_path(output)
    if output.exists():
        raise FileExistsError("Frozen releases cannot be overwritten")
    candidates = read_rows(localized)
    old = {
        row["id"]: dict(row)
        for row in read_rows(prior_samples)
        if row["dataset"] == "product"
    }
    old.update({row["id"]: row for row in read_rows(prior_localized)})
    historical = set(old)
    if historical & {row["id"] for row in candidates}:
        raise ValueError("Candidates contain previously used IDs")
    rows = [*old.values(), *candidates]
    hashes: dict[str, int] = {}
    flags: dict[str, list[str]] = {}
    for row in rows:
        identifier = row["id"]
        row["normalized_title"] = normalized_title(row.get("text", ""))
        issues = []
        if (
            not row.get("text")
            or not row.get("attributes")
            or row.get("category") not in PRODUCT_RULES
        ):
            issues.append("missing_required_metadata")
        multiple = [
            key
            for key, values in row.get("attributes", {}).items()
            if len(set(values)) > 1
        ]
        if multiple:
            issues.append("multi_value_attributes_require_sku_review")
        row["multi_value_keys"] = multiple
        if row.get("image_path"):
            base = (
                prior_localized.parent if identifier in historical else localized.parent
            )
            path = (base / row["image_path"]).resolve()
            if (
                not path.is_relative_to(base.resolve())
                or sha256_file(path) != row["image_sha256"]
            ):
                raise ValueError(f"Image checksum/path failure: {identifier}")
            hashes[identifier] = difference_hash(path)
        else:
            issues.append("missing_local_image")
        flags[identifier] = issues
    groups, edges = duplicate_groups(rows, hashes)
    group_ids = {}
    excluded = {}
    by_id = {row["id"]: row for row in rows}
    for members in groups:
        gid = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:20]
        for key in members:
            group_ids[key] = gid
        fresh = [key for key in members if key not in historical]
        if historical.intersection(members):
            excluded.update({key: "matches_previous_development_data" for key in fresh})
        elif fresh:
            valid = [
                key
                for key in fresh
                if by_id[key].get("image_path")
                and "missing_required_metadata" not in flags[key]
            ]
            if valid:
                representative = min(valid, key=lambda key: order_key(key, seed))
                excluded.update(
                    {
                        key: "duplicate_group_nonrepresentative"
                        for key in fresh
                        if key != representative
                    }
                )
    counts: Counter = Counter()
    selected = []
    decisions = []
    for row in sorted(candidates, key=lambda item: order_key(item["id"], seed)):
        key = row["id"]
        group = f"{row['split']}/{row['category']}"
        reason = excluded.get(key)
        if not reason and (
            not row.get("image_path") or "missing_required_metadata" in flags[key]
        ):
            reason = "missing_required_data"
        if not reason and counts[group] >= per_group:
            reason = "stratum_quota_reached"
        decisions.append(
            {
                "id": key,
                "selected": reason is None,
                "reason": reason,
                "group_id": group_ids[key],
                "flags": flags[key],
            }
        )
        if reason:
            continue
        counts[group] += 1
        origin = localized.parent / row["image_path"]
        target = ensure_output_path(output / "images" / origin.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        if sha256_file(target) != row["image_sha256"]:
            raise OSError("Copied image checksum mismatch")
        selected.append(
            {
                **row,
                "image_path": target.relative_to(output).as_posix(),
                "group_id": group_ids[key],
                "dhash64": f"{hashes[key]:016x}",
                "project_role": "heldout" if row["split"] == "test" else "development",
                "evaluation_eligible": True,
                "quality_flags": flags[key],
                "category_review_status": "heuristic_pending_human_review",
            }
        )
    if not selected:
        raise ValueError("No valid products available")
    write_rows(output / "products.jsonl", selected)
    write_json(output / "decisions.json", decisions)
    write_json(
        output / "duplicate_audit.json",
        {
            "groups": [group for group in groups if len(group) > 1],
            "edges": edges,
            "dhash_threshold": 4,
            "policy": "Conservatively exclude near-duplicate candidates; dHash is not confirmed identity and can miss transformed copies.",
        },
    )
    split_ids = {
        role: [row["id"] for row in selected if row["project_role"] == role]
        for role in ("development", "heldout")
    }
    write_json(output / "splits.json", split_ids)
    report = {
        "schema_version": 1,
        "seed": seed,
        "candidate_count": len(candidates),
        "downloaded_candidates": sum(bool(row.get("image_path")) for row in candidates),
        "selected": len(selected),
        "counts": dict(counts),
        "per_split_category_target": per_group,
        "shortfalls": {
            f"{split}/{category}": per_group - counts[f"{split}/{category}"]
            for split in ("train", "test")
            for category in PRODUCT_RULES
            if counts[f"{split}/{category}"] < per_group
        },
        "roles": {key: len(value) for key, value in split_ids.items()},
        "decisions": dict(Counter(item["reason"] or "selected" for item in decisions)),
        "quality_flags": dict(
            Counter(flag for row in selected for flag in row["quality_flags"])
        ),
        "duplicate_edges": len(edges),
        "prior_records_excluded": len(historical),
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (localized, prior_samples, prior_localized)
        },
        "scope": "Frozen engineering dataset. Heldout images/IDs have automatic quality checks only; no human relevance labels or final quality scores exist. Source-prefix, category heuristics and download availability bias remain.",
    }
    write_json(output / "quality_report.json", report)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return report
