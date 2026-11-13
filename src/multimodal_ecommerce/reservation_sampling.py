"""Hash-prioritized metadata sampling for an unseen, source-range-frozen reservation."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.taxonomy import normalize_label


def sample_reservation_metadata(
    source: Iterable[tuple[str, dict]], policy: dict, excluded: set[str]
) -> tuple[list[dict], dict[str, int], int]:
    """Select candidates without image semantics, relevance labels or model scores.

    Args:
        source: Streaming native source IDs and metadata records.
        policy: Frozen source range, seed, per-type budget and predefined fine types.
        excluded: All recorded development, consumed and older reserved family keys.

    Returns:
        Frozen hash-priority queue, eligible counts per fine type and scanned records.

    Raises:
        ValueError: Source range or per-type sampling budget is invalid.
    """
    start, stop = policy.get("prefix_scan_start", 0), policy["prefix_scan_limit"]
    budget = policy["queue_per_fine_type"]
    if not 0 <= start < stop or budget < 1:
        raise ValueError("A nonempty source range and positive queue budget required")
    fine = {
        label: category
        for category, labels in policy["fine_types"].items()
        for label in labels
    }
    candidates: dict[str, list[dict]] = {label: [] for label in fine}
    counts: Counter[str] = Counter()
    scanned = 0
    for position, (sid, raw) in enumerate(source, 1):
        if position > stop:
            break
        scanned = position
        if position <= start:
            continue
        label = normalize_label(str(raw.get("label") or ""))
        if label not in fine or not raw.get("title") or not raw.get("url"):
            continue
        row = dict(
            id=f"product/{policy['source_split']}/{sid}",
            source_id=str(sid),
            dataset="product",
            split=policy["source_split"],
            label=label,
            category=fine[label],
            text=raw["title"],
            image_url=raw["url"],
            attributes=parse_pv(raw.get("pv") or ""),
            source_position=position,
        )
        keys = product_family_keys(row)
        if keys & excluded:
            continue
        counts[label] += 1
        row["family_keys"] = sorted(keys)
        row["priority"] = hashlib.sha256(f"{policy['seed']}/{sid}".encode()).hexdigest()
        candidates[label].append(row)
        if len(candidates[label]) > 2 * budget:
            candidates[label] = sorted(candidates[label], key=lambda r: r["priority"])[
                :budget
            ]
    queue = [
        row
        for label in fine
        for row in sorted(candidates[label], key=lambda r: r["priority"])[:budget]
    ]
    return queue, dict(counts), scanned
