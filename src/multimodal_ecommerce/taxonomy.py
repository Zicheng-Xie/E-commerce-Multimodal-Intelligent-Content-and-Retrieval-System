"""Explicit, versioned taxonomy decisions without modifying frozen v1 data."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import ensure_output_path, sha256_file, write_json


def normalize_label(value: str) -> str:
    """Normalize width and surrounding whitespace, not semantic substrings.

    Args:
        value: Input value to validate or transform.

    Returns:
        Text produced by this operation.
    """
    return unicodedata.normalize("NFKC", value).strip()


def load_policy(path: Path) -> dict:
    """Reject overlapping exact/context keys and invalid category targets.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Invalid context category.
    """
    policy = json.loads(path.read_text(encoding="utf-8"))
    categories = set(policy["category_scope"])
    seen = set()
    for label, target in policy["exact_labels"].items():
        key = normalize_label(label)
        if key in seen or target not in categories | {"out_of_scope"}:
            raise ValueError(f"Invalid or duplicate taxonomy entry: {label}")
        seen.add(key)
    for label, rules in policy["context_labels"].items():
        key = normalize_label(label)
        if key in seen:
            raise ValueError(f"Overlapping taxonomy entry: {label}")
        seen.add(key)
        for rule in rules:
            if rule["category"] not in categories | {"out_of_scope"}:
                raise ValueError("Invalid context category")
            re.compile(rule["pattern"], re.IGNORECASE)
    return policy


def resolve_category(row: dict, policy: dict) -> dict:
    """Return inspectable evidence; ambiguous/unknown labels are quarantined.

    Context is used only for explicitly listed ambiguous fine labels. Conflicting
    matches abstain instead of resolving by arbitrary priority. The result is an
    AI-authored project taxonomy decision, never a human relevance judgment.

    Args:
        row: Row used by this operation.
        policy: Policy used by this operation.

    Returns:
        Mapping with fields: id, label, previous_category, category, status,
        taxonomy_version, review_origin, human_reviewed, evidence.
    """
    label = normalize_label(row.get("label", ""))
    title = unicodedata.normalize("NFKC", row.get("text", ""))
    contexts = {normalize_label(k): v for k, v in policy["context_labels"].items()}
    exact = {normalize_label(k): v for k, v in policy["exact_labels"].items()}
    result = {
        "id": row["id"],
        "label": row.get("label", ""),
        "previous_category": row.get("category"),
        "category": None,
        "status": "needs_review",
        "taxonomy_version": policy["version"],
        "review_origin": "ai_taxonomy_review",
        "human_reviewed": False,
        "evidence": [],
    }
    if label in contexts:
        matches = []
        for rule in contexts[label]:
            match = re.search(rule["pattern"], title, re.IGNORECASE)
            if match:
                matches.append({**rule, "matched_text": match.group()})
        targets = {match["category"] for match in matches}
        result["evidence"] = matches
        if len(targets) != 1:
            result["reason"] = "ambiguous_context" if targets else "unresolved_context"
            return result
        target = targets.pop()
    elif label in exact:
        target = exact[label]
        result["evidence"] = [{"rule": "exact_fine_label", "matched_text": label}]
    else:
        result["reason"] = "unknown_fine_label"
        return result
    result.update(
        category=None if target == "out_of_scope" else target,
        status="out_of_scope" if target == "out_of_scope" else "mapped",
        reason="explicit_project_scope",
    )
    return result


def release_taxonomy(dataset: Path, policy_path: Path, output: Path) -> dict:
    """Copy accepted assets into a new release; preserve IDs, roles and lineage.

    Args:
        dataset: Dataset used by this operation.
        policy_path: Policy path used by this operation.
        output: Output used by this operation.

    Returns:
        Mapping with fields: version, source_dataset, source_dataset_sha256,
        policy_sha256, source_count, accepted_count, quarantined_count, status_counts,
        before_category_counts, after_category_counts, after_role_category_counts,
        changed_category_ids, human_reviewed, limitations.

    Raises:
        FileExistsError: Taxonomy releases must use a new directory.
        ValueError: Duplicate product IDs; Image path escapes dataset; Copied image hash
            mismatch.
    """
    output = ensure_output_path(output, dataset.parent)
    if output.exists():
        raise FileExistsError("Taxonomy releases must use a new directory")
    policy = load_policy(policy_path)
    rows = read_rows(dataset)
    dataset_digest = sha256_file(dataset)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate product IDs")
    decisions = [resolve_category(row, policy) for row in rows]
    copies: dict[Path, str] = {}
    kept, quarantine = [], []
    # Validate all source assets before writing any release.
    for row, decision in zip(rows, decisions):
        if decision["status"] != "mapped":
            quarantine.append({"source_record": row, "decision": decision})
            continue
        relative = Path(row["image_path"])
        source = (dataset.parent / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(
            dataset.parent.resolve()
        ):
            raise ValueError("Image path escapes dataset")
        digest = sha256_file(source)
        if digest != row["image_sha256"]:
            raise ValueError(f"Changed source image: {row['id']}")
        copies[relative] = digest
        kept.append(
            {
                **row,
                "category": decision["category"],
                "category_rule": policy["version"],
                "category_review_status": "ai_reviewed_pending_human",
                "taxonomy_decision": decision,
                "lineage": {
                    "source_dataset": str(dataset.resolve()),
                    "source_dataset_sha256": dataset_digest,
                    "previous_category": row["category"],
                    "previous_category_rule": row["category_rule"],
                },
            }
        )
    output.mkdir(parents=True)
    for relative, digest in copies.items():
        destination = ensure_output_path(output / relative, dataset.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset.parent / relative, destination)
        if sha256_file(destination) != digest:
            raise ValueError("Copied image hash mismatch")
    write_rows(output / "products.jsonl", kept)
    write_rows(output / "quarantine.jsonl", quarantine)
    write_rows(output / "decisions.jsonl", decisions)
    write_json(output / "taxonomy.json", policy)
    summary = {
        "version": policy["version"],
        "source_dataset": str(dataset.resolve()),
        "source_dataset_sha256": dataset_digest,
        "policy_sha256": sha256_file(policy_path),
        "source_count": len(rows),
        "accepted_count": len(kept),
        "quarantined_count": len(quarantine),
        "status_counts": dict(Counter(d["status"] for d in decisions)),
        "before_category_counts": dict(Counter(r["category"] for r in rows)),
        "after_category_counts": dict(Counter(r["category"] for r in kept)),
        "after_role_category_counts": dict(
            Counter(f"{r['project_role']}/{r['category']}" for r in kept)
        ),
        "changed_category_ids": [
            d["id"]
            for d in decisions
            if d["status"] == "mapped" and d["category"] != d["previous_category"]
        ],
        "human_reviewed": False,
        "limitations": [
            "AI-authored taxonomy and context rules require human confirmation.",
            "Category decisions do not certify attributes, SKU alignment or relevance.",
            "Historical v1 index, copy outputs and review packets are unchanged.",
            "Old roles are preserved for traceability; inspected heldout data is not a new blind test.",
            "Quarantine records reference the source release; they are not indexable products.",
        ],
    }
    write_json(output / "report.json", summary)
    write_json(
        output / "manifest.json",
        {
            "files": {
                str(p.relative_to(output).as_posix()): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
        },
    )
    return summary
