"""Correct insufficient metadata coverage before any heldout pixels are opened."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import ijson

import grounded_blind_dataset as original
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import product_family_keys
from multimodal_ecommerce.business_review_jobs import canonical_hash
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.module_loading import load_project_module
from multimodal_ecommerce.safety import PROJECT_ROOT, PROTECTED_ROOT, sha256_file

ROOT = original.ROOT
SOURCE = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_test_id_label.json"
ACTUAL_POLICY: dict = dict(
    original.POLICY, scan_after=0, scan_limit=2000000, source_split="test"
)


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    original.guard()
    if (ROOT / "source_range_execution_protocol.json").exists():
        guard()
        return
    assert (
        not original.QUEUE.exists() and not (ROOT / "acquisition_receipt.json").exists()
    )
    shortage = load(ROOT / "metadata_sampling_shortage_evidence.json")
    assert (
        shortage["missing_buckets"]
        and shortage["pixels_opened"] == shortage["predictions"] == 0
    )
    paths = [
        PROJECT_ROOT / "scripts/grounded_blind_source_recovery.py",
        ROOT / "release_protocol.json",
        ROOT / "metadata_sampling_shortage_evidence.json",
    ]
    once(
        ROOT / "source_range_execution_protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            source=rel(SOURCE),
            source_sha256=sha256_file(SOURCE),
            policy=ACTUAL_POLICY,
            reason="Original train tail produced three metadata candidates and missed nine predefined strata; no new pixels or predictions existed. Before opening any new image, use the official test source uniformly for all unchanged strata/seed/counts, excluding all recorded project IDs, families and fingerprints. No category is substituted or selected from relevance outcomes.",
            changed_before_new_metadata_and_pixels=True,
            candidate_and_model_unchanged=True,
            buckets_seed_counts_and_relevance_unchanged=True,
            source_split_corrected=True,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    original.guard()
    p = load(ROOT / "source_range_execution_protocol.json")
    verify_files(PROJECT_ROOT, p["files"])
    assert p["policy"] == ACTUAL_POLICY


@dataclass(frozen=True)
class GroundedBlindSourceRecoveryTestScanInputs:
    """Input values for test scan.

    Attributes:
        source: Source used by this operation.
        prior: Prior used by this operation.
        after: After used by this operation.
        limit: Limit used by this operation.
        seed: Seed used to reproduce randomized ordering.
        buckets: Buckets used by this operation.
        classifier: Classifier used by this operation.
        count: Count used by this operation.
    """

    source: Path
    prior: dict
    after: int
    limit: int
    seed: int
    buckets: Sequence[str]
    classifier: Callable[[str], str | None]
    count: int


def test_scan(options: GroundedBlindSourceRecoveryTestScanInputs) -> tuple[list, dict]:
    """Verify that scan.

    Args:
        options: Validated input fields described by
            GroundedBlindSourceRecoveryTestScanInputs.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    source = options.source
    prior = options.prior
    after = options.after
    limit = options.limit
    seed = options.seed
    buckets = options.buckets
    classifier = options.classifier
    count = options.count
    assert source == SOURCE and after == 0 and limit == 2000000
    groups: dict = {b: [] for b in buckets}
    excluded = set(prior["family_keys"])
    with source.open("rb") as handle:
        for number, (sid, raw) in enumerate(ijson.kvitems(handle, ""), 1):
            if number > after and raw.get("url"):
                bucket = classifier(str(raw.get("title") or ""))
                if bucket:
                    row = dict(
                        id=f"product/test/{sid}",
                        source_id=str(sid),
                        dataset="product",
                        split="test",
                        text=raw["title"],
                        image_url=raw["url"],
                        attributes=parse_pv(raw.get("pv") or ""),
                        bucket=bucket,
                        source_position=number,
                        source_record_sha256=canonical_hash(raw),
                        project_role="reserved_new_grounded_AI_blind_never_development",
                    )
                    keys = product_family_keys(row)
                    if not keys & excluded:
                        row.update(
                            family_keys=sorted(keys),
                            priority=hashlib.sha256(
                                f"{seed}:{row['id']}".encode()
                            ).hexdigest(),
                        )
                        groups[bucket].append(row)
            if number == limit:
                break
    selected = []
    seen = set(excluded)
    for bucket, rows in groups.items():
        picked = 0
        for row in sorted(rows, key=lambda r: r["priority"]):
            if not set(row["family_keys"]) & seen:
                selected.append(row)
                seen.update(row["family_keys"])
                picked += 1
                if picked == count:
                    break
    return selected, dict(
        scanned=number,
        eligible={b: len(g) for b, g in groups.items()},
        actual_source_split="test",
        after=after,
        limit=limit,
    )


def reserve() -> None:
    """Apply the reserve workflow step."""
    guard()
    module = load_project_module(
        "grounded_blind_actual_source",
        PROJECT_ROOT / "scripts/grounded_blind_dataset.py",
    )
    module.SOURCE = SOURCE
    module.POLICY = ACTUAL_POLICY
    module.scan = test_scan
    module.guard = guard
    module.reserve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "reserve", "verify"])
    a = parser.parse_args().action
    if a == "prepare":
        prepare()
    elif a == "reserve":
        reserve()
    else:
        guard()
        original.checked("reservation_receipt.json")
        print(dict(actual_source_verified=True))
