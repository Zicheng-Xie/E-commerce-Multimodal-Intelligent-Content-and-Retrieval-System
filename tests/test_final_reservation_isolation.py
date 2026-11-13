"""Retired-command barriers, source-range sampling, and frozen reservation integrity."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import module2_final_reservation as reservation
from multimodal_ecommerce import safety
from multimodal_ecommerce.evaluation_access import (
    require_command_access,
    require_data_access,
)
from multimodal_ecommerce.reservation_sampling import sample_reservation_metadata


def test_consumed_flag_cannot_be_bypassed_by_inference_ready_flags() -> None:
    """Once consumed, even apparently ready evaluator metadata cannot reopen data."""
    registry = dict(
        rules=[
            dict(
                path_prefix="used/",
                role="old_result",
                consumed=True,
                model_inference_allowed=True,
                formal_acceptance_ready=True,
            )
        ],
        exclusion_metadata_allowlist=["used/ids.json"],
    )
    for purpose in ("development", "training", "tuning", "acceptance_inference"):
        with pytest.raises(PermissionError, match="already consumed"):
            require_data_access([Path("used/data.json")], purpose, registry)
    require_data_access([Path("used/ids.json")], "exclusion_metadata", registry)
    with pytest.raises(PermissionError):
        require_data_access([Path("used/labels.json")], "exclusion_metadata", registry)


def test_every_registered_retired_command_is_blocked_before_execution() -> None:
    """The real policy covers every legacy command and all protected dataset purposes."""
    root = safety.PROJECT_ROOT
    registry = json.loads((root / "configs/evaluation_registry.json").read_text())
    assert len(registry["sealed_evaluation_commands"]) == 18
    for rule in registry["sealed_evaluation_commands"]:
        with pytest.raises(PermissionError, match="Retired evaluation command"):
            require_command_access(root / rule["script"], registry)
    require_command_access(root / "scripts/module2_mainline_conflicts.py", registry)
    for name in (
        "artifacts/module2_final_acceptance_20261110/gallery.reserved.json",
        "artifacts/module2_independent_business_blind/judgments.ai.jsonl",
    ):
        for purpose in ("development", "training", "tuning", "acceptance_inference"):
            with pytest.raises(PermissionError):
                require_data_access([Path(name)], purpose, registry)


def test_retired_cli_stops_before_loading_its_frozen_experiment() -> None:
    """Exercise the actual bootstrap rather than only testing its policy helper."""
    result = subprocess.run(
        [sys.executable, "scripts/blind_retrieval.py", "predict"],
        cwd=safety.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Retired evaluation command" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_new_queue_uses_only_frozen_source_range_and_ignores_model_scores() -> None:
    """Source bounds and family exclusions must dominate any incidental score fields."""
    rows = [
        (
            str(i),
            dict(
                label="耳机",
                title=f"耳机{i}",
                url=f"https://example.org/{i}.jpg",
                pv="",
                model_score=i,
            ),
        )
        for i in range(12)
    ]
    policy = dict(
        seed=7,
        source_split="test",
        prefix_scan_start=3,
        prefix_scan_limit=10,
        queue_per_fine_type=3,
        fine_types={"3C": ["耳机"]},
    )
    first, counts, scanned = sample_reservation_metadata(rows, policy, {"id:4"})
    assert scanned == 10 and counts == {"耳机": 6}
    assert len(first) == 3 and all(3 < r["source_position"] <= 10 for r in first)
    assert all(r["source_id"] != "4" for r in first)
    for _, row in rows:
        row["model_score"] = -999
        row["AI_relevance"] = "different"
    second, _, _ = sample_reservation_metadata(rows, policy, {"id:4"})
    assert first == second
    old_ids = {row["id"] for row in first}
    third, _, _ = sample_reservation_metadata(rows, policy, {"title:耳机5", "id:4"})
    assert all(r["source_id"] not in {"4", "5"} for r in third)
    assert len(old_ids) == 3


def test_empty_source_range_is_rejected() -> None:
    """Do not silently fall back to an already sampled source prefix."""
    with pytest.raises(ValueError):
        sample_reservation_metadata(
            [],
            dict(prefix_scan_start=10, prefix_scan_limit=10, queue_per_fine_type=3),
            set(),
        )


def _fixture(root: Path) -> Path:
    folder = root / "reserved"
    (folder / "media").mkdir(parents=True)
    rng = random.Random(1)
    rows, files = [], {}
    for index in range(330):
        content = str(index).encode()
        digest = hashlib.sha256(content).hexdigest()
        name = f"media/{index}.jpg"
        (folder / name).write_bytes(content)
        files[name] = digest
        rows.append(
            dict(
                id=f"product/test/{index}",
                source_id=str(index),
                source_position=400001 + index,
                family_keys=[f"id:{index}"],
                image_path=name,
                fingerprints=dict(
                    sha256=digest, pixels=digest, dhash=rng.getrandbits(64)
                ),
            )
        )

    def save(name: str, value: object) -> None:
        """Write synthetic fixture metadata without real evaluation content.

        Args:
            name: Fixture filename inside the temporary reservation.
            value: JSON fixture content.
        """
        (folder / name).write_text(json.dumps(value))

    save("frozen_media_manifest.json", dict(gallery=300, image_queries=30, files=files))
    save("gallery.reserved.json", rows[:300])
    save("image_queries.reserved.json", rows[300:])
    save("candidates.reserved.json", rows)
    save(
        "technical_downloads.json",
        {
            r["id"]: dict(status="downloaded", fingerprints=r["fingerprints"])
            for r in rows
        },
    )
    save("reservation_protocol.json", {})
    prior = root / "prior.json"
    prior.write_text(
        json.dumps(
            dict(
                ids=[],
                product_source_ids=[],
                family_keys=[],
                image_fingerprints=[],
                counts={},
            )
        )
    )
    return prior


def test_seal_is_immutable_and_rejects_changed_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat verification may not bless altered reserved media or overwrite its seal.

    Args:
        tmp_path: Temporary artifact root without real evaluation data.
        monkeypatch: Isolate all output and source paths from the real project.
    """
    prior = _fixture(tmp_path)
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reservation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reservation, "ROOT", tmp_path / "reserved")
    monkeypatch.setattr(reservation, "CONTROL", tmp_path / "control")
    reservation._seal(prior, {})
    sealed = (reservation.ROOT / "seal.json").read_bytes()
    reservation._seal(prior, {})
    assert (reservation.ROOT / "seal.json").read_bytes() == sealed
    (reservation.ROOT / "media/0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen identity"):
        reservation._seal(prior, {})
    with pytest.raises(ValueError, match="Sealed acceptance contents changed"):
        reservation._verify()
    assert (reservation.ROOT / "seal.json").read_bytes() == sealed


def test_seal_rejects_overlap_with_prior_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid image hashes do not excuse a previously exposed product family.

    Args:
        tmp_path: Temporary artifact root without real evaluation data.
        monkeypatch: Isolate all output and source paths from the real project.
    """
    prior = _fixture(tmp_path)
    value = json.loads(prior.read_text())
    value["family_keys"] = ["id:5"]
    prior.write_text(json.dumps(value))
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reservation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reservation, "ROOT", tmp_path / "reserved")
    with pytest.raises(ValueError, match="Prior exposure"):
        reservation._seal(prior, {})
    assert not (reservation.ROOT / "seal.json").exists()
