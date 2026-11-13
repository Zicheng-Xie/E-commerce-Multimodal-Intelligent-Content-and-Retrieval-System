"""Meaningful leakage, immutable-input, and one-shot unblinding checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from multimodal_ecommerce import blind_evaluation as blind
from multimodal_ecommerce import holdout_data as heldout
from multimodal_ecommerce import safety
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.safety import sha256_file, write_json


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return workspace for this workflow.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Returns:
        Resolved artifact path.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(heldout, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(heldout, "PROTECTED_ROOT", tmp_path / "source")
    monkeypatch.setattr(blind, "PROJECT_ROOT", tmp_path)
    return tmp_path


def picture(path: Path, seed: int) -> None:
    """Apply the picture workflow step.

    Args:
        path: Path of the input or output artifact used by this operation.
        seed: Seed used to reproduce randomized ordering.

    Notes:
        Writes derived artifacts through Image.fromarray(pixels).save,
        path.parent.mkdir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_fingerprint_detects_reencoding_and_close_hash(workspace: Path) -> None:
    """Verify that fingerprint detects reencoding and close hash.

    Args:
        workspace: Workspace used by this operation.

    Notes:
        Writes derived artifacts through source.save.
    """
    first, second = workspace / "a.png", workspace / "b.bmp"
    picture(first, 1)
    with Image.open(first) as source:
        source.save(second)
    a, b = heldout.fingerprints(first), heldout.fingerprints(second)
    assert a["sha256"] != b["sha256"] and a["pixels"] == b["pixels"]
    assert heldout.is_exposed(b, [a])
    close: dict = {"sha256": "new", "pixels": "new", "dhash": a["dhash"] ^ 15}
    assert heldout.is_exposed(close, [a], 4)
    assert not heldout.is_exposed(close, [a], 3)


def test_preparation_excludes_old_pair_and_sanitizes(workspace: Path) -> None:
    """Verify that preparation excludes old pair and sanitizes.

    Args:
        workspace: Workspace used by this operation.

    Notes:
        Writes derived artifacts through bundle.mkdir, write_json.
    """
    bundle = workspace / "suite"
    bundle.mkdir()
    source = workspace / "source/服饰分割/validation"
    for number, (pair, origin) in enumerate(
        ((99, "user"), (99, "shop"), (1, "user"), (1, "shop"), (2, "shop")), 1
    ):
        name = f"{number:06}"
        write_json(
            source / "annos" / (name + ".json"),
            {
                "pair_id": pair,
                "source": origin,
                "item1": {"style": 1, "category_id": 1},
            },
        )
        picture(source / "image" / (name + ".jpg"), number)
    recipe: dict = {
        "data_policy": {
            "source_split": "validation",
            "scan_limit": 5,
            "category_ids": [1],
            "queries_per_category": 1,
            "gallery_size": 2,
            "seed": 42,
            "max_positive_shop_views": 10,
            "dhash_threshold": 4,
        }
    }
    exposure: dict = {
        "ids": [],
        "deepfashion_pair_ids_conservative_cross_split": ["99"],
        "image_fingerprints": [],
    }
    write_json(bundle / "recipe.json", recipe)
    write_json(bundle / "exposure.json", exposure)
    report = heldout.prepare_same_item(bundle, recipe, exposure)
    assert report["query_count"] == 1 and report["gallery_count"] == 2
    assert report["pre_inference_exclusions"]["known_id_or_pair_group"] == 2
    lineage = blind.read_json(bundle / "dataset/evaluator/lineage.json")
    assert all(r["pair_group"] != "99" for r in lineage)
    for role in ("queries", "gallery"):
        folder = bundle / "dataset/public" / role
        rows = read_rows(folder / "products.jsonl")
        blind.validate_public(rows, folder)
        assert all("source_id" not in r and "pair_group" not in r for r in rows)
    blind.verify_files(
        bundle / "dataset", blind.read_json(bundle / "dataset/manifest.json")["files"]
    )
    with pytest.raises(FileExistsError):
        heldout.prepare_same_item(bundle, recipe, exposure)


def fake_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return fake bundle for this workflow.

    Args:
        root: Root directory used to resolve the operation artifacts.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Returns:
        Numeric result computed from the supplied values.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    bundle = root / "blind"
    recipe: dict = {"methods": list(blind.METHODS), "scope_limit": "synthetic fixture"}
    write_json(bundle / "recipe.json", recipe)
    monkeypatch.setattr(blind, "verify_recipe", lambda *args, **kwargs: recipe)
    for role, ids in (("queries", ["q1", "q2"]), ("gallery", ["g1", "g2", "g3"])):
        folder = bundle / "dataset/public" / role
        rows = []
        for n, identifier in enumerate(ids):
            image = folder / "images" / (identifier + ".jpg")
            picture(image, n + 1)
            rows.append(
                {
                    "id": identifier,
                    "image_path": "images/" + image.name,
                    "image_sha256": sha256_file(image),
                    "evaluation_eligible": True,
                    "text": None,
                    "category": "",
                    "attributes": {},
                }
            )
        write_rows(folder / "products.jsonl", rows)
    write_json(bundle / "dataset/audit.json", {"recorded_exposure_audit_passed": True})
    write_json(
        bundle / "dataset/evaluator/qrels.json",
        [
            {"query_id": q, "pair_group": q, "relevant_ids": [g]}
            for q, g in (("q1", "g1"), ("q2", "g2"))
        ],
    )
    write_json(
        bundle / "dataset/manifest.json", {"files": blind.file_map(bundle / "dataset")}
    )
    return bundle


def predictions(bundle: Path) -> None:
    """Apply the predictions workflow step.

    Args:
        bundle: Bundle used by this operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    rows = [
        {
            "query_id": q,
            "strategy": s,
            "ranked_ids": ["g1", "g2", "g3"],
            "scores": [3.0, 2.0, 1.0],
        }
        for q in ("q1", "q2")
        for s in blind.METHODS
    ]
    write_json(bundle / "predictions/rankings.json", rows)
    write_json(
        bundle / "predictions/timings.json",
        [{"strategy": s, "warm_query_seconds": 0.1} for s in blind.METHODS],
    )
    write_json(
        bundle / "prediction_receipt.json",
        {
            "recipe_sha256": sha256_file(bundle / "recipe.json"),
            "dataset_manifest_sha256": sha256_file(bundle / "dataset/manifest.json"),
            "files": blind.file_map(bundle / "predictions"),
        },
    )


def test_blind_predictions_never_parse_truth(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that blind predictions never parse truth.

    Args:
        workspace: Workspace used by this operation.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    bundle = fake_bundle(workspace, monkeypatch)
    real_read = blind.read_json

    def guarded_read(path: Path) -> dict:
        """Return guarded read for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        assert "evaluator" not in path.parts
        return real_read(path)

    monkeypatch.setattr(blind, "read_json", guarded_read)

    def fake_build(embedder: QwenEmbedder, gallery: object, output: Path) -> None:
        """Apply the fake build workflow step.

        Args:
            embedder: Embedder used by this operation.
            gallery: Gallery used by this operation.
            output: Output used by this operation.

        Notes:
            Writes derived artifacts through (output / 'marker').write_text,
            output.mkdir.
        """
        output.mkdir(parents=True)
        (output / "marker").write_text("synthetic")

    monkeypatch.setattr(blind, "build_regional_index", fake_build)
    index = SimpleNamespace(
        records=[{"id": "g1"}, {"id": "g2"}, {"id": "g3"}],
        vectors=np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]),
    )
    monkeypatch.setattr(blind.RegionalIndex, "load", lambda _: (index, None, {}))
    monkeypatch.setattr(
        blind, "selected_scores", lambda *args: (np.array([1.0, 0.0, -1.0]), {})
    )

    class Encoder:
        """Supply deterministic vectors for the isolated retrieval scenario under
        test.
        """

        def encode(self, **kwargs: object) -> np.ndarray:
            """Encode the configured inputs into model feature representations.

            Args:
                **kwargs: Extra arguments accepted to match the patched dependency.

            Returns:
                Computed array in the order of the supplied records.
            """
            assert set(kwargs) == {"image", "query"} and kwargs["query"] is True
            return np.array([1.0, 0.0])

        def diagnostics(self) -> dict:
            """Describe the active backend and recorded runtime measurements.

            Returns:
                Mapping with fields: synthetic.
            """
            return {"synthetic": True}

        def close(self) -> None:
            """Release the resources held by this component."""
            pass

    report = blind.predict(bundle, Encoder)
    assert report["truth_read"] is False
    with pytest.raises(FileExistsError):
        blind.predict(bundle, Encoder)


def test_one_time_unblind_and_no_development_release(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that one time unblind and no development release.

    Args:
        workspace: Workspace used by this operation.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    bundle = fake_bundle(workspace, monkeypatch)
    predictions(bundle)
    report = blind.score(bundle)
    assert report["formal_module2_acceptance"] is False
    assert report["selected_after_results"] is False
    assert report["paired_candidate_minus_baseline"]["recall@10"]["estimate"] == 0
    assert blind.read_json(bundle / "score.claim.json")["consumed"] is True
    with pytest.raises(FileExistsError):
        blind.score(bundle)
    public = blind.read_json(bundle / "public_status.json")
    assert public["development_metrics_releasable"] is False
    assert public["formal_acceptance_metrics"] is None


def test_tampered_predictions_rejected_before_truth(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that tampered predictions rejected before truth.

    Args:
        workspace: Workspace used by this operation.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    bundle = fake_bundle(workspace, monkeypatch)
    predictions(bundle)
    with (bundle / "predictions/rankings.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Frozen file changed"):
        blind.score(bundle)
    assert not (bundle / "score.claim.json").exists()


@pytest.mark.parametrize(
    "change", ["missing_query", "missing_candidate", "nan", "ascending"]
)
def test_invalid_complete_rankings_rejected(change: object) -> None:
    """Verify that invalid complete rankings rejected.

    Args:
        change: Change used by this operation.
    """
    rows: list[dict] = [
        {
            "query_id": "q",
            "strategy": "global",
            "ranked_ids": ["a", "b"],
            "scores": [2.0, 1.0],
        }
    ]
    if change == "missing_query":
        rows = []
    elif change == "missing_candidate":
        rows[0]["ranked_ids"] = ["a"]
    elif change == "nan":
        rows[0]["scores"] = [float("nan"), 1.0]
    else:
        rows[0]["scores"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        blind.validate_rankings(rows, {"q"}, {"a", "b"}, ["global"])


def test_frozen_recipe_rejects_changed_source(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that frozen recipe rejects changed source.

    Args:
        workspace: Workspace used by this operation.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through source.write_text, write_json.
    """
    bundle = workspace / "suite"
    source = workspace / "implementation.py"
    source.write_text("frozen")
    exposure = bundle / "exposure.json"
    write_json(exposure, {"files": {"implementation.py": sha256_file(source)}})
    recipe: dict = {
        "methods": list(blind.METHODS),
        "implementation": {"implementation.py": sha256_file(source)},
        "runtime": blind.runtime_versions(),
        "exposure_sha256": sha256_file(exposure),
    }
    write_json(bundle / "recipe.json", recipe)
    write_json(
        bundle / "freeze_receipt.json",
        {"recipe_sha256": sha256_file(bundle / "recipe.json")},
    )
    assert blind.verify_recipe(bundle, checkpoint=False) == recipe
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        blind.verify_recipe(bundle, checkpoint=False)


def test_manifest_escape_and_label_input_rejected(workspace: Path) -> None:
    """Verify that manifest escape and label input rejected.

    Args:
        workspace: Workspace used by this operation.
    """
    with pytest.raises(ValueError, match="Escaped"):
        blind.verify_files(workspace, {"../escape": "digest"})
    row: dict = {
        "id": "x",
        "image_path": "a",
        "image_sha256": "x",
        "evaluation_eligible": True,
        "text": None,
        "category": "source answer",
        "attributes": {},
    }
    with pytest.raises(ValueError, match="Labels"):
        blind.validate_public([row], workspace)


@pytest.mark.parametrize("wrong_order", [False, True])
def test_saved_vector_replay_detects_wrong_rank(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, wrong_order: object
) -> None:
    """Verify that saved vector replay detects wrong rank.

    Args:
        workspace: Workspace used by this operation.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
        wrong_order: Wrong order used by this operation.

    Notes:
        Writes derived artifacts through np.save, output.mkdir, write_json.
    """
    from multimodal_ecommerce import blind_verification as replay

    bundle = fake_bundle(workspace, monkeypatch)
    monkeypatch.setattr(replay, "verify_recipe", blind.verify_recipe)
    real_read = replay.read_json

    def guarded_read(path: Path) -> dict:
        """Return guarded read for this workflow.

        Args:
            path: Path of the input or output artifact used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        assert "evaluator" not in path.parts
        return real_read(path)

    monkeypatch.setattr(replay, "read_json", guarded_read)
    index = SimpleNamespace(
        records=[{"id": key} for key in ("g1", "g2", "g3")],
        vectors=np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]),
    )
    monkeypatch.setattr(replay.RegionalIndex, "load", lambda _: (index, None, {}))
    monkeypatch.setattr(
        replay, "selected_scores", lambda *args: (np.array([1.0, 0.0, -1.0]), {})
    )
    output = bundle / "predictions"
    output.mkdir()
    np.save(
        output / "query_vectors.npy",
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        allow_pickle=False,
    )
    ids = ["g2", "g1", "g3"] if wrong_order else ["g1", "g2", "g3"]
    write_json(
        output / "rankings.json",
        [
            {
                "query_id": q,
                "strategy": s,
                "ranked_ids": ids,
                "scores": [1.0, 0.0, -1.0],
            }
            for q in ("q1", "q2")
            for s in blind.METHODS
        ],
    )
    write_json(
        bundle / "prediction_receipt.json",
        {
            "recipe_sha256": sha256_file(bundle / "recipe.json"),
            "dataset_manifest_sha256": sha256_file(bundle / "dataset/manifest.json"),
            "files": blind.file_map(output),
        },
    )
    if wrong_order:
        with pytest.raises(ValueError, match="Replayed ranking"):
            replay.replay_predictions(bundle)
    else:
        result = replay.replay_predictions(bundle)
        assert (
            result["complete_rankings"] == 4 and result["quality_truth_parsed"] is False
        )
