"""An extended gallery may reorder old IDs; cached vectors must follow identity."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from multimodal_ecommerce import safety


@pytest.mark.parametrize("corrupt_cache", [False, True])
def test_cached_extension_aligns_all_banks_and_checks_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_cache: object
) -> None:
    """Verify that cached extension aligns all banks and checks checksum.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
        corrupt_cache: Corrupt cache used by this operation.

    Notes:
        Writes derived artifacts through (parent / 'index').mkdir, (root /
        'profiles.public.jsonl').write_text, cache.mkdir, np.savez_compressed.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    stage = importlib.import_module("advance_visual_coverage")
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    root, parent = tmp_path / "new", tmp_path / "parent"
    cache = root / "embedding_cache"
    cache.mkdir(parents=True)
    (parent / "index").mkdir(parents=True)
    monkeypatch.setattr(stage, "ROOT", root)
    monkeypatch.setattr(stage, "PARENT", parent)
    monkeypatch.setattr(stage, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        stage, "check", lambda: dict(region_policy=dict(max_views=7), embedding={})
    )

    def no_model() -> None:
        """Apply the no model workflow step."""
        pytest.fail("A complete cache must not start a model")

    monkeypatch.setattr(stage, "QwenEmbedder", no_model)
    old = [
        dict(
            id=cid,
            title=cid,
            image_path=str(tmp_path / (cid + ".png")),
            image_sha256=cid,
            views=[],
        )
        for cid in ["a", "b"]
    ]
    base = SimpleNamespace(
        records=old,
        vectors=np.stack([np.full((7, 2048), 1), np.full((7, 2048), 2)]),
    )
    base_titles = np.stack([np.full(2048, 3), np.full(2048, 4)])
    monkeypatch.setattr(stage, "load_expanded_index", lambda path: (base, base_titles))
    np.savez_compressed(
        parent / "index/vectors.npz",
        profiles=np.stack([np.full(2048, 5), np.full(2048, 6)]),
    )
    stage.write_json(parent / "index/profile_texts.json", ["profile-a", "profile-b"])
    stage.write_json(parent / "index/manifest.json", dict(files={}, upstream_files={}))
    new: dict = dict(id="c", title="c", image_path=str(tmp_path / "c.png"))
    documents = [new, *reversed(old)]
    stage.write_json(root / "inputs.public.json", dict(catalog=documents))
    (root / "profiles.public.jsonl").write_text(
        json.dumps(dict(id="c")) + "\n", encoding="utf-8"
    )
    asset = cache / (stage.key("c") + ".npz")
    np.savez_compressed(
        asset,
        regions=np.full((2, 2048), 7),
        title=np.full(2048, 8),
        profile=np.full(2048, 9),
    )
    stage.write_json(
        asset.with_suffix(".json"),
        dict(
            **new,
            image_sha256="c",
            views=[{}, {}],
            profile_text="profile-c",
            vector_asset=asset.name,
            vector_sha256="changed" if corrupt_cache else stage.sha256_file(asset),
        ),
    )
    if corrupt_cache:
        with pytest.raises(AssertionError):
            stage.encode()
        assert not (root / "index/manifest.json").exists()
        return
    stage.encode()
    with np.load(root / "index/vectors.npz", allow_pickle=False) as archive:
        assert archive["regions"][:, 0, 0].tolist() == [7, 2, 1]
        assert archive["texts"][:, 0].tolist() == [8, 4, 3]
        assert archive["profiles"][:, 0].tolist() == [9, 6, 5]
    assert [r["id"] for r in stage.load(root / "index/records.json")] == ["c", "b", "a"]
    assert stage.load(root / "index/profile_texts.json") == [
        "profile-c",
        "profile-b",
        "profile-a",
    ]
