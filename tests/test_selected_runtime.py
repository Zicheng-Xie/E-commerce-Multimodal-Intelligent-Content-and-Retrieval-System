"""General runtime invariance, isolation, cache binding and exact interrupted resume."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multimodal_ecommerce import (
    evaluation_access,
    safety,
    selected_acceptance,
    selected_release,
    selected_runtime_io,
)
from multimodal_ecommerce.selected_engine import SelectedEngine
from multimodal_ecommerce.selected_models import validate_value
from multimodal_ecommerce.selected_runtime_io import (
    ContentCache,
    RuntimeAccess,
    atomic_json,
    digest,
    load,
    public_records,
)


class _Models:
    def __init__(self, policy: dict) -> None:
        """Bind a synthetic backend without loading local model assets.

        Args:
            policy: Test-only policy identity.
        """
        self.policy = policy
        self.events: list[dict] = []
        self.calls: list[str] = []
        self.scores = 0
        self.fail_after: int | None = None

    def identity(self) -> dict:
        """Return a synthetic content-bound model identity.

        Returns:
            Test-only identity, never represented as actual model inference.
        """
        return dict(policy_sha256=digest(self.policy), assets_sha256={}, synthetic=True)

    def profile(self, row: dict, last: bool = False) -> dict:
        """Provide one synthetic public product observation.

        Args:
            row: Public product record.
            last: Test residency flag.

        Returns:
            Valid candidate-only profile.
        """
        self.calls.append("profile")
        return dict(
            sale_item="耳机",
            role="main",
            host_object="",
            display_props=[],
            function="听音乐",
            appearance="黑色头戴结构",
            primary_unclear=False,
            evidence="商品可见",
        )

    def describe(self, row: dict, last: bool = False) -> dict:
        """Return a query-only synthetic caption.

        Args:
            row: Public image query.
            last: Test residency flag.

        Returns:
            Structured caption.
        """
        self.calls.append("describe")
        return dict(product="耳机", appearance="黑色头戴结构", unclear=False)

    def detect(self, row: dict) -> dict:
        """Abstain from cropping a synthetic product.

        Args:
            row: Public product record.

        Returns:
            Explicit whole-image fallback.
        """
        self.calls.append("detect")
        return dict(box=None)

    def embed(
        self, text: str | None = None, image: Path | None = None, query: bool = False
    ) -> np.ndarray:
        """Encode synthetic content independently of file names and arbitrary IDs.

        Args:
            text: Optional text.
            image: Optional image.
            query: Query instruction flag.

        Returns:
            Deterministic finite unit vector of the real dimension.
        """
        self.calls.append("embed")
        assert text or image is not None
        payload = text.encode() if text else image.read_bytes() if image else b""
        seed = int(hashlib.sha256(payload + str(query).encode()).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).normal(size=2048).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def regions(self, row: dict) -> dict:
        """Return synthetic baseline views.

        Args:
            row: Public product record.

        Returns:
            Aligned test vectors and metadata.
        """
        return dict(
            vectors=[self.embed(image=Path(row["image_path"])).tolist()],
            views=[dict(kind="global")],
        )

    def score(self, docs: list[dict], query: dict, caption: str) -> np.ndarray:
        """Score public content; optionally simulate interruption after a whole query.

        Args:
            docs: Public product inputs.
            query: Public query.
            caption: Query-only caption.

        Returns:
            Synthetic logits.
        """
        self.calls.append("score")
        self.scores += len(docs)
        if self.fail_after is not None and self.scores > self.fail_after:
            raise RuntimeError("simulated interruption")
        return np.array(
            [
                int(
                    hashlib.sha256(
                        (d["text"] + str(query.get("text")) + caption).encode()
                    ).hexdigest()[:10],
                    16,
                )
                / 1e12
                for d in docs
            ]
        )

    def close(self) -> None:
        """No real resources are held by the test double."""


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, Path, Path]:
    """Isolate runtime tests from actual models and all protected datasets.

    Args:
        tmp_path: Temporary project root.
        monkeypatch: Restore module roots after the test.

    Returns:
        Root, frozen prompt policy, public catalog and public queries.
    """
    policy = load(safety.PROJECT_ROOT / "configs/module2_selected_runtime.json")
    for module in (
        safety,
        evaluation_access,
        selected_runtime_io,
        selected_acceptance,
        selected_release,
    ):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    atomic_json(
        tmp_path / "configs/evaluation_registry.json",
        dict(
            rules=[
                dict(path_prefix="data/", role="development"),
                dict(path_prefix="artifacts/", role="development"),
                dict(
                    path_prefix="artifacts/heldout/",
                    role="reserved",
                    model_inference_allowed=False,
                ),
                dict(path_prefix="artifacts/gold/", role="reserved"),
            ]
        ),
    )
    records = []
    for i in range(25):
        path = tmp_path / f"data/{i}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(
            np.random.default_rng(i).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        ).save(path)
        records.append(dict(id=f"p{i:02}", title=f"耳机商品{i}", image_path=str(path)))
    catalog, queries = tmp_path / "data/catalog.json", tmp_path / "data/queries.json"
    atomic_json(catalog, records[:24])
    atomic_json(
        queries,
        [
            dict(id="text", mode="text", text="耳机"),
            dict(id="image", mode="image", image_path=records[24]["image_path"]),
        ],
    )
    return tmp_path, policy, catalog, queries


def test_index_predict_and_all_baselines_cover_the_full_gallery(
    workspace: tuple,
) -> None:
    """Real interfaces preserve every candidate and return stable cached outputs.

    Args:
        workspace: Isolated public input fixture.
    """
    root, policy, catalog, queries = workspace
    models = _Models(policy)
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=models)
    engine.index(catalog, root / "artifacts/index")
    result = engine.predict(
        root / "artifacts/index", queries, root / "artifacts/output"
    )
    for row in result["queries"].values():
        assert len(row["ranking"]) == len(set(row["ranking"])) == 24
        assert row["top10"] == row["ranking"][:10]
        assert set(row["baselines"]) == {"global", "region_title"}
        assert all(set(v) == set(row["ranking"]) for v in row["baselines"].values())
    calls = len(models.calls)
    assert (
        engine.predict(root / "artifacts/index", queries, root / "artifacts/output")
        == result
    )
    assert len(models.calls) == calls


def test_product_and_query_ids_do_not_change_non_tied_rankings(
    workspace: tuple,
) -> None:
    """Renaming IDs affects joins and output IDs, not model features or selection.

    Args:
        workspace: Isolated public inputs.
    """
    root, policy, catalog, queries = workspace
    first = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    first.index(catalog, root / "artifacts/i1")
    before = first.predict(root / "artifacts/i1", queries, root / "artifacts/o1")
    rows = load(catalog)
    mapping = {r["id"]: "renamed_" + r["id"] for r in rows}
    for row in rows:
        row["id"] = mapping[row["id"]]
    atomic_json(catalog, rows)
    q = load(queries)
    for row in q:
        row["id"] = "renamed_" + row["id"]
    atomic_json(queries, q)
    second = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    second.index(catalog, root / "artifacts/i2")
    after = second.predict(root / "artifacts/i2", queries, root / "artifacts/o2")
    assert all(
        [mapping[c] for c in row["ranking"]]
        == after["queries"]["renamed_" + key]["ranking"]
        for key, row in before["queries"].items()
    )


def test_interrupted_batch_resumes_only_missing_cross_scores(workspace: tuple) -> None:
    """A completed query is never recomputed during exact input/model continuation.

    Args:
        workspace: Isolated public inputs.
    """
    root, policy, catalog, queries = workspace
    models = _Models(policy)
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=models)
    engine.index(catalog, root / "artifacts/index")
    models.fail_after = 20
    with pytest.raises(RuntimeError, match="interruption"):
        engine.predict(root / "artifacts/index", queries, root / "artifacts/output")
    assert len(list((root / "artifacts/output/queries").glob("*.json"))) == 1
    next_models = _Models(policy)
    resumed = SelectedEngine(policy, root / "artifacts/cache", backend=next_models)
    result = resumed.predict(
        root / "artifacts/index", queries, root / "artifacts/output"
    )
    assert len(result["queries"]) == 2 and next_models.scores == 20
    atomic_json(queries, [dict(id="different", mode="text", text="新的查询")])
    with pytest.raises(ValueError, match="another immutable batch"):
        resumed.predict(root / "artifacts/index", queries, root / "artifacts/output")


@pytest.mark.parametrize(
    "defect", ["grade", "duplicate_id", "duplicate_image", "missing_image"]
)
def test_public_manifest_rejects_labels_or_invalid_sources(
    workspace: tuple, defect: str
) -> None:
    """Invalid public input is rejected before any model output can be produced.

    Args:
        workspace: Isolated public inputs.
        defect: Invalid or private field to inject.
    """
    _, _, catalog, _ = workspace
    rows = load(catalog)
    if defect == "grade":
        rows[0]["grade"] = 1
    elif defect == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif defect == "duplicate_image":
        rows[1]["image_path"] = rows[0]["image_path"]
    else:
        rows[0]["image_path"] = "data/not-found.png"
    atomic_json(catalog, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        public_records(catalog, "catalog", RuntimeAccess())


def test_cache_model_preprocessing_changes_and_tampering_are_detected(
    workspace: tuple,
) -> None:
    """Changing a model/prompt identity cannot reuse observations from another release.

    Args:
        workspace: Isolated output root.
    """
    root, _, _, _ = workspace
    cache = ContentCache(root / "artifacts/cache", dict(model="a", prompt="p"))
    key = cache.key("caption", dict(image="hash"))
    cache.put(key, dict(product="耳机"))
    assert cache.get(key) == dict(product="耳机")
    changed = ContentCache(cache.root, dict(model="a", prompt="changed"))
    assert changed.get(changed.key("caption", dict(image="hash"))) is None
    path = cache.root / (digest(key) + ".json")
    value = load(path)
    value["value"]["product"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="altered"):
        cache.get(key)


def test_index_and_source_mutations_fail_visibly(workspace: tuple) -> None:
    """Changing a public image or typed index metadata invalidates the frozen index.

    Args:
        workspace: Isolated public inputs.
    """
    root, policy, catalog, queries = workspace
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    folder = root / "artifacts/index"
    engine.index(catalog, folder)
    path = folder / "manifest.json"
    value = load(path)
    value["supported_types"]["p00"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="manifest was altered"):
        engine.predict(folder, queries, root / "artifacts/output")


def test_invalid_observation_and_model_mismatch_are_rejected(workspace: tuple) -> None:
    """No malformed VLM answer or substituted model is silently accepted.

    Args:
        workspace: Isolated public inputs.
    """
    root, policy, catalog, queries = workspace
    with pytest.raises(ValueError):
        validate_value(dict(sale_item="耳机"), policy["profile"]["schema"])
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    engine.index(catalog, root / "artifacts/index")
    changed = copy.deepcopy(policy)
    changed["teacher_digest"] = "different"
    other = SelectedEngine(changed, root / "artifacts/cache", backend=_Models(changed))
    with pytest.raises(ValueError, match="model or vector"):
        other.predict(root / "artifacts/index", queries, root / "artifacts/output")


def test_human_answer_changes_cannot_affect_inference(workspace: tuple) -> None:
    """Private labels are inaccessible to the ranker and absent from every feature key.

    Args:
        workspace: Isolated public inputs and reserved private gold path.
    """
    root, policy, catalog, queries = workspace
    gold = root / "artifacts/gold/answers.json"
    atomic_json(gold, dict(grade=1))
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    engine.index(catalog, root / "artifacts/index")
    before = engine.predict(root / "artifacts/index", queries, root / "artifacts/o1")
    atomic_json(gold, dict(grade=0))
    after = engine.predict(root / "artifacts/index", queries, root / "artifacts/o2")
    assert before["queries"] == after["queries"]
    with pytest.raises(PermissionError):
        RuntimeAccess().check(gold)


def test_invalid_vectors_and_modified_source_images_fail(workspace: tuple) -> None:
    """Neither nonunit embeddings nor mutated original media can be used silently.

    Args:
        workspace: Public content and scoped development outputs.
    """
    root, policy, catalog, queries = workspace
    models = _Models(policy)
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=models)
    engine.index(catalog, root / "artifacts/index")
    row = load(catalog)[0]
    Path(row["image_path"]).write_bytes(b"changed original")
    with pytest.raises(ValueError, match="source image changed"):
        engine.predict(root / "artifacts/index", queries, root / "artifacts/output")
    key = engine.cache.key(
        "query_embedding", dict(text="bad", image_sha256=None, query=True)
    )
    engine.cache.put(key, [0.0] * 2048)
    with pytest.raises(ValueError, match="unit embedding"):
        engine._embed(dict(text="bad"), "query_embedding", True)


def test_runtime_cannot_overwrite_protected_data(workspace: tuple) -> None:
    """Cache and result directories must obey the same isolation as model inputs.

    Args:
        workspace: Synthetic development inputs and protected destination rules.
    """
    root, policy, catalog, _ = workspace
    with pytest.raises(PermissionError):
        SelectedEngine(policy, root / "artifacts/gold/cache", backend=_Models(policy))
    engine = SelectedEngine(policy, root / "artifacts/cache", backend=_Models(policy))
    with pytest.raises(PermissionError):
        engine.index(catalog, root / "artifacts/heldout/new_index")
