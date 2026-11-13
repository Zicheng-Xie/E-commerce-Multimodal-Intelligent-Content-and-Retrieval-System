"""Live selected-policy search over the complete public development gallery."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.retrieval_safeguards import compound_caption_conflict
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.selected_models import LocalModels, validate_value
from multimodal_ecommerce.selected_ranking import finish_ranking, select_pool
from multimodal_ecommerce.selected_runtime_io import (
    ContentCache,
    RuntimeAccess,
    digest,
    load,
)
from multimodal_ecommerce.simple_baseline import stable_order
from multimodal_ecommerce.visual_query import CAPTION_SCHEMA, caption_text

ROOT = PROJECT_ROOT / "artifacts/module2_selected_runtime_20261110/showcase"


def reserved_signatures() -> list[dict]:
    """Read exclusion hashes only, without loading reserved media or query text.

    Returns:
        Byte/pixel/perceptual signatures corresponding to the sealed media hashes.
    """
    registry = load(PROJECT_ROOT / "configs/evaluation_registry.json")
    signatures: dict = {}
    for date in ("20261105", "20261110"):
        folder = PROJECT_ROOT / ("artifacts/module2_final_acceptance_" + date)
        seal = folder / "seal.json"
        exclusion = folder / "future_selection_exclusions.json"
        if not seal.exists() or not exclusion.exists():
            continue
        # Seal and exclusion metadata are used solely to deny model access.
        hashes = {
            v
            for k, v in load(seal)["files"].items()
            if Path(k).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
        require_data_access([exclusion], "exclusion_metadata", registry)
        for row in load(exclusion)["image_fingerprints"]:
            if row["sha256"] in hashes:
                signatures[row["sha256"]] = row
        missing = hashes - set(signatures)
        if missing:
            raise ValueError("Reserved image exclusion signatures are incomplete")
    return list(signatures.values())


class ShowcaseEngine:
    """Serialize MPS stages and reuse exact content observations, never judgments."""

    def __init__(self, root: Path = ROOT, backend: Any = None) -> None:
        """Verify the public bundle and fixed model identity before accepting searches.

        Args:
            root: Owned demonstration output directory.
            backend: Actual fixed local models or an injected synthetic test backend.
        """
        self.root = RuntimeAccess().output(root)
        self.access = RuntimeAccess()
        self.index = root / "public_index"
        self.manifest = load(self.access.check(self.index / "manifest.json"))
        expected = self.manifest["manifest_sha256"]
        if (
            digest({k: v for k, v in self.manifest.items() if k != "manifest_sha256"})
            != expected
        ):
            raise ValueError("Public gallery manifest changed")
        if self.manifest["labels_exported"] is not False:
            raise ValueError("Only public label-free gallery inputs are allowed")
        if sha256_file(self.index / "vectors.npz") != self.manifest["vectors_sha256"]:
            raise ValueError("Public gallery vectors changed")
        self.records = self.manifest["products"]
        self.ids = [r["id"] for r in self.records]
        if len(self.ids) != len(set(self.ids)) or len(self.ids) < 20:
            raise ValueError(
                "A unique complete gallery with at least twenty products is required"
            )
        self.docs = {r["id"]: r for r in self.records}
        self.assets: dict[str, Path] = {}
        self.asset_hashes: dict[str, str] = {}
        self.tokens = {}
        for row in self.records:
            if set(row) != {"id", "title", "image_path", "image_sha256"}:
                raise ValueError(
                    "Gallery records must contain only public product fields"
                )
            self.tokens[row["id"]] = self._asset(row)
        self.samples = []
        for row in self.manifest["samples"]:
            self.samples.append(dict(name=row["name"], token=self._asset(row)))
        with np.load(self.index / "vectors.npz", allow_pickle=False) as bank:
            self.whole, self.crop, self.titles = [
                bank[k].copy() for k in ("whole", "crop", "titles")
            ]
        for bank in (self.whole, self.crop, self.titles):
            if (
                bank.shape != (len(self.ids), 2048)
                or not np.isfinite(bank).all()
                or not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-3)
            ):
                raise ValueError("Invalid actual product vector bank")
        policy = load(PROJECT_ROOT / "configs/module2_selected_runtime.json")
        self.models = backend if backend is not None else LocalModels(policy)
        identity = self.models.identity()
        if identity != self.manifest["model_policy"]:
            raise ValueError("Live model identity differs from the public index")
        self.cache = ContentCache(
            self.access.output(root / "cache"),
            dict(
                models=identity,
                index=self.manifest["content_id"],
                implementation_sha256=sha256_file(Path(__file__)),
            ),
        )
        self.protected = reserved_signatures()
        self.lock = threading.Lock()

    def _asset(self, row: dict) -> str:
        path = self.access.check(PROJECT_ROOT / row["image_path"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Published source image changed")
        token = row["image_sha256"][:32]
        self.assets[token], self.asset_hashes[token] = path, row["image_sha256"]
        return token

    def card(self, cid: str, rank: int | None = None) -> dict:
        """Return original product facts without fabricated prices or relevance scores.

        Args:
            cid: Public catalog identifier.
            rank: Optional returned position.

        Returns:
            Original title, ID and a bounded local media URL.
        """
        row = self.docs[cid]
        return dict(
            id=cid,
            title=row["title"],
            rank=rank,
            image_url="/media/" + self.tokens[cid],
        )

    def _similarities(
        self, vector: np.ndarray, image: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return coarse visual scores and title scores for the query."""
        coarse = vector @ self.whole.T
        if not image:
            coarse = 0.5 * coarse + 0.5 * (vector @ self.crop.T)
        return coarse, vector @ self.titles.T

    def upload(self, data: bytes) -> dict:
        """Validate a bounded uploaded raster and block sealed byte/pixel/near matches.

        Args:
            data: Original image upload bytes.

        Returns:
            Local image token and preview URL.
        """
        if not data or len(data) > 8 * 1024 * 1024:
            raise ValueError("图片应小于 8 MB")
        checksum = hashlib.sha256(data).hexdigest()
        if any(r["sha256"] == checksum for r in self.protected):
            raise ValueError("该图片属于封存验收数据，不能用于演示检索")
        with Image.open(io.BytesIO(data)) as image:
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
                image.format or ""
            )
            if (
                suffix is None
                or image.width * image.height > 20_000_000
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("仅支持 2000 万像素以内的单张 JPG、PNG 或 WebP")
            image.verify()
        path = self.access.output(self.root / "uploads") / (checksum + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if is_exposed(fingerprints(path), self.protected, 4):
            path.unlink(missing_ok=True)
            raise ValueError("该图片与封存验收数据存在重复风险，已阻止模型读取")
        token = self._asset(dict(image_path=str(path), image_sha256=checksum))
        return dict(token=token, image_url="/media/" + token)

    def _observe(self, stage: str, content: dict, create: Callable[[], Any]) -> Any:
        key = self.cache.key(stage, content)
        value = self.cache.get(key)
        if value is None:
            value = create()
            self.cache.put(key, value)
        return value

    def _score_one(self, doc: dict, query: dict, description: str) -> float:
        return float(
            self.models.score(
                [
                    dict(
                        id="document",
                        text=doc["title"],
                        image_path=str(PROJECT_ROOT / doc["image_path"]),
                    )
                ],
                query,
                description,
            )[0]
        )

    def search(self, request: dict, progress: Callable[[str], None]) -> dict:
        """Run live arbitrary text/image queries through the frozen selected policy.

        Args:
            request: Exactly one public text or registered image token.
            progress: Callback for the browser's asynchronous progress display.

        Returns:
            Current Top10, complete ranking, actual timing and provenance.
        """
        if set(request) not in ({"text"}, {"image_token"}):
            raise ValueError("请提供文字或一张图片")
        value = next(iter(request.values()))
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError("请输入 1 至 512 字的商品需求")
        query: dict = dict(mode="text" if "text" in request else "image")
        if "text" in request:
            query["text"] = value.strip()
        else:
            if value not in self.assets:
                raise ValueError("未知图片，请重新上传")
            query.update(
                image_path=str(self.assets[value]),
                image_sha256=self.asset_hashes[value],
            )
        started = time.perf_counter()
        with self.lock:
            try:
                return self._search(query, progress, started)
            finally:
                self.models.close()

    def _search(
        self, query: dict, progress: Callable[[str], None], started: float
    ) -> dict:
        image = query["mode"] == "image"
        obs: dict = {}
        if image:
            path = self.access.check(Path(query["image_path"]))
            if sha256_file(path) != query["image_sha256"]:
                raise ValueError("查询图片已改变，请重新选择")
            progress("正在识别图片中的商品主体…")
            obs["caption"] = self._observe(
                "query_caption",
                dict(image_sha256=query["image_sha256"]),
                lambda: self.models.describe(query, True),
            )
            validate_value(obs["caption"], CAPTION_SCHEMA)
        self.models.close()
        progress("正在全图库编码检索…")
        vector = np.asarray(
            self._observe(
                "query_embedding",
                {k: query.get(k) for k in ("mode", "text", "image_sha256")},
                lambda: self.models.embed(
                    text=query.get("text"),
                    image=Path(query["image_path"]) if image else None,
                    query=True,
                ).tolist(),
            ),
            dtype=np.float32,
        )
        if (
            vector.shape != (2048,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3)
        ):
            raise ValueError("模型未返回有效查询向量")
        coarse, title_scores = self._similarities(vector, image)
        excluded = {
            r["id"]
            for r in self.records
            if r["image_sha256"] == query.get("image_sha256")
        }
        base = [c for c in stable_order(self.ids, coarse) if c not in excluded]
        if image:
            obs["supported_types"] = self.manifest["supported_types"]
        else:
            obs["title_order"] = [
                c
                for c in stable_order(self.ids, title_scores)
                if c not in excluded
            ]
        self.models.close()
        proposal, active = select_pool(base, query, obs)
        description = caption_text(obs["caption"]) if image else ""
        obs["scores"] = {}
        if not image or active and not compound_caption_conflict(obs["caption"]):
            for number, cid in enumerate(proposal[:20], 1):
                progress(f"正在图文精排候选 {number}/20…")
                doc = self.docs[cid]
                content = dict(
                    query={k: query.get(k) for k in ("mode", "text", "image_sha256")},
                    description=description,
                    title=doc["title"],
                    image_sha256=doc["image_sha256"],
                )
                obs["scores"][cid] = float(
                    self._observe(
                        "cross_logit",
                        content,
                        partial(self._score_one, doc, query, description),
                    )
                )
        ranking, trace = finish_ranking(
            base, query, obs, {c: self.docs[c]["title"] for c in base}
        )
        return dict(
            results=[self.card(c, i) for i, c in enumerate(ranking[:10], 1)],
            ranking=ranking,
            products=len(self.ids),
            searched=len(base),
            processing_seconds=time.perf_counter() - started,
            caption=obs.get("caption"),
            trace=trace,
            strategy="primary_public_conflict_matched_v1",
            formal_metrics=None,
            query_sha256=digest(query),
            index_id=self.manifest["content_id"],
        )
