"""General selected retrieval over public catalog/query inputs and real model stages."""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.regional_retrieval import region_configuration
from multimodal_ecommerce.retrieval_safeguards import compound_caption_conflict
from multimodal_ecommerce.safety import ensure_output_path, sha256_file
from multimodal_ecommerce.selected_models import (
    LocalModels,
    supported_type,
    validate_value,
)
from multimodal_ecommerce.selected_ranking import finish_ranking, select_pool
from multimodal_ecommerce.selected_runtime_io import (
    ContentCache,
    RuntimeAccess,
    atomic_json,
    digest,
    load,
    public_records,
)
from multimodal_ecommerce.simple_baseline import stable_order
from multimodal_ecommerce.visual_query import CAPTION_SCHEMA, caption_text


def _vector(value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if (
        vector.shape != (2048,)
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3)
    ):
        raise ValueError("A finite real unit embedding of dimension 2048 is required")
    return vector


def _usage(started: float) -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        seconds=time.perf_counter() - started,
        process_peak_rss_mib=rss / (2**20 if sys.platform == "darwin" else 1024),
    )


class SelectedEngine:
    """The selected policy with injectable model observations, never evaluator gold."""

    def __init__(
        self,
        policy: dict,
        cache: Path,
        backend: Any = None,
        access: RuntimeAccess | None = None,
    ) -> None:
        """Bind fixed policy, model stages and content-keyed cache.

        Args:
            policy: Frozen prompts and model preprocessing policy.
            cache: Owned content cache directory.
            backend: Actual local models, or a synthetic test double.
            access: Ordinary development or scoped acceptance access.
        """
        self.policy = policy
        self.access = access if access is not None else RuntimeAccess()
        self.models = backend if backend is not None else LocalModels(policy)
        self.cache = ContentCache(self.access.output(cache), self.models.identity())
        self.access.validate_model(self.cache.identity)

    def _observation(self, stage: str, content: dict, create: Any) -> Any:
        key = self.cache.key(stage, content)
        value = self.cache.get(key)
        if value is None:
            value = create()
            self.cache.put(key, value)
        return value

    def _embed(self, row: dict, stage: str, query: bool = False) -> np.ndarray:
        text = row.get("text")
        content = dict(
            text=text, image_sha256=None if text else row["image_sha256"], query=query
        )
        value = self._observation(
            stage,
            content,
            lambda: self.models.embed(
                text=text, image=None if text else Path(row["image_path"]), query=query
            ).tolist(),
        )
        return _vector(value)

    def index(self, catalog_path: Path, output: Path) -> dict:
        """Build a new whole/crop/title index from public catalog records.

        Args:
            catalog_path: Public catalog array with real image files.
            output: New immutable index directory.

        Returns:
            Index identity and runtime measurements.
        """
        started = time.perf_counter()
        records = sorted(
            public_records(catalog_path, "catalog", self.access), key=lambda r: r["id"]
        )
        output = self.access.output(output)
        content_id = digest(dict(records=records, model_policy=self.cache.identity))
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            existing = self._index(output)
            if existing["content_id"] != content_id:
                raise ValueError(
                    "Existing index is bound to different catalog/model inputs"
                )
            return existing
        kinds, crops = {}, {}
        try:
            for i, row in enumerate(records):
                profile = self._observation(
                    "sale_profile",
                    dict(image_sha256=row["image_sha256"], title=row["title"]),
                    lambda r=row, last=i == len(records) - 1: self.models.profile(
                        r, last
                    ),
                )
                validate_value(profile, self.policy["profile"]["schema"])
                kinds[row["id"]] = supported_type(row["title"], profile)
                if (i + 1) % 5 == 0:
                    print(f"Candidate profiles {i + 1}/{len(records)}", flush=True)
            self.models.close()
            for row in records:
                crops[row["id"]] = self._observation(
                    "subject_crop",
                    dict(image_sha256=row["image_sha256"], title=row["title"]),
                    lambda r=row: self.models.detect(r),
                )
            self.models.close()
            whole, crop, titles, regions, counts = [], [], [], [], []
            for i, row in enumerate(records):
                full = self._embed(row, "image_embedding")
                whole.append(full)
                titles.append(self._embed(dict(text=row["title"]), "title_embedding"))
                views = self._observation(
                    "baseline_regions",
                    dict(image_sha256=row["image_sha256"]),
                    lambda r=row: self.models.regions(r),
                )
                region_vectors = np.asarray(views["vectors"], dtype=np.float32)
                if region_vectors.ndim != 2 or not 1 <= len(region_vectors) <= 7:
                    raise ValueError("One to seven actual baseline views required")
                for v in region_vectors:
                    _vector(v)
                padded = np.zeros((7, 2048), dtype=np.float32)
                padded[: len(region_vectors)] = region_vectors
                regions.append(padded)
                counts.append(len(region_vectors))
                decision = crops[row["id"]]
                if decision["box"] is None:
                    crop.append(full)
                else:
                    key = dict(image_sha256=row["image_sha256"], crop=decision)

                    def encode_crop(r: dict = row, d: dict = decision) -> list:
                        """Encode an actual validated crop without changing its source.

                        Args:
                            r: Public source product.
                            d: Frozen detector geometry.

                        Returns:
                            Actual crop embedding as JSON-compatible floats.
                        """
                        with Image.open(r["image_path"]) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                            box = d["box"]
                            if list(image.size) != d["source_size"] or not (
                                0 <= box[0] < box[2] <= image.width
                                and 0 <= box[1] < box[3] <= image.height
                            ):
                                raise ValueError(
                                    "Crop geometry differs from its source image"
                                )
                            path = ensure_output_path(
                                self.cache.root / "crops" / (digest(key) + ".png")
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            image.crop(tuple(box)).save(path)
                        return self.models.embed(image=path).tolist()

                    crop.append(
                        _vector(self._observation("crop_embedding", key, encode_crop))
                    )
                if (i + 1) % 5 == 0:
                    print(f"Index vectors {i + 1}/{len(records)}", flush=True)
        finally:
            self.models.close()
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "vectors.npz",
            whole=np.stack(whole),
            crop=np.stack(crop),
            titles=np.stack(titles),
            regions=np.stack(regions),
        )
        manifest = dict(
            format="selected_index_v1",
            content_id=content_id,
            records=records,
            supported_types=kinds,
            region_counts=counts,
            model_policy=self.cache.identity,
            vectors_sha256=sha256_file(output / "vectors.npz"),
            source_manifest_sha256=sha256_file(catalog_path),
            runtime=_usage(started),
            model_events=self.models.events,
            formal_metrics=None,
        )
        manifest["data_id"] = digest(
            dict(
                content=content_id,
                vectors=manifest["vectors_sha256"],
                supported_types=kinds,
                region_counts=counts,
            )
        )
        manifest["manifest_sha256"] = digest(manifest)
        atomic_json(manifest_path, manifest)
        return manifest

    def _index(self, folder: Path) -> dict:
        path = self.access.check(folder / "manifest.json")
        manifest = load(path)
        checksum = manifest.pop("manifest_sha256")
        if digest(manifest) != checksum:
            raise ValueError("Index manifest was altered")
        manifest["manifest_sha256"] = checksum
        if (
            manifest["model_policy"] != self.cache.identity
            or sha256_file(self.access.check(folder / "vectors.npz"))
            != manifest["vectors_sha256"]
        ):
            raise ValueError("Index model or vector identity changed")
        for row in manifest["records"]:
            if (
                sha256_file(self.access.check(Path(row["image_path"])))
                != row["image_sha256"]
            ):
                raise ValueError("Indexed source image changed")
        return manifest

    def predict(self, index: Path, queries_path: Path, output: Path) -> dict:
        """Run new queries and resume only checkpoints with exactly matching inputs.

        Args:
            index: Existing immutable whole/crop/title index.
            queries_path: Public queries without relevance labels.
            output: Owned predictions/checkpoints directory.

        Returns:
            Full rankings, Top10, model provenance and runtime measurements.
        """
        started = time.perf_counter()
        manifest = self._index(index)
        queries = public_records(queries_path, "queries", self.access)
        docs = {r["id"]: r for r in manifest["records"]}
        ids = list(docs)
        with np.load(index / "vectors.npz", allow_pickle=False) as arrays:
            regions = arrays["regions"].copy()
            whole, crop, titles = (
                arrays[k].copy() for k in ("whole", "crop", "titles")
            )
        for bank in (whole, crop, titles):
            if bank.shape != (len(ids), 2048):
                raise ValueError("Index vectors do not align with catalog records")
            for v in bank:
                _vector(v)
        if regions.shape != (len(ids), 7, 2048) or not np.isfinite(regions).all():
            raise ValueError("Regional baseline vectors do not align")
        observations: dict = {}
        prepared: dict = {}
        results: dict = {}
        output = self.access.output(output)
        batch_id = digest(
            dict(
                index=manifest["data_id"],
                queries=queries,
                policy=self.cache.identity,
            )
        )
        old_run = output / "predictions.json"
        if old_run.exists():
            old = load(old_run)
            checksum = old.pop("result_sha256")
            if old["batch_id"] != batch_id or digest(old) != checksum:
                raise ValueError("Prediction output belongs to another immutable batch")
            old["result_sha256"] = checksum
            return old
        try:
            image_queries = [q for q in queries if q["mode"] == "image"]
            for i, q in enumerate(image_queries):
                caption = self._observation(
                    "query_caption",
                    dict(image_sha256=q["image_sha256"]),
                    lambda r=q, last=i == len(image_queries) - 1: self.models.describe(
                        r, last
                    ),
                )
                validate_value(caption, CAPTION_SCHEMA)
                observations[q["id"]] = dict(
                    caption=caption, supported_types=manifest["supported_types"]
                )
            self.models.close()
            for q in queries:
                qid = q["id"]
                vector = self._embed(q, "query_embedding", True)
                scores = vector @ whole.T
                if q["mode"] == "text":
                    scores = 0.5 * scores + 0.5 * (vector @ crop.T)
                excluded = set(q["exclusions"]) | {
                    cid
                    for cid, row in docs.items()
                    if row["image_sha256"] == q.get("image_sha256")
                }
                if not set(q["exclusions"]) <= set(ids):
                    raise ValueError(
                        "Query exclusions refer to foreign catalog products"
                    )
                base = [cid for cid in stable_order(ids, scores) if cid not in excluded]
                obs = observations.setdefault(qid, {})
                sims = np.einsum("d,nvd->nv", vector, regions)
                mask = (
                    np.arange(7)[None, :]
                    < np.asarray(manifest["region_counts"])[:, None]
                )
                cfg = region_configuration()
                logits = np.where(mask, sims / cfg["attention_temperature"], -np.inf)
                attention = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                attention /= np.sum(attention, axis=1, keepdims=True)
                attended = cfg["global_weight"] * (vector @ whole.T) + (
                    1 - cfg["global_weight"]
                ) * (attention * sims).sum(axis=1)
                baselines = {
                    "global": [
                        c
                        for c in stable_order(ids, vector @ whole.T)
                        if c not in excluded
                    ],
                    "region_title": [
                        c
                        for c in stable_order(
                            ids, 0.8 * attended + 0.2 * (vector @ titles.T)
                        )
                        if c not in excluded
                    ],
                }
                if q["mode"] == "text":
                    obs["title_order"] = [
                        cid
                        for cid in stable_order(ids, vector @ titles.T)
                        if cid not in excluded
                    ]
                proposal, active = select_pool(base, q, obs)
                prepared[qid] = dict(
                    base=base, proposal=proposal, active=active, baselines=baselines
                )
            self.models.close()
            for q in queries:
                qid, item = q["id"], prepared[q["id"]]
                checkpoint = (
                    output
                    / "queries"
                    / (digest(dict(batch=batch_id, query=qid)) + ".json")
                )
                if checkpoint.exists():
                    saved = load(checkpoint)
                    if (
                        saved["batch_id"] != batch_id
                        or saved["query_id"] != qid
                        or saved["result_sha256"] != digest(saved["result"])
                    ):
                        raise ValueError("Query checkpoint was altered")
                    results[qid] = saved["result"]
                    continue
                obs = observations[qid]
                description = (
                    caption_text(obs["caption"]) if q["mode"] == "image" else ""
                )
                needs_scores = (
                    q["mode"] == "text"
                    or item["active"]
                    and not compound_caption_conflict(obs["caption"])
                )
                obs["scores"] = {}
                if needs_scores:
                    for cid in item["proposal"][:20]:
                        doc = docs[cid]
                        content = dict(
                            query={
                                k: q.get(k) for k in ("mode", "text", "image_sha256")
                            },
                            description=description,
                            document=dict(
                                title=doc["title"], image_sha256=doc["image_sha256"]
                            ),
                        )
                        value = self._observation(
                            "cross_logit",
                            content,
                            lambda d=doc: float(
                                self.models.score(
                                    [
                                        dict(
                                            id="document",
                                            image_path=d["image_path"],
                                            text=d["title"],
                                        )
                                    ],
                                    q,
                                    description,
                                )[0]
                            ),
                        )
                        obs["scores"][cid] = float(value)
                ranked, trace = finish_ranking(
                    item["base"],
                    q,
                    obs,
                    {cid: docs[cid]["title"] for cid in item["base"]},
                )
                result = dict(
                    ranking=ranked,
                    top10=ranked[:10],
                    trace=trace,
                    baselines=item["baselines"],
                )
                results[qid] = result
                atomic_json(
                    checkpoint,
                    dict(
                        batch_id=batch_id,
                        query_id=qid,
                        result=result,
                        result_sha256=digest(result),
                    ),
                )
                print(f"Predicted {len(results)}/{len(queries)} queries", flush=True)
        finally:
            self.models.close()
        result = dict(
            format="selected_predictions_v1",
            batch_id=batch_id,
            index_content_id=manifest["content_id"],
            queries=results,
            model_policy=self.cache.identity,
            inputs=dict(
                index_sha256=sha256_file(index / "manifest.json"),
                query_manifest_sha256=sha256_file(queries_path),
            ),
            runtime=_usage(started),
            cache_hits=self.cache.hits,
            model_events=self.models.events,
            formal_metrics=None,
        )
        result["result_sha256"] = digest(result)
        atomic_json(old_run, result)
        return result
