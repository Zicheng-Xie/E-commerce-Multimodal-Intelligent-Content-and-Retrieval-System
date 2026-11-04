"""Resident, local-only development retrieval with protected acceptance inputs."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageOps

from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_constraints import constrain_scores
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.embedding import QwenEmbedder
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.regional_retrieval import (
    RegionalIndex,
    RegionalSearchRequest,
    pixel_roi,
)
from multimodal_ecommerce.runtime_device import project_path
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    predict_ranker,
)

BASE = PROJECT_ROOT / "artifacts/module2_machine_assist"
BUNDLE = BASE / "semantic_ranker"
RUNTIME = PROJECT_ROOT / "artifacts/module2_continuation/service/runtime"
METHODS = {
    "semantic_profile_blend",
    "constraint_aware_fusion",
    "balanced_logistic",
    "hard_negative_ranknet",
    "refined_logistic",
    "refined_ranknet",
}


def validate_search(value: Any) -> dict:
    """Validate search.

    Args:
        value: Input value to validate or transform.

    Returns:
        Mapping with fields: top_k, method, roi.

    Raises:
        ValueError: Unsupported query fields; Provide exactly one text or image query;
            Query must be nonempty text; Text query exceeds 512 characters; top_k must
            be an integer from 1 to 50; Unknown retrieval method; ROI needs four finite
            numbers and an image query; ROI must be ordered normalized coordinates.
    """
    if not isinstance(value, dict) or set(value) - {
        "text",
        "image_id",
        "image_path",
        "roi",
        "top_k",
        "method",
    }:
        raise ValueError("Unsupported query fields")
    modalities = [
        key for key in ["text", "image_id", "image_path"] if value.get(key) is not None
    ]
    if len(modalities) != 1:
        raise ValueError("Provide exactly one text or image query")
    key = modalities[0]
    if not isinstance(value[key], str) or not value[key].strip():
        raise ValueError("Query must be nonempty text")
    if key == "text" and len(value[key]) > 512:
        raise ValueError("Text query exceeds 512 characters")
    top_k = value.get("top_k", 10)
    method = value.get("method", "semantic_profile_blend")
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer from 1 to 50")
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("Unknown retrieval method")
    roi = value.get("roi")
    if roi is not None:
        if (
            key == "text"
            or not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in roi)
        ):
            raise ValueError("ROI needs four finite numbers and an image query")
        if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError("ROI must be ordered normalized coordinates")
    return {key: value[key].strip(), "top_k": top_k, "method": method, "roi": roi}


class ResidentRetrieval:
    """Hash-verify and load the model/index once, embedding every new query."""

    def __init__(self) -> None:
        """Initialize the component and validate its configured dependencies.

        Raises:
            ValueError: Published-title features required; Resident feature alignment
                mismatch; Acceptance registry changed.

        Notes:
            Writes derived artifacts through RUNTIME.mkdir, write_json.
        """
        started = time.perf_counter()
        self.lock = threading.Lock()
        self.instance_id = uuid.uuid4().hex[:12]
        self.closed = False
        self.requests = 0
        RUNTIME.mkdir(parents=True, exist_ok=True)
        for name in [
            "training_receipt.json",
            "feature_receipt.json",
            "runtime_receipt.json",
        ]:
            verify_files(
                BUNDLE, json.loads((BUNDLE / name).read_text(encoding="utf-8"))["files"]
            )
        self.cached = json.loads(
            (BUNDLE / "runtime_fingerprints.json").read_text(encoding="utf-8")
        )
        verify_files(PROJECT_ROOT, self.cached["source_files"])
        loaded_index, titles, _ = RegionalIndex.load(
            PROJECT_ROOT / "data/index_regions_products_v1"
        )
        self.index = cast(RegionalIndex, loaded_index)
        if titles is None:
            raise ValueError("Published-title features required")
        self.titles = titles
        self.metadata = json.loads(
            (BUNDLE / "feature_metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata["gallery_ids"] != [
            r["id"]
            # pylint: disable-next=no-member
            for r in self.index.records
        ] or set(self.cached["records"]) != set(self.metadata["gallery_ids"]):
            raise ValueError("Resident feature alignment mismatch")
        with np.load(BUNDLE / "profile_vectors.npz", allow_pickle=False) as bank:
            self.profiles = bank["vectors"].copy()
        with np.load(BUNDLE / "features.npz", allow_pickle=False) as bank:
            self.colors = bank["color_bank"].copy()
        self.models = json.loads(
            (BUNDLE / "candidate_models.json").read_text(encoding="utf-8")
        )
        refined = PROJECT_ROOT / "artifacts/module2_continuation/refined_ranker"
        if (refined / "receipt.json").exists():
            verify_files(
                refined,
                json.loads((refined / "receipt.json").read_text(encoding="utf-8"))[
                    "files"
                ],
            )
            models = json.loads(
                (refined / "candidate_models.json").read_text(encoding="utf-8")
            )
            self.models.update(
                refined_logistic=models["balanced_logistic"],
                refined_ranknet=models["hard_negative_ranknet"],
            )
        self.registry = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
        dataset = PROJECT_ROOT / "artifacts/module2_business_acceptance/dataset"
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        if sha256_file(self.registry) != manifest["reserved_registry_sha256"]:
            raise ValueError("Acceptance registry changed")
        verify_files(
            dataset,
            {"evaluator/lineage.jsonl": manifest["files"]["evaluator/lineage.jsonl"]},
        )
        self.registry_hash = sha256_file(self.registry)
        self.forbidden = [
            r["fingerprints"] for r in read_rows(dataset / "evaluator/lineage.jsonl")
        ]
        context = json.loads(
            (BASE / "development_inputs.json").read_text(encoding="utf-8")
        )
        self.image_queries = {
            q["id"]: project_path(q["image_path"])
            for q in context["queries"]
            if q["image_path"]
        }
        self.text_queries = [
            dict(id=q["id"], text=q["text"]) for q in context["queries"] if q["text"]
        ]
        self.assets: dict[str, Path] = {}
        self.images = {}
        self.focus_images = {}
        self.sample_images = {}
        # pylint: disable-next=no-member
        for row in self.index.records:
            self.images[row["id"]] = self._asset(
                row["id"] + ":main", Path(row["image_path"])
            )
            for view in row["views"]:
                self.focus_images[row["id"], view["kind"]] = self._asset(
                    row["id"] + ":" + view["kind"], Path(view["image_path"])
                )
        for qid, path in self.image_queries.items():
            self.sample_images[qid] = self._asset("query:" + qid, path)
        self.encoder = QwenEmbedder()
        self.encoder.encode(text="商品检索", query=True)
        self.startup_seconds = time.perf_counter() - started
        write_json(
            RUNTIME / "status.json",
            dict(
                state="ready",
                startup_seconds=self.startup_seconds,
                # pylint: disable-next=no-member
                products=len(self.index.records),
                resident=True,
                formal_acceptance=False,
            ),
        )

    def _asset(self, key: str, path: Path) -> str:
        token = hashlib.sha256(key.encode()).hexdigest()[:24]
        self.assets[token] = path
        return "/media/" + token

    def samples(self) -> dict:
        """Return samples for this workflow.

        Returns:
            Mapping with fields: text_queries, image_queries, methods,
            formal_acceptance.
        """
        return dict(
            text_queries=self.text_queries,
            image_queries=[
                dict(id=qid, image_url=self.sample_images[qid])
                for qid in self.image_queries
            ],
            methods=["semantic_profile_blend", "constraint_aware_fusion", *self.models],
            formal_acceptance=False,
        )

    def health(self) -> dict:
        """Return health for this workflow.

        Returns:
            Mapping with fields: state, instance_id, resident, products, requests,
            startup_seconds, result_cache, human_quality_validated, formal_acceptance.
        """
        return dict(
            state="closed" if self.closed else "ready",
            instance_id=self.instance_id,
            resident=not self.closed,
            # pylint: disable-next=no-member
            products=len(self.index.records),
            requests=self.requests,
            startup_seconds=self.startup_seconds,
            result_cache=False,
            human_quality_validated=False,
            formal_acceptance=False,
        )

    def search(self, value: dict) -> dict:
        """Return search for this workflow.

        Args:
            value: Input value to validate or transform.

        Returns:
            Mapping with fields: results, method, roi, resident, embedding_calls,
            result_cache_used, human_quality_validated, formal_acceptance,
            formal_metrics, processing_seconds.

        Raises:
            RuntimeError: Retrieval engine closed; Acceptance registry changed; restart
                to refresh guards.
            ValueError: Requested candidate is not available; Unknown development image
                ID; Image is missing or exceeds 20 MiB; Image dimensions or animation
                unsupported; Reserved acceptance image or duplicate risk is blocked.

        Notes:
            Writes derived artifacts through crop.save.
        """
        started = time.perf_counter()
        args = validate_search(value)
        with self.lock:
            if self.closed:
                raise RuntimeError("Retrieval engine closed")
            if (
                args["method"]
                not in {"semantic_profile_blend", "constraint_aware_fusion"}
                and args["method"] not in self.models
            ):
                raise ValueError("Requested candidate is not available")
            if sha256_file(self.registry) != self.registry_hash:
                raise RuntimeError(
                    "Acceptance registry changed; restart to refresh guards"
                )
            image = None
            signature = None
            temporary = None
            text = args.get("text")
            if args.get("image_id"):
                if args["image_id"] not in self.image_queries:
                    raise ValueError("Unknown development image ID")
                image = self.image_queries[args["image_id"]]
            elif args.get("image_path"):
                image = Path(args["image_path"])
                image = image if image.is_absolute() else PROJECT_ROOT / image
                image = image.resolve()
                image.relative_to(PROJECT_ROOT)
            if image is not None:
                if not image.is_file() or image.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError("Image is missing or exceeds 20 MiB")
                with Image.open(image) as source:
                    if (
                        source.width * source.height > 20_000_000
                        or getattr(source, "n_frames", 1) != 1
                    ):
                        raise ValueError("Image dimensions or animation unsupported")
                signature = fingerprints(image)
                if is_exposed(signature, self.forbidden, 4):
                    raise ValueError(
                        "Reserved acceptance image or duplicate risk is blocked"
                    )
            try:
                if args["roi"] is not None:
                    assert image is not None
                    with Image.open(image) as source:
                        crop = ImageOps.exif_transpose(source).convert("RGB")
                        crop = crop.crop(pixel_roi(args["roi"], crop.size))
                    with tempfile.NamedTemporaryFile(
                        dir=RUNTIME, suffix=".png", delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                    crop.save(temporary)
                    image = temporary
                query = self.encoder.encode(text=text, image=image, query=True)
                x = feature_matrix(
                    SemanticRankerFeatureMatrixInputs(
                        self.index,
                        self.titles,
                        self.profiles,
                        self.metadata["profile_texts"],
                        query,
                        text=text,
                        image=image,
                        color_bank=self.colors,
                    )
                )
                scores = (
                    0.5 * x[:, 1] + 0.2 * x[:, 3] + 0.3 * x[:, 4]
                    if args["method"]
                    in {"semantic_profile_blend", "constraint_aware_fusion"}
                    else predict_ranker(x, self.models[args["method"]])
                )
                constraints = None
                if args["method"] == "constraint_aware_fusion":
                    scores, constraints = constrain_scores(
                        text,
                        [
                            r["title"]
                            # pylint: disable-next=no-member
                            for r in self.index.records
                        ],
                        self.metadata["profile_texts"],
                        scores,
                    )
                excluded = {
                    cid
                    for cid, sig in self.cached["records"].items()
                    if signature is not None
                    and (
                        signature["sha256"] == sig["sha256"]
                        or signature["pixels"] == sig["pixels"]
                    )
                }
                ranked_rows: list[dict] = self.index.search(
                    RegionalSearchRequest(
                        query,
                        # pylint: disable-next=no-member
                        top_k=len(self.index.records),
                        strategy="attention",
                        exclude_ids=excluded,
                    )
                )
                positions = {
                    r["id"]: i
                    # pylint: disable-next=no-member
                    for i, r in enumerate(self.index.records)
                }
                # pylint: disable-next=no-member
                ranked_rows.sort(
                    key=lambda r: (-float(scores[positions[r["id"]]]), r["id"])
                )
                cards = []
                for rank, row in enumerate(ranked_rows[: args["top_k"]], 1):
                    i = positions[row["id"]]
                    cards.append(
                        dict(
                            rank=rank,
                            id=row["id"],
                            title=row["title"],
                            category=row["category"],
                            score=float(scores[i]),
                            image_url=self.images[row["id"]],
                            focus_image_url=self.focus_images[
                                row["id"], row["focus_view"]["kind"]
                            ],
                            focus_kind=row["focus_view"]["kind"],
                            constraint_evidence=(
                                constraints[i] if constraints is not None else None
                            ),
                            components=dict(
                                global_image=float(x[i, 0]),
                                regional_attention=float(x[i, 1]),
                                published_title=float(x[i, 3]),
                                semantic_profile=float(x[i, 4]),
                            ),
                        )
                    )
                self.requests += 1
                result = dict(
                    results=cards,
                    method=args["method"],
                    roi=args["roi"],
                    resident=True,
                    embedding_calls=1,
                    result_cache_used=False,
                    human_quality_validated=False,
                    formal_acceptance=False,
                    formal_metrics=None,
                    processing_seconds=time.perf_counter() - started,
                )
                record = dict(
                    request=self.requests,
                    instance_id=self.instance_id,
                    modality="text" if text else "image",
                    method=args["method"],
                    query_sha256=hashlib.sha256(
                        json.dumps(args, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest(),
                    top_ids=[r["id"] for r in cards],
                    processing_seconds=result["processing_seconds"],
                    formal_acceptance=False,
                )
                with (RUNTIME / "requests.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                return result
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def close(self) -> None:
        """Release the resources held by this component.

        Notes:
            Writes derived artifacts through write_json.
        """
        with self.lock:
            self.encoder.close()
            self.closed = True
            write_json(
                RUNTIME / "status.json",
                dict(state="closed", requests=self.requests, formal_acceptance=False),
            )
