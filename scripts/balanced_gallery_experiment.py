"""Same 315-product gallery, exact full-model evidence and frozen fusion models.

All queries and both galleries remain exposed development. Native relation
labels enter only evaluation, after complete rankings have been persisted.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import balanced_pair_review as review
import balanced_supplement as acquisition
import balanced_supplement_features as supplement
import recover_balanced_media as recovery
import run  # noqa: F401
from caption_roi_ranker import CaptionRoiRankerFuseCropsInputs, fuse_crops
from compare_fgclip2_retrieval import read_array
from dual_context_pair_cache import valid_identity
from dual_path_ranker import DualPathRankerFuseContextsInputs, fuse_contexts
from learned_fusion_experiment import load, once, rel
from learned_fusion_model import FEATURE_NAMES, predict
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.contracts import FeatureBanks
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.fgclip_retrieval import (
    FgclipRetrievalAuxiliaryScoresInputs,
    auxiliary_scores,
)
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.intent_retrieval import (
    balanced_pool,
    normalized_ranks,
    public_evidence,
    recall_channels,
    rerank_complete,
)
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.query_fusion import (
    QueryFusionVisualChannelsInputs,
    visual_channels,
)
from multimodal_ecommerce.regional_retrieval import RegionalIndex
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerFeatureMatrixInputs,
    feature_matrix,
    ranked_ids,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_subject_experiment import bank_path, pair_identity
from querywise_paired_bounds import paired_querywise

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_gallery"
DATA = PROJECT_ROOT / "artifacts/module2_primary_subjects"
WHOLE = PROJECT_ROOT / "artifacts/module2_dual_path_development"
CROP = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
TRAINED = PROJECT_ROOT / "artifacts/module2_learned_fusion"
COHORTS = ("tune", "regression", "fresh")
NEW = (
    "frozen_pointwise_family_masked",
    "frozen_ranknet_family_masked",
    "frozen_ranknet_blend_family_masked",
)
PRIMARY = NEW[2]
BASELINES = (
    "global",
    "region_title",
    "cross20",
    "dual_context_union",
    "whole_crop_context_union",
)
METHODS = (*BASELINES, *NEW)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [
    (PRIMARY, NEW[1]),
    (NEW[1], NEW[0]),
]
SOURCES = [
    "scripts/balanced_gallery_experiment.py",
    "tests/test_balanced_gallery_experiment.py",
    "scripts/balanced_supplement_features.py",
    "scripts/balanced_pair_review.py",
    "scripts/balanced_supplement.py",
    "scripts/recover_balanced_media.py",
    "scripts/learned_fusion_experiment.py",
    "scripts/learned_fusion_model.py",
    "scripts/primary_subject_experiment.py",
    "scripts/caption_roi_ranker.py",
    "scripts/dual_path_ranker.py",
    "scripts/dual_context_pair_cache.py",
    "scripts/compare_fgclip2_retrieval.py",
    "scripts/querywise_paired_bounds.py",
]


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(ROOT / name, dict(files={rel(p): sha256_file(p) for p in paths}, **extra))


def checked(name: str) -> Any:
    """Return checked for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    result = load(ROOT / name)
    verify_files(PROJECT_ROOT, result["files"])
    return result


def frozen_folds() -> dict:
    """Return frozen folds for this workflow.

    Returns:
        Mapping with fields: queries, gallery, query_families, gallery_families.
    """
    meta = load(TRAINED / "feature_metadata.json")
    with np.load(TRAINED / "feature_bank.npz", allow_pickle=False) as b:
        qfold, gfold = b["query_folds"], b["gallery_folds"]
    return dict(
        queries=dict(zip(meta["query_ids"], qfold.tolist())),
        gallery=dict(zip(meta["gallery_ids"], gfold.tolist())),
        query_families=dict(zip(meta["query_ids"], meta["query_families"])),
        gallery_families=dict(zip(meta["gallery_ids"], meta["gallery_families"])),
    )


def check_new_families() -> list:
    """Return check new families for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    old = load(
        PROJECT_ROOT
        / "artifacts/module2_roi_semantic_gate/post_experiment_exclusion_snapshot.json"
    )
    rows = read_rows(recovery.ROOT / "selected.evaluator.jsonl")
    for row in rows:
        assert row["family_keys"] and not set(row["family_keys"]) & set(
            old["family_keys"]
        )
        assert row["id"] not in old["ids"]
        assert not is_exposed(row["fingerprints"], old["image_fingerprints"], 4)
    return [r["id"] for r in rows]


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    supplement.guard()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        ROOT / "engineering_checks.pre.json",
        supplement.ROOT / "protocol.json",
        review.ROOT / "protocol.json",
        recovery.ROOT / "selected.evaluator.jsonl",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
        TRAINED / "fold_models.json",
        TRAINED / "fold_audits.json",
        TRAINED / "feature_metadata.json",
        TRAINED / "feature_bank.npz",
    ]
    for root in (DATA, WHOLE, CROP, TRAINED):
        seal = load(root / "completion_receipt.json")
        verify_files(root, seal["files"])
        verify_files(PROJECT_ROOT, seal["source_files"])
        verify_files(PROJECT_ROOT, seal["project_files"])
        paths.append(root / "completion_receipt.json")
    once(ROOT / "public_fold_map.json", frozen_folds())
    paths.append(ROOT / "public_fold_map.json")
    new_ids = check_new_families()
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            methods=METHODS,
            primary=PRIMARY,
            comparisons=COMPARISONS,
            gallery_per_cohort=315,
            queries=40,
            fixed_k=10,
            precision_required=0.60,
            recall_preferred=0.85,
            additions=new_ids,
            all_19_technical_successes_retained=True,
            query_and_original_gallery_features="exact original vectors; all full-gallery public scores and pools recomputed on 315 products",
            neural_models="unchanged complete Qwen3.5 labeler, Qwen embedding/reranker, DINO and FG; no simulated, missing or reduced signals",
            candidate_profiles="candidate-only original image and title, independent of queries and relevance judgments",
            pretrained_fusion="same 18 frozen stage27 fold models; no refit, new labels never used in prediction",
            old_candidate_fold="original held query fold and original held gallery fold",
            new_candidate_fold="source families verified absent from all prior training exposure; equal calibrated vote of three gallery-fold models at held query fold",
            blend="0.65 frozen RankNet rank + 0.35 equal original/new base rank",
            scope="All/Chinese text/image for A/B/C; same complete gallery and labels; include nulls and zero-positive queries",
            comparability="315-gallery numbers only compare methods in this root; sealed 296-gallery histories remain separate",
            no_after_result_tuning=True,
            model_hyperparameter_search=False,
            default_promotion_allowed=False,
            independent_acceptance=False,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="comparison_rules_frozen_full_features_pending",
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    protocol = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    assert protocol["methods"] == list(METHODS) and protocol["primary"] == PRIMARY
    supplement.guard()


def feature_banks(
    cohort: str,
) -> tuple[RegionalIndex, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return feature banks for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    folder = supplement.ROOT / ("tune" if cohort == "fresh" else cohort) / "index"
    manifest = load(folder / "manifest.json")
    verify_files(folder, manifest["files"])
    records = load(folder / "records.json")
    assert len(records) == 315 and len({d["id"] for d in records}) == 315
    assert records[:296] == load(
        DATA / ("tune" if cohort == "fresh" else cohort) / "index/records.json"
    )
    for doc in records:
        for view in doc["views"]:
            assert sha256_file(Path(view["image_path"])) == view["sha256"]
    with np.load(folder / "vectors.npz", allow_pickle=False) as b:
        arrays = {k: b[k].copy() for k in ("regions", "texts", "profiles")}
    original = DATA / ("tune" if cohort == "fresh" else cohort) / "index/vectors.npz"
    with np.load(original, allow_pickle=False) as old:
        assert all(np.array_equal(arrays[k][:296], old[k]) for k in arrays)
    return (
        RegionalIndex(records, arrays["regions"], manifest["regions"]),
        arrays["texts"],
        arrays["profiles"],
        load(folder / "profile_texts.json"),
        np.load(folder / "colors.npy", allow_pickle=False),
    )


def features(
    cohort: str, query: dict, banks: FeatureBanks, *, new_caption: bool = False
) -> dict:
    """Return features for this workflow.

    Args:
        cohort: Cohort used by this operation.
        query: Query record or representation used by this operation.
        banks: Banks used by this operation.
        new_caption: New caption used by this operation.

    Returns:
        Mapping with fields: ids, base, pool, features, description, offsets.
    """
    index, titles, profiles, texts, colors = banks
    ids = [d["id"] for d in index.records]
    with np.load(
        bank_path(cohort, query["id"]) / "vectors.npz", allow_pickle=False
    ) as b:
        views, caption = b["views"].copy(), b["caption"].copy()
    image = Path(query["image_path"]) if query["image_path"] else None
    matrix = feature_matrix(
        SemanticRankerFeatureMatrixInputs(
            index,
            titles,
            profiles,
            texts,
            views[0],
            text=query["text"],
            image=image,
            color_bank=colors,
        )
    )
    description, offsets = "", None
    if image:
        if new_caption:
            description = load(
                DATA / "query_captions" / cohort / (query["id"] + ".json")
            )["text"]
            with np.load(
                DATA / "new_query_features" / cohort / (query["id"] + ".npz"),
                allow_pickle=False,
            ) as b:
                caption = b["caption"].copy()
        else:
            description = load(bank_path(cohort, query["id"]) / "description.json")[
                "text"
            ]
        fusion = visual_channels(
            QueryFusionVisualChannelsInputs(
                index,
                titles,
                profiles,
                matrix,
                views,
                caption,
                description,
                np.arange(len(ids)),
            )
        )
        base, pool = np.asarray(fusion["fused_scores"]), fusion["pool"]
    else:
        base, channels = recall_channels(ids, matrix)
        offsets = np.array(
            [
                public_evidence(query["text"], d["title"], p)["offset"]
                for d, p in zip(index.records, texts)
            ]
        )
        channels.append(ranked_ids(ids, normalized_ranks(ids, base) + offsets))
        pool = balanced_pool(channels, 60)
    return dict(
        ids=ids,
        base=base.tolist(),
        pool=pool[:20],
        features=matrix.tolist(),
        description=description,
        offsets=None if offsets is None else offsets.tolist(),
    )


def fg_banks(cohort: str) -> dict:
    """Return fg banks for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Mapping with fields: ids, titles, whole, dense, roi, roi_dense.
    """
    data = load(DATA / "inputs.json")["cohorts"][cohort]
    paths, roi = load(DATA / "feature_paths.json"), load(
        DATA / "roi_feature_index.json"
    )
    rows = [
        dict(
            id=d["id"],
            whole=paths["images"][d["image_key"]],
            title=paths["texts"][d["title_key"]],
            roi=roi[d["roi_request"]]["file"],
        )
        for d in data["gallery"]
    ]
    rows += load(supplement.ROOT / "addition_FG_paths.json")
    return dict(
        ids=[r["id"] for r in rows],
        titles=np.stack(
            [read_array(PROJECT_ROOT / r["title"], "vector") for r in rows]
        ),
        whole=np.stack(
            [read_array(PROJECT_ROOT / r["whole"], "global_vector") for r in rows]
        ),
        dense=[read_array(PROJECT_ROOT / r["whole"], "dense") for r in rows],
        roi=np.stack(
            [read_array(PROJECT_ROOT / r["roi"], "global_vector") for r in rows]
        ),
        roi_dense=[read_array(PROJECT_ROOT / r["roi"], "dense") for r in rows],
    )


def fg_scores(cohort: str, query: dict, bank: dict) -> dict:
    """Return fg scores for this workflow.

    Args:
        cohort: Cohort used by this operation.
        query: Query record or representation used by this operation.
        bank: Bank used by this operation.

    Returns:
        Mapping with fields: ids, whole, roi.
    """
    paths, roi = load(DATA / "feature_paths.json"), load(
        DATA / "roi_feature_index.json"
    )
    q = next(
        r
        for r in load(DATA / "inputs.json")["cohorts"][cohort]["queries"]
        if r["id"] == query["id"]
    )
    text = (
        read_array(PROJECT_ROOT / paths["texts"][q["text_key"]], "vector")
        if q["text_key"]
        else None
    )
    box = (
        read_array(PROJECT_ROOT / paths["box_texts"][q["text_key"]], "vector")
        if q["text_key"]
        else None
    )
    image = (
        read_array(PROJECT_ROOT / paths["images"][q["image_key"]], "global_vector")
        if q["image_key"]
        else None
    )
    whole = auxiliary_scores(
        FgclipRetrievalAuxiliaryScoresInputs(
            bank["ids"],
            bank["whole"],
            bank["titles"],
            bank["dense"],
            query_image=image,
            query_text=text,
            query_box=box,
        )
    )
    regional = (
        auxiliary_scores(
            FgclipRetrievalAuxiliaryScoresInputs(
                bank["ids"],
                bank["roi"],
                bank["titles"],
                bank["roi_dense"],
                query_image=read_array(
                    PROJECT_ROOT / roi[q["roi_request"]]["file"], "global_vector"
                ),
                query_text=text,
                query_box=box,
            )
        )
        if image is not None
        else None
    )
    return dict(ids=bank["ids"], whole=whole, roi=regional)


def crop_inputs(cohort: str, query: dict, docs: dict) -> tuple[dict, dict]:
    """Return crop inputs for this workflow.

    Args:
        cohort: Cohort used by this operation.
        query: Query record or representation used by this operation.
        docs: Docs used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    assets = load(CROP / "crop_inputs.json")["cohorts"][cohort]
    q = next(a for a in assets["queries"] if a["id"] == query["id"])
    originals = {a["id"]: a for a in assets["gallery"]}
    additions = {d["id"]: i for i, d in enumerate(supplement.selected())}
    actual = {}
    for cid, doc in docs.items():
        if cid in originals:
            selected = PROJECT_ROOT / originals[cid]["selected_image"]
            assert sha256_file(selected) == originals[cid]["selected_sha256"]
        else:
            i = additions[cid]
            detection = load(supplement.ROOT / f"detections/{i:03d}/result.json")
            selected = (
                supplement.ROOT / f"detections/{i:03d}/zoom.png"
                if detection["decision"]["status"] == "roi"
                else Path(doc["image_path"])
            )
        actual[cid] = dict(doc, image_path=str(selected))
    selected = PROJECT_ROOT / q["selected_image"]
    assert sha256_file(selected) == q["selected_sha256"]
    return dict(query, image_path=str(selected)), actual


def add_cache(cache: dict, raw: dict, path: Path) -> None:
    """Apply the add cache workflow step.

    Args:
        cache: Cache used by this operation.
        raw: Raw used by this operation.
        path: Path of the input or output artifact used by this operation.

    Notes:
        Updates the supplied cache in place.
    """
    assert len(raw["pool"]) == len(set(raw["pool"])) == len(raw["logits"])
    assert set(raw["pair_identities"]) == set(raw["pool"])
    for cid, score in zip(raw["pool"], raw["logits"]):
        identity = raw["pair_identities"][cid]
        assert valid_identity(identity) and math.isfinite(score)
        key = identity["sha256"]
        record = dict(
            identity=identity,
            logit=score,
            source_file=rel(path),
            source_sha256=sha256_file(path),
        )
        if key in cache:
            assert cache[key]["identity"] == identity and cache[key]["logit"] == score
        else:
            cache[key] = record


def plan() -> None:
    """Return plan for this workflow.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    for name in (
        "qwen_receipt.json",
        "detection_receipt.json",
        "fg_receipt.json",
        "profile_receipt.json",
    ):
        supplement.guard_receipt(name)
    if (ROOT / "plan_receipt.json").exists():
        checked("plan_receipt.json")
        return
    jobs = []
    cache: dict = {}
    paths = []
    for source in (DATA, WHOLE, CROP):
        seal = load(source / "completion_receipt.json")
        for c in COHORTS:
            for path in sorted((source / c / "runs").glob("*/*cross.json")):
                assert seal["files"][
                    path.relative_to(source).as_posix()
                ] == sha256_file(path)
                add_cache(cache, load(path), path)
                paths.append(path)
    for c in COHORTS:
        banks = feature_banks(c)
        docs = {
            d["id"]: dict(id=d["id"], title=d["title"], image_path=d["image_path"])
            for d in banks[0].records
        }
        once(ROOT / c / "catalog.public.json", list(docs.values()))
        once(ROOT / c / "queries.public.json", load(DATA / c / "queries.public.json"))
        fgb = fg_banks(c)
        assert fgb["ids"] == list(docs)
        for q in load(DATA / c / "queries.public.json"):
            folder = ROOT / c / "runs" / q["id"]
            a = features(c, q, banks)
            once(folder / "features.json", a)
            if q["image_path"]:
                b = features(c, q, banks, new_caption=True)
                once(folder / "caption_features.json", b)
                pool = list(dict.fromkeys(a["pool"] + b["pool"]))
                actual_q, actual_docs = crop_inputs(c, q, docs)
                contexts = [
                    ("old", q, a["description"], docs),
                    ("new", q, b["description"], docs),
                    ("crop", actual_q, b["description"], actual_docs),
                ]
                once(folder / "fg_scores.json", fg_scores(c, q, fgb))
            else:
                pool = a["pool"]
                contexts = [("old", q, a["description"], docs)]
            for context, query, description, documents in contexts:
                for cid in pool:
                    identity = pair_identity(query, description, documents[cid])
                    jobs.append(
                        dict(
                            cohort=c,
                            query_id=q["id"],
                            context=context,
                            candidate_id=cid,
                            identity=identity,
                            query=query,
                            description=description,
                            document=documents[cid],
                        )
                    )
            once(folder / "shared_pool.json", pool)
    once(ROOT / "pair_plan.json", jobs)
    once(ROOT / "exact_cache.json", cache)
    paths += [ROOT / "pair_plan.json", ROOT / "exact_cache.json"]
    paths += [p for c in COHORTS for p in sorted((ROOT / c).rglob("*.json"))]
    receipt(
        "plan_receipt.json",
        paths,
        planned_context_pairs=len(jobs),
        new_labels_used=False,
    )


def infer() -> bool:
    """Run model inference using the configured inputs.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("plan_receipt.json")
    cache = load(ROOT / "exact_cache.json")
    engine = None
    jobs = load(ROOT / "pair_plan.json")
    new, reused = 0, 0
    try:
        for i, job in enumerate(jobs):
            identity = job["identity"]
            key = identity["sha256"]
            path = ROOT / "pairs" / (key + ".json")
            if not path.exists():
                if key in cache:
                    value = cache[key]
                    assert value["identity"] == identity
                    assert (
                        sha256_file(PROJECT_ROOT / value["source_file"])
                        == value["source_sha256"]
                    )
                    once(path, dict(value, reused=True))
                else:
                    if engine is None:
                        if not acquisition.gpu_available():
                            return False
                        engine = DescribedImageReranker()
                    doc = job["document"]
                    query = job["query"]
                    pair = [
                        dict(
                            id=doc["id"],
                            text=doc["title"],
                            image_path=doc["image_path"],
                        )
                    ]
                    image = Path(query["image_path"]) if query["image_path"] else None
                    values = (
                        engine.score_described(
                            pair, query_image=image, description=job["description"]
                        )
                        if image and job["description"]
                        else engine.score(
                            pair, query_image=image, query_text=query["text"]
                        )
                    )
                    assert len(values) == 1 and np.isfinite(values).all()
                    once(
                        path,
                        dict(
                            identity=identity,
                            logit=float(values[0]),
                            reused=False,
                            timing=engine.adapter.timings[-1],
                        ),
                    )
            record = load(path)
            assert record["identity"] == identity and math.isfinite(record["logit"])
            reused += int(record["reused"])
            new += int(not record["reused"])
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(dict(context_pairs=i + 1, total=len(jobs)), flush=True)
        once(
            ROOT / "reranker_runtime.json",
            engine.diagnostics() if engine else dict(no_new_forward_calls=True),
        )
    finally:
        if engine:
            engine.close()
    receipt(
        "reranker_receipt.json",
        [ROOT / "reranker_runtime.json", *sorted((ROOT / "pairs").glob("*.json"))],
        context_positions=len(jobs),
        new_context_positions=new,
        reused_context_positions=reused,
        unique_real_new_calls=sum(
            not load(p)["reused"] for p in (ROOT / "pairs").glob("*.json")
        ),
    )
    return True


@dataclass(frozen=True)
class BalancedGalleryExperimentTransferScoresInputs:
    """Input values for transfer scores.

    Attributes:
        x: X used by this operation.
        pool: Pool used by this operation.
        query_key: Query key used by this operation.
        folds: Folds used by this operation.
        models: Models used by this operation.
        new_ids: New ids used by this operation.
    """

    x: np.ndarray
    pool: list[str]
    query_key: str
    folds: dict
    models: dict
    new_ids: list[str]


def transfer_scores(options: BalancedGalleryExperimentTransferScoresInputs) -> dict:
    """No labels or gradient fitting. Old cells keep both original held folds.

    Args:
        options: Validated input fields described by
            BalancedGalleryExperimentTransferScoresInputs.

    Returns:
        Result mapping produced by the workflow.
    """
    x = options.x
    pool = options.pool
    query_key = options.query_key
    folds = options.folds
    models = options.models
    new_ids = options.new_ids
    assert x.shape == (len(pool), len(FEATURE_NAMES)) and np.isfinite(x).all()
    hq = folds["queries"][query_key]
    out: dict = {name: [] for name in ("pointwise", "ranknet")}
    assert set(pool) <= set(folds["gallery"]) | set(new_ids)
    for name in out:
        for i, cid in enumerate(pool):
            held = [folds["gallery"][cid]] if cid in folds["gallery"] else [0, 1, 2]
            out[name].append(
                float(
                    np.mean(
                        [
                            predict(x[i : i + 1], models[f"{name}_q{hq}_g{hg}"])[0]
                            for hg in held
                        ]
                    )
                )
            )
    return out


def orders() -> tuple[dict, dict]:
    """Return orders for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    folds = load(ROOT / "public_fold_map.json")
    models = load(TRAINED / "fold_models.json")
    outputs, details = {}, {}
    for c in COHORTS:
        output, evidence = {}, {}
        for q in load(ROOT / c / "queries.public.json"):
            folder = ROOT / c / "runs" / q["id"]
            a = load(folder / "features.json")
            ids = a["ids"]
            pool = load(folder / "shared_pool.json")
            context: dict = {}
            for job in (
                j
                for j in load(ROOT / "pair_plan.json")
                if j["cohort"] == c and j["query_id"] == q["id"]
            ):
                context.setdefault(job["context"], {})[job["candidate_id"]] = load(
                    ROOT / "pairs" / (job["identity"]["sha256"] + ".json")
                )["logit"]
            old20 = [context["old"][cid] for cid in a["pool"]]
            baseline = rerank_complete(
                ids,
                np.asarray(a["base"]),
                a["pool"],
                np.asarray(old20),
                np.asarray(a["offsets"]) if a["offsets"] is not None else None,
            )
            matrix = np.asarray(a["features"])
            result = dict(
                global_=ranked_ids(ids, matrix[:, 0]),
                region_title=ranked_ids(ids, 0.8 * matrix[:, 1] + 0.2 * matrix[:, 3]),
                cross20=baseline,
            )
            result["global"] = result.pop("global_")
            if q["image_path"]:
                b = load(folder / "caption_features.json")
                fg = load(folder / "fg_scores.json")
                assert fg["ids"] == ids == b["ids"]
                ce = [
                    np.array([context[key][cid] for cid in pool])
                    for key in ("old", "new", "crop")
                ]
                whole = np.asarray(fg["whole"]["fused_scores"])
                roi = np.asarray(fg["roi"]["fused_scores"])
                aux = 0.5 * normalized_ranks(ids, whole) + 0.5 * normalized_ranks(
                    ids, roi
                )
                common = (ids, pool, ce[0], ce[1], a["base"], b["base"], aux, baseline)
                dual, _ = fuse_contexts(DualPathRankerFuseContextsInputs(*common))
                crop, _ = fuse_crops(
                    CaptionRoiRankerFuseCropsInputs(
                        ids,
                        pool,
                        ce[0],
                        ce[1],
                        ce[2],
                        a["base"],
                        b["base"],
                        aux,
                        baseline,
                    )
                )
                result["dual_context_union"] = dual["dual_context_union"]
                result["whole_crop_context_union"] = crop["whole_crop_context_union"]
                selected = [ids.index(cid) for cid in pool]
                extras = np.column_stack(
                    [
                        *[normalized_ranks(pool, v) for v in ce],
                        *[
                            normalized_ranks(ids, row["base"])[selected]
                            for row in (a, b)
                        ],
                        *[normalized_ranks(ids, v)[selected] for v in (whole, roi)],
                    ]
                )
                x = np.column_stack([matrix[selected], extras])
                learned = transfer_scores(
                    BalancedGalleryExperimentTransferScoresInputs(
                        x,
                        pool,
                        c + "/" + q["id"],
                        folds,
                        models,
                        load(ROOT / "protocol.json")["additions"],
                    )
                )
                scores = [
                    learned["pointwise"],
                    learned["ranknet"],
                    0.65 * normalized_ranks(pool, learned["ranknet"])
                    + 0.35 * (0.5 * extras[:, 3] + 0.5 * extras[:, 4]),
                ]
                tail = [cid for cid in baseline if cid not in set(pool)]
                result.update(
                    {
                        name: ranked_ids(pool, score) + tail
                        for name, score in zip(NEW, scores)
                    }
                )
                evidence[q["id"]] = dict(
                    pool=pool,
                    feature_names=FEATURE_NAMES,
                    features=x.tolist(),
                    scores={
                        name: np.asarray(score).tolist()
                        for name, score in zip(NEW, scores)
                    },
                )
            else:
                result.update({name: list(baseline) for name in (*BASELINES[3:], *NEW)})
                evidence[q["id"]] = dict(
                    route="this_315_gallery_original_cross20_exact"
                )
            assert set(result) == set(METHODS) and all(
                len(v) == len(set(v)) == 315 and set(v) == set(ids)
                for v in result.values()
            )
            output[q["id"]] = result
        outputs[c], details[c] = output, evidence
    return outputs, details


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("reranker_receipt.json")
    outputs, details = orders()
    paths = []
    for c in COHORTS:
        for name, value in [
            ("predictions.json", outputs[c]),
            ("runtime_scores.json", details[c]),
        ]:
            path = ROOT / c / name
            once(path, value)
            paths.append(path)
    receipt(
        "prediction_receipt.json",
        paths,
        complete_rankings=320,
        galley_per_cohort=315,
        new_relation_labels_used=False,
        formal_metrics=None,
    )


def labels(cohort: str) -> list[dict]:
    """Return labels for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    receipt = load(review.ROOT / "label_receipt.json")
    verify_files(review.ROOT, receipt["files"])
    original = read_rows(DATA / cohort / "judgments.ai.jsonl")
    added = [
        dict(
            query_id=r["query_key"].split("/", 1)[1],
            candidate_id=r["candidate_id"],
            grade=r["grade"],
            reason=r["reason"],
            source_file=rel(review.ROOT / r["source_file"]),
            review_origin="model",
            human_label=False,
        )
        for r in read_rows(review.ROOT / "judgments.ai.jsonl")
        if r["query_key"].split("/", 1)[0] == cohort
    ]
    return original + added


def evaluated(cohort: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    rankings = load(ROOT / cohort / "predictions.json")
    qrels = labels(cohort)
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(ROOT / cohort / "queries.public.json")
    }
    return evaluate_intervals(rankings, qrels, modes), paired_querywise(
        rankings, qrels, modes, COMPARISONS
    )


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once, receipt, write_rows.
    """
    guard()
    checked("prediction_receipt.json")
    paths = []
    for c in COHORTS:
        path = ROOT / c / "judgments.ai.jsonl"
        rows = labels(c)
        if path.exists():
            assert read_rows(path) == rows
        else:
            write_rows(path, rows)
        metrics, paired = evaluated(c)
        for name, value in [
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ]:
            target = ROOT / c / name
            once(target, value)
            paths.append(target)
        paths.append(path)
    receipt(
        "evaluation_receipt.json",
        paths,
        original_AI_relations_unchanged=11840,
        new_AI_relations=760,
        total_AI_relations=12600,
        human_labels=0,
        formal_metrics=None,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, receipt, report.write_text, write_json.
    """
    guard()
    checked("evaluation_receipt.json")
    outputs, details = orders()
    gates: dict = {}
    lines = [
        "# 315件同图库内部AI开发对照",
        "",
        "K10，40条原查询，11840原关系加760新增AI关系。人工0；全部开发曝光，正式指标null。",
        "18个旧折模型固定，不训练新增关系；旧商品保持原家族折，新商品通过来源家族/像素隔离后在被验证查询折的三模型等权投票。",
        "所有完整模型输入均为真实推理或严格字节一致缓存；无缺失分数填充。",
        "315图库内方法可互比，296图库历史不能与此表拼接。缓存重放时延不是在线端到端验收时延。",
        "",
    ]
    for c in COHORTS:
        assert outputs[c] == load(ROOT / c / "predictions.json") and details[c] == load(
            ROOT / c / "runtime_scores.json"
        )
        metrics, paired = evaluated(c)
        assert metrics == load(
            ROOT / c / "internal_metrics.ai.development.json"
        ) and paired == load(ROOT / c / "internal_paired_querywise.json")
        gates[c] = {}
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            gates[c][scope] = {}
            for name, values in methods.items():
                cells = [
                    (
                        "–"
                        if bounds[0] is None
                        else f"{bounds[0]*100:.2f}–{bounds[1]*100:.2f}"
                    )
                    for bounds in values["macro_ranges"].values()
                ]
                lines.append("|" + "|".join([scope, name, *cells]) + "|")
            for name in NEW:
                gates[c][scope][name] = dict(
                    precision_60=methods[name]["precision_60_lower_bound_met"],
                    recall_85=methods[name]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in BASELINES
                        for v in paired["scopes"][scope][name][b].values()
                    ),
                )
    primary = [scope[PRIMARY] for group in gates.values() for scope in group.values()]
    decision = dict(
        primary=PRIMARY,
        checks=gates,
        development_precision_target_met=all(g["precision_60"] for g in primary),
        development_recall_preference_met=all(g["recall_85"] for g in primary),
        development_all_metric_nonregression=all(g["nonregression"] for g in primary),
        formal_metrics=None,
        goal_met=False,
        default_strategy_changed=False,
        independent_AI_blind_predictions=0,
    )
    once(ROOT / "decision.json", decision)
    report = ROOT / "internal_comparison.ai.development.md"
    text = "\n".join(lines) + "\n"
    if report.exists():
        assert report.read_text(encoding="utf-8") == text
    else:
        report.write_text(text, encoding="utf-8")
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_complete_rankings=320,
            original_vectors_unchanged=True,
            new_labels_used_in_training_or_prediction=False,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        load(supplement.ROOT / "post_features_exclusion_snapshot.json"),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="same_gallery_development_comparison_completed",
            methods=8,
            rankings=320,
            gallery_per_cohort=315,
            AI_relations=12600,
            human_labels=0,
            formal_metrics=None,
            quality_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    paths = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and p.name != "completion_receipt.json"
    ]
    receipt(
        "completion_receipt.json",
        paths,
        source_files={s: sha256_file(PROJECT_ROOT / s) for s in SOURCES},
        formal_metrics=None,
        completed=True,
    )
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["prepare", "plan", "infer", "prediction", "evaluation", "finish"],
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        with recovery.locked() as acquired:
            if acquired:
                print(
                    dict(action=args.action, result=globals()[args.action]()),
                    flush=True,
                )
            else:
                print(dict(another_runner_holds_lock=True), flush=True)
