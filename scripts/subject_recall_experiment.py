"""Full-gallery family-masked proposals followed by unchanged full Qwen CE.

The proposal layer uses only complete Qwen/FG/public-attribute channels available
for every gallery product. It has no CE columns and never fills missing CE scores.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import balanced_supplement as acquisition
import run  # noqa: F401
import subject_attribute_experiment as previous
from finalize_balanced_gallery import report_cells
from learned_fusion_experiment import load, np_once, once, rel
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import FEATURE_NAMES as QWEN_FEATURES
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerTrainingMaskInputs,
    ranked_ids,
    training_mask,
)
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_subject_experiment import pair_identity
from primary_subject_ranker import penalty
from querywise_paired_bounds import paired_querywise
from subject_attribute_ranker import (
    ATTRIBUTE_NAMES,
    POLICY,
    fit,
    predict,
    public_descriptor,
)

ROOT = PROJECT_ROOT / "artifacts/module2_subject_recall_learning"
DATA = previous.parent.ROOT
COHORTS = previous.COHORTS
NEW = (
    "proposal_only_oof_ablation",
    "learned_recall_whole_union",
    "learned_recall_dual_union",
)
PRIMARY = NEW[2]
METHODS = (*previous.METHODS, *NEW)
BASELINES = (
    "cross20",
    "dual_context_union",
    "whole_crop_context_union",
    previous.parent.PRIMARY,
    previous.PRIMARY,
)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [
    (PRIMARY, NEW[1]),
    (PRIMARY, NEW[0]),
]
FEATURE_NAMES = [
    *QWEN_FEATURES,
    "old_base_rank",
    "new_base_rank",
    "whole_FG_rank",
    "ROI_FG_rank",
    *ATTRIBUTE_NAMES[:13],
    "whole_minus_ROI_FG_rank",
    "whole_ROI_FG_disagreement",
    "regional_minus_global_cosine",
]
SOURCES = [
    "scripts/subject_recall_experiment.py",
    "tests/test_subject_recall_experiment.py",
    "scripts/subject_attribute_ranker.py",
    "scripts/subject_attribute_experiment.py",
    "scripts/primary_subject_ranker.py",
    "scripts/primary_subject_experiment.py",
    "scripts/querywise_paired_bounds.py",
    "scripts/finalize_balanced_gallery.py",
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
    r = load(ROOT / name)
    verify_files(PROJECT_ROOT, r["files"])
    return r


def public_vector(q: dict, candidate: dict) -> list:
    """Return public vector for this workflow.

    Args:
        q: Q used by this operation.
        candidate: Candidate used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    q, d = public_descriptor(q), public_descriptor(candidate)

    def pair(a: str | None, b: str | None) -> list[float]:
        """Return pair for this workflow.

        Args:
            a: A used by this operation.
            b: B used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        known = a is not None and b is not None
        return [float(known), float(known and a == b), float(known and a != b)]

    known = bool(q["phone_models"] and d["phone_models"])
    match = bool(set(q["phone_models"]) & set(d["phone_models"]))
    structured = (
        penalty(
            {k: q[k] for k in ("family", "subtype", "structures")},
            {k: d[k] for k in ("family", "subtype", "structures")},
        )["reason"]
        == "explicit incompatible visible structures"
    )
    result = [
        *pair(q["family"], d["family"]),
        *pair(q["subtype"], d["subtype"]),
        *pair(q["role"], d["role"]),
        float(known),
        float(known and match),
        float(known and not match),
        float(structured),
    ]
    assert len(result) == 13
    return result


def stable_union(original: list[str], proposal: list[str]) -> list:
    """Return stable union for this workflow.

    Args:
        original: Original used by this operation.
        proposal: Proposal used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    assert (
        len(original) == len(set(original))
        and len(proposal) == len(set(proposal)) == 20
    )
    assert len(original) <= 40
    result = list(dict.fromkeys([*original, *proposal]))
    assert set(original) <= set(result) and len(result) <= 60
    return result


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    previous.verify()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    paths += [
        ROOT / "engineering_checks.pre.json",
        previous.ROOT / "completion_receipt.json",
        previous.ROOT / "feature_metadata.json",
        previous.ROOT / "feature_bank.npz",
        previous.ROOT / "post_experiment_exclusion_snapshot.json",
        DATA / "exact_cache.json",
        DATA / "protocol.json",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
    ]
    paths += sorted((DATA / "pairs").glob("*.json"))
    for c in COHORTS:
        paths += [
            DATA / c / n
            for n in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
            )
        ]
        paths += [
            previous.ROOT / c / n
            for n in ("predictions.json", "internal_metrics.ai.development.json")
        ]
        for q in load(DATA / c / "queries.public.json"):
            if q["image_path"]:
                paths += [
                    DATA / c / "runs" / q["id"] / n
                    for n in (
                        "features.json",
                        "caption_features.json",
                        "fg_scores.json",
                        "shared_pool.json",
                    )
                ]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            role="All exposed AI development, not independent acceptance",
            methods=METHODS,
            primary=PRIMARY,
            features=FEATURE_NAMES,
            training=POLICY,
            hypothesis="Train on all 315 gallery products per image query to propose candidates beyond the previous small pool, then score every old/new union candidate with both original full whole-image CE contexts",
            training_scope="7560 query/gallery cells, family masks unchanged from preceding experiment; labels remain frozen, null excluded from all fitting and hard-negative mining",
            proposal="Full-gallery OOF calibrated RankNet score, top20; stable union retains every previous original20/new20 candidate, at most60",
            proposal_ablation="Proposal-only scores are an explicit no-CE component ablation, not the main method or replacement for full Qwen",
            primary_score=".65 equal original/new full CE ranks on shared union + .35 full-gallery proposal rank; common original cross20 tail",
            whole_control=".65 original full CE rank + .35 full-gallery proposal rank on same union",
            CE_evidence="Every union candidate needs real complete-model scores in both contexts; byte/config/instruction/description identities required for reuse, otherwise real inference",
            no_missing_CE_columns_or_imputation=True,
            hyperparameter_search=False,
            after_result_tuning=False,
            text_route="16 text queries retain this 315-gallery original cross20",
            new_pixel_generation=False,
            ROI="Inherited full regional/FG channels remain proposal evidence; no coordinate repair claim",
            K=10,
            precision_required=0.60,
            recall_preferred=0.85,
            human_labels=0,
            formal_metrics=None,
            default_promotion_allowed=False,
            independent_acceptance=False,
        ),
    )
    status("protocol_frozen")


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    previous.verify()


def status(phase: str, **extra: object) -> None:
    """Apply the status workflow step.

    Args:
        phase: Phase used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through write_json.
    """
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=phase == "completed",
            phase=phase,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
            **extra,
        ),
    )


def bank() -> None:
    """Apply the bank workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    meta = load(previous.ROOT / "feature_metadata.json")
    gids = meta["gallery_ids"]
    gpos = {cid: i for i, cid in enumerate(gids)}
    with np.load(previous.ROOT / "feature_bank.npz", allow_pickle=False) as b:
        qfold, gfold = b["query_folds"].copy(), b["gallery_folds"].copy()
    shape = (24, len(gids))
    x = np.zeros((*shape, len(FEATURE_NAMES)))
    y = np.full(shape, np.nan)
    valid = np.zeros(shape, bool)
    base = np.zeros(shape)
    for qi, q in enumerate(meta["queries"]):
        c, qid = q["cohort"], q["id"]
        folder = DATA / c / "runs" / qid
        old, new, fg = [
            load(folder / n)
            for n in ("features.json", "caption_features.json", "fg_scores.json")
        ]
        ids = old["ids"]
        assert ids == new["ids"] == fg["ids"] and len(ids) == 315
        original = np.asarray(old["features"])
        assert original.shape == (315, len(QWEN_FEATURES))
        oldbase, newbase = normalized_ranks(ids, old["base"]), normalized_ranks(
            ids, new["base"]
        )
        whole, roi = normalized_ranks(
            ids, fg["whole"]["fused_scores"]
        ), normalized_ranks(ids, fg["roi"]["fused_scores"])
        query = previous.query_profile(c, qid)
        profiles = previous.public_profiles(c)
        labels = {
            r["candidate_id"]: r["grade"]
            for r in read_rows(DATA / c / "judgments.ai.jsonl")
            if r["query_id"] == qid
        }
        for i, cid in enumerate(ids):
            gi = gpos[cid]
            x[qi, gi] = np.r_[
                original[i],
                oldbase[i],
                newbase[i],
                whole[i],
                roi[i],
                public_vector(query, profiles[cid]),
                whole[i] - roi[i],
                abs(whole[i] - roi[i]),
                original[i, 1] - original[i, 0],
            ]
            y[qi, gi] = np.nan if labels[cid] is None else labels[cid]
            valid[qi, gi] = True
            base[qi, gi] = 0.5 * oldbase[i] + 0.5 * newbase[i]
    assert int(valid.sum()) == 7560 and np.isfinite(x[valid]).all()
    np_once(
        ROOT / "feature_bank.npz",
        features=x,
        labels=y,
        valid=valid,
        base=base,
        query_folds=qfold,
        gallery_folds=gfold,
    )
    once(
        ROOT / "feature_metadata.json",
        dict(
            meta,
            feature_names=FEATURE_NAMES,
            feature_scope="All 315 real gallery products per image query; no CE columns",
        ),
    )
    receipt(
        "bank_receipt.json",
        [ROOT / "feature_bank.npz", ROOT / "feature_metadata.json"],
        complete_real_feature_rows=7560,
        features=len(FEATURE_NAMES),
        known=int(np.isfinite(y[valid]).sum()),
        unknown=int(np.isnan(y[valid]).sum()),
        CE_features=0,
        imputed_CE_scores=0,
    )


def train_values() -> tuple[np.ndarray, dict, list, dict]:
    """Train values.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    meta = load(ROOT / "feature_metadata.json")
    with np.load(ROOT / "feature_bank.npz", allow_pickle=False) as b:
        x, y, valid, base, qf, gf = [
            b[k].copy()
            for k in (
                "features",
                "labels",
                "valid",
                "base",
                "query_folds",
                "gallery_folds",
            )
        ]
    scores = np.full(valid.shape, np.nan)
    models = {}
    audits = []
    for hq in range(3):
        for hg in range(3):
            allowed, evidence = training_mask(
                SemanticRankerTrainingMaskInputs(
                    meta["query_ids"],
                    qf,
                    meta["query_families"],
                    meta["gallery_families"],
                    gf,
                    hq,
                    hg,
                )
            )
            allowed &= valid
            model = fit(x, y, allowed, base, names=FEATURE_NAMES)
            key = f"q{hq}_g{hg}"
            models[key] = model
            held = valid & (qf[:, None] == hq) & (gf[None, :] == hg)
            scores[held] = predict(x, model)[held]
            audits.append(
                dict(
                    model=key,
                    **evidence,
                    training_cells=[
                        [meta["query_ids"][q], meta["gallery_ids"][g]]
                        for q, g in np.argwhere(allowed & np.isfinite(y))
                    ],
                )
            )
    assert np.isfinite(scores[valid]).all()
    candidate = fit(x, y, valid, base, names=FEATURE_NAMES)
    return scores, models, audits, candidate


def train() -> None:
    """Fit the configured model on the allowed training data.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("bank_receipt.json")
    scores, models, audits, candidate = train_values()
    np_once(ROOT / "oof_scores.npz", scores=scores)
    for name, value in [
        ("fold_models.json", models),
        ("fold_audits.json", audits),
        ("all_development_candidate.json", candidate),
    ]:
        once(ROOT / name, value)
    receipt(
        "training_receipt.json",
        [
            ROOT / n
            for n in (
                "oof_scores.npz",
                "fold_models.json",
                "fold_audits.json",
                "all_development_candidate.json",
            )
        ],
        fold_models=9,
        all_development_candidates=1,
        OOF_not_independent_acceptance=True,
    )


def plan() -> None:
    """Apply the plan workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("training_receipt.json")
    meta = load(ROOT / "feature_metadata.json")
    positions = {cid: i for i, cid in enumerate(meta["gallery_ids"])}
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        scores = b["scores"].copy()
    jobs = []
    proposals = []
    cache = load(DATA / "exact_cache.json")
    for path in sorted((DATA / "pairs").glob("*.json")):
        raw = load(path)
        identity = raw["identity"]
        key = identity["sha256"]
        record = dict(
            identity=identity,
            logit=raw["logit"],
            source_file=rel(path),
            source_sha256=sha256_file(path),
        )
        if key in cache:
            assert (
                cache[key]["identity"] == identity
                and cache[key]["logit"] == raw["logit"]
            )
        else:
            cache[key] = record
    for qi, q in enumerate(meta["queries"]):
        c, qid = q["cohort"], q["id"]
        docs = {d["id"]: d for d in load(DATA / c / "catalog.public.json")}
        ids = list(docs)
        old, new = [
            load(DATA / c / "runs" / qid / n)
            for n in ("features.json", "caption_features.json")
        ]
        values = scores[qi, [positions[cid] for cid in ids]]
        proposed = ranked_ids(ids, values)
        prior = load(DATA / c / "runs" / qid / "shared_pool.json")
        pool = stable_union(prior, proposed[:20])
        proposals.append(
            dict(
                cohort=c,
                query_id=qid,
                ids=ids,
                scores=values.tolist(),
                proposal_order=proposed,
                old_pool=prior,
                pool=pool,
            )
        )
        for context, description in [
            ("old", old["description"]),
            ("new", new["description"]),
        ]:
            public = dict(id=qid, text=q["text"], image_path=q["image_path"])
            for cid in pool:
                identity = pair_identity(public, description, docs[cid])
                jobs.append(
                    dict(
                        cohort=c,
                        query_id=qid,
                        context=context,
                        candidate_id=cid,
                        query=public,
                        description=description,
                        document=docs[cid],
                        identity=identity,
                    )
                )
    once(ROOT / "proposals.public.json", proposals)
    once(ROOT / "pair_plan.json", jobs)
    once(ROOT / "exact_cache.json", cache)
    receipt(
        "plan_receipt.json",
        [
            ROOT / n
            for n in ("proposals.public.json", "pair_plan.json", "exact_cache.json")
        ],
        query_unions=24,
        context_positions=len(jobs),
        additional_pool_positions=sum(
            len(p["pool"]) - len(p["old_pool"]) for p in proposals
        ),
        planned_missing_unique_pairs=len(
            {
                j["identity"]["sha256"]
                for j in jobs
                if j["identity"]["sha256"] not in cache
            }
        ),
        old_candidates_preserved=True,
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
    engine = None
    cache = load(ROOT / "exact_cache.json")
    jobs = load(ROOT / "pair_plan.json")
    try:
        for i, job in enumerate(jobs):
            key = job["identity"]["sha256"]
            path = ROOT / "pairs" / (key + ".json")
            if not path.exists():
                if key in cache:
                    raw = cache[key]
                    assert raw["identity"] == job["identity"]
                    assert (
                        sha256_file(PROJECT_ROOT / raw["source_file"])
                        == raw["source_sha256"]
                    )
                    once(path, dict(raw, reused=True))
                else:
                    if engine is None:
                        if not acquisition.gpu_available():
                            status("waiting_natural_GPU_release")
                            return False
                        engine = DescribedImageReranker()
                        status("full_Qwen_reranking")
                    doc = job["document"]
                    pairs = [
                        dict(
                            id=doc["id"],
                            text=doc["title"],
                            image_path=doc["image_path"],
                        )
                    ]
                    image = PROJECT_ROOT / job["query"]["image_path"]
                    values = (
                        engine.score_described(
                            pairs, query_image=image, description=job["description"]
                        )
                        if job["description"]
                        else engine.score(pairs, query_image=image, query_text=None)
                    )
                    assert len(values) == 1 and np.isfinite(values).all()
                    once(
                        path,
                        dict(
                            identity=job["identity"],
                            logit=float(values[0]),
                            reused=False,
                            timing=engine.adapter.timings[-1],
                        ),
                    )
            row = load(path)
            assert row["identity"] == job["identity"] and math.isfinite(row["logit"])
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(dict(context_positions=i + 1, total=len(jobs)), flush=True)
        once(
            ROOT / "reranker_runtime.json",
            engine.diagnostics() if engine else dict(no_new_forward_calls=True),
        )
    finally:
        if engine:
            engine.close()
    paths = sorted((ROOT / "pairs").glob("*.json"))
    receipt(
        "reranker_receipt.json",
        [ROOT / "reranker_runtime.json", *paths],
        context_positions=len(jobs),
        unique_pairs=len(paths),
        unique_real_new_calls=sum(not load(p)["reused"] for p in paths),
        missing_scores_imputed=0,
    )
    return True


def orders() -> tuple[dict, dict]:
    """Return orders for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    proposals = {
        (p["cohort"], p["query_id"]): p for p in load(ROOT / "proposals.public.json")
    }
    jobs = load(ROOT / "pair_plan.json")
    outputs, details = {}, {}
    for c in COHORTS:
        output = load(previous.ROOT / c / "predictions.json")
        evidence = {}
        for q in load(DATA / c / "queries.public.json"):
            qid = q["id"]
            baseline = output[qid]["cross20"]
            if q["image_path"]:
                p = proposals[c, qid]
                ids, pool = p["ids"], p["pool"]
                context: dict = {}
                for job in (
                    j for j in jobs if j["cohort"] == c and j["query_id"] == qid
                ):
                    row = load(ROOT / "pairs" / (job["identity"]["sha256"] + ".json"))
                    context.setdefault(job["context"], {})[job["candidate_id"]] = row[
                        "logit"
                    ]
                ce = [
                    normalized_ranks(pool, [context[key][cid] for cid in pool])
                    for key in ("old", "new")
                ]
                base = normalized_ranks(ids, p["scores"])[
                    [ids.index(cid) for cid in pool]
                ]
                tail = [cid for cid in baseline if cid not in set(pool)]
                new = {
                    NEW[0]: p["proposal_order"],
                    NEW[1]: ranked_ids(pool, 0.65 * ce[0] + 0.35 * base) + tail,
                    PRIMARY: ranked_ids(
                        pool, 0.65 * (0.5 * ce[0] + 0.5 * ce[1]) + 0.35 * base
                    )
                    + tail,
                }
                evidence[qid] = dict(
                    pool=pool,
                    base=base.tolist(),
                    original_context_rank=ce[0].tolist(),
                    new_context_rank=ce[1].tolist(),
                    score_is_probability=False,
                )
            else:
                new = {m: list(baseline) for m in NEW}
                evidence[qid] = dict(route="this_gallery_cross20_exact")
            output[qid].update(new)
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
            p = ROOT / c / name
            once(p, value)
            paths.append(p)
    receipt(
        "prediction_receipt.json",
        paths,
        methods=14,
        rankings=560,
        unchanged_parent_rankings=440,
    )


def evaluated(c: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        c: C used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / c / "predictions.json")
    labels = read_rows(DATA / c / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(DATA / c / "queries.public.json")
    }
    return evaluate_intervals(ranks, labels, modes), paired_querywise(
        ranks, labels, modes, COMPARISONS
    )


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("prediction_receipt.json")
    paths = []
    for c in COHORTS:
        metrics, paired = evaluated(c)
        for name, value in [
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ]:
            p = ROOT / c / name
            once(p, value)
            paths.append(p)
    receipt(
        "evaluation_receipt.json",
        paths,
        AI_relations=12600,
        human_labels=0,
        formal_metrics=None,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, receipt, report.write_text.
    """
    guard()
    checked("evaluation_receipt.json")
    outputs, details = orders()
    scores, models, audits, candidate = train_values()
    assert (
        models == load(ROOT / "fold_models.json")
        and audits == load(ROOT / "fold_audits.json")
        and candidate == load(ROOT / "all_development_candidate.json")
    )
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        assert np.array_equal(b["scores"], scores, equal_nan=True)
    gates: dict = {}
    losses = []
    lines = [
        "# 全图库候选学习与完整双上下文精排：内部AI开发对照",
        "",
        "315图库、40原查询、12600原AI关系、K10。全部曝光开发，正式指标null。",
        "候选层用全图库真实Qwen/FG/公开属性特征，独立列定义不含CE、不补造CE；9个家族排除折模型进行OOF候选提议。",
        "旧候选完整保留，追加提议top20；共享并集的每一候选都取得完整Qwen两个原整图上下文分数或严格身份缓存。",
        "proposal_only为明确移除CE的组件消融；主方法始终包含完整精排。不存在通过消融替代主方法的成绩。",
        "16文本保持原cross20，旧11方法440排序及原标签保持不变。未知范围不覆盖AI误标，OOF拼接不等于可部署单模型独立验收。",
        "",
    ]
    for c in COHORTS:
        assert outputs[c] == load(ROOT / c / "predictions.json") and details[c] == load(
            ROOT / c / "runtime_scores.json"
        )
        oldrank = load(previous.ROOT / c / "predictions.json")
        for qid, methods in outputs[c].items():
            assert set(methods) == set(METHODS) and all(
                len(v) == len(set(v)) == 315 for v in methods.values()
            )
            assert all(methods[m] == oldrank[qid][m] for m in previous.METHODS)
            if qid.startswith("text_"):
                assert all(methods[m] == methods["cross20"] for m in NEW)
        metrics, paired = evaluated(c)
        assert metrics == load(
            ROOT / c / "internal_metrics.ai.development.json"
        ) and paired == load(ROOT / c / "internal_paired_querywise.json")
        old = load(previous.ROOT / c / "internal_metrics.ai.development.json")
        gates[c] = {}
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            assert all(methods[m] == old["scopes"][scope][m] for m in previous.METHODS)
            gates[c][scope] = {}
            for m in METHODS:
                lines.append(
                    "|" + "|".join([scope, m, *report_cells(methods[m])]) + "|"
                )
            for m in NEW:
                gates[c][scope][m] = dict(
                    precision_60=methods[m]["precision_60_lower_bound_met"],
                    recall_85=methods[m]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in BASELINES
                        for v in paired["scopes"][scope][m][b].values()
                    ),
                )
        for row in paired["query_pair_bounds"]:
            if row["candidate"] == PRIMARY:
                bad = [
                    m
                    for m, v in row["metrics"].items()
                    if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
                ]
                if bad:
                    losses.append(
                        dict(
                            cohort=c,
                            query_id=row["query_id"],
                            baseline=row["baseline"],
                            strict_regression_metrics=bad,
                        )
                    )
    primary = [scope[PRIMARY] for group in gates.values() for scope in group.values()]
    decision = dict(
        primary=PRIMARY,
        checks=gates,
        development_precision_target_met=all(g["precision_60"] for g in primary),
        development_recall_preference_met=all(g["recall_85"] for g in primary),
        development_all_metric_nonregression=all(g["nonregression"] for g in primary),
        goal_met=False,
        default_strategy_changed=False,
        independent_AI_blind_predictions=0,
        formal_metrics=None,
    )
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_strict_regressions.json", losses)
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
            replayed_rankings=560,
            unchanged_parent_rankings=440,
            refitted_identical_models=10,
            AI_relations_unchanged=12600,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        load(previous.ROOT / "post_experiment_exclusion_snapshot.json"),
    )
    status(
        "completed",
        methods=14,
        rankings=560,
        gallery_per_cohort=315,
        AI_relations=12600,
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
        completed=True,
        formal_metrics=None,
    )
    verify()
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    checked("completion_receipt.json")


def run_all(wait: bool = False) -> None:
    """Return run all for this workflow.

    Args:
        wait: Wait used by this operation.
    """
    guard()
    for name, stage in [
        ("bank_receipt.json", bank),
        ("training_receipt.json", train),
        ("plan_receipt.json", plan),
    ]:
        if (ROOT / name).exists():
            checked(name)
        else:
            stage()
    while not (ROOT / "reranker_receipt.json").exists():
        if infer():
            break
        if not wait:
            return
        print(dict(waiting_for_natural_GPU_release=True), flush=True)
        time.sleep(60)
    for name, stage in [
        ("prediction_receipt.json", prediction),
        ("evaluation_receipt.json", evaluation),
        ("completion_receipt.json", finish),
    ]:
        if (ROOT / name).exists():
            checked(name)
        else:
            stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "worker", "verify"])
    action = parser.parse_args().action
    with previous.locked() as acquired:
        if not acquired:
            raise RuntimeError("Another subject-learning process holds the lock")
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            try:
                run_all(wait=action == "worker")
            except Exception as exc:
                once(
                    ROOT / "failure.json",
                    dict(
                        type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                        formal_metrics=None,
                    ),
                )
                status("engineering_failure")
                raise
