"""Family-masked supervised subject/attribute fusion on exposed development."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import balanced_gallery_experiment as parent
import balanced_supplement_features as supplement
import finalize_balanced_gallery as finalized
import run  # noqa: F401
from finalize_balanced_gallery import report_cells
from learned_fusion_experiment import load, np_once, once, rel
from learned_fusion_model import FEATURE_NAMES as ORIGINAL_FEATURES
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows
from multimodal_ecommerce.intent_retrieval import normalized_ranks
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerTrainingMaskInputs,
    assign_folds,
    ranked_ids,
    training_mask,
)
from querywise_paired_bounds import paired_querywise
from subject_attribute_ranker import (
    FEATURE_NAMES,
    POLICY,
    attribute_vector,
    fit,
    predict,
    public_descriptor,
)

ROOT = PROJECT_ROOT / "artifacts/module2_subject_attribute_learning"
COHORTS = parent.COHORTS
NEW = (
    "refit_ranknet_oof",
    "subject_attribute_ranknet_oof",
    "subject_attribute_blend_oof",
)
PRIMARY = NEW[2]
METHODS = (*parent.METHODS, *NEW)
BASELINES = (*parent.BASELINES, parent.PRIMARY)
COMPARISONS = [(m, b) for m in NEW for b in BASELINES] + [
    (NEW[1], NEW[0]),
    (PRIMARY, NEW[1]),
]
SOURCES = [
    "scripts/subject_attribute_experiment.py",
    "scripts/subject_attribute_ranker.py",
    "tests/test_subject_attribute_ranker.py",
    "scripts/learned_fusion_model.py",
    "scripts/learned_fusion_experiment.py",
    "scripts/primary_subject_ranker.py",
    "scripts/querywise_paired_bounds.py",
    "scripts/finalize_balanced_gallery.py",
    "scripts/balanced_gallery_experiment.py",
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


def public_profiles(cohort: str) -> dict:
    """Return public profiles for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    profiles = {
        r["id"]: r for r in read_rows(parent.DATA / cohort / "profiles.public.jsonl")
    }
    profiles.update(
        {
            r["id"]: r
            for r in read_rows(supplement.ROOT / "profiles.candidate_only.jsonl")
        }
    )
    docs = load(parent.ROOT / cohort / "catalog.public.json")
    assert set(profiles) == {d["id"] for d in docs}
    return {
        d["id"]: dict(
            product=profiles[d["id"]]["product_type"],
            appearance=profiles[d["id"]]["appearance"],
            visible_text=d["title"],
            primary_unclear=profiles[d["id"]].get("primary_unclear", False),
        )
        for d in docs
    }


def query_profile(cohort: str, qid: str) -> dict:
    """Return query profile for this workflow.

    Args:
        cohort: Cohort used by this operation.
        qid: Qid used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return load(parent.DATA / "query_captions" / cohort / (qid + ".json"))["value"]


def new_family_groups(rows: list[dict]) -> dict:
    """Return new family groups for this workflow.

    Args:
        rows: Records processed by this operation.

    Returns:
        Result mapping produced by the workflow.
    """
    roots = {r["id"]: r["id"] for r in rows}
    owners: dict = {}

    def find(key: str) -> str:
        """Return find for this workflow.

        Args:
            key: Key used by this operation.

        Returns:
            Text produced by this operation.
        """
        while roots[key] != key:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key

    for row in sorted(rows, key=lambda r: r["id"]):
        for key in row["family_keys"]:
            if key in owners:
                left, right = find(row["id"]), find(owners[key])
                roots[max(left, right)] = min(left, right)
            else:
                owners[key] = row["id"]
    return {key: find(key) for key in roots}


def diagnosis() -> dict:
    """Oracle counts and public conflicts diagnose development, not performance.

    Returns:
        Mapping with fields: query_diagnosis,
        positive_label_public_attribute_conflicts, scope, top6_reference, label_policy,
        ROI_visual_observations, human_labels, formal_metrics.
    """
    rows = []
    conflicts = []
    for cohort in COHORTS:
        runtime = load(parent.ROOT / cohort / "runtime_scores.json")
        ranks = load(parent.ROOT / cohort / "predictions.json")
        profiles = public_profiles(cohort)
        labels = read_rows(parent.ROOT / cohort / "judgments.ai.jsonl")
        for query in load(parent.ROOT / cohort / "queries.public.json"):
            if not query["image_path"]:
                continue
            qid = query["id"]
            grades = {
                r["candidate_id"]: r["grade"] for r in labels if r["query_id"] == qid
            }
            pool = runtime[qid]["pool"]
            known = sum(v == 1 for v in grades.values())
            unknown = sum(v is None for v in grades.values())
            pool_known = sum(grades[c] == 1 for c in pool)
            top = sum(grades[c] == 1 for c in ranks[qid][parent.PRIMARY][:10])
            rows.append(
                dict(
                    cohort=cohort,
                    query_id=qid,
                    known_gallery_positive=known,
                    unknown=unknown,
                    known_pool_positive=pool_known,
                    known_primary_top10_positive=top,
                    known_coverage_gap=max(0, 6 - known),
                    known_recall_pool_gap=min(10, known) - min(10, pool_known),
                    known_ranking_gap=min(10, pool_known) - top,
                    oracle_recall_upper_if_any_positive=(
                        min(10, known) / known if known else 1.0 if unknown else None
                    ),
                    oracle_precision_capacity=[
                        min(10, known) / 10,
                        min(10, known + unknown) / 10,
                    ],
                )
            )
            q = query_profile(cohort, qid)
            for cid in grades:
                if grades[cid] != 1:
                    continue
                qd, dd = public_descriptor(q), public_descriptor(profiles[cid])
                if qd["family"] and dd["family"]:
                    model_conflict = bool(
                        qd["phone_models"]
                        and dd["phone_models"]
                        and not set(qd["phone_models"]) & set(dd["phone_models"])
                    )
                    material_conflict = (
                        dd["role"] == "material" and qd["role"] != "material"
                    )
                    if model_conflict or material_conflict:
                        conflicts.append(
                            dict(
                                cohort=cohort,
                                query_id=qid,
                                candidate_id=cid,
                                existing_grade=1,
                                public_model_conflict=model_conflict,
                                public_material_role_conflict=material_conflict,
                                annotation_changed=False,
                                interpretation="Public consistency conflict with broad existing AI business relevance; not proof of incorrect label",
                            )
                        )
    return dict(
        query_diagnosis=rows,
        positive_label_public_attribute_conflicts=conflicts,
        scope="Exposed development; known-count oracle gaps, not measured retrieval improvements; unknowns not assigned",
        top6_reference="Per-query diagnostic reference only, not a replacement for macro P@10>=.60",
        label_policy="All 12600 existing AI grades remain unchanged; compatibility is soft evidence and a separate diagnostic",
        ROI_visual_observations=[
            dict(
                query="fresh/image_03",
                observation="Original depicts upper-right brown organizer; inherited selected zoom depicts lower-left handbag prop; this experiment changes fusion reliability, not crop coordinates",
            ),
            dict(
                query="fresh/image_02",
                observation="Original clearly shows Huawei Mate10; frozen query-only visible_text also captures it; existing relevance includes other phone models",
            ),
        ],
        human_labels=0,
        formal_metrics=None,
    )


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once.
    """
    finalized.verify()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    once(ROOT / "diagnosis.before_training.json", diagnosis())
    paths = [PROJECT_ROOT / s for s in SOURCES]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        ROOT / "engineering_checks.pre.json",
        ROOT / "diagnosis.before_training.json",
        finalized.ROOT / "completion_receipt.json",
        finalized.ROOT / "post_experiment_exclusion_snapshot.json",
        parent.ROOT / "public_fold_map.json",
        parent.ROOT / "protocol.json",
        supplement.ROOT / "profiles.candidate_only.jsonl",
        supplement.ROOT / "profile_receipt.json",
        PROJECT_ROOT
        / "artifacts/module2_balanced_media_recovery/selected.evaluator.jsonl",
    ]
    assets = load(parent.CROP / "crop_inputs.json")
    paths.append(parent.CROP / "crop_inputs.json")
    for asset in assets["cohorts"]["fresh"]["queries"]:
        if asset["id"] in ("image_02", "image_03"):
            paths.extend(
                PROJECT_ROOT / asset[key] for key in ("source_image", "selected_image")
            )
    for c in COHORTS:
        paths += [
            parent.ROOT / c / n
            for n in (
                "runtime_scores.json",
                "predictions.json",
                "judgments.ai.jsonl",
                "catalog.public.json",
                "queries.public.json",
                "internal_metrics.ai.development.json",
            )
        ]
        paths.append(parent.DATA / c / "profiles.public.jsonl")
        paths += sorted((parent.DATA / "query_captions" / c).glob("*.json"))
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            role="Adaptive, fully exposed AI development; gradient-family masks are not independent acceptance",
            methods=METHODS,
            primary=PRIMARY,
            comparisons=COMPARISONS,
            hypotheses=[
                "Refit existing complete 20-channel evidence with added frozen business labels",
                "Public primary/type/role/model signals and whole-crop/FG disagreement help avoid accessory and scene-prop false matches",
            ],
            recipes=POLICY,
            base_feature_names=ORIGINAL_FEATURES,
            structured_feature_names=FEATURE_NAMES,
            training="3 query-family folds x 3 gallery-source-family folds, no held family in scaler/mining/gradient/calibration; existing folds preserved, new families grouped before fold assignment",
            confirmed_hard_negatives="At most 8 grade-0 training candidates per query by fixed base; null never a negative",
            blend=".65 structured RankNet rank + .35 equal old/new base rank",
            attributes="Soft evidence; no deterministic model/type hard filtering, no grade rewriting; historical public profiles remain AI development inputs, not independently annotated truth",
            ROI="Learn whole/local disagreement reliability from real frozen full-model channels; inherited crop coordinates unchanged, known wrong organizer crop is not claimed repaired",
            data=dict(
                gallery_per_cohort=315,
                queries=40,
                old_AI_relations=11840,
                added_AI_relations=760,
                total_AI_relations=12600,
                K=10,
            ),
            parent_8_methods_unchanged=True,
            text_route="New image methods retain this gallery's cross20 exactly for 16 text queries",
            full_neural_models="Original full-model feature/CE/FG scores reused exactly; no neural forward calls, no score imputation or backbone replacement",
            hyperparameter_search=False,
            after_result_tuning=False,
            default_promotion_allowed=False,
            independent_acceptance=False,
            human_labels=0,
            formal_metrics=None,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    finalized.verify()


def bank() -> None:
    """Return bank for this workflow.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    if (ROOT / "bank_receipt.json").exists():
        checked("bank_receipt.json")
        return
    previous = load(parent.ROOT / "public_fold_map.json")
    rows = read_rows(
        PROJECT_ROOT
        / "artifacts/module2_balanced_media_recovery/selected.evaluator.jsonl"
    )
    assert set(parent.check_new_families()) == {r["id"] for r in rows}
    newgroups = new_family_groups(rows)
    newfolds = assign_folds(list(newgroups.values()), 3, POLICY["seed"])
    docs = {}
    queries = []
    for c in COHORTS:
        for d in load(parent.ROOT / c / "catalog.public.json"):
            docs[d["id"]] = d
        queries += [
            dict(q, cohort=c, qualified=c + "/" + q["id"])
            for q in load(parent.ROOT / c / "queries.public.json")
            if q["image_path"]
        ]
    gids = sorted(docs)
    positions = {cid: i for i, cid in enumerate(gids)}
    qfamilies = [previous["query_families"][q["qualified"]] for q in queries]
    gfamilies = [
        previous["gallery_families"].get(cid, newgroups.get(cid)) for cid in gids
    ]
    qfold = np.array([previous["queries"][q["qualified"]] for q in queries])
    gfold = np.array(
        [
            (
                previous["gallery"][cid]
                if cid in previous["gallery"]
                else newfolds[newgroups[cid]]
            )
            for cid in gids
        ]
    )
    shape = (len(queries), len(gids))
    x = np.zeros((*shape, len(FEATURE_NAMES)))
    y = np.full(shape, np.nan)
    valid = np.zeros(shape, bool)
    base = np.zeros(shape)
    pools = []
    public_evidence: dict = {}
    for qi, q in enumerate(queries):
        c, qid = q["cohort"], q["id"]
        entry = load(parent.ROOT / c / "runtime_scores.json")[qid]
        assert entry["feature_names"] == ORIGINAL_FEATURES
        profile = query_profile(c, qid)
        profiles = public_profiles(c)
        labels = {
            r["candidate_id"]: r["grade"]
            for r in read_rows(parent.ROOT / c / "judgments.ai.jsonl")
            if r["query_id"] == qid
        }
        pool = entry["pool"]
        pools.append(pool)
        public_evidence[q["qualified"]] = {}
        for pi, cid in enumerate(pool):
            original = np.asarray(entry["features"][pi])
            additional, evidence = attribute_vector(original, profile, profiles[cid])
            gi = positions[cid]
            x[qi, gi] = np.r_[original, additional]
            y[qi, gi] = np.nan if labels[cid] is None else labels[cid]
            valid[qi, gi] = True
            base[qi, gi] = (
                0.5 * original[ORIGINAL_FEATURES.index("old_full_gallery_base_rank")]
                + 0.5 * original[ORIGINAL_FEATURES.index("new_full_gallery_base_rank")]
            )
            public_evidence[q["qualified"]][cid] = evidence
    assert len(queries) == 24 and len(gids) == 411 and int(valid.sum()) == 504
    assert np.isfinite(x[valid]).all() and all(g is not None for g in gfamilies)
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
            query_ids=[q["qualified"] for q in queries],
            queries=queries,
            gallery_ids=gids,
            query_families=qfamilies,
            gallery_families=gfamilies,
            pools=pools,
            new_family_groups=newgroups,
            feature_names=FEATURE_NAMES,
        ),
    )
    once(ROOT / "public_attribute_evidence.json", public_evidence)
    receipt(
        "bank_receipt.json",
        [
            ROOT / "feature_bank.npz",
            ROOT / "feature_metadata.json",
            ROOT / "public_attribute_evidence.json",
        ],
        actual_full_evidence_rows=int(valid.sum()),
        known=int(np.isfinite(y[valid]).sum()),
        unknown=int(np.isnan(y[valid]).sum()),
        features=len(FEATURE_NAMES),
        new_neural_calls=0,
    )


def train_values() -> tuple[dict, dict, list, dict]:
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
    variants = dict(
        refit=(x[:, :, : len(ORIGINAL_FEATURES)], ORIGINAL_FEATURES),
        structured=(x, FEATURE_NAMES),
    )
    scores = {name: np.full(valid.shape, np.nan) for name in variants}
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
            held = valid & (qf[:, None] == hq) & (gf[None, :] == hg)
            for name, (matrix, names) in variants.items():
                key = f"{name}_q{hq}_g{hg}"
                model = fit(matrix, y, allowed, base, names=names)
                scores[name][held] = predict(matrix, model)[held]
                models[key] = model
                audits.append(
                    dict(
                        model=key,
                        **evidence,
                        validation_cells=int(held.sum()),
                        training_cells=[
                            [meta["query_ids"][q], meta["gallery_ids"][g]]
                            for q, g in np.argwhere(allowed & np.isfinite(y))
                        ],
                    )
                )
    assert all(np.isfinite(v[valid]).all() for v in scores.values())
    candidates = {
        name: fit(matrix, y, valid, base, names=names)
        for name, (matrix, names) in variants.items()
    }
    return scores, models, audits, candidates


def train() -> None:
    """Fit the configured model on the allowed training data.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("bank_receipt.json")
    scores, models, audits, candidates = train_values()
    for name, value in [
        ("fold_models.json", models),
        ("fold_audits.json", audits),
        ("all_development_candidates.json", candidates),
    ]:
        once(ROOT / name, value)
    np_once(ROOT / "oof_scores.npz", **scores)
    receipt(
        "training_receipt.json",
        [
            ROOT / n
            for n in (
                "fold_models.json",
                "fold_audits.json",
                "all_development_candidates.json",
                "oof_scores.npz",
            )
        ],
        fold_models=18,
        full_development_candidates=2,
        OOF_is_multiple_model_stitch_not_deployable_single_model=True,
        new_labels_in_supervised_training=True,
        formal_metrics=None,
    )


def orders() -> tuple[dict, dict]:
    """Return orders for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    meta = load(ROOT / "feature_metadata.json")
    qpos = {qid: i for i, qid in enumerate(meta["query_ids"])}
    gpos = {cid: i for i, cid in enumerate(meta["gallery_ids"])}
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        refit, structured = b["refit"].copy(), b["structured"].copy()
    with np.load(ROOT / "feature_bank.npz", allow_pickle=False) as b:
        base = b["base"].copy()
    outputs, details = {}, {}
    for c in COHORTS:
        previous = load(parent.ROOT / c / "predictions.json")
        output, evidence = {}, {}
        for query in load(parent.ROOT / c / "queries.public.json"):
            qid = query["id"]
            baseline = previous[qid]["cross20"]
            if query["image_path"]:
                qi = qpos[c + "/" + qid]
                pool = meta["pools"][qi]
                gi = [gpos[cid] for cid in pool]
                values = [
                    refit[qi, gi],
                    structured[qi, gi],
                    0.65 * normalized_ranks(pool, structured[qi, gi])
                    + 0.35 * base[qi, gi],
                ]
                tail = [cid for cid in baseline if cid not in set(pool)]
                new = {name: ranked_ids(pool, v) + tail for name, v in zip(NEW, values)}
                evidence[qid] = dict(
                    pool=pool,
                    scores={
                        name: dict(zip(pool, v.tolist()))
                        for name, v in zip(NEW, values)
                    },
                    score_is_probability=False,
                )
            else:
                new = {name: list(baseline) for name in NEW}
                evidence[qid] = dict(route="this_gallery_cross20_exact")
            output[qid] = dict(previous[qid], **new)
        outputs[c], details[c] = output, evidence
    return outputs, details


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    checked("training_receipt.json")
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
        methods=11,
        rankings=440,
        unchanged_parent_rankings=320,
    )


def evaluated(cohort: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        cohort: Cohort used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    ranks = load(ROOT / cohort / "predictions.json")
    labels = read_rows(parent.ROOT / cohort / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(parent.ROOT / cohort / "queries.public.json")
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
        all_queries_and_unknowns_retained=True,
        fixed_K=10,
        formal_metrics=None,
    )


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Notes:
        Writes derived artifacts through once, receipt, report.write_text.
    """
    guard()
    checked("evaluation_receipt.json")
    output, details = orders()
    scores, models, audits, candidates = train_values()
    assert (
        models == load(ROOT / "fold_models.json")
        and audits == load(ROOT / "fold_audits.json")
        and candidates == load(ROOT / "all_development_candidates.json")
    )
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        assert all(np.array_equal(b[k], v, equal_nan=True) for k, v in scores.items())
    lines = [
        "# 主售属性与困难负例融合：内部AI开发对照",
        "",
        "315件同图库、40查询、12600冻结AI关系，人工0，K10。全部曝光开发，OOF不作为独立验收。",
        "3个新方法隔离训练、标准化、困难负例和校准的来源家族；18个折模型和2个全开发候选固定配置可重放。",
        "原8方法320排序保留。属性与型号为软信号，AI业务标签不重写；已知内胆包错误ROI坐标不变，本轮检验融合一致性，不宣称定位修复。",
        "新增神经前向0；所有原骨干、整图/裁切精排和FG证据均为封存真实模型推理。",
        "共享未知范围不是置信区间，不能覆盖AI误标；历史公开描述来自已有AI开发材料，训练家族掩码不能把它变成独立验收。",
        "",
    ]
    gates: dict = {}
    regressions = []
    for c in COHORTS:
        assert output[c] == load(ROOT / c / "predictions.json") and details[c] == load(
            ROOT / c / "runtime_scores.json"
        )
        previous = load(parent.ROOT / c / "predictions.json")
        for qid, methods in output[c].items():
            assert set(methods) == set(METHODS) and all(
                len(v) == len(set(v)) == 315 for v in methods.values()
            )
            assert all(methods[m] == previous[qid][m] for m in parent.METHODS)
            if qid.startswith("text_"):
                assert all(methods[m] == methods["cross20"] for m in NEW)
        metrics, paired = evaluated(c)
        assert metrics == load(
            ROOT / c / "internal_metrics.ai.development.json"
        ) and paired == load(ROOT / c / "internal_paired_querywise.json")
        old = load(parent.ROOT / c / "internal_metrics.ai.development.json")
        gates[c] = {}
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        for scope, methods in metrics["scopes"].items():
            assert all(methods[m] == old["scopes"][scope][m] for m in parent.METHODS)
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
                        for baseline in BASELINES
                        for v in paired["scopes"][scope][m][baseline].values()
                    ),
                )
        for row in paired["query_pair_bounds"]:
            if row["candidate"] == PRIMARY:
                strict = [
                    metric
                    for metric, v in row["metrics"].items()
                    if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
                ]
                if strict:
                    regressions.append(
                        dict(
                            cohort=c,
                            query_id=row["query_id"],
                            baseline=row["baseline"],
                            strict_regression_metrics=strict,
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
    once(ROOT / "internal_strict_regressions.json", regressions)
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
            replayed_rankings=440,
            unchanged_parent_rankings=320,
            refitted_identical_models=20,
            original_AI_relations_unchanged=11840,
            new_AI_relations_unchanged=760,
            new_neural_calls=0,
            human_labels=0,
            formal_metrics=None,
        ),
    )
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        load(finalized.ROOT / "post_experiment_exclusion_snapshot.json"),
    )
    once(
        ROOT / "public_status.json",
        dict(
            completed=True,
            phase="subject_attribute_family_training_completed",
            methods=11,
            rankings=440,
            gallery_per_cohort=315,
            AI_relations=12600,
            quality_metrics=None,
            formal_metrics=None,
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
        completed=True,
        formal_metrics=None,
    )
    verify()
    print({k: v for k, v in decision.items() if k.startswith("development_")})


def verify() -> None:
    """Verify the recorded artifacts and their consistency."""
    guard()
    checked("completion_receipt.json")


@contextmanager
def locked() -> Iterator[bool]:
    """Return locked for this workflow.

    Yields:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = PROJECT_ROOT / "tmp/module2_subject_attribute_learning.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as h:
        h.seek(0)
        if not h.read(1):
            h.write(b"0")
            h.flush()
        h.seek(0)
        try:
            msvcrt.locking(h.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in locked", exc_info=True
            )
            yield False
            return
        try:
            yield True
        finally:
            h.seek(0)
            msvcrt.locking(h.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    with locked() as acquired:
        if not acquired:
            raise RuntimeError("Another subject-attribute experiment holds the lock")
        if action == "prepare":
            prepare()
        elif action == "verify":
            verify()
            print(dict(seal_verified=True))
        else:
            guard()
            for name, stage in [
                ("bank_receipt.json", bank),
                ("training_receipt.json", train),
                ("prediction_receipt.json", prediction),
                ("evaluation_receipt.json", evaluation),
                ("completion_receipt.json", finish),
            ]:
                if (ROOT / name).exists():
                    checked(name)
                else:
                    stage()
