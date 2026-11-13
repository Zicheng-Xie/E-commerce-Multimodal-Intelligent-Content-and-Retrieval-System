"""Family-masked supervised fusion of frozen full multimodal model evidence."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import run  # noqa: E402,F401
from learned_fusion_model import FEATURE_NAMES, POLICY, fit, predict  # noqa: E402
from multimodal_ecommerce.blind_evaluation import verify_files  # noqa: E402
from multimodal_ecommerce.business_acceptance import product_family_keys  # noqa: E402
from multimodal_ecommerce.business_evaluation import METRICS  # noqa: E402
from multimodal_ecommerce.dataset_release import read_rows  # noqa: E402
from multimodal_ecommerce.holdout_data import fingerprints  # noqa: E402
from multimodal_ecommerce.intent_retrieval import normalized_ranks  # noqa: E402
from multimodal_ecommerce.interval_retrieval_metrics import (  # noqa: E402
    evaluate_intervals,
)
from multimodal_ecommerce.safety import (  # noqa: E402
    PROJECT_ROOT,
    sha256_file,
    write_json,
)
from multimodal_ecommerce.semantic_ranker import (  # noqa: E402 - local path bootstrap must run first.; noqa: E402
    SemanticRankerTrainingMaskInputs,
    assign_folds,
    ranked_ids,
    training_mask,
)
from querywise_paired_bounds import paired_querywise  # noqa: E402
from roi_semantic_gate_experiment import COMPARISONS as OLD_COMPARISONS  # noqa: E402
from roi_semantic_gate_experiment import DISPLAY as OLD_DISPLAY  # noqa: E402
from roi_semantic_gate_experiment import METHODS as OLD_METHODS  # noqa: E402

ROOT = PROJECT_ROOT / "artifacts/module2_learned_fusion"
PARENT = PROJECT_ROOT / "artifacts/module2_roi_semantic_gate"
DATA = PROJECT_ROOT / "artifacts/module2_primary_subjects"
WHOLE = PROJECT_ROOT / "artifacts/module2_dual_path_development"
CROP = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
COHORTS = ("tune", "regression", "fresh")
NEW = ("learned_pointwise_oof", "learned_ranknet_oof", "learned_ranknet_blend_oof")
PRIMARY = "learned_ranknet_blend_oof"
METHODS = (*OLD_METHODS, *NEW)
DISPLAY = dict(
    OLD_DISPLAY,
    **{
        NEW[0]: "家族隔离有监督点式融合",
        NEW[1]: "家族隔离困难负例RankNet融合",
        NEW[2]: "家族隔离RankNet加原base融合",
    },
)
BASELINES = (
    "cross20",
    "global",
    "region_title",
    "dual_context_union",
    "whole_crop_context_union",
)
COMPARISONS = list(
    dict.fromkeys(
        OLD_COMPARISONS
        + [(m, b) for m in NEW for b in BASELINES]
        + [(NEW[2], NEW[1]), (NEW[1], NEW[0])]
    )
)
SOURCES = [
    "scripts/learned_fusion_experiment.py",
    "scripts/learned_fusion_model.py",
    "tests/test_learned_fusion_model.py",
    "scripts/querywise_paired_bounds.py",
    "tests/test_querywise_paired_bounds.py",
]
METADATA = [
    "artifacts/module2_structure_development/selected.evaluator.jsonl",
    "artifacts/module2_primary_form_coverage/fresh/selected.evaluator.jsonl",
    "data/module2_primary_form_candidates/records.jsonl",
    "data/module2_subject_coverage_candidates/records.jsonl",
    "data/module2_primary_subject_candidates/records.jsonl",
]


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    """Return rel for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def once(path: Path, value: object) -> None:
    """Persist a new result without replacing an existing artifact.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        assert load(path) == value, str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".pending")
        if tmp.exists():
            assert load(tmp) == value
        else:
            with tmp.open("x", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        tmp.replace(path)


def parent_guard() -> None:
    """Apply the parent guard workflow step."""
    for root in (PARENT, DATA, WHOLE, CROP):
        receipt = load(root / "completion_receipt.json")
        verify_files(root, receipt["files"])
        verify_files(PROJECT_ROOT, receipt["source_files"])
        verify_files(PROJECT_ROOT, receipt["project_files"])


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    parent_guard()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    paths = [PROJECT_ROOT / s for s in SOURCES + METADATA]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        ROOT / "engineering_checks.pre.json",
        PARENT / "completion_receipt.json",
        PARENT / "post_experiment_exclusion_snapshot.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
    ]
    for cohort in COHORTS:
        paths += [
            DATA / cohort / n
            for n in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
                "runtime_scores.json",
            )
        ]
        paths += [
            PARENT / cohort / n
            for n in ("predictions.json", "internal_metrics.ai.development.json")
        ]
        for q in load(DATA / cohort / "queries.public.json"):
            if q["image_path"]:
                paths.append(Path(q["image_path"]))
                paths += [
                    DATA / cohort / "runs" / q["id"] / n
                    for n in ("features.json", "caption_features.json")
                ]
                paths += [
                    WHOLE / cohort / "runs" / q["id"] / n
                    for n in ("old_cross.json", "new_cross.json")
                ]
                paths.append(CROP / cohort / "runs" / q["id"] / "crop_cross.json")
        paths += [
            Path(d["image_path"]) for d in load(DATA / cohort / "catalog.public.json")
        ]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            sources=SOURCES,
            policy=POLICY,
            features=FEATURE_NAMES,
            primary=PRIMARY,
            methods=METHODS,
            comparisons=COMPARISONS,
            K=10,
            original_AI_relations=11840,
            image_training_queries=24,
            unchanged_text_queries=16,
            family_policy="Connected source ID, normalized title, merchant/model metadata, byte/pixel identity and dHash<=4 across actual query/gallery sources; three query-family and three gallery-family folds; every scaler, mining and gradient excludes both held families and held-query source families in both roles.",
            training_evidence="Full original Qwen/FG scores plus 13 public full-model features; exactly 504 fixed image candidate positions, no imputation outside the union.",
            hyperparameters="Existing 400-step/0.1/0.05 recipe, no validation search; logistic and hard-negative pairwise controls. Primary fixed 0.65 rank(RankNet)+0.35 original/new full-gallery base vote.",
            unknowns="null excluded from training, retained in all evaluation bounds",
            OOF_scope="Group-masked internal development, all earlier data exposed; per-cell cross-fit rankings stitch nine models and are not a deployable single model or independent acceptance.",
            deployment_candidate="Two separate all-development fitted fusion models stored without reporting their in-sample quality; no auto default promotion.",
            original_neural_models_unchanged=True,
            new_neural_forward_calls=0,
            new_AI_relevance_labels=0,
            human_labels=0,
            formal_metrics=None,
            independent_acceptance_started=False,
            default_strategy_changed=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="frozen_before_fitting",
            formal_metrics=None,
            quality_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    print({"protocol_frozen": True, "methods": len(METHODS)}, flush=True)


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    protocol = load(ROOT / "protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    assert protocol["policy"] == POLICY and protocol["features"] == FEATURE_NAMES
    for path in ROOT.glob("*receipt.json"):
        receipt = load(path)
        if "files" in receipt:
            verify_files(ROOT, receipt["files"])


def receipt(name: str, paths: list[Path], **extra: object) -> None:
    """Record file hashes and provenance for the completed stage.

    Args:
        name: Name used by this operation.
        paths: Paths used by this operation.
        **extra: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through once.
    """
    once(
        ROOT / name,
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths},
            **extra,
        ),
    )


def np_once(path: Path, **arrays: np.ndarray) -> None:
    """Return np once for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.
        **arrays: Named options forwarded to the underlying operation.

    Notes:
        Writes derived artifacts through np.savez_compressed.
    """
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            assert all(
                np.array_equal(saved[k], v, equal_nan=True) for k, v in arrays.items()
            )
        return
    pending = path.with_name(path.name + ".pending")
    if not pending.exists():
        with pending.open("xb") as handle:
            np.savez_compressed(handle, allow_pickle=False, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    with np.load(pending, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        assert all(
            np.array_equal(saved[k], v, equal_nan=True) for k, v in arrays.items()
        )
    pending.replace(path)


def bank() -> None:
    """Apply the bank workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    metadata_rows = [r for s in METADATA for r in read_rows(PROJECT_ROOT / s)]
    metadata = {r["id"]: r for r in metadata_rows}
    image_rows = {
        str(Path(r["image_path"]).resolve()): r
        for r in metadata_rows
        if r.get("image_path")
    }
    docs: dict = {}
    queries = []
    for c in COHORTS:
        for d in load(DATA / c / "catalog.public.json"):
            if d["id"] in docs:
                assert docs[d["id"]] == d
            docs[d["id"]] = d
        for q in load(DATA / c / "queries.public.json"):
            if q["image_path"]:
                queries.append(dict(q, cohort=c, qualified=c + "/" + q["id"]))
    assert len(queries) == 24 and all(k in metadata for k in docs)
    nodes = {k: dict(row=metadata[k], image=d["image_path"]) for k, d in docs.items()}
    qnodes = []
    for q in queries:
        row = image_rows[str(Path(q["image_path"]).resolve())]
        nodes[row["id"]] = dict(row=row, image=q["image_path"])
        qnodes.append(row["id"])
    parents = {k: k for k in nodes}

    def find(k: str) -> str:
        """Return find for this workflow.

        Args:
            k: K used by this operation.

        Returns:
            Count or index produced by this operation.
        """
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    def join(a: str, b: str) -> None:
        """Apply the join workflow step.

        Args:
            a: A used by this operation.
            b: B used by this operation.
        """
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    owners: dict
    owners, signatures, links = {}, {}, []
    for key, item in sorted(nodes.items()):
        signature = fingerprints(Path(item["image"]))
        signatures[key] = signature
        keys = product_family_keys(item["row"]) | set(
            item["row"].get("family_keys", [])
        )
        keys |= {"bytes:" + signature["sha256"], "pixels:" + signature["pixels"]}
        for public_key in sorted(keys):
            if public_key in owners:
                join(key, owners[public_key])
                links.append([key, owners[public_key], public_key])
            else:
                owners[public_key] = key
    ordered = sorted(nodes)
    for i, a in enumerate(ordered):
        for b in ordered[:i]:
            if (signatures[a]["dhash"] ^ signatures[b]["dhash"]).bit_count() <= 4:
                join(a, b)
                links.append([a, b, "dHash<=4"])
    gids = sorted(docs)
    qfamilies, gfamilies = [find(k) for k in qnodes], [find(k) for k in gids]
    qmap = assign_folds(qfamilies, 3, POLICY["seed"])
    gmap = assign_folds(gfamilies, 3, POLICY["seed"])
    qfold, gfold = np.array([qmap[f] for f in qfamilies]), np.array(
        [gmap[f] for f in gfamilies]
    )
    shape = (24, len(gids))
    x, y, valid, base = (
        np.zeros((*shape, len(FEATURE_NAMES))),
        np.full(shape, np.nan),
        np.zeros(shape, bool),
        np.zeros(shape),
    )
    positions = {k: i for i, k in enumerate(gids)}
    pools = []
    for qi, q in enumerate(queries):
        c, qid = q["cohort"], q["id"]
        old, new = [
            load(DATA / c / "runs" / qid / n)
            for n in ("features.json", "caption_features.json")
        ]
        ce_old = load(WHOLE / c / "runs" / qid / "old_cross.json")
        ce_new = load(WHOLE / c / "runs" / qid / "new_cross.json")
        crop = load(CROP / c / "runs" / qid / "crop_cross.json")
        pool = ce_old["pool"]
        assert pool == ce_new["pool"] == crop["pool"] and old["ids"] == new["ids"]
        ids, signals = old["ids"], np.asarray(old["features"])
        fg = load(DATA / c / "runtime_scores.json")[qid]
        extras = np.column_stack(
            [
                *[
                    normalized_ranks(pool, row["logits"])
                    for row in (ce_old, ce_new, crop)
                ],
                *[
                    normalized_ranks(ids, row["base"])[[ids.index(k) for k in pool]]
                    for row in (old, new)
                ],
                *[
                    normalized_ranks(ids, fg[key]["fused_scores"])[
                        [ids.index(k) for k in pool]
                    ]
                    for key in ("whole_scores", "roi_scores")
                ],
            ]
        )
        labels = {
            r["candidate_id"]: r["grade"]
            for r in read_rows(DATA / c / "judgments.ai.jsonl")
            if r["query_id"] == qid
        }
        for pi, cid in enumerate(pool):
            gi, local = positions[cid], ids.index(cid)
            x[qi, gi] = np.r_[signals[local], extras[pi]]
            y[qi, gi] = np.nan if labels[cid] is None else labels[cid]
            valid[qi, gi] = True
            base[qi, gi] = 0.5 * extras[pi, 3] + 0.5 * extras[pi, 4]
        pools.append(pool)
    assert int(valid.sum()) == 504 and np.isfinite(x[valid]).all()
    meta = dict(
        query_ids=[q["qualified"] for q in queries],
        queries=queries,
        gallery_ids=gids,
        query_families=qfamilies,
        gallery_families=gfamilies,
        query_source_ids=qnodes,
        public_family_links=links,
        pools=pools,
        original_feature_names=FEATURE_NAMES,
        roles="All exposed AI development; not acceptance",
    )
    np_once(
        ROOT / "feature_bank.npz",
        features=x,
        labels=y,
        valid=valid,
        base=base,
        query_folds=qfold,
        gallery_folds=gfold,
    )
    once(ROOT / "feature_metadata.json", meta)
    receipt(
        "bank_receipt.json",
        [ROOT / "feature_bank.npz", ROOT / "feature_metadata.json"],
        actual_full_model_feature_rows=504,
        features=len(FEATURE_NAMES),
        known=int(np.isfinite(y[valid]).sum()),
        unknown=int(np.isnan(y[valid]).sum()),
        query_families=len(set(qfamilies)),
        gallery_families=len(set(gfamilies)),
        new_neural_calls=0,
        unknowns_imputed=False,
    )


def train_values() -> tuple[dict, dict, list[dict], dict]:
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
    scores = {name: np.full(valid.shape, np.nan) for name in ("pointwise", "ranknet")}
    models, audits = {}, []
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
            for name in scores:
                key = f"{name}_q{hq}_g{hg}"
                model = fit(x, y, allowed, base, pairwise=name == "ranknet")
                values = predict(x, model)
                scores[name][held] = values[held]
                models[key] = model
                cells = np.argwhere(allowed & np.isfinite(y))
                audits.append(
                    dict(
                        model=key,
                        **evidence,
                        validation_cells=int(held.sum()),
                        training_cells=[
                            [meta["query_ids"][q], meta["gallery_ids"][g]]
                            for q, g in cells
                        ],
                    )
                )
    assert all(np.isfinite(values[valid]).all() for values in scores.values())
    candidates = {
        name: fit(x, y, valid, base, pairwise=name == "ranknet") for name in scores
    }
    return scores, models, audits, candidates


def train() -> None:
    """Fit the configured model on the allowed training data.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
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
        cross_fitted_models=18,
        all_development_candidates=2,
        evaluation_uses_all_fit_candidates=False,
        hyperparameter_search=False,
        OOF_rankings_are_nine_model_stitches=True,
        formal_metrics=None,
    )


def orders() -> tuple[dict, dict]:
    """Return orders for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    meta = load(ROOT / "feature_metadata.json")
    positions = {cid: i for i, cid in enumerate(meta["gallery_ids"])}
    qpos = {qid: i for i, qid in enumerate(meta["query_ids"])}
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        point, pair = b["pointwise"].copy(), b["ranknet"].copy()
    with np.load(ROOT / "feature_bank.npz", allow_pickle=False) as b:
        base = b["base"].copy()
    output, details = {}, {}
    for c in COHORTS:
        previous = load(PARENT / c / "predictions.json")
        out, evidence = {}, {}
        for q in load(DATA / c / "queries.public.json"):
            qid = q["id"]
            baseline = previous[qid]["cross20"]
            if q["image_path"]:
                qi = qpos[c + "/" + qid]
                pool = meta["pools"][qi]
                gi = [positions[k] for k in pool]
                values = [
                    point[qi, gi],
                    pair[qi, gi],
                    0.65 * normalized_ranks(pool, pair[qi, gi]) + 0.35 * base[qi, gi],
                ]
                tail = [cid for cid in baseline if cid not in set(pool)]
                new = {name: ranked_ids(pool, s) + tail for name, s in zip(NEW, values)}
                evidence[qid] = dict(
                    scores={
                        name: dict(zip(pool, s.tolist()))
                        for name, s in zip(NEW, values)
                    },
                    pool=pool,
                    same_cross20_tail=True,
                    score_is_probability=False,
                )
            else:
                new = {name: baseline for name in NEW}
                evidence[qid] = dict(route="original_text_exact")
            out[qid] = dict(previous[qid], **new)
        output[c], details[c] = out, evidence
    return output, details


def prediction() -> None:
    """Apply the prediction workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    rankings, details = orders()
    paths = []
    for c in COHORTS:
        for n, value in [
            ("predictions.json", rankings[c]),
            ("runtime_scores.json", details[c]),
        ]:
            p = ROOT / c / n
            once(p, value)
            paths.append(p)
    receipt(
        "prediction_receipt.json",
        paths,
        full_rankings=880,
        preserved_parent_rankings=760,
        unchanged_text_queries=16,
        direct_training_scores_reported_as_quality=False,
    )


def evaluated(c: str) -> tuple[dict, dict]:
    """Return evaluated for this workflow.

    Args:
        c: C used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    rankings = load(ROOT / c / "predictions.json")
    labels = read_rows(DATA / c / "judgments.ai.jsonl")
    modes = {
        q["id"]: "image" if q["image_path"] else "text"
        for q in load(DATA / c / "queries.public.json")
    }
    return evaluate_intervals(rankings, labels, modes), paired_querywise(
        rankings, labels, modes, COMPARISONS
    )


def evaluation() -> None:
    """Apply the evaluation workflow step.

    Notes:
        Writes derived artifacts through once, receipt.
    """
    guard()
    paths = []
    for c in COHORTS:
        metrics, paired = evaluated(c)
        for n, value in [
            ("internal_metrics.ai.development.json", metrics),
            ("internal_paired_querywise.json", paired),
        ]:
            p = ROOT / c / n
            once(p, value)
            paths.append(p)
        print({"evaluated": c, "role": "exposed_group_masked_development"}, flush=True)
    receipt(
        "evaluation_receipt.json",
        paths,
        original_AI_relations=11840,
        new_labels=0,
        all_queries_retained=True,
        formal_metrics=None,
    )


def verification() -> None:
    """Apply the verification workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    guard()
    rankings, details = orders()
    scores, models, audits, candidates = train_values()
    assert models == load(ROOT / "fold_models.json") and audits == load(
        ROOT / "fold_audits.json"
    )
    assert candidates == load(ROOT / "all_development_candidates.json")
    with np.load(ROOT / "oof_scores.npz", allow_pickle=False) as b:
        assert all(np.array_equal(b[k], v, equal_nan=True) for k, v in scores.items())
    for c in COHORTS:
        assert rankings[c] == load(ROOT / c / "predictions.json")
        assert details[c] == load(ROOT / c / "runtime_scores.json")
        previous = load(PARENT / c / "predictions.json")
        for qid, methods in rankings[c].items():
            assert set(methods) == set(METHODS)
            assert all(len(v) == len(set(v)) == 296 for v in methods.values())
            assert all(methods[k] == previous[qid][k] for k in OLD_METHODS)
            if qid.startswith("text_"):
                assert all(methods[k] == methods["cross20"] for k in NEW)
        metrics, paired = evaluated(c)
        assert metrics == load(ROOT / c / "internal_metrics.ai.development.json")
        assert paired == load(ROOT / c / "internal_paired_querywise.json")
        old = load(PARENT / c / "internal_metrics.ai.development.json")
        for scope in old["scopes"]:
            assert all(
                metrics["scopes"][scope][k] == old["scopes"][scope][k]
                for k in OLD_METHODS
            )
    once(
        ROOT / "verification.json",
        dict(
            passed=True,
            replayed_rankings=880,
            unchanged_parent_rankings=760,
            refitted_identical_models=20,
            held_query_and_gallery_families_excluded=True,
            formal_metrics=None,
        ),
    )


def report() -> None:
    """Summarize the recorded results of this development workflow.

    Notes:
        Writes derived artifacts through (ROOT /
        'internal_comparison.ai.development.md').write_text, once, receipt.
    """
    guard()
    assert load(ROOT / "verification.json")["passed"]
    lines = [
        "# 家族隔离有监督融合：内部AI开发对照",
        "",
        "固定296图库/40查询/K10，11840原AI关系不变，人工0。全部已曝光开发，OOF不是独立验收。",
        "使用完整原模型真实分数作为输入，新增融合层训练；没有替换、简化或补造神经模型分数。",
        "18个交叉拟合模型用于逐单元未见家族开发预测，2个全开发候选单独保存，其训练集成绩不用于质量。",
        "所有宏平均范围和共享未知配对边界均报告；不是置信区间，也不能覆盖AI误标。",
        "",
    ]
    gates: dict = {}
    for c in COHORTS:
        metrics = load(ROOT / c / "internal_metrics.ai.development.json")
        paired = load(ROOT / c / "internal_paired_querywise.json")
        lines += [
            f"## {c}",
            "",
            "|范围|方法|P@10|R@10|mAP|NDCG@10|MRR|Hit@1|",
            "|---|---|---|---|---|---|---|---|",
        ]
        gates[c] = {}
        for scope, methods in metrics["scopes"].items():
            gates[c][scope] = {}
            for name in METHODS:
                ranges = methods[name]["ranges"]
                cells = [
                    " / ".join(
                        "undefined" if v is None else f"{v*100:.2f}" for v in ranges[m]
                    )
                    for m in METRICS
                ]
                lines.append("|" + "|".join([scope, DISPLAY[name], *cells]) + "|")
            for name in NEW:
                gates[c][scope][name] = dict(
                    precision_60=methods[name]["precision_60_lower_bound_met"],
                    recall_85=methods[name]["recall_85_lower_bound_met"],
                    nonregression=all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for baseline in BASELINES
                        for v in paired["scopes"][scope][name][baseline].values()
                    ),
                )
        lines += [
            "",
            "逐查询、共享未知标签配对增量详见同目录internal_paired_querywise.json；所有阴性与退化保留。",
            "",
        ]
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
    (ROOT / "internal_comparison.ai.development.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    receipt(
        "report_receipt.json",
        [ROOT / "decision.json", ROOT / "internal_comparison.ai.development.md"],
        quality_numbers_internal_only=True,
        browser_interaction_tested=False,
    )
    print(
        {k: v for k, v in decision.items() if k.startswith("development_")}, flush=True
    )


def seal() -> None:
    """Seal the completed outputs with their integrity metadata.

    Notes:
        Writes derived artifacts through once, target.parent.mkdir, write_json.
    """
    guard()
    parent_guard()
    snapshot = load(PARENT / "post_experiment_exclusion_snapshot.json")
    once(
        ROOT / "post_experiment_exclusion_snapshot.json",
        dict(
            snapshot,
            parent_snapshot_sha256=sha256_file(
                PARENT / "post_experiment_exclusion_snapshot.json"
            ),
            new_supervised_images=0,
            development_only_no_new_independent_test=True,
        ),
    )
    paths = [PROJECT_ROOT / s for s in SOURCES] + sorted(
        (PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")
    )
    sources = {rel(p): sha256_file(p) for p in paths}
    for p in paths:
        target = ROOT / "source_release" / rel(p)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        assert sha256_file(target) == sha256_file(p)
    once(ROOT / "source_release/manifest.json", sources)
    decision = load(ROOT / "decision.json")
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=True,
            sealed=True,
            role="exposed_AI_group_masked_supervised_fusion_not_acceptance",
            methods=22,
            full_rankings=880,
            original_AI_relations=11840,
            human_labels=0,
            cross_fit_models=18,
            all_development_candidates=2,
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            development_precision_target_met=decision[
                "development_precision_target_met"
            ],
            development_recall_preference_met=decision[
                "development_recall_preference_met"
            ],
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    assert not any(ROOT.rglob("*.pending"))
    files = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "completion_receipt.json"
    }
    once(
        ROOT / "completion_receipt.json",
        dict(
            files=files,
            source_files=sources,
            project_files=snapshot["files"],
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
        ),
    )
    print({"sealed_files": len(files), "source_snapshots": len(sources)}, flush=True)


def completed() -> bool:
    """Return completed for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if not (ROOT / "completion_receipt.json").exists():
        return False
    value = load(ROOT / "completion_receipt.json")
    verify_files(ROOT, value["files"])
    verify_files(PROJECT_ROOT, value["source_files"])
    verify_files(PROJECT_ROOT, value["project_files"])
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    parent_guard()
    assert load(ROOT / "public_status.json")["sealed"]
    return True


@contextmanager
def locked() -> Iterator[bool]:
    """Return locked for this workflow.

    Yields:
        Values produced by this operation, in the stated return order.

    Notes:
        Writes derived artifacts through path.parent.mkdir.
    """
    from multimodal_ecommerce import platform_lock as msvcrt

    path = PROJECT_ROOT / "tmp/module2_learned_fusion.lock"
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("xb") as first:
            first.write(b"0")
    except FileExistsError:
        logging.getLogger(__name__).debug("Reusing existing worker lock file")
    with path.open("r+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in locked", exc_info=True
            )
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def run_all() -> None:
    """Return run all for this workflow."""
    if completed():
        print({"seal_verified": True}, flush=True)
        return
    with locked() as acquired:
        if not acquired:
            print({"another_runner_holds_lock": True}, flush=True)
            return
        if completed():
            return
        prepare()
        for action, marker in [
            (bank, "bank_receipt.json"),
            (train, "training_receipt.json"),
            (prediction, "prediction_receipt.json"),
            (evaluation, "evaluation_receipt.json"),
            (verification, "verification.json"),
            (report, "report_receipt.json"),
        ]:
            guard()
            if not (ROOT / marker).exists():
                print({"stage": action.__name__}, flush=True)
                action()
        seal()
        assert completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "verify"])
    action = parser.parse_args().action
    if action == "verify":
        assert completed()
        print({"seal_verified": True})
    else:
        run_all()
