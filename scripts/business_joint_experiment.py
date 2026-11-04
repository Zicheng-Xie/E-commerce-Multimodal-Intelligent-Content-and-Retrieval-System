"""Complete DINOv2 joint-image attention, family-masked head, fixed Qwen fusion."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import run  # noqa: F401
from business_grounding_audit import COHORTS, DATA, PARENT
from business_grounding_audit import ROOT as CYCLE
from business_grounding_audit import load, once, relative, rows
from business_joint_model import POLICY, blend, fit_head
from multimodal_ecommerce.interval_retrieval_metrics import evaluate_intervals
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.semantic_ranker import (
    SemanticRankerTrainingMaskInputs,
    training_mask,
)
from querywise_paired_bounds import paired_querywise

ROOT = PROJECT_ROOT / "artifacts/module2_business_joint_attention"
SOURCE = PROJECT_ROOT / "artifacts/module2_subject_recall_learning"
METRIC = PROJECT_ROOT / "artifacts/module2_business_metric_learning"
NEW = (
    "business_metric_parent_pool_fusion",
    "business_joint_attention_only_ablation",
    "business_joint_attention_fusion",
)
PRIMARY = NEW[2]
BASELINES = ("global", "region_title", "cross20", "grounded_whole_zoom_union")


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        ValueError: This control was specified for a globally disabled unproven mask
            branch; Frozen24-image-query metadata changed.

    Notes:
        Writes derived artifacts through ROOT.mkdir, once.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assessment = load(CYCLE / "semantic_quality_assessment.json")
    if assessment["new_masks_in_primary_training_allowed"]:
        raise ValueError(
            "This control was specified for a globally disabled unproven mask branch"
        )
    public = load(CYCLE / "views_inputs.public.json")
    images = {
        r["key"]: {"image_path": r["image_path"], "sha256": r["source_sha256"]}
        for r in public
    }
    once(ROOT / "images.public.json", images)
    meta = load(SOURCE / "feature_metadata.json")
    if len(meta["query_ids"]) != 24:
        raise ValueError("Frozen24-image-query metadata changed")
    pools: defaultdict = defaultdict(list)
    for r in load(PARENT / "pair_plan.json"):
        key = r["cohort"] + "/" + r["query_id"]
        if r["candidate_id"] not in pools[key]:
            pools[key].append(r["candidate_id"])
    for c in COHORTS:
        parent = load(PARENT / c / "predictions.json")
        for q in load(DATA / c / "queries.public.json"):
            if q["image_path"]:
                p = pools[c + "/" + q["id"]]
                assert set(
                    parent[q["id"]]["grounded_whole_zoom_union"][: len(p)]
                ) == set(p)
    once(ROOT / "frozen_parent_pools.json", dict(pools))
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/business_joint_model.py",
        PROJECT_ROOT / "tests/test_business_joint_model.py",
        PROJECT_ROOT / "scripts/business_grounding_audit.py",
        PROJECT_ROOT / "scripts/querywise_paired_bounds.py",
        ROOT / "images.public.json",
        ROOT / "frozen_parent_pools.json",
        CYCLE / "semantic_quality_assessment.json",
        CYCLE / "semantic_assessment_receipt.json",
        SOURCE / "feature_metadata.json",
        SOURCE / "feature_bank.npz",
        METRIC / "OOF_scores.npz",
        METRIC / "training_receipt.json",
        PROJECT_ROOT / "configs/dino_model.json",
        PROJECT_ROOT / "configs/module2_current_target.json",
        PARENT / "completion_receipt.json",
    ]
    paths += [PROJECT_ROOT / r["image_path"] for r in images.values()]
    paths += sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py"))
    paths += [
        ROOT / "protocol.pre_execution.json",
        ROOT / "pre_execution_revision.json",
        ROOT / "source_history/business_joint_experiment.pre_execution.py",
    ]
    for c in COHORTS:
        paths += [
            DATA / c / n
            for n in (
                "catalog.public.json",
                "queries.public.json",
                "judgments.ai.jsonl",
            )
        ]
        paths.append(PARENT / c / "predictions.json")
    once(
        ROOT / "protocol.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "policy": POLICY,
            "primary": PRIMARY,
            "methods": [*BASELINES, *NEW],
            "data": "same315 per cohort,40 queries,12600 unchanged AI relations, fixedK10; all original queries retained",
            "frozen_backbone": "complete official pinned DINOv2-small, all12 layers float32, all512 patch tokens jointly interact before CLS extraction; no learned attention parameter update",
            "STIR_scope": "Distinct frozen-DINOv2 business adaptation inspired by concatenated-pixel STIR; not its paper checkpoint or end-to-end reproduction",
            "training": "class-balanced explicit business grades; unknown and held family cells never enter loss; nine source-family masks + separate all-dev candidate,400 head steps each",
            "fusion": "within exact fully Qwen-scored parent pool, .8 parent rank + .2 auxiliary rank; unchanged full parent tail; no new Qwen calls or missing score imputation",
            "main_method_predeclared": True,
            "post_result_selection": False,
            "text": "original315-gallery cross20 exactly",
            "formal_metrics": None,
            "human_labels": 0,
            "default_strategy_changed": False,
        },
    )
    print(
        json.dumps(
            {"joint_attention_protocol_frozen": True, "new_quality_results": None}
        ),
        flush=True,
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    for name, digest in load(ROOT / "protocol.json")["files"].items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError("Joint source/input changed: " + name)


def composition_fingerprint(picture: Image.Image) -> dict:
    """Return composition fingerprint for this workflow.

    Args:
        picture: Picture used by this operation.

    Returns:
        Mapping with fields: sha256, pixels, dhash.

    Notes:
        Writes derived artifacts through picture.save.
    """
    from PIL import Image

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    small = list(
        picture.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    )
    dhash = 0
    for y in range(8):
        for x in range(8):
            dhash = (dhash << 1) | (small[y * 9 + x] > small[y * 9 + x + 1])
    return {
        "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "pixels": hashlib.sha256(
            str(picture.size).encode() + picture.tobytes()
        ).hexdigest(),
        "dhash": dhash,
    }


def encode() -> None:
    """Encode the configured inputs into model feature representations.

    Raises:
        ValueError: Pinned full DINO checkpoint changed; Original default processor
            shape changed; Incomplete joint vision backbone loading; Complete12-layer
            backbone required; All512 patch tokens must participate; Missing real joint
            attention cells.

    Notes:
        Writes derived artifacts through np.savez_compressed, once, write_json.
    """
    guard()
    if (ROOT / "feature_receipt.json").exists():
        return
    from gpu_admission import gpu_available

    available, info = gpu_available()
    if not available:
        write_json(ROOT / "waiting.json", info)
        print(json.dumps({"waiting": True, "reason": info}), flush=True)
        return
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModel

    config = load(PROJECT_ROOT / "configs/dino_model.json")
    model_path = Path(config["path"])
    for name, digest in config["files"].items():
        if sha256_file(model_path / name) != digest:
            raise ValueError("Pinned full DINO checkpoint changed")
    images = load(ROOT / "images.public.json")
    meta = load(SOURCE / "feature_metadata.json")
    with np.load(SOURCE / "feature_bank.npz", allow_pickle=False) as bank:
        valid = bank["valid"].copy()
    processor = AutoImageProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False
    )
    pixel_cache = {}
    rgb_cache = {}
    for key, r in images.items():
        with Image.open(PROJECT_ROOT / r["image_path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        tensor = processor(images=image, return_tensors="pt")["pixel_values"][0]
        if tuple(tensor.shape) != (3, 224, 224):
            raise ValueError("Original default processor shape changed")
        pixel_cache[key] = tensor
        mean = np.asarray(processor.image_mean).reshape(3, 1, 1)
        std = np.asarray(processor.image_std).reshape(3, 1, 1)
        rgb_cache[key] = (
            np.clip(np.rint((tensor.numpy() * std + mean) * 255), 0, 255)
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )
    model, loading = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Incomplete joint vision backbone loading")
    model.eval().to("cuda")
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    if model.config.num_hidden_layers != 12:
        raise ValueError("Complete12-layer backbone required")
    features = np.full((*valid.shape, 384), np.nan, dtype=np.float32)
    cells = np.argwhere(valid)
    calls = []
    exposure = []
    try:
        for start in range(0, len(cells), 4):
            batch = cells[start : start + 4]
            pixels = []
            for q, g in batch:
                qid = meta["query_ids"][q]
                gid = meta["gallery_ids"][g]
                pixels.append(torch.cat([pixel_cache[qid], pixel_cache[gid]], dim=1))
                picture = Image.fromarray(
                    np.concatenate([rgb_cache[qid], rgb_cache[gid]], axis=0)
                )
                exposure.append(
                    {
                        "id": "business_joint_input/"
                        + hashlib.sha256((qid + "|" + gid).encode()).hexdigest(),
                        "source_ids": [qid, gid],
                        "source_sha256": [images[qid]["sha256"], images[gid]["sha256"]],
                        "composition": "official per-image processor then224x448 RGB vertical input tensor; image not saved to disk",
                        "fingerprints": composition_fingerprint(picture),
                        "role": "exposed development transformed model input; never unseen acceptance",
                    }
                )
            tensor = torch.stack(pixels).to("cuda")
            torch.cuda.synchronize()
            beg = time.perf_counter()
            with torch.inference_mode():
                out = model(pixel_values=tensor)
            torch.cuda.synchronize()
            if tuple(out.last_hidden_state.shape[1:]) != (513, 384):
                raise ValueError("All512 patch tokens must participate")
            values = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            for (q, g), vector in zip(batch, values):
                features[q, g] = vector
            calls.append(
                {
                    "pairs": len(batch),
                    "input_shape": list(tensor.shape),
                    "tokens": 513,
                    "seconds": time.perf_counter() - beg,
                }
            )
            if (start + len(batch)) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "real_joint_pairs_completed": start + len(batch),
                            "total": len(cells),
                        }
                    ),
                    flush=True,
                )
        if not np.isfinite(features[valid]).all():
            raise ValueError("Missing real joint attention cells")
        np.savez_compressed(ROOT / "joint_features.npz", features=features, valid=valid)
        runtime = {
            "loading": loading,
            "parameters": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "layers": model.config.num_hidden_layers,
            "actual_joint_pairs": len(cells),
            "full_batch_forward_calls": len(calls),
            "calls": calls,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        once(ROOT / "joint_runtime.json", runtime)
        once(ROOT / "joint_input_exposure.json", exposure)
        once(
            ROOT / "feature_receipt.json",
            {
                "files": {
                    relative(ROOT / n): sha256_file(ROOT / n)
                    for n in (
                        "joint_features.npz",
                        "joint_runtime.json",
                        "joint_input_exposure.json",
                    )
                },
                "complete": True,
                "formal_metrics": None,
            },
        )
    finally:
        del model
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "complete_joint_attention_features": True,
                "actual_pairs": len(cells),
                "formal_metrics": None,
            }
        ),
        flush=True,
    )


def checked(name: str) -> Any:
    """Return checked for this workflow.

    Args:
        name: Name used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    value = load(ROOT / name)
    for file, digest in value["files"].items():
        if sha256_file(PROJECT_ROOT / file) != digest:
            raise ValueError("Joint evidence changed: " + file)
    return value


def train() -> None:
    """Fit the configured model on the allowed training data.

    Raises:
        ValueError: Changed attention head checkpoint; Incomplete or leaking joint OOF.

    Notes:
        Writes derived artifacts through (ROOT / 'models').mkdir, np.save,
        np.savez_compressed, once.
    """
    guard()
    checked("feature_receipt.json")
    if (ROOT / "training_receipt.json").exists():
        checked("training_receipt.json")
        return
    from safetensors.torch import save_file

    meta = load(SOURCE / "feature_metadata.json")
    with np.load(SOURCE / "feature_bank.npz", allow_pickle=False) as bank:
        y, valid, qf, gf = [
            bank[k].copy() for k in ("labels", "valid", "query_folds", "gallery_folds")
        ]
    with np.load(ROOT / "joint_features.npz", allow_pickle=False) as bank:
        features = bank["features"].copy()
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    audits = []
    (ROOT / "models").mkdir(exist_ok=True)
    for hq in range(3):
        for hg in range(3):
            name = f"q{hq}_g{hg}"
            checkpoint = ROOT / "models" / (name + ".safetensors")
            record = ROOT / "models" / (name + ".json")
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
            if record.exists():
                saved = load(record)
                if (
                    sha256_file(checkpoint) != saved["checkpoint_sha256"]
                    or sha256_file(ROOT / "models" / (name + ".npy"))
                    != saved["scores_sha256"]
                ):
                    raise ValueError("Changed attention head checkpoint")
                scores = np.load(ROOT / "models" / (name + ".npy"), allow_pickle=False)
            else:
                started = time.perf_counter()
                model, scores, trace = fit_head(
                    features, y, allowed, valid, POLICY["seed"] + hq * 3 + hg
                )
                save_file(model.state_dict(), checkpoint)
                np.save(ROOT / "models" / (name + ".npy"), scores, allow_pickle=False)
                once(
                    record,
                    {
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "scores_sha256": sha256_file(ROOT / "models" / (name + ".npy")),
                        "steps": POLICY["steps"],
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "seconds": time.perf_counter() - started,
                        "trace": trace,
                    },
                )
            held = valid & (qf[:, None] == hq) & (gf[None, :] == hg)
            oof[held] = scores[held]
            audits.append(
                {
                    "model": name,
                    "family_audit": evidence,
                    "known_training_cells": int((allowed & np.isfinite(y)).sum()),
                    "unknown_excluded": int((allowed & ~np.isfinite(y)).sum()),
                    "training_held_overlap": bool((allowed & held).any()),
                }
            )
            print(
                json.dumps(
                    {"joint_head_fold_completed": name, "full_steps": POLICY["steps"]}
                ),
                flush=True,
            )
    if not np.isfinite(oof[valid]).all() or any(
        r["training_held_overlap"] for r in audits
    ):
        raise ValueError("Incomplete or leaking joint OOF")
    np.savez_compressed(ROOT / "OOF_scores.npz", scores=oof, valid=valid)
    model, _, trace = fit_head(features, y, valid, valid, POLICY["seed"] + 100)
    save_file(model.state_dict(), ROOT / "models/all_development.safetensors")
    once(
        ROOT / "models/all_development.json",
        {
            "steps": POLICY["steps"],
            "trace": trace,
            "training_only_not_quality_evaluation": True,
        },
    )
    once(ROOT / "family_training_audits.json", audits)
    paths = sorted((ROOT / "models").glob("*")) + [
        ROOT / "OOF_scores.npz",
        ROOT / "family_training_audits.json",
    ]
    once(
        ROOT / "training_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "fold_models": 9,
            "all_dev_candidates": 1,
            "formal_metrics": None,
        },
    )


def predictions() -> dict:
    """Return predictions for this workflow.

    Returns:
        Result mapping produced by the workflow.

    Raises:
        ValueError: Missing real complete auxiliary scores.
    """
    guard()
    checked("training_receipt.json")
    meta = load(SOURCE / "feature_metadata.json")
    qi = {q: i for i, q in enumerate(meta["query_ids"])}
    gi = {g: i for i, g in enumerate(meta["gallery_ids"])}
    pools = load(ROOT / "frozen_parent_pools.json")
    with np.load(ROOT / "OOF_scores.npz", allow_pickle=False) as bank:
        joint = bank["scores"].copy()
    with np.load(METRIC / "OOF_scores.npz", allow_pickle=False) as bank:
        metric = bank["scores"].copy()
    outputs = {}
    for c in COHORTS:
        parent = load(PARENT / c / "predictions.json")
        queries = load(DATA / c / "queries.public.json")
        ids = [r["id"] for r in load(DATA / c / "catalog.public.json")]
        out = {}
        for q in queries:
            key = q["id"]
            out[key] = {m: parent[key][m] for m in BASELINES}
            if q["image_path"]:
                qq = qi[c + "/" + key]
                pool = pools[c + "/" + key]
                jj = {cid: float(joint[qq, gi[cid]]) for cid in ids}
                mm = {cid: float(metric[qq, gi[cid]]) for cid in pool}
                if not np.isfinite([*jj.values(), *mm.values()]).all():
                    raise ValueError("Missing real complete auxiliary scores")
                old = parent[key]["grounded_whole_zoom_union"]
                out[key][NEW[0]] = blend(pool, old, mm)
                out[key][NEW[1]] = sorted(ids, key=lambda cid: (-jj[cid], cid))
                out[key][NEW[2]] = blend(pool, old, {cid: jj[cid] for cid in pool})
            else:
                out[key].update({m: parent[key]["cross20"] for m in NEW})
            assert all(
                len(v) == len(set(v)) == 315 and set(v) == set(ids)
                for v in out[key].values()
            )
        outputs[c] = out
    return outputs


def evaluate() -> None:
    """Apply the evaluate workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    comparisons = [(m, b) for m in NEW for b in BASELINES] + [(PRIMARY, NEW[0])]
    for c, out in predictions().items():
        once(ROOT / c / "predictions.json", out)
        labels = rows(DATA / c / "judgments.ai.jsonl")
        modes = {
            q["id"]: "image" if q["image_path"] else "text"
            for q in load(DATA / c / "queries.public.json")
        }
        once(
            ROOT / c / "internal_metrics.ai.development.json",
            evaluate_intervals(out, labels, modes),
        )
        once(
            ROOT / c / "internal_paired_querywise.json",
            paired_querywise(out, labels, modes, comparisons),
        )
    paths = list(ROOT.glob("*/predictions.json")) + list(ROOT.glob("*/internal_*.json"))
    once(
        ROOT / "evaluation_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "rankings": 280,
            "AI_relations_unchanged": 12600,
            "formal_metrics": None,
        },
    )


def replay_heads() -> None:
    """Apply the replay heads workflow step."""
    import torch
    from safetensors.torch import load_file

    from business_joint_model import PairHead
    from business_metric_model import Projection

    torch.set_num_threads(4)
    with np.load(ROOT / "joint_features.npz", allow_pickle=False) as bank:
        features, valid = bank["features"].copy(), bank["valid"].copy()
    with np.load(METRIC / "full_embedding_bank.npz", allow_pickle=False) as bank:
        q, g = [
            torch.tensor(bank[k].copy(), dtype=torch.float32)
            for k in ("queries", "gallery")
        ]
    for root, cls in ((ROOT, PairHead), (METRIC, Projection)):
        for p in sorted((root / "models").glob("q*.safetensors")):
            model = cls()
            model.load_state_dict(load_file(p), strict=True)
            model.eval()
            with torch.no_grad():
                if cls is PairHead:
                    scores = np.full(valid.shape, np.nan, dtype=np.float32)
                    scores[valid] = model(torch.tensor(features[valid])).numpy()
                else:
                    scores = (model(q) @ model(g).T).numpy()
            expected = np.load(p.with_suffix(".npy"), allow_pickle=False)
            if not np.allclose(scores, expected, atol=1e-6, rtol=1e-6, equal_nan=True):
                raise ValueError("Actual trained head replay differs: " + p.name)


def finish() -> None:
    """Finalize the recorded outputs of the workflow.

    Raises:
        ValueError: Immutable joint report changed; Broken visual replay source;
            Immutable replay page changed; Ranking replay differs.

    Notes:
        Writes derived artifacts through once, report.write_text, visual.write_text,
        write_json.
    """
    guard()
    checked("evaluation_receipt.json")
    replay_heads()
    metrics = ("precision@10", "recall@10", "ap", "ndcg@10", "mrr", "hit@1")
    gates: dict = {}
    losses = []
    lines = [
        "# Internal AI development: joint-image attention business adaptation",
        "",
        "All data exposed; not independent acceptance. Same315 products per cohort,40 queries,12600 unchanged AI relations,K10. OOF combines family-masked models and is not a deployable single-model acceptance result.",
        "Complete frozen DINOv2 joint-image attention plus trained business head; not STIR paper reproduction. New masks disabled globally following failed subject QA. Original complete Qwen scores retained; no imputation.",
        "Cells show shared-unknown-label ranges in percent; not confidence intervals and not bounds on AI annotation error.",
        "",
    ]
    for c in COHORTS:
        result = load(ROOT / c / "internal_metrics.ai.development.json")
        paired = load(ROOT / c / "internal_paired_querywise.json")
        gates[c] = {}
        for scope, methods in result["scopes"].items():
            lines += [
                f"## {c} / {scope}",
                "",
                "| Method | P@10 | R@10 | AP | NDCG@10 | MRR | Hit@1 |",
                "|---|---|---|---|---|---|---|",
            ]
            for m, values in methods.items():
                cells = [
                    " / ".join(
                        "undefined" if x is None else f"{x*100:.2f}"
                        for x in values["ranges"][metric]
                    )
                    for metric in metrics
                ]
                lines.append("| " + " | ".join([m, *cells]) + " |")
            lines.append("")
            gates[c][scope] = {
                m: {
                    "precision_60": methods[m]["precision_60_lower_bound_met"],
                    "recall_85": methods[m]["recall_85_lower_bound_met"],
                    "nonregression": all(
                        v["nonregression_proven_for_shared_unknown_labels"]
                        for b in ("global", "region_title")
                        for metric, v in paired["scopes"][scope][m][b].items()
                        if metric != "precision@10"
                    ),
                }
                for m in NEW
            }
        for r in paired["query_pair_bounds"]:
            bad = [
                m
                for m, v in r["metrics"].items()
                if v["delta_range"][1] is not None and v["delta_range"][1] < -1e-12
            ]
            if r["candidate"] == PRIMARY and bad:
                losses.append(
                    {
                        "cohort": c,
                        "query_id": r["query_id"],
                        "baseline": r["baseline"],
                        "strict_regression_metrics": bad,
                    }
                )
    primary = [s[PRIMARY] for groups in gates.values() for s in groups.values()]
    decision = {
        "primary": PRIMARY,
        "checks": gates,
        "development_precision_target_met": all(r["precision_60"] for r in primary),
        "development_recall_preference_met": all(r["recall_85"] for r in primary),
        "development_all_metric_nonregression": all(
            r["nonregression"] for r in primary
        ),
        "formal_metrics": None,
        "goal_met": False,
        "default_strategy_changed": False,
        "new_independent_AI_blind_predictions": 0,
    }
    once(ROOT / "decision.json", decision)
    once(ROOT / "internal_strict_regressions.json", losses)
    text = "\n".join(lines) + "\n"
    report = ROOT / "internal_comparison.ai.development.md"
    if report.exists() and report.read_text(encoding="utf-8") != text:
        raise ValueError("Immutable joint report changed")
    report.write_text(text, encoding="utf-8")
    import copy
    import html
    import os
    from urllib.parse import quote

    public = load(CYCLE / "views_inputs.public.json")
    profiles = {r["key"]: r["profile"]["product_type"] for r in public}
    markup = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>模块二内部开发对照</title><style>body{font:16px sans-serif;max-width:1500px;margin:24px auto}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.card{border:1px solid #ddd;padding:10px}.card img{width:100%;height:170px;object-fit:contain}summary{padding:12px;font-weight:bold}</style><h1>模块二内部AI开发结果回放</h1><p>同图库、同查询、同标签、K10；AI评审，人工0。开发结果不能当作正式验收。新掩码因质检未通过全局停用。</p>'
    ]
    references = []
    for c in COHORTS:
        docs = {r["id"]: r for r in load(DATA / c / "catalog.public.json")}
        for qid, methods in load(ROOT / c / "predictions.json").items():
            for method, rank in methods.items():
                markup.append(
                    "<details><summary>"
                    + html.escape(c + "/" + qid + " / " + method)
                    + '</summary><div class="cards">'
                )
                for k, cid in enumerate(rank[:10], 1):
                    d = docs[cid]
                    source = Path(d["image_path"])
                    references.append(source)
                    src = quote(
                        Path(os.path.relpath(source, ROOT)).as_posix(), safe="/"
                    )
                    markup.append(
                        '<div class="card"><b>#'
                        + str(k)
                        + '</b><img src="'
                        + src
                        + '"><div>'
                        + html.escape(d["title"])
                        + "</div><small>"
                        + html.escape(profiles[cid])
                        + "</small></div>"
                    )
                markup.append("</div></details>")
    if not all(p.exists() for p in references):
        raise ValueError("Broken visual replay source")
    markup.append("</html>")
    page = "\n".join(markup)
    visual = ROOT / "internal_comparison.html"
    if visual.exists() and visual.read_text(encoding="utf-8") != page:
        raise ValueError("Immutable replay page changed")
    visual.write_text(page, encoding="utf-8")
    once(
        ROOT / "visual_replay_check.json",
        {
            "static_image_references_verified": len(references),
            "browser_interaction_tested": False,
            "actual_rankings": 280,
        },
    )
    parent_snapshot = CYCLE / "post_views_exclusion_snapshot.json"
    snapshot = copy.deepcopy(load(parent_snapshot))
    derived = load(ROOT / "joint_input_exposure.json")
    images = {r["sha256"]: r for r in snapshot["image_fingerprints"]}
    images.update({r["fingerprints"]["sha256"]: r["fingerprints"] for r in derived})
    snapshot.update(
        parent_snapshot_sha256=sha256_file(parent_snapshot),
        ids=sorted(set(snapshot["ids"]) | {r["id"] for r in derived}),
        image_fingerprints=list(images.values()),
        joint_attention_transformed_inputs=len(derived),
        scope="All prior development and consumed blind exclusions plus every real joint-image transformed model input; never unseen acceptance",
    )
    snapshot["files"][relative(ROOT / "joint_input_exposure.json")] = sha256_file(
        ROOT / "joint_input_exposure.json"
    )
    snapshot["counts"].update(
        records=len(snapshot["ids"]), unique_image_bytes=len(images)
    )
    once(ROOT / "post_experiment_exclusion_snapshot.json", snapshot)
    regenerated = predictions()
    if any(load(ROOT / c / "predictions.json") != regenerated[c] for c in COHORTS):
        raise ValueError("Ranking replay differs")
    once(
        ROOT / "verification.json",
        {
            "passed": True,
            "head_checkpoints_replayed": 18,
            "rankings_replayed": 280,
            "AI_relations_unchanged": 12600,
            "formal_metrics": None,
        },
    )
    once(
        ROOT / "completion_receipt.json",
        {
            "files": {
                relative(p): sha256_file(p)
                for p in sorted(ROOT.rglob("*"))
                if p.is_file()
                and p.name
                not in ("completion_receipt.json", "public_status.json", "runtime.lock")
            },
            "source_files": load(ROOT / "protocol.json")["files"],
            "completed": True,
            "formal_metrics": None,
        },
    )
    write_json(
        ROOT / "public_status.json",
        {
            "phase": "joint_attention_and_fixed_fusion_completed",
            "completed": True,
            "methods": 7,
            "rankings": 280,
            "development_precision_target_met": decision[
                "development_precision_target_met"
            ],
            "development_recall_preference_met": decision[
                "development_recall_preference_met"
            ],
            "development_all_metric_nonregression": decision[
                "development_all_metric_nonregression"
            ],
            "quality_metrics": None,
            "formal_metrics": None,
            "default_strategy_changed": False,
            "goal_met": False,
        },
    )
    print(
        json.dumps(
            {
                "joint_attention_fusion_completed": True,
                "formal_metrics": None,
                "default_strategy_changed": False,
            }
        ),
        flush=True,
    )


def worker() -> None:
    """Return worker for this workflow.

    Notes:
        Writes derived artifacts through write_json.
    """
    lock = ROOT / "runtime.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("business_joint_experiment worker\n")
    try:
        encode()
        if not (ROOT / "feature_receipt.json").exists():
            return
        train()
        evaluate()
        finish()
    except Exception as error:
        write_json(
            ROOT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "formal_metrics": None,
            },
        )
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "verify"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "worker":
        worker()
    else:
        guard()
        checked("completion_receipt.json")
        replay_heads()
        assert all(
            load(ROOT / c / "predictions.json") == v for c, v in predictions().items()
        )
        print(
            json.dumps(
                {
                    "sealed_joint_rank_and_model_replay_verified": True,
                    "formal_metrics": None,
                }
            ),
            flush=True,
        )
