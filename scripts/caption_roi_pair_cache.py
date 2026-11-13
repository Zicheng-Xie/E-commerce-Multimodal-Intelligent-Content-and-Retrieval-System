"""Exact full-model cache for the frozen caption-guided crop interaction arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run  # noqa: F401
from dual_context_pair_cache import cached_logit, gpu_available, memory_available
from dual_context_pair_cache import once as frozen_once
from dual_context_pair_cache import source_file, valid_identity
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.contracts import FileDigests
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_subject_experiment import pair_identity
from subject_roi_experiment import load

ROOT = PROJECT_ROOT / "artifacts/module2_caption_roi_development"
PARENT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
DATA = PROJECT_ROOT / "artifacts/module2_primary_subjects"
COHORTS = ("tune", "regression", "fresh")
CONTEXT = "new_caption_actual_crop"


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def once(path: Path, value: object) -> None:
    """Canonical JSON makes immutable retries independent of tuple/list inputs.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.
    """
    if (ROOT / "completion_receipt.json").exists():
        assert Path(path).exists(), "Preserve the completed crop experiment"
    frozen_once(path, json.loads(json.dumps(value, ensure_ascii=False)))


def local_path(value: str | Path) -> Path:
    """Return local path for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Resolved artifact path.
    """
    path = (PROJECT_ROOT / value).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    return path


def pin_public_files(mapping: dict, files: FileDigests) -> None:
    """Only public/model/pixel evidence belongs in the inference dependency map.

    Args:
        mapping: Mapping used by this operation.
        files: Files used by this operation.
    """
    for name, checksum in mapping.items():
        lowered = name.lower()
        if Path(name).suffix.lower() != ".py":
            assert not any(
                word in lowered
                for word in ("qrels", "metrics", "review_part", "labels")
            ), f"Non-public inference dependency: {name}"
        path = local_path(name)
        assert sha256_file(path) == checksum, f"Changed crop dependency: {name}"
        key = relative(path)
        assert key not in files or files[key] == checksum
        files[key] = checksum


def asset_context(
    asset: dict, public: dict, public_field: str, files: FileDigests
) -> tuple[dict, dict]:
    """Validate actual input bytes and retain original/ROI provenance separately.

    Args:
        asset: Asset used by this operation.
        public: Public used by this operation.
        public_field: Public field used by this operation.
        files: Files used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    assert asset[public_field] == public and asset["id"] == public["id"]
    source = local_path(asset["source_image"])
    assert source == Path(public["image_path"]).resolve()
    selected = local_path(asset["selected_image"])
    assert sha256_file(source) == asset["source_sha256"]
    assert sha256_file(selected) == asset["selected_sha256"]
    assert files[relative(source)] == asset["source_sha256"]
    assert files[relative(selected)] == asset["selected_sha256"]
    result_path = local_path(asset["result_file"])
    result_digest = sha256_file(result_path)
    assert files[relative(result_path)] == result_digest
    result = load(result_path)
    assert result["source_sha256"] == asset["source_sha256"]
    assert result["request_id"] == asset["roi_request"]
    assert result["decision"]["status"] == asset["status"]
    if asset["status"] == "roi":
        assert asset["crop_sha256"] == asset["selected_sha256"]
        assert result["crop"]["zoom_sha256"] == asset["selected_sha256"]
        native = local_path(asset["native_image"])
        assert sha256_file(native) == asset["native_sha256"]
        assert files[relative(native)] == asset["native_sha256"]
        assert result["crop"]["native_sha256"] == asset["native_sha256"]
    else:
        assert asset["status"] == "global_fallback"
        assert asset["selected_sha256"] == asset["source_sha256"]
        assert asset["crop_sha256"] is None
        assert asset.get("native_image") is None
        assert asset.get("native_sha256") is None
    context = {
        key: asset.get(key)
        for key in (
            "source_image",
            "source_sha256",
            "selected_image",
            "selected_sha256",
            "status",
            "roi_request",
            "result_file",
            "crop_sha256",
            "native_image",
            "native_sha256",
        )
    }
    context["result_sha256"] = result_digest
    return context, dict(public, image_path=str(selected))


@dataclass(frozen=True)
class CaptionRoiPairCacheAddCacheInputs:
    """Input values for add cache.

    Attributes:
        raw: Raw used by this operation.
        path: Path of the input or output artifact used by this operation.
        query: Query record or representation used by this operation.
        description: Description used by this operation.
        documents: Documents used by this operation.
        cache: Cache used by this operation.
    """

    raw: dict
    path: Path
    query: dict
    description: str
    documents: dict
    cache: dict


def add_cache(options: CaptionRoiPairCacheAddCacheInputs) -> None:
    """Apply the add cache workflow step.

    Args:
        options: Validated input fields described by CaptionRoiPairCacheAddCacheInputs.

    Notes:
        Updates the supplied cache in place.
    """
    raw = options.raw
    path = options.path
    query = options.query
    description = options.description
    documents = options.documents
    cache = options.cache
    assert len(raw["pool"]) == len(set(raw["pool"])) == len(raw["logits"])
    assert set(raw["pair_identities"]) == set(raw["pool"])
    for index, (cid, score) in enumerate(zip(raw["pool"], raw["logits"])):
        identity = raw["pair_identities"][cid]
        assert math.isfinite(score) and valid_identity(identity)
        assert identity == pair_identity(query, description, documents[cid])
        record = dict(
            cache_file=relative(path),
            cache_sha256=sha256_file(path),
            source_index=index,
            identity_sha256=identity["sha256"],
            logit=score,
        )
        previous = cache.get(identity["sha256"])
        if previous is None:
            cache[identity["sha256"]] = (identity, record)
        else:
            assert previous[0] == identity
            assert previous[1]["logit"] == score, "Conflicting exact cached logits"


def prepare() -> Any | dict:
    """Freeze the public-only pair protocol after crop detection has completed.

    Returns:
        Mapping with fields: version, query_unions, jobs, text_references,
        image_query_context_jobs, unchanged_text_query_pairs.
    """
    if (ROOT / "pair_cache_protocol.json").exists():
        return guard()
    assert not (ROOT / "completion_receipt.json").exists()
    seals = {root: load(root / "completion_receipt.json") for root in (DATA, PARENT)}
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/dual_context_pair_cache.py",
        PROJECT_ROOT / "scripts/primary_subject_experiment.py",
        PROJECT_ROOT / "scripts/caption_roi_assets.py",
        PROJECT_ROOT / "scripts/subject_roi_experiment.py",
        PROJECT_ROOT / "scripts/run.py",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
        *[root / "completion_receipt.json" for root in (DATA, PARENT)],
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
        *[
            ROOT / name
            for name in (
                "crop_inputs.json",
                "detection_receipt.json",
                "crop_asset_protocol.json",
                "crop_plan.json",
            )
        ],
    ]
    files = {relative(path): sha256_file(path) for path in paths}
    inputs = load(ROOT / "crop_inputs.json")
    assert inputs["parent"] == relative(PARENT)
    assert inputs["whole_parent"] == relative(DATA)
    assert set(inputs["cohorts"]) == set(COHORTS)
    pin_public_files(inputs["files"], files)
    detector_receipt = load(ROOT / "detection_receipt.json")
    verify_files(ROOT, detector_receipt["files"])
    assert detector_receipt["files"]["crop_inputs.json"] == sha256_file(
        ROOT / "crop_inputs.json"
    )
    for name, digest in detector_receipt["files"].items():
        files[relative(ROOT / name)] = digest
    data_protocol = source_file(
        DATA / "comparison_protocol.json", DATA, seals[DATA], files
    )
    parent_protocol = source_file(
        PARENT / "pair_cache_protocol.json", PARENT, seals[PARENT], files
    )
    implementation = [
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "scripts/primary_subject_experiment.py",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    for path in implementation:
        source_key, digest = relative(path), sha256_file(path)
        assert data_protocol["files"][source_key] == digest
        assert parent_protocol["files"][source_key] == digest
    vendor = PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py"
    assert parent_protocol["files"][relative(vendor)] == sha256_file(vendor)
    parent_plan = source_file(PARENT / "pair_plan.json", PARENT, seals[PARENT], files)
    unions = parent_plan["query_unions"]
    assert len(unions) == 24 and sum(len(u["union_pool"]) for u in unions) == 504
    assert len({(u["cohort"], u["query_id"]) for u in unions}) == 24
    cache: dict
    queries, documents, descriptions, cache, text_references = {}, {}, {}, {}, []
    actual_docs: dict
    contexts: dict
    actual_queries, actual_docs, contexts = {}, {}, {}
    for cohort in COHORTS:
        public_queries = source_file(
            DATA / cohort / "queries.public.json", DATA, seals[DATA], files
        )
        catalog = source_file(
            DATA / cohort / "catalog.public.json", DATA, seals[DATA], files
        )
        documents[cohort] = {doc["id"]: doc for doc in catalog}
        assert len(documents[cohort]) == len(catalog) == 296
        assets = inputs["cohorts"][cohort]
        assert len(assets["queries"]) == 8
        assert [a["id"] for a in assets["gallery"]] == [d["id"] for d in catalog]
        query_assets = {a["id"]: a for a in assets["queries"]}
        assert len(query_assets) == 8
        actual_docs[cohort], contexts[cohort] = {}, {"queries": {}, "gallery": {}}
        for asset in assets["gallery"]:
            cid = asset["id"]
            context, actual = asset_context(
                asset, documents[cohort][cid], "public_doc", files
            )
            actual_docs[cohort][cid] = actual
            contexts[cohort]["gallery"][cid] = context
        for query in public_queries:
            qid = query["id"]
            if not query["image_path"]:
                path = DATA / cohort / "runs" / qid / "cross.json"
                raw = source_file(path, DATA, seals[DATA], files)
                assert len(raw["pool"]) == len(raw["logits"]) == 20
                text_references.append(
                    dict(cohort=cohort, query_id=qid, file=relative(path), pairs=20)
                )
                continue
            key = (cohort, qid)
            asset = query_assets[qid]
            context, actual = asset_context(asset, query, "public_query", files)
            queries[key], actual_queries[key] = query, actual
            contexts[cohort]["queries"][qid] = context
            feature = source_file(
                DATA / cohort / "runs" / qid / "caption_features.json",
                DATA,
                seals[DATA],
                files,
            )
            description = feature["description"]
            assert isinstance(description, str)
            assert asset["new_description"] == description
            descriptions[key] = description
            for root, name in (
                (DATA, "caption_cross.json"),
                (PARENT, "new_cross.json"),
            ):
                path = root / cohort / "runs" / qid / name
                raw = source_file(path, root, seals[root], files)
                add_cache(
                    CaptionRoiPairCacheAddCacheInputs(
                        raw, path, query, description, documents[cohort], cache
                    )
                )
    assert len(queries) == 24
    assert (
        len(text_references) == 16 and sum(r["pairs"] for r in text_references) == 320
    )
    jobs = []
    for union in unions:
        cohort, qid = union["cohort"], union["query_id"]
        key, pool = (cohort, qid), union["union_pool"]
        assert len(pool) == len(set(pool))
        assert pool == union["old_pool"] + [
            cid for cid in union["new_pool"] if cid not in set(union["old_pool"])
        ]
        query, description = actual_queries[key], descriptions[key]
        docs = {cid: actual_docs[cohort][cid] for cid in pool}
        identities = {cid: pair_identity(query, description, docs[cid]) for cid in pool}
        hits = {}
        for cid, identity in identities.items():
            if identity["sha256"] in cache:
                cached_identity, record = cache[identity["sha256"]]
                assert identity == cached_identity
                hits[cid] = record
        pair_contexts = {
            cid: dict(
                query=contexts[cohort]["queries"][qid],
                candidate=contexts[cohort]["gallery"][cid],
                description_sha256=hashlib.sha256(
                    description.encode("utf-8")
                ).hexdigest(),
            )
            for cid in pool
        }
        jobs.append(
            dict(
                id=f"{cohort}/{qid}/crop",
                cohort=cohort,
                query_id=qid,
                context=CONTEXT,
                query=query,
                description=description,
                pool=pool,
                documents=docs,
                pair_identities=identities,
                cache_sources=hits,
                missing_ids=[cid for cid in pool if cid not in hits],
                crop_contexts=pair_contexts,
                output_file=f"{cohort}/runs/{qid}/crop_cross.json",
            )
        )
    counts = pair_counts(jobs)
    assert counts["total_pairs"] == 504
    plan = dict(
        version=1,
        query_unions=unions,
        jobs=jobs,
        text_references=text_references,
        image_query_context_jobs=24,
        **counts,
        unchanged_text_query_pairs=320,
    )
    once(ROOT / "pair_plan.json", plan)
    files[relative(ROOT / "pair_plan.json")] = sha256_file(ROOT / "pair_plan.json")
    once(
        ROOT / "pair_cache_protocol.json",
        dict(
            files=files,
            identity_function="primary_subject_experiment.pair_identity",
            cache_roots=[relative(DATA), relative(PARENT)],
            cache_contexts=["DATA new caption", "PARENT new_cross"],
            reuse_requires_full_identity_and_actual_image_bytes=True,
            source_and_ROI_context_kept_separately_from_identity=True,
            fixed_parent_query_unions=True,
            image_query_context_jobs=24,
            expected_counts=counts,
            selected_inputs="actual query zoom + candidate zoom; whole image fallback",
            candidate_title="unchanged sealed public catalog title",
            description="exact sealed DATA caption_features.description",
            instruction="unchanged full BusinessReranker.BUSINESS_INSTRUCTION",
            score_kind="raw yes-minus-no logits, not probability",
            GPU_admission="memory <1500 MiB then empty /api/ps; no unload/keepalive change",
            no_qrels_metrics_or_native_reviews_read=True,
            independent_acceptance=False,
            formal_metrics=None,
        ),
    )
    print(dict(prepared=True, image_query_context_jobs=24, **counts), flush=True)
    return plan


def pair_counts(jobs: list[dict]) -> dict:
    """Return pair counts for this workflow.

    Args:
        jobs: Jobs used by this operation.

    Returns:
        Mapping with fields: total_pairs, exact_reused_pairs, new_pair_positions,
        new_full_reranker_pairs.
    """
    missing = [
        j["pair_identities"][cid]["sha256"] for j in jobs for cid in j["missing_ids"]
    ]
    return dict(
        total_pairs=sum(len(j["pool"]) for j in jobs),
        exact_reused_pairs=sum(len(j["cache_sources"]) for j in jobs),
        new_pair_positions=len(missing),
        new_full_reranker_pairs=len(set(missing)),
    )


def guard() -> Any:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "pair_cache_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    plan = load(ROOT / "pair_plan.json")
    assert protocol["expected_counts"] == pair_counts(plan["jobs"])
    assert plan["query_unions"] == load(PARENT / "pair_plan.json")["query_unions"]
    return plan


def new_pair_path(identity: dict) -> Path:
    """Return new pair path for this workflow.

    Args:
        identity: Identity used by this operation.

    Returns:
        Numeric result computed from the supplied values.
    """
    return ROOT / "new_pair_scores" / (identity["sha256"] + ".json")


def completed_pair(identity: dict) -> Any:
    """Return completed pair for this workflow.

    Args:
        identity: Identity used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    value = load(new_pair_path(identity))
    assert valid_identity(identity) and value["pair_identity"] == identity
    assert value["full_model"] is True and math.isfinite(value["logit"])
    assert len(value["timings"]) == 1
    assert value["timings"][0]["document_id"] == identity["inputs"]["candidate_id"]
    assert (
        value["crop_context"]["query"]["selected_sha256"]
        == identity["inputs"]["query_image_sha256"]
    )
    assert (
        value["crop_context"]["candidate"]["selected_sha256"]
        == identity["inputs"]["candidate_image_sha256"]
    )
    return value


def raw_value(job: dict) -> dict:
    """Return raw value for this workflow.

    Args:
        job: Job used by this operation.

    Returns:
        Mapping with fields: job_id, context, description, pool, logits,
        pair_identities, new_pair_ids, reused_pair_ids, cache_sources, new_pair_files,
        crop_contexts, timings.
    """
    logits, pairs, timings = [], {}, []
    for cid in job["pool"]:
        identity = job["pair_identities"][cid]
        assert (
            pair_identity(job["query"], job["description"], job["documents"][cid])
            == identity
        )
        if cid in job["cache_sources"]:
            score = cached_logit(job["cache_sources"][cid], identity)
        else:
            pair = completed_pair(identity)
            score = pair["logit"]
            timings.extend(pair["timings"])
            path = new_pair_path(identity)
            pairs[cid] = dict(file=relative(path), sha256=sha256_file(path))
        logits.append(score)
    return dict(
        job_id=job["id"],
        context=job["context"],
        description=job["description"],
        pool=job["pool"],
        logits=logits,
        pair_identities=job["pair_identities"],
        new_pair_ids=job["missing_ids"],
        reused_pair_ids=[cid for cid in job["pool"] if cid in job["cache_sources"]],
        cache_sources=job["cache_sources"],
        new_pair_files=pairs,
        crop_contexts=job["crop_contexts"],
        timings=timings,
    )


def infer() -> bool:
    """Resume atomic pairs; return False when the user's GPU remains occupied.

    Returns:
        Whether the condition checked by this operation holds.
    """
    plan = guard()
    if (ROOT / "reranker_receipt.json").exists():
        verify()
        return True
    outstanding = any(
        not new_pair_path(job["pair_identities"][cid]).exists()
        for job in plan["jobs"]
        for cid in job["missing_ids"]
    )
    if outstanding and not memory_available():
        return False
    engine, runtime = None, None
    try:
        for job in plan["jobs"]:
            output = ROOT / job["output_file"]
            if output.exists():
                assert load(output) == raw_value(job)
                continue
            for cid in job["missing_ids"]:
                identity = job["pair_identities"][cid]
                path = new_pair_path(identity)
                if path.exists():
                    completed_pair(identity)
                    continue
                if engine is None:
                    if not gpu_available():
                        return False
                    engine = DescribedImageReranker()
                doc = job["documents"][cid]
                assert pair_identity(job["query"], job["description"], doc) == identity
                pair = [dict(id=cid, image_path=doc["image_path"], text=doc["title"])]
                before = len(engine.adapter.timings)
                if job["description"]:
                    scores = engine.score_described(
                        pair,
                        query_image=Path(job["query"]["image_path"]),
                        description=job["description"],
                    )
                else:
                    scores = engine.score(
                        pair,
                        query_image=Path(job["query"]["image_path"]),
                        query_text=job["query"]["text"],
                    )
                assert len(scores) == 1 and math.isfinite(float(scores[0]))
                value = dict(
                    pair_identity=identity,
                    logit=float(scores[0]),
                    timings=engine.adapter.timings[before:],
                    source_job=job["id"],
                    crop_context=job["crop_contexts"][cid],
                    full_model=True,
                )
                assert len(value["timings"]) == 1
                once(path, value)
                completed_pair(identity)
            once(output, raw_value(job))
            print(
                dict(completed_job=job["id"], candidates=len(job["pool"])), flush=True
            )
        runtime = engine.diagnostics() if engine else dict(loaded_this_invocation=False)
    finally:
        if engine is not None:
            engine.close()
    completed_files = [
        dict(file=relative(path), sha256=sha256_file(path))
        for path in sorted((ROOT / "new_pair_scores").glob("*.json"))
    ]
    runtime_path = ROOT / "reranker_runtime.json"
    if runtime_path.exists():
        assert load(runtime_path)["all_completed_new_pairs"] == completed_files
    else:
        once(
            runtime_path,
            dict(
                invocation=runtime,
                all_completed_new_pairs=completed_files,
                no_Ollama_model_unloaded=True,
                no_keepalive_changed=True,
            ),
        )
    files = [
        runtime_path,
        *[ROOT / job["output_file"] for job in plan["jobs"]],
        *sorted((ROOT / "new_pair_scores").glob("*.json")),
    ]
    once(
        ROOT / "reranker_receipt.json",
        dict(
            files={
                path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files
            },
            **pair_counts(plan["jobs"]),
            image_query_context_jobs=24,
            original_text_pairs_untouched=320,
            text_inference_calls=0,
            fixed_parent_candidates=True,
            qrels_used_in_runtime=False,
            formal_metrics=None,
        ),
    )
    verify()
    return True


def verify() -> dict:
    """Verify the recorded artifacts and their consistency.

    Returns:
        Mapping with fields: passed, image_query_context_jobs,
        actual_crop_identities_replayed, cached_logits_replayed_exactly,
        new_logits_have_real_full_model_timings, original_text_pairs_untouched,
        quality_metrics_computed, independent_acceptance, formal_metrics.

    Notes:
        Writes derived artifacts through once.
    """
    plan = guard()
    receipt = load(ROOT / "reranker_receipt.json")
    verify_files(ROOT, receipt["files"])
    seen_new: set = set()
    assert len(plan["jobs"]) == len(plan["query_unions"]) == 24
    for job, union in zip(plan["jobs"], plan["query_unions"]):
        assert (job["cohort"], job["query_id"]) == (union["cohort"], union["query_id"])
        assert job["pool"] == union["union_pool"]
        assert job["context"] == CONTEXT
        assert set(job["pair_identities"]) == set(job["documents"]) == set(job["pool"])
        assert set(job["crop_contexts"]) == set(job["pool"])
        raw = load(ROOT / job["output_file"])
        assert raw == raw_value(job)
        assert len(raw["pool"]) == len(set(raw["pool"])) == len(raw["logits"])
        assert not set(raw["new_pair_ids"]) & set(raw["reused_pair_ids"])
        assert set(raw["new_pair_ids"]) | set(raw["reused_pair_ids"]) == set(
            raw["pool"]
        )
        assert len(raw["timings"]) == len(raw["new_pair_ids"])
        for cid, identity in job["pair_identities"].items():
            assert valid_identity(identity)
            context = job["crop_contexts"][cid]
            assert (
                context["query"]["selected_sha256"]
                == identity["inputs"]["query_image_sha256"]
            )
            assert (
                context["candidate"]["selected_sha256"]
                == identity["inputs"]["candidate_image_sha256"]
            )
            assert (
                context["description_sha256"]
                == hashlib.sha256(job["description"].encode("utf-8")).hexdigest()
            )
        seen_new.update(
            job["pair_identities"][cid]["sha256"] for cid in job["missing_ids"]
        )
    assert {path.stem for path in (ROOT / "new_pair_scores").glob("*.json")} == seen_new
    counts = pair_counts(plan["jobs"])
    assert counts["total_pairs"] == 504
    assert counts["exact_reused_pairs"] + counts["new_pair_positions"] == 504
    assert all(receipt[key] == value for key, value in counts.items())
    for identity_key in seen_new:
        pair = load(ROOT / "new_pair_scores" / (identity_key + ".json"))
        origin = next(job for job in plan["jobs"] if job["id"] == pair["source_job"])
        cid = pair["pair_identity"]["inputs"]["candidate_id"]
        assert origin["pair_identities"][cid] == pair["pair_identity"]
        assert origin["crop_contexts"][cid] == pair["crop_context"]
    assert len(plan["text_references"]) == 16
    result = dict(
        passed=True,
        **counts,
        image_query_context_jobs=24,
        actual_crop_identities_replayed=True,
        cached_logits_replayed_exactly=True,
        new_logits_have_real_full_model_timings=True,
        original_text_pairs_untouched=320,
        quality_metrics_computed=False,
        independent_acceptance=False,
        formal_metrics=None,
    )
    once(ROOT / "pair_cache_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "infer", "verify"])
    action = parser.parse_args().action
    result = globals()[action]()
    if action == "prepare":
        result = dict(prepared=True, **pair_counts(result["jobs"]))
    elif action == "infer":
        result = dict(completed=result, waiting=not result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
