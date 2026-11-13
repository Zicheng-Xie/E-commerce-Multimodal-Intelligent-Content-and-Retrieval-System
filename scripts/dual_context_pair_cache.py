"""Exact full-Qwen pair cache for frozen old/new image-query context unions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.contracts import FileDigests
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file
from multimodal_ecommerce.visual_query import DescribedImageReranker
from primary_subject_experiment import pair_identity
from subject_roi_experiment import load

PARENT = PROJECT_ROOT / "artifacts/module2_primary_subjects"
PREVIOUS = PROJECT_ROOT / "artifacts/module2_subject_coverage_expansion"
ROOT = PROJECT_ROOT / "artifacts/module2_dual_path_development"
COHORTS = ("tune", "regression", "fresh")
IDLE_BELOW_MIB = 1500


def relative(path: Path) -> str:
    """Represent a workspace path relative to the project root.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Text produced by this operation.
    """
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def once(path: Path, value: object) -> None:
    """Publish complete JSON atomically; validate an existing result verbatim.

    Args:
        path: Path of the input or output artifact used by this operation.
        value: Input value to validate or transform.
    """
    path = ensure_output_path(path)
    if path.exists():
        assert load(path) == value, f"Preserve completed artifact: {path.name}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        assert load(pending) == value, f"Inspect incomplete artifact: {pending.name}"
    else:
        with pending.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
    pending.replace(path)


def valid_identity(identity: dict) -> bool:
    """Return valid identity for this workflow.

    Args:
        identity: Identity used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if set(identity) != {"sha256", "inputs"}:
        return False
    digest = hashlib.sha256(
        json.dumps(identity["inputs"], ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return identity["sha256"] == digest


def source_file(path: Path, sealed_root: Path, sealed: dict, files: FileDigests) -> Any:
    """Return source file for this workflow.

    Args:
        path: Path of the input or output artifact used by this operation.
        sealed_root: Sealed root used by this operation.
        sealed: Sealed used by this operation.
        files: Files used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.

    Notes:
        Updates the supplied files in place.
    """
    key = path.relative_to(sealed_root).as_posix()
    checksum = sha256_file(path)
    assert sealed["files"][key] == checksum, f"Changed frozen public input: {key}"
    files[relative(path)] = checksum
    return load(path)


def prepare() -> Any | dict:
    """Prepare the inputs and protocol for this development workflow.

    Returns:
        Mapping with fields: version, query_unions, jobs, text_references,
        image_query_context_jobs, full_image_context_pairs, exact_cached_pairs,
        new_full_pairs, unchanged_text_query_pairs.

    Notes:
        Writes derived artifacts through once.
    """
    if (ROOT / "pair_cache_protocol.json").exists():
        return guard()
    roots = [PARENT, PREVIOUS]
    seals = {root: load(root / "completion_receipt.json") for root in roots}
    paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/primary_subject_experiment.py",
        PROJECT_ROOT / "configs/reranker_model.json",
        PROJECT_ROOT / "vendor/qwen3_vl_reranker/qwen3_vl_reranker.py",
        *[root / "completion_receipt.json" for root in roots],
        *[root / "comparison_protocol.json" for root in roots],
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    files = {relative(p): sha256_file(p) for p in paths}
    implementation = [
        PROJECT_ROOT / "configs/reranker_model.json",
        *sorted((PROJECT_ROOT / "src/multimodal_ecommerce").glob("*.py")),
    ]
    for root in roots:
        protocol = source_file(
            root / "comparison_protocol.json", root, seals[root], files
        )
        assert all(
            protocol["files"][relative(p)] == sha256_file(p) for p in implementation
        ), "Current full model adapter differs from a cache's frozen implementation"
    assert load(PARENT / "comparison_protocol.json")["files"][
        "scripts/primary_subject_experiment.py"
    ] == sha256_file(PROJECT_ROOT / "scripts/primary_subject_experiment.py")
    cache: dict
    queries, documents, features, cache = {}, {}, {}, {}
    text_references = []
    for cohort in COHORTS:
        query_path = PARENT / cohort / "queries.public.json"
        catalog_path = PARENT / cohort / "catalog.public.json"
        public_queries = source_file(query_path, PARENT, seals[PARENT], files)
        catalog = source_file(catalog_path, PARENT, seals[PARENT], files)
        assert len(catalog) == 296
        by_id = {d["id"]: d for d in catalog}
        assert len(by_id) == len(catalog)
        documents[cohort] = by_id
        for doc in catalog:
            image = Path(doc["image_path"])
            files[relative(image)] = sha256_file(image)
        for query in public_queries:
            qid = query["id"]
            queries[(cohort, qid)] = query
            if not query["image_path"]:
                path = PARENT / cohort / "runs" / qid / "cross.json"
                raw = source_file(path, PARENT, seals[PARENT], files)
                assert (
                    len(raw["pool"])
                    == len(set(raw["pool"]))
                    == len(raw["logits"])
                    == 20
                )
                text_references.append(
                    dict(
                        cohort=cohort,
                        query_id=qid,
                        file=relative(path),
                        sha256=sha256_file(path),
                        pairs=20,
                    )
                )
                continue
            image = Path(query["image_path"])
            files[relative(image)] = sha256_file(image)
            for root in roots:
                for context, feature_name, cross_name in [
                    ("old", "features.json", "cross.json"),
                    ("new", "caption_features.json", "caption_cross.json"),
                ]:
                    path = root / cohort / "runs" / qid / cross_name
                    if not path.exists():
                        assert root == PREVIOUS and context == "new"
                        continue
                    feature_path = path.with_name(feature_name)
                    feature = source_file(feature_path, root, seals[root], files)
                    raw = source_file(path, root, seals[root], files)
                    assert (
                        len(raw["pool"])
                        == len(set(raw["pool"]))
                        == len(raw["logits"])
                        == 20
                    )
                    assert set(raw["pair_identities"]) == set(raw["pool"])
                    if root == PARENT:
                        features[(cohort, qid, context)] = (feature, raw)
                    for index, (cid, score) in enumerate(
                        zip(raw["pool"], raw["logits"])
                    ):
                        identity = raw["pair_identities"][cid]
                        assert math.isfinite(score) and valid_identity(identity)
                        assert identity == pair_identity(
                            query, feature["description"], by_id[cid]
                        )
                        record = dict(
                            cache_file=relative(path),
                            cache_sha256=sha256_file(path),
                            source_index=index,
                            identity_sha256=identity["sha256"],
                            logit=score,
                        )
                        previous = cache.get(identity["sha256"])
                        if previous is not None:
                            assert (
                                previous["logit"] == score
                            ), "Same exact identity has conflicting cached logits"
                        else:
                            cache[identity["sha256"]] = record
    jobs, unions = [], []
    for (cohort, qid), query in queries.items():
        if not query["image_path"]:
            continue
        old_feature, old_raw = features[(cohort, qid, "old")]
        new_feature, new_raw = features[(cohort, qid, "new")]
        union = old_raw["pool"] + [
            cid for cid in new_raw["pool"] if cid not in set(old_raw["pool"])
        ]
        assert len(union) == len(set(union)) and 20 <= len(union) <= 40
        assert old_feature["description"] != new_feature["description"]
        unions.append(
            dict(
                cohort=cohort,
                query_id=qid,
                old_pool=old_raw["pool"],
                new_pool=new_raw["pool"],
                union_pool=union,
            )
        )
        for context, feature in [("old", old_feature), ("new", new_feature)]:
            identities = {
                cid: pair_identity(
                    query, feature["description"], documents[cohort][cid]
                )
                for cid in union
            }
            hits = {
                cid: cache[identity["sha256"]]
                for cid, identity in identities.items()
                if identity["sha256"] in cache
            }
            jobs.append(
                dict(
                    id=f"{cohort}/{qid}/{context}",
                    cohort=cohort,
                    query_id=qid,
                    context=context,
                    query=query,
                    description=feature["description"],
                    pool=union,
                    pair_identities=identities,
                    cache_sources=hits,
                    missing_ids=[cid for cid in union if cid not in hits],
                    output_file=f"{cohort}/runs/{qid}/{context}_cross.json",
                )
            )
    requested = sum(len(j["pool"]) for j in jobs)
    reused = sum(len(j["cache_sources"]) for j in jobs)
    new = sum(len(j["missing_ids"]) for j in jobs)
    assert len(unions) == 24 and sum(len(q["union_pool"]) for q in unions) == 504
    assert len(jobs) == 48 and (requested, reused, new) == (1008, 973, 35)
    assert (
        len(text_references) == 16 and sum(r["pairs"] for r in text_references) == 320
    )
    plan = dict(
        version=1,
        query_unions=unions,
        jobs=jobs,
        text_references=text_references,
        image_query_context_jobs=48,
        full_image_context_pairs=requested,
        exact_cached_pairs=reused,
        new_full_pairs=new,
        unchanged_text_query_pairs=320,
    )
    once(ROOT / "pair_plan.json", plan)
    files[relative(ROOT / "pair_plan.json")] = sha256_file(ROOT / "pair_plan.json")
    once(
        ROOT / "pair_cache_protocol.json",
        dict(
            files=files,
            identity_function="primary_subject_experiment.pair_identity",
            cache_roots=[relative(root) for root in roots],
            reuse_legacy_scores_without_pair_identities=False,
            union_order="original twenty then new-context-only candidates in new-pool order",
            image_query_context_jobs=48,
            full_image_context_pairs=1008,
            expected_exact_cached_pairs=973,
            expected_new_full_pairs=35,
            original_text_queries=16,
            original_text_pairs_untouched=320,
            instruction="unchanged full BusinessReranker.BUSINESS_INSTRUCTION",
            score_kind="raw yes-minus-no logits, not probability",
            no_qrels_metrics_or_native_reviews_read=True,
            GPU_admission="nvidia-smi memory <1500 MiB, then one empty /api/ps immediately before first missing pair; no unload or keep-alive mutation",
            formal_metrics=None,
            independent_acceptance=False,
        ),
    )
    print(
        dict(
            planned_jobs=48,
            image_pairs=1008,
            exact_cached=973,
            new_full_pairs=35,
            text_pairs_untouched=320,
        ),
        flush=True,
    )
    return plan


def guard() -> Any:
    """Validate the frozen inputs before continuing the workflow.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    protocol = load(ROOT / "pair_cache_protocol.json")
    verify_files(PROJECT_ROOT, protocol["files"])
    return load(ROOT / "pair_plan.json")


def memory_available() -> bool:
    """Return memory available for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode:
            return False
        memory = float(result.stdout.strip().splitlines()[0])
        return 0 <= memory < IDLE_BELOW_MIB
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        logging.getLogger(__name__).debug(
            "Handled exception in memory_available", exc_info=True
        )
        return False


def gpu_available() -> bool:
    """Return gpu available for this workflow.

    Returns:
        Whether the condition checked by this operation holds.
    """
    if not memory_available():
        return False
    client = OllamaClient(timeout=10)
    try:
        resident = client._request("/api/ps").get("models")
    except (OSError, RuntimeError):
        logging.getLogger(__name__).debug(
            "Handled exception in gpu_available", exc_info=True
        )
        return False
    finally:
        client.close()
    return isinstance(resident, list) and not resident


def cached_logit(record: dict, identity: dict) -> float:
    """Return cached logit for this workflow.

    Args:
        record: Record used by this operation.
        identity: Identity used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    source = PROJECT_ROOT / record["cache_file"]
    assert sha256_file(source) == record["cache_sha256"]
    raw, index = load(source), record["source_index"]
    cid = identity["inputs"]["candidate_id"]
    assert record["identity_sha256"] == identity["sha256"]
    assert raw["pool"][index] == cid and raw["pair_identities"][cid] == identity
    value = raw["logits"][index]
    assert value == record["logit"] and math.isfinite(value)
    return value


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
    path = new_pair_path(identity)
    value = load(path)
    assert value["pair_identity"] == identity
    assert value["full_model"] is True
    assert math.isfinite(value["logit"])
    assert len(value["timings"]) == 1
    assert value["timings"][0]["document_id"] == identity["inputs"]["candidate_id"]
    return value


def raw_value(job: dict) -> dict:
    """Return raw value for this workflow.

    Args:
        job: Job used by this operation.

    Returns:
        Mapping with fields: job_id, context, description, pool, logits,
        pair_identities, new_pair_ids, reused_pair_ids, cache_sources, new_pair_files,
        timings.
    """
    logits, pairs, timings = [], {}, []
    for cid in job["pool"]:
        identity = job["pair_identities"][cid]
        if cid in job["cache_sources"]:
            value = cached_logit(job["cache_sources"][cid], identity)
        else:
            pair = completed_pair(identity)
            value = pair["logit"]
            timings.extend(pair["timings"])
            path = new_pair_path(identity)
            pairs[cid] = dict(file=relative(path), sha256=sha256_file(path))
        logits.append(value)
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
        timings=timings,
    )


def infer() -> bool:
    """Run model inference using the configured inputs.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once.
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
        print(dict(waiting_for_natural_GPU_release=True, models_started=0), flush=True)
        return False
    documents = {
        c: {d["id"]: d for d in load(PARENT / c / "catalog.public.json")}
        for c in COHORTS
    }
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
                        print(
                            dict(
                                waiting_for_natural_GPU_release=True, models_started=0
                            ),
                            flush=True,
                        )
                        return False
                    engine = DescribedImageReranker()
                doc = documents[job["cohort"]][cid]
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
                once(
                    path,
                    dict(
                        pair_identity=identity,
                        logit=float(scores[0]),
                        timings=engine.adapter.timings[before:],
                        source_job=job["id"],
                        full_model=True,
                    ),
                )
            once(output, raw_value(job))
            print(
                dict(
                    completed_job=job["id"],
                    candidates=len(job["pool"]),
                    new_pairs=len(job["missing_ids"]),
                ),
                flush=True,
            )
        runtime = (
            engine.diagnostics()
            if engine
            else dict(timings=[], loaded_this_invocation=False)
        )
    finally:
        if engine is not None:
            engine.close()
    runtime_path = ROOT / "reranker_runtime.json"
    completed_files = [
        dict(file=relative(p), sha256=sha256_file(p))
        for p in sorted((ROOT / "new_pair_scores").glob("*.json"))
    ]
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
        ROOT / "reranker_runtime.json",
        *[ROOT / j["output_file"] for j in plan["jobs"]],
        *sorted((ROOT / "new_pair_scores").glob("*.json")),
    ]
    once(
        ROOT / "reranker_receipt.json",
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files},
            total_pairs=1008,
            new_full_reranker_pairs=35,
            exact_reused_pairs=973,
            image_query_context_jobs=48,
            original_text_pairs_untouched=320,
            text_inference_calls=0,
            qrels_used_in_runtime=False,
            formal_metrics=None,
        ),
    )
    verify()
    return True


def verify() -> dict:
    """Verify the recorded artifacts and their consistency.

    Returns:
        Mapping with fields: passed, full_image_context_pairs, exact_cached_pairs,
        new_full_reranker_pairs, image_query_context_jobs,
        old_new_context_identities_replayed, cached_logits_replayed_exactly,
        new_logits_have_real_full_model_timings, original_text_pairs_untouched,
        quality_metrics_computed, formal_metrics.

    Notes:
        Writes derived artifacts through once.
    """
    plan = guard()
    receipt = load(ROOT / "reranker_receipt.json")
    verify_files(ROOT, receipt["files"])
    new, reused, total = 0, 0, 0
    seen_new: set = set()
    documents = {
        c: {d["id"]: d for d in load(PARENT / c / "catalog.public.json")}
        for c in COHORTS
    }
    for job in plan["jobs"]:
        for cid, identity in job["pair_identities"].items():
            assert valid_identity(identity)
            assert (
                pair_identity(
                    job["query"], job["description"], documents[job["cohort"]][cid]
                )
                == identity
            )
        raw = load(ROOT / job["output_file"])
        assert raw == raw_value(job)
        assert not set(raw["new_pair_ids"]) & set(raw["reused_pair_ids"])
        assert set(raw["new_pair_ids"]) | set(raw["reused_pair_ids"]) == set(
            raw["pool"]
        )
        assert len(raw["pool"]) == len(set(raw["pool"])) == len(raw["logits"])
        assert len(raw["timings"]) == len(raw["new_pair_ids"])
        new += len(raw["new_pair_ids"])
        reused += len(raw["reused_pair_ids"])
        total += len(raw["pool"])
        seen_new.update(
            job["pair_identities"][cid]["sha256"] for cid in raw["new_pair_ids"]
        )
    assert (total, new, reused) == (1008, 35, 973)
    assert len(seen_new) == 35
    assert {p.stem for p in (ROOT / "new_pair_scores").glob("*.json")} == seen_new
    assert receipt["total_pairs"] == total
    assert receipt["new_full_reranker_pairs"] == new
    assert receipt["exact_reused_pairs"] == reused
    assert len(plan["text_references"]) == 16
    for reference in plan["text_references"]:
        assert sha256_file(PROJECT_ROOT / reference["file"]) == reference["sha256"]
    value = dict(
        passed=True,
        full_image_context_pairs=total,
        exact_cached_pairs=reused,
        new_full_reranker_pairs=new,
        image_query_context_jobs=48,
        old_new_context_identities_replayed=True,
        cached_logits_replayed_exactly=True,
        new_logits_have_real_full_model_timings=True,
        original_text_pairs_untouched=320,
        quality_metrics_computed=False,
        formal_metrics=None,
    )
    once(ROOT / "pair_cache_verification.json", value)
    print(value, flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "infer", "verify"])
    action = parser.parse_args().action
    if action == "infer":
        finished = infer()
        print(dict(completed=finished, waiting=not finished), flush=True)
    else:
        globals()[action]()
