"""Native single-image-query relation review after frozen context-budget failure."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import balanced_supplement as base
import recover_balanced_media as recovery
import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.machine_teacher import resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_pair_review"


OPTIONS: dict[str, int] = dict(
    num_ctx=16384, num_predict=1024, temperature=0, seed=2026102412
)
once, load, rel = base.once, base.load, base.rel


def batches() -> list:
    """Return batches for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    result: list = []
    for mode, qs in base.query_batches():
        if mode == "image":
            result.extend((mode, [q]) for q in qs)
        else:
            result.append((mode, qs))
    assert len(result) == 28
    assert sum(len(qs) for _, qs in result) == 40
    return result


def payload(doc: dict, mode: str, qs: list[dict]) -> tuple[dict, dict]:
    """Return payload for this workflow.

    Args:
        doc: Doc used by this operation.
        mode: Mode used by this operation.
        qs: Qs used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    if mode == "image":
        assert len(qs) == 1
    original, evidence = base.payload(doc, mode, qs)
    request, identity = copy.deepcopy(original), copy.deepcopy(evidence)
    request["options"] = dict(OPTIONS)
    identity["options"] = dict(OPTIONS)
    identity["execution_change"] = (
        "single image query, full native original pair, larger context; no resizing or model reduction"
    )
    return request, identity


def parsed(raw: dict, count: int) -> Any:
    """Return parsed for this workflow.

    Args:
        raw: Raw used by this operation.
        count: Count used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    assert isinstance(raw, dict) and raw.get("model") == base.MODEL
    assert raw.get("done") is True and raw.get("done_reason") == "stop"
    assert type(raw.get("prompt_eval_count")) is int
    assert 0 <= raw["prompt_eval_count"] < OPTIONS["num_ctx"] - OPTIONS["num_predict"]
    assert (
        type(raw.get("eval_count")) is int
        and 0 <= raw["eval_count"] < OPTIONS["num_predict"]
    )
    assert raw["message"]["role"] == "assistant"

    def unique(pairs: list[tuple[str, object]]) -> dict:
        """Return unique for this workflow.

        Args:
            pairs: Pairs used by this operation.

        Returns:
            Result mapping produced by the workflow.

        Raises:
            ValueError: Duplicate native JSON key.
        """
        values = {}
        for k, v in pairs:
            if k in values:
                raise ValueError("Duplicate native JSON key")
            values[k] = v
        return values

    value = json.loads(raw["message"]["content"], object_pairs_hook=unique)
    assert set(value) == {
        "grades",
        "evidence",
        "candidate_product_type",
        "candidate_appearance",
    }
    assert isinstance(value["grades"], list) and isinstance(value["evidence"], list)
    assert len(value["grades"]) == len(value["evidence"]) == count
    assert all(v is None or type(v) is int and v in (0, 1) for v in value["grades"])
    assert all(
        isinstance(v, str) and 0 < len(v.strip()) <= 100 for v in value["evidence"]
    )
    for k, size in [("candidate_product_type", 100), ("candidate_appearance", 300)]:
        assert isinstance(value[k], str) and 0 < len(value[k].strip()) <= size
    return value


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    recovery.guard()
    if (ROOT / "protocol.json").exists():
        guard()
        return
    assert load(ROOT / "engineering_checks.pre.json")["passed"]
    failure = recovery.ROOT / "attempts/000_08/001.json"
    old = load(failure)
    assert old["exception"] is not None and old["result"] is None
    assert (
        old["raw"]["prompt_eval_count"]
        >= base.OPTIONS["num_ctx"] - base.OPTIONS["num_predict"]
    )
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_balanced_pair_review.py",
        ROOT / "engineering_checks.pre.json",
        recovery.ROOT / "protocol.json",
        recovery.ROOT / "acquisition_receipt.json",
        recovery.ROOT / "selected.evaluator.jsonl",
        failure,
        base.ROOT / "protocol.json",
        base.ROOT / "model_manifest.json",
    ]
    paths += [PROJECT_ROOT / n for n in load(base.ROOT / "protocol.json")["files"]]
    paths += [
        Path(r["image_path"])
        for r in read_rows(recovery.ROOT / "selected.evaluator.jsonl")
    ]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            original_protocol_sha256=sha256_file(base.ROOT / "protocol.json"),
            options=OPTIONS,
            same_full_model_digest=load(base.ROOT / "model_manifest.json")["digest"],
            reason="Prior four-image request had 7370 prompt tokens beyond the frozen 7168 reserve and ambiguous image references; original partial records and failure remain unchanged, no complete label set was made",
            mode="one image query per native original-image pair; four text queries per native candidate image",
            same_candidate_set=True,
            same_40_queries=True,
            same_business_relevance_rubric=True,
            full_resolution_native_images=True,
            model_layers_or_weights_reduced=False,
            previous_partial_judgments_reused=False,
            new_rankings_seen=False,
            unknowns="valid semantic null retained; engineering failure never becomes a grade",
            keep_alive="10m; natural expiry; never unload any model",
            human_labels=0,
            formal_metrics=None,
            independent_acceptance=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="pair_review_frozen",
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
    assert protocol["options"] == OPTIONS
    recovery.guard()


def jobs() -> list:
    """Return jobs for this workflow.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    return [
        (i, b, d, mode, qs)
        for i, d in enumerate(read_rows(recovery.ROOT / "selected.evaluator.jsonl"))
        for b, (mode, qs) in enumerate(batches())
    ]


def validate(record: dict, doc: dict, mode: str, qs: list[dict]) -> Any:
    """Return validate for this workflow.

    Args:
        record: Record used by this operation.
        doc: Doc used by this operation.
        mode: Mode used by this operation.
        qs: Qs used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    _, identity = payload(doc, mode, qs)
    assert record["identity"] == identity
    assert record["exception"] is None
    result = parsed(record["raw"], len(qs))
    assert record["result"] == result
    return result


def review() -> bool:
    """Return review for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, write_json, write_rows.
    """
    prepare()
    if (ROOT / "label_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
        return True
    planned = jobs()
    pending = []
    for i, b, doc, mode, qs in planned:
        path = ROOT / f"native/{i:03d}_{b:02d}.json"
        if path.exists():
            validate(load(path), doc, mode, qs)
        else:
            pending.append((i, b, doc, mode, qs))
    if pending and not base.gpu_available():
        print(
            {
                "waiting_for_natural_GPU_release": True,
                "pending_native_requests": len(pending),
            },
            flush=True,
        )
        return False
    client = None
    try:
        if pending:
            client = OllamaClient(timeout=240)
            model = resolve_local_teacher(
                client, Path(load(base.ROOT / "model_manifest.json")["model_root"])
            )
            assert (
                model["digest"]
                == load(ROOT / "protocol.json")["same_full_model_digest"]
            )
            once(ROOT / "runtime_model_manifest.json", model)
        for i, b, doc, mode, qs in pending:
            request, identity = payload(doc, mode, qs)
            raw, error, result = None, None, None
            start = time.perf_counter()
            try:
                assert client is not None
                raw = client._request("/api/chat", request)
                result = parsed(raw, len(qs))
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Handled exception in review", exc_info=True
                )
                error = dict(type=type(exc).__name__, message=str(exc))
            record = dict(
                identity=identity,
                raw=raw,
                exception=error,
                result=result,
                seconds=time.perf_counter() - start,
                review_origin="model",
                human_label=False,
                new_rankings_seen=False,
            )
            folder = ROOT / f"attempts/{i:03d}_{b:02d}"
            attempt = folder / f"{len(list(folder.glob('*.json')))+1:03d}.json"
            once(attempt, record)
            if error is not None:
                print({"engineering_failure": True, "file": str(attempt)}, flush=True)
                return False
            once(
                ROOT / f"native/{i:03d}_{b:02d}.json",
                dict(record, attempt_file=rel(attempt)),
            )
            if b == len(batches()) - 1:
                print(
                    {"AI_reviewed_products": i + 1, "total": len(planned) // 28},
                    flush=True,
                )
    finally:
        if client is not None:
            client.close()
    labels, profiles = [], []
    docs = read_rows(recovery.ROOT / "selected.evaluator.jsonl")
    for i, b, doc, mode, qs in planned:
        path = ROOT / f"native/{i:03d}_{b:02d}.json"
        result = validate(load(path), doc, mode, qs)
        if b == 0:
            profiles.append(
                dict(
                    id=doc["id"],
                    product_type=result["candidate_product_type"],
                    appearance=result["candidate_appearance"],
                    source_file=path.relative_to(ROOT).as_posix(),
                    review_origin="model",
                    human_label=False,
                )
            )
        for q, grade, reason in zip(qs, result["grades"], result["evidence"]):
            labels.append(
                dict(
                    query_key=q["key"],
                    candidate_id=doc["id"],
                    grade=grade,
                    reason=reason,
                    source_file=path.relative_to(ROOT).as_posix(),
                    review_origin="model",
                    human_label=False,
                    new_rankings_seen=False,
                )
            )
    assert len(labels) == len(docs) * 40 and len(
        {(r["query_key"], r["candidate_id"]) for r in labels}
    ) == len(labels)
    for name, rows in [
        ("judgments.ai.jsonl", labels),
        ("profiles.public.jsonl", profiles),
    ]:
        path = ROOT / name
        if path.exists():
            assert read_rows(path) == rows
        else:
            write_rows(path, rows)
    paths = [
        ROOT / n
        for n in (
            "judgments.ai.jsonl",
            "profiles.public.jsonl",
            "runtime_model_manifest.json",
        )
    ]
    paths += sorted((ROOT / "native").glob("*.json")) + sorted(
        (ROOT / "attempts").rglob("*.json")
    )
    once(
        ROOT / "label_receipt.json",
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths},
            AI_reviewed_products=len(docs),
            new_AI_relations=len(labels),
            native_successful_requests=len(planned),
            semantic_unknown=sum(r["grade"] is None for r in labels),
            human_labels=0,
            old_11840_relations_unchanged=True,
            previous_partial_batch_labels_not_reused=True,
            labels_frozen_before_new_gallery_predictions=True,
            added_to_comparison_gallery=False,
            full_new_encoding_and_same_gallery_comparison_pending=True,
            formal_metrics=None,
        ),
    )
    snapshot = load(recovery.ROOT / "post_acquisition_exclusion_snapshot.json")
    once(
        ROOT / "post_review_exclusion_snapshot.json",
        dict(
            snapshot,
            scope="All raw supplement pixels and source families remain development-exposed, including native AI review; not independent acceptance",
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="AI_labels_frozen_encoding_pending",
            AI_reviewed_products=len(docs),
            new_AI_relations=len(labels),
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "review"])
    args = parser.parse_args()
    with recovery.locked() as acquired:
        if acquired:
            result = globals()[args.action]()
            if args.action == "review":
                print({"review_completed": result}, flush=True)
        else:
            print({"another_runner_holds_lock": True}, flush=True)
