"""One transparent retry of sandbox-denied original URLs, with separate receipts."""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx

import balanced_supplement as base
import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.holdout_data import is_exposed
from multimodal_ecommerce.machine_teacher import resolve_local_teacher
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_balanced_media_recovery"
MEDIA = PROJECT_ROOT / "data/module2_balanced_recovery_media/records.jsonl"
once, load, rel = base.once, base.load, base.rel


def eligible(row: dict) -> bool:
    """Return eligible for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Whether the condition checked by this operation holds.
    """
    return row.get("error_type") == "ConnectError" and "10013" in row.get("error", "")


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Notes:
        Writes derived artifacts through once, write_json.
    """
    if (ROOT / "protocol.json").exists():
        guard()
        return
    base.guard()
    original = read_rows(base.MEDIA)
    assert len(original) == 48 and all(eligible(r) for r in original)
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tests/test_balanced_media_recovery.py",
        base.ROOT / "protocol.json",
        base.ROOT / "metadata_receipt.json",
        base.ROOT / "acquisition_receipt.json",
        base.ROOT / "download_cache.json",
        base.ROOT / "post_acquisition_exclusion_snapshot.json",
        base.ROOT / "model_manifest.json",
        base.QUEUE,
        base.MEDIA,
    ]
    paths += [PROJECT_ROOT / n for n in load(base.ROOT / "protocol.json")["files"]]
    once(
        ROOT / "protocol.json",
        dict(
            files={rel(p): sha256_file(p) for p in paths},
            reason="All 48 original GET attempts failed with WinError10013 socket permission before image content; immutable original evidence preserved",
            retry_eligible="Exactly original ConnectError10013; no HTTP rejection, timeout or other source error retried",
            same_original_URLs=True,
            new_candidate_selection=False,
            additional_attempts_per_original_URL=1,
            respect_proxy=True,
            workers=4,
            unchanged_original_sampling_and_review_protocol_sha256=sha256_file(
                base.ROOT / "protocol.json"
            ),
            new_AI_labels_before_ranking=True,
            human_labels=0,
            formal_metrics=None,
            default_strategy_changed=False,
            independent_acceptance=False,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="frozen_sandbox_network_recovery",
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )


def guard() -> None:
    """Validate the frozen inputs before continuing the workflow."""
    verify_files(PROJECT_ROOT, load(ROOT / "protocol.json")["files"])
    base.guard()
    if (ROOT / "acquisition_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "acquisition_receipt.json")["files"])
        verify_files(
            PROJECT_ROOT,
            load(ROOT / "post_acquisition_exclusion_snapshot.json")["files"],
        )


def acquire() -> None:
    """Return acquire for this workflow.

    Notes:
        Writes derived artifacts through once, write_json, write_rows.
    """
    prepare()
    if (ROOT / "acquisition_receipt.json").exists():
        return
    rows = read_rows(base.QUEUE)
    failures = {r["id"]: r for r in read_rows(base.MEDIA)}
    assert all(eligible(failures[r["id"]]) for r in rows)
    target = ROOT / "download_cache.json"
    cache = load(target) if target.exists() else {}
    pending = [r for r in rows if r["id"] not in cache]
    with httpx.Client(timeout=25, trust_env=True) as client:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for row, result in zip(
                pending,
                executor.map(lambda r: base.fetch_one(client, r, ROOT), pending),
            ):
                cache[row["id"]] = result
                write_json(target, cache)
    prior = load(base.ROOT / "post_acquisition_exclusion_snapshot.json")
    signatures = list(prior["image_fingerprints"])
    attempts, chosen = [], []
    for row in rows:
        result = cache[row["id"]]
        duplicate = result["status"] == "downloaded" and is_exposed(
            result["fingerprints"], signatures, 4
        )
        selected = result["status"] == "downloaded" and not duplicate
        attempts.append(
            dict(
                row,
                **result,
                original_attempt_error="ConnectError10013",
                total_attempt_number=2,
                technical_duplicate=bool(duplicate),
                ready_for_AI_review=selected,
            )
        )
        if result["status"] == "downloaded":
            signatures.append(result["fingerprints"])
        if selected:
            chosen.append(dict(row, **result))
    write_rows(MEDIA, attempts)
    write_rows(ROOT / "selected.evaluator.jsonl", chosen)
    once(
        ROOT / "post_acquisition_exclusion_snapshot.json",
        dict(
            prior,
            files=dict(prior["files"], **{rel(MEDIA): sha256_file(MEDIA)}),
            image_fingerprints=signatures,
            scope="All original failures and all retry metadata/raw images excluded from future unseen tests",
        ),
    )
    paths = [
        target,
        ROOT / "selected.evaluator.jsonl",
        ROOT / "post_acquisition_exclusion_snapshot.json",
    ]
    paths += [Path(r["image_path"]) for r in attempts if r["status"] == "downloaded"]
    once(
        ROOT / "acquisition_receipt.json",
        dict(
            files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths},
            retried_original_URLs=len(rows),
            downloaded=sum(r["status"] == "downloaded" for r in attempts),
            failed=sum(r["status"] != "downloaded" for r in attempts),
            ready_for_AI_review=len(chosen),
            technical_duplicates=sum(r["technical_duplicate"] for r in attempts),
            by_bucket=dict(Counter(r["bucket"] for r in chosen)),
            failure_types=dict(
                Counter(
                    (
                        "HTTP420"
                        if "420" in r.get("error", "")
                        else r.get("error_type", "unknown")
                    )
                    for r in attempts
                    if r["status"] != "downloaded"
                )
            ),
            original_receipts_unchanged=True,
            added_to_gallery=False,
            labels_created=0,
            formal_metrics=None,
        ),
    )
    print({"recovered": len(chosen), "attempted": len(rows)}, flush=True)


def review() -> bool:
    """Return review for this workflow.

    Returns:
        Whether the condition checked by this operation holds.

    Notes:
        Writes derived artifacts through once, write_json, write_rows.
    """
    guard()
    if (ROOT / "label_receipt.json").exists():
        verify_files(ROOT, load(ROOT / "label_receipt.json")["files"])
        return True
    docs = read_rows(ROOT / "selected.evaluator.jsonl")
    batches = base.query_batches()
    if not docs:
        once(
            ROOT / "no_usable_images.json",
            dict(new_labels=0, missing_images_not_labeled=True),
        )
        return True
    pending = [
        (i, b, d, mode, qs)
        for i, d in enumerate(docs)
        for b, (mode, qs) in enumerate(batches)
        if not (ROOT / f"native/{i:03d}_{b:02d}.json").exists()
    ]
    if pending and not base.gpu_available():
        print({"waiting_for_natural_GPU_release": True}, flush=True)
        return False
    client = None
    try:
        if pending:
            client = OllamaClient(timeout=240)
            model = resolve_local_teacher(
                client, Path(load(base.ROOT / "model_manifest.json")["model_root"])
            )
            assert model["digest"] == load(base.ROOT / "model_manifest.json")["digest"]
            once(ROOT / "runtime_model_manifest.json", model)
        for i, b, doc, mode, qs in pending:
            request, identity = base.payload(doc, mode, qs)
            response, error, result = None, None, None
            start = time.perf_counter()
            try:
                assert client is not None
                response = client._request("/api/chat", request)
                result = base.parsed(response, len(qs))
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Handled exception in review", exc_info=True
                )
                error = dict(type=type(exc).__name__, message=str(exc))
            record = dict(
                identity=identity,
                raw=response,
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
            if b == len(batches) - 1:
                print({"AI_reviewed_products": i + 1, "total": len(docs)}, flush=True)
    finally:
        if client is not None:
            client.close()
    judgments, profiles = [], []
    for i, doc in enumerate(docs):
        for b, (mode, qs) in enumerate(batches):
            path = ROOT / f"native/{i:03d}_{b:02d}.json"
            record = load(path)
            _, identity = base.payload(doc, mode, qs)
            assert record["identity"] == identity
            result = base.parsed(record["raw"], len(qs))
            assert record["result"] == result
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
                judgments.append(
                    dict(
                        query_key=q["key"],
                        candidate_id=doc["id"],
                        grade=grade,
                        reason=reason,
                        review_origin="model",
                        human_label=False,
                        new_rankings_seen=False,
                        source_file=path.relative_to(ROOT).as_posix(),
                    )
                )
    assert len(judgments) == 40 * len(docs)
    for name, rows in [
        ("judgments.ai.jsonl", judgments),
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
            new_AI_relations=len(judgments),
            native_successful_requests=len(docs) * len(batches),
            unknown_relations=sum(r["grade"] is None for r in judgments),
            human_labels=0,
            original_11840_relations_unchanged=True,
            added_to_comparison_gallery=False,
            complete_new_encoding_and_same_gallery_comparison_pending=True,
            formal_metrics=None,
        ),
    )
    write_json(
        ROOT / "public_status.json",
        dict(
            completed=False,
            phase="AI_labels_frozen_encoding_pending",
            acquired=len(docs),
            new_AI_relations=len(judgments),
            quality_metrics=None,
            formal_metrics=None,
            goal_met=False,
            default_strategy_changed=False,
            independent_AI_blind_predictions=0,
        ),
    )
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

    path = PROJECT_ROOT / "tmp/module2_balanced_recovery.lock"
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("xb") as f:
            f.write(b"0")
    except FileExistsError:
        logging.getLogger(__name__).debug("Reusing existing worker lock file")
    with path.open("r+b") as f:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            logging.getLogger(__name__).debug(
                "Handled exception in locked", exc_info=True
            )
            yield False
            return
        try:
            yield True
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "acquire", "review"])
    args = parser.parse_args()
    with locked() as acquired:
        if acquired:
            result = globals()[args.action]()
            if args.action == "review":
                print({"review_completed": result}, flush=True)
        else:
            print({"another_runner_holds_lock": True}, flush=True)
