"""Replay the reviewed text route and isolate two subject-conflict diagnostics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import run  # noqa: F401
from module2_development_benchmark import Evidence, load
from module2_shared_gallery_fusion_review import (
    FUSION,
    REVIEW,
    TEXT_REVIEW,
    _apply_answers,
)
from module2_shared_gallery_precision import (
    CAPTIONS,
    POOL,
    ROOT,
    reviewed_inputs,
    worker_lock,
)
from module2_shared_gallery_retrieval import _checked_media, _score
from multimodal_ecommerce.ai_review_metrics import optional_mean_bounds
from multimodal_ecommerce.business_constraints import entity
from multimodal_ecommerce.business_retrieval import BUSINESS_INSTRUCTION
from multimodal_ecommerce.evaluation_access import require_data_access
from multimodal_ecommerce.human_audit_ui import render_review
from multimodal_ecommerce.interval_retrieval_metrics import query_bounds
from multimodal_ecommerce.machine_teacher import MODEL
from multimodal_ecommerce.ollama_client import OllamaClient
from multimodal_ecommerce.safety import PROJECT_ROOT, sha256_file, write_json
from multimodal_ecommerce.simple_baseline import rerank_head
from multimodal_ecommerce.visual_query import (
    CAPTION_OPTIONS,
    CAPTION_PROMPT,
    CAPTION_SCHEMA,
    DescribedImageReranker,
    caption_text,
    describe_query,
)

OUTPUT = ROOT / "modality_route"
CASE_REVIEW = OUTPUT / "human_review"
TOP = 10
TROUSERS = "text/长裤"


def _inputs(evidence: Evidence) -> tuple[dict, dict]:
    manifest, baseline, _ = reviewed_inputs(evidence)
    crop = evidence.json(ROOT / "crop_results.json")
    orders = evidence.json(ROOT / "crop_rankings.json", crop["rankings_sha256"])
    if orders["whole"] != baseline:
        raise ValueError("The original full-image ranking changed")
    queries = manifest["query_manifest"]
    receipt = evidence.json(TEXT_REVIEW / "progress.json")
    for directory in (REVIEW, TEXT_REVIEW):
        tasks_path = directory / "tasks.private.json"
        tasks = evidence.json(
            tasks_path,
            receipt["inputs"][tasks_path.relative_to(PROJECT_ROOT).as_posix()],
        )
        for path in (directory / "answers").glob("*.csv"):
            evidence.digest(
                path, receipt["inputs"][path.relative_to(PROJECT_ROOT).as_posix()]
            )
        if not _apply_answers(evidence, directory, tasks, queries)["complete"]:
            raise ValueError("Both ranking comparison packets must be complete")
    return manifest, orders


def _grade(order: list[str], query: dict) -> dict[str, int | None]:
    return {cid: query["judgments"].get(cid, {}).get("grade") for cid in order}


def _route_metrics(orders: dict, queries: dict) -> dict:
    baseline = orders["whole"]
    route = {
        key: orders[FUSION if query["mode"] == "text" else "whole"][key]
        for key, query in queries.items()
    }
    deltas = {}
    for scope in ("all", "text", "image"):
        mandatory, optional = [], []
        hits, queries_count = 0, 0
        for key, query in queries.items():
            if scope not in ("all", query["mode"]):
                continue
            queries_count += 1
            grades = _grade(baseline[key], query)
            before, after = set(baseline[key][:TOP]), set(route[key][:TOP])
            if any(grades[cid] is None for cid in before ^ after):
                raise ValueError(
                    "Resolved swap judgments are required for the routed comparison"
                )
            change = sum(grades[cid] == 1 for cid in after - before) - sum(
                grades[cid] == 1 for cid in before - after
            )
            hits += change
            positives = sum(grade == 1 for grade in grades.values())
            uncertain = sum(grade is None for grade in grades.values())
            if positives:
                ends = (change / positives, change / (positives + uncertain))
                mandatory.append((min(ends), max(ends)))
            elif uncertain:
                optional.append((0.0, 0.0))
        deltas[scope] = dict(
            net_top10_hits=hits,
            precision_at_10_delta=hits / (TOP * queries_count),
            paired_recall_at_10_delta=optional_mean_bounds(mandatory, optional),
        )
    return dict(
        baseline=_score(baseline, queries), route=_score(route, queries), paired=deltas
    )


def _documents(evidence: Evidence, catalog: dict, order: list[str]) -> list[dict]:
    return [
        dict(
            id=cid,
            text=catalog[cid]["title"],
            image_path=str(_checked_media(evidence, catalog[cid])),
        )
        for cid in order[:POOL]
    ]


def _precision_evidence(evidence: Evidence) -> tuple[dict, dict]:
    prior = evidence.json(ROOT / "precision_results.json")
    protocol = evidence.json(ROOT / "precision_protocol.json", prior["protocol_sha256"])
    model_config = PROJECT_ROOT / "configs/reranker_model.json"
    if (
        sha256_file(model_config) != protocol["reranker_config_sha256"]
        or protocol["instruction"] != BUSINESS_INSTRUCTION
        or protocol["pool_size"] != POOL
    ):
        raise ValueError("The original reranker configuration or pool budget changed")
    return prior, protocol


def _trouser(evidence: Evidence, orders: dict, query: dict, catalog: dict) -> dict:
    prior, protocol = _precision_evidence(evidence)
    whole, fused = orders["whole"][TROUSERS], orders[FUSION][TROUSERS]
    archived = prior["query_results"][TROUSERS]
    job = protocol["jobs"][TROUSERS]
    if (
        job["query_text"] != query["text"]
        or job["query_image"] is not None
        or job["documents"] != _documents(evidence, catalog, whole)
    ):
        raise ValueError(
            "The archived text job differs from the original query or documents"
        )
    slug = hashlib.sha256(TROUSERS.encode()).hexdigest()
    cached = evidence.json(ROOT / "pair_cache" / f"{slug}.json")
    old = cached["scores"]
    if (
        cached["identity"] != prior["protocol_sha256"]
        or rerank_head(whole, old, POOL)[:POOL] != archived["reranked_top20"]
    ):
        raise ValueError("Archived logits cannot reproduce the original text rerank")
    docs = _documents(evidence, catalog, fused)
    scores = {row["id"]: old[row["id"]] for row in docs if row["id"] in old}
    new = [row for row in docs if row["id"] not in scores]
    runtime = None
    if new:
        model = DescribedImageReranker()
        try:
            values = model.score(new, query_text=query["text"])
            scores.update(
                (row["id"], float(score))
                for row, score in zip(new, values, strict=True)
            )
            runtime = model.diagnostics()
        finally:
            model.close()
    reranked = rerank_head(fused, scores, POOL)
    whole_reranked = rerank_head(whole, old, POOL)
    grades = _grade(whole, query)
    variants = dict(
        whole=whole,
        fusion=fused,
        whole_reranked=whole_reranked,
        fusion_reranked=reranked,
    )
    moved = set(whole[:TOP]) ^ set(fused[:TOP])
    return dict(
        scope="one preselected text error diagnostic",
        variants={
            name: dict(top20=order[:POOL], metrics=query_bounds(order, grades))
            for name, order in variants.items()
        },
        original_fusion_swaps=[
            dict(
                candidate_id=cid,
                grade=grades[cid],
                ranks={name: order.index(cid) + 1 for name, order in variants.items()},
            )
            for cid in sorted(moved)
        ],
        scores=scores,
        reused_scores=len(scores) - len(new),
        new_scores=len(new),
        runtime=runtime,
    )


def _kind(text: str) -> str | None:
    kind = entity(text)
    return "footwear" if kind in ("sports_shoes", "other_shoes") else kind


def _caption_evidence(evidence: Evidence, query: dict) -> tuple[str, str]:
    summary = evidence.json(CAPTIONS / "summary.json")
    identity = summary["protocol_sha256"]
    plan = evidence.json(CAPTIONS / "protocol.json", identity)
    job = next(row for row in plan["jobs"] if row["qid"] == "fresh_chair")
    if job["image_sha256"] != query["image_sha256"] or _checked_media(
        evidence, query
    ) != Path(job["image"]):
        raise ValueError("The two caption inputs must be the original query image")
    kinds = set()
    for model in ("4b", "9b"):
        path = f"{model}/{job['qid']}.json"
        record = evidence.json(CAPTIONS / path, summary["prediction_sha256"][path])
        value = record["result"]["value"]
        if record["identity"]["protocol_sha256"] != identity or not caption_text(value):
            raise ValueError("The query caption is unclear or from another protocol")
        kinds.add(_kind(value["product"]))
    if len(kinds) != 1 or None in kinds:
        raise ValueError("The two independent query models must agree on product type")
    kind = kinds.pop()
    assert kind is not None
    return kind, plan["model_info"][MODEL]["digest"]


def _observe(evidence: Evidence, documents: list[dict], digest: str) -> dict:
    protocol = dict(
        model=MODEL,
        model_digest=digest,
        prompt=CAPTION_PROMPT,
        options=CAPTION_OPTIONS,
        schema=CAPTION_SCHEMA,
        images={
            doc["id"]: evidence.digest(Path(doc["image_path"])) for doc in documents
        },
    )
    path = OUTPUT / "candidate_subjects.json"
    require_data_access([path], "development", evidence.registry)
    saved: dict = (
        load(path) if path.exists() else dict(protocol=protocol, candidates={})
    )
    if saved["protocol"] != protocol or not set(saved["candidates"]) <= set(
        protocol["images"]
    ):
        raise ValueError("Candidate subject evidence belongs to another protocol")
    pending = [doc for doc in documents if doc["id"] not in saved["candidates"]]
    if pending:
        client = OllamaClient(model=MODEL)
        try:
            live = client.health()
            digests = {
                item["digest"] for item in live["models"] if item["name"] == MODEL
            }
            if digests != {digest}:
                raise ValueError("Local candidate vision model digest differs")
            for index, doc in enumerate(pending, 1):
                result = describe_query(
                    client, Path(doc["image_path"]), last=index == len(pending)
                )
                saved["candidates"][doc["id"]] = dict(
                    value=result["value"], seconds=result["wall_seconds"]
                )
                temporary = path.with_suffix(".partial")
                write_json(temporary, saved)
                temporary.replace(path)
                if index % 5 == 0 or index == len(pending):
                    print(
                        f"Candidate original-image observations: {len(saved['candidates'])}/{len(documents)}.",
                        flush=True,
                    )
        finally:
            client.close()
    return saved["candidates"]


def _chair(evidence: Evidence, orders: dict, queries: dict, catalog: dict) -> dict:
    prior = evidence.json(ROOT / "chair_subject_results.json")
    evidence.digest(ROOT / "chair_subject_protocol.json", prior["protocol_sha256"])
    key = prior["query"]
    query = queries[key]
    baseline = orders["whole"][key]
    head = prior["described_top20"]
    if len(head) != POOL or set(head) != set(baseline[:POOL]):
        raise ValueError("The described rerank must cover the original Top20")
    query_kind, digest = _caption_evidence(evidence, query)
    docs = _documents(evidence, catalog, baseline)
    observations = _observe(evidence, docs, digest)
    conflicts = {}
    for doc in docs:
        cid = doc["id"]
        title = _kind(doc["text"])
        value = observations[cid]["value"]
        visual = _kind(value["product"]) if caption_text(value) else None
        if title is not None and title == visual and title != query_kind:
            conflicts[cid] = title
    revised = [cid for cid in head if cid not in conflicts] + [
        cid for cid in head if cid in conflicts
    ]
    tail = baseline[POOL:]
    before, after = head + tail, revised + tail
    negative = prior["reviewed_negative"]
    grades = _grade(baseline, query)
    return dict(
        scope="one preselected image error diagnostic, same candidates and model scores",
        query=key,
        query_kind=query_kind,
        corroborated_conflicts=conflicts,
        confirmed_wrong_product_ranks=[
            before.index(negative) + 1,
            after.index(negative) + 1,
        ],
        before=dict(top20=head, metrics=query_bounds(before, grades)),
        after=dict(top20=revised, metrics=query_bounds(after, grades)),
        entering=sorted(set(after[:TOP]) - set(before[:TOP])),
        leaving=sorted(set(before[:TOP]) - set(after[:TOP])),
    )


def _run(evidence: Evidence) -> None:
    """Run the all-query replay and two scoped local model diagnostics."""
    with worker_lock():
        manifest, orders = _inputs(evidence)
        queries = manifest["query_manifest"]
        catalog = {row["id"]: row for row in manifest["catalog"]}
        route = _route_metrics(orders, queries)
        print(f"Replayed the modality route over {len(queries)} queries.", flush=True)
        text_result = _trouser(evidence, orders, queries[TROUSERS], catalog)
        print(
            f"Trousers: {text_result['reused_scores']} reused and {text_result['new_scores']} new scores.",
            flush=True,
        )
        chair = _chair(evidence, orders, queries, catalog)
        evidence.digest(OUTPUT / "candidate_subjects.json")
        write_json(
            OUTPUT / "results.json",
            dict(
                role="exposed_development_modality_route_and_case_diagnostics",
                route_policy={"text": FUSION, "image": "whole"},
                route=route,
                trousers=text_result,
                chair=chair,
                inputs=evidence.files,
                source_sha256=sha256_file(Path(__file__)),
                formal_metrics=None,
            ),
        )
        print(
            f"Finished fixed Top20 chair comparison. Mismatch ranks: {chair['confirmed_wrong_product_ranks']}."
        )


def _review(evidence: Evidence, score_only: bool) -> None:
    manifest, orders = _inputs(evidence)
    result = evidence.json(OUTPUT / "results.json")
    chair = result["chair"]
    qid = chair["query"]
    query = manifest["query_manifest"][qid]
    catalog = {row["id"]: row for row in manifest["catalog"]}
    head = orders["whole"][qid][:POOL]
    before, after = [chair[name]["top20"] for name in ("before", "after")]
    if any(len(order) != POOL or set(order) != set(head) for order in (before, after)):
        raise ValueError("Review comparison must preserve the frozen Top20")
    grades = _grade(head, query)
    image = str(_checked_media(evidence, query))
    changes = set(before[:TOP]) ^ set(after[:TOP])
    tasks = []
    for cid in sorted(changes):
        if grades[cid] is not None or query["judgments"].get(cid, {}).get("human"):
            continue
        doc = catalog[cid]
        tasks.append(
            dict(
                task_id=hashlib.sha256(
                    f"{qid}\n{cid}\nsubject-gate".encode()
                ).hexdigest()[:24],
                query_id=qid,
                candidate_id=cid,
                query=dict(text=None, image_path=image),
                candidate=dict(
                    title=doc["title"], image_path=str(_checked_media(evidence, doc))
                ),
            )
        )
    path = CASE_REVIEW / "tasks.private.json"
    if path.exists() and evidence.json(path) != tasks:
        raise ValueError("Existing case review tasks differ")
    if score_only and not path.exists():
        raise ValueError("Prepare the case review before importing")
    if not score_only:
        render_review(
            tasks,
            CASE_REVIEW,
            title="模块二 · 餐椅图搜换位复核",
            instructions=(
                "按原标准核对商品主体、用途与可见结构。"
                + "选相关、不相关或证据不足，写一句理由，导出 CSV 放进本目录 answers 文件夹。"
            ),
        )
        if not path.exists():
            write_json(path, tasks)
            evidence.digest(path)
    progress = _apply_answers(evidence, CASE_REVIEW, tasks, manifest["query_manifest"])
    grades = _grade(orders["whole"][qid], query)
    entrants = [grades[cid] for cid in set(after[:TOP]) - set(before[:TOP])]
    exits = [grades[cid] for cid in set(before[:TOP]) - set(after[:TOP])]
    net = entrants.count(1) - exits.count(1)
    tail = orders["whole"][qid][POOL:]
    write_json(
        CASE_REVIEW / "progress.json",
        dict(
            reviewed=progress["completed"],
            expected=progress["expected"],
            paired_precision_delta=[
                (net - exits.count(None)) / TOP,
                (net + entrants.count(None)) / TOP,
            ],
            before=query_bounds(before + tail, grades),
            after=query_bounds(after + tail, grades),
            inputs=evidence.files,
            formal_metrics=None,
        ),
    )
    print(f"Chair case review: {progress['completed']}/{progress['expected']}.")


def main() -> None:
    """Run the modality experiment or prepare and import the focused human review."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-review", action="store_true")
    mode.add_argument("--score-review", action="store_true")
    args = parser.parse_args()
    evidence = Evidence(load(PROJECT_ROOT / "configs/evaluation_registry.json"))
    if args.prepare_review or args.score_review:
        _review(evidence, args.score_review)
    else:
        _run(evidence)


if __name__ == "__main__":
    main()
