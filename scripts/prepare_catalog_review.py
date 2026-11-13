"""Prepare isolated development candidates and prioritize actual human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import ijson

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.business_acceptance import (
    prior_product_keys,
    product_family_keys,
)
from multimodal_ecommerce.business_review_jobs import ROOT as ORIGINAL
from multimodal_ecommerce.business_review_jobs import canonical_hash, page
from multimodal_ecommerce.catalog_preparation import (
    BUCKETS,
    POLICY,
    merge_catalog_qa,
    metadata_bucket,
    priority_order,
)
from multimodal_ecommerce.data import parse_pv
from multimodal_ecommerce.dataset_release import order_key, read_rows, write_rows
from multimodal_ecommerce.holdout_data import fingerprints, is_exposed
from multimodal_ecommerce.media import fetch_image, inspect_image
from multimodal_ecommerce.safety import (
    PROJECT_ROOT,
    PROTECTED_ROOT,
    sha256_file,
    write_json,
)

ROOT = PROJECT_ROOT / "artifacts/module2_catalog_review"
REGISTRY = PROJECT_ROOT / "data/module2_development_expansion/records.jsonl"


def load(path: Path) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        path: Path of the input or output artifact used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def register_exposure(rows: list[dict], cache: dict) -> None:
    """Apply the register exposure workflow step.

    Args:
        rows: Records processed by this operation.
        cache: Cache used by this operation.

    Notes:
        Writes derived artifacts through REGISTRY.parent.mkdir, write_rows.
    """
    registry = []
    for row in rows:
        current = cache.get(row["id"], {})
        value = {
            **row,
            "project_role": "exposed_development_metadata_candidate_not_relevance_truth",
            "image_path": None,
        }
        if current.get("status") == "downloaded":
            value.update(
                image_path=os.path.relpath(
                    ROOT / current["path"], REGISTRY.parent
                ).replace("\\", "/"),
                image_sha256=current["fingerprints"]["sha256"],
                image_fingerprints=current["fingerprints"],
            )
        registry.append(value)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    write_rows(REGISTRY, registry)


def prepare() -> None:
    """Prepare the inputs and protocol for this development workflow.

    Raises:
        FileExistsError: Preserve the existing development preparation.

    Notes:
        Writes derived artifacts through ROOT.mkdir, write_json, write_rows.
    """
    if (ROOT / "preparation_protocol.json").exists() or REGISTRY.exists():
        raise FileExistsError("Preserve the existing development preparation")
    source = PROTECTED_ROOT / "多模态电商数据/product1m_product5m_train_id_label.json"
    prior = load(
        PROJECT_ROOT / "artifacts/module2_business_acceptance/prior_exposure.json"
    )
    verify_files(PROJECT_ROOT, prior["files"])
    reserved = PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl"
    reserved_rows = read_rows(reserved)
    old_keys = prior_product_keys()
    for row in reserved_rows:
        old_keys.update(row["family_keys"])
    lineage = (
        PROJECT_ROOT
        / "artifacts/module2_business_acceptance/dataset/evaluator/lineage.jsonl"
    )
    previous = {r["sha256"]: r for r in prior["image_fingerprints"]}
    for row in read_rows(lineage):
        previous[row["fingerprints"]["sha256"]] = row["fingerprints"]
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        ROOT / "prior_exclusion.json",
        dict(
            family_keys=sorted(old_keys),
            image_fingerprints=list(previous.values()),
            prior_exposure_sha256=sha256_file(
                PROJECT_ROOT
                / "artifacts/module2_business_acceptance/prior_exposure.json"
            ),
            reserved_registry_sha256=sha256_file(reserved),
            reserved_lineage_sha256=sha256_file(lineage),
            data_files={
                p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                for p in sorted((PROJECT_ROOT / "data").rglob("*.jsonl"))
            },
        ),
    )
    code = [
        "scripts/prepare_catalog_review.py",
        "src/multimodal_ecommerce/catalog_preparation.py",
        "src/multimodal_ecommerce/business_acceptance.py",
        "src/multimodal_ecommerce/media.py",
        "src/multimodal_ecommerce/holdout_data.py",
    ]
    write_json(
        ROOT / "preparation_protocol.json",
        dict(
            policy=POLICY,
            source=str(source.relative_to(PROJECT_ROOT)),
            source_sha256=sha256_file(source),
            prior_exclusion_sha256=sha256_file(ROOT / "prior_exclusion.json"),
            implementation={n: sha256_file(PROJECT_ROOT / n) for n in code},
            formal_metrics=None,
            human_labels=0,
            goal_met=False,
        ),
    )
    candidates: dict = {key: [] for key in BUCKETS}
    scanned = 0
    with source.open("rb") as handle:
        for sid, raw in ijson.kvitems(handle, ""):
            scanned += 1
            label = str(raw.get("label") or "")
            title = str(raw.get("title") or "")
            bucket = metadata_bucket(label, title)
            if bucket and raw.get("url"):
                row = dict(
                    id=f"product/train/{sid}",
                    source_id=str(sid),
                    dataset="product",
                    split="train",
                    label=label,
                    text=title,
                    image_url=raw["url"],
                    attributes=parse_pv(raw.get("pv") or ""),
                    bucket=bucket,
                    source_position=scanned,
                    source_record_sha256=canonical_hash(raw),
                )
                keys = product_family_keys(row)
                if not keys & old_keys:
                    row.update(
                        family_keys=sorted(keys),
                        priority=order_key(row["id"], POLICY["seed"]),
                    )
                    candidates[bucket].append(row)
            if scanned >= POLICY["scan_limit"]:
                break
    assert scanned == POLICY["scan_limit"]
    rows = []
    seen = set(old_keys)
    counts = {}
    for bucket, values in candidates.items():
        chosen = []
        for row in sorted(values, key=lambda r: r["priority"]):
            if set(row["family_keys"]) & seen:
                continue
            chosen.append(row)
            seen.update(row["family_keys"])
            if len(chosen) >= POLICY["metadata_budget_per_bucket"]:
                break
        if len(chosen) < POLICY["per_bucket"]:
            raise ValueError("Insufficient metadata for " + bucket)
        counts[bucket] = dict(
            eligible_before_family_dedup=len(values), queued=len(chosen)
        )
        rows.extend(chosen)
    write_rows(ROOT / "source_candidates.jsonl", rows)
    register_exposure(rows, {})
    write_json(
        ROOT / "metadata_receipt.json",
        dict(
            source_candidates_sha256=sha256_file(ROOT / "source_candidates.jsonl"),
            scanned=scanned,
            queued=len(rows),
            counts=counts,
            model_inferences=0,
            not_relevance_labels=True,
        ),
    )
    print(
        json.dumps(
            dict(
                scanned=scanned, queued=len(rows), target_images=64, model_inferences=0
            )
        )
    )


def fetch_one(row: dict) -> dict:
    """Return fetch one for this workflow.

    Args:
        row: Row used by this operation.

    Returns:
        Mapping with fields: id, status, path, url, fingerprints, width, height,
        error_type, error.

    Notes:
        Writes derived artifacts through folder.mkdir, temporary.write_bytes.
    """
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            content, url = fetch_image(client, row["image_url"])
        info = inspect_image(content)
        folder = ROOT / "dataset/media"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (info["sha256"] + info["suffix"])
        if not path.exists():
            temporary = path.with_name(
                path.name
                + "."
                + hashlib.sha256(row["id"].encode()).hexdigest()[:16]
                + ".tmp"
            )
            temporary.write_bytes(content)
            temporary.replace(path)
        assert sha256_file(path) == info["sha256"]
        return dict(
            id=row["id"],
            status="downloaded",
            path=path.relative_to(ROOT).as_posix(),
            url=url,
            fingerprints=fingerprints(path),
            width=info["width"],
            height=info["height"],
        )
    except (httpx.HTTPError, OSError, ValueError) as error:
        logging.getLogger(__name__).debug(
            "Handled exception in fetch_one", exc_info=True
        )
        return dict(
            id=row["id"],
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )


def download() -> None:
    """Apply the download workflow step.

    Raises:
        FileExistsError: Preserve the candidate catalog.
        ValueError: Insufficient unique decodable candidate images.

    Notes:
        Writes derived artifacts through write_json, write_rows.
    """
    if (ROOT / "dataset/manifest.json").exists():
        raise FileExistsError("Preserve the candidate catalog")
    protocol = load(ROOT / "preparation_protocol.json")
    verify_files(PROJECT_ROOT, protocol["implementation"])
    assert (
        protocol["policy"] == POLICY
        and sha256_file(ROOT / "prior_exclusion.json")
        == protocol["prior_exclusion_sha256"]
    )
    receipt = load(ROOT / "metadata_receipt.json")
    assert (
        sha256_file(ROOT / "source_candidates.jsonl")
        == receipt["source_candidates_sha256"]
    )
    prior = load(ROOT / "prior_exclusion.json")
    assert (
        sha256_file(PROJECT_ROOT / "data/module2_reserved_acceptance/records.jsonl")
        == prior["reserved_registry_sha256"]
    )
    rows = read_rows(ROOT / "source_candidates.jsonl")
    cache_path = ROOT / "download_cache.json"
    cache = load(cache_path) if cache_path.exists() else {}
    if not cache:
        first = fetch_one(rows[0])
        if first["status"] == "failed" and first["error_type"] in [
            "ConnectError",
            "ConnectTimeout",
        ]:
            raise ConnectionError(first["error"])
        cache[rows[0]["id"]] = first
        write_json(cache_path, cache)
    chosen: dict = {k: [] for k in BUCKETS}
    by_bucket = {k: [r for r in rows if r["bucket"] == k] for k in BUCKETS}
    offsets = {k: 0 for k in BUCKETS}
    used_keys = set(prior["family_keys"])
    selected_fingerprints: list = []
    reasons: Counter = Counter()
    while any(len(v) < POLICY["per_bucket"] for v in chosen.values()):
        pending = []
        for bucket in BUCKETS:
            needed = POLICY["per_bucket"] - len(chosen[bucket])
            if needed <= 0:
                continue
            start = offsets[bucket]
            pending.extend(by_bucket[bucket][start : start + needed])
            offsets[bucket] = min(len(by_bucket[bucket]), start + needed)
        if not pending:
            raise ValueError("Insufficient unique decodable candidate images")
        missing = [r for r in pending if r["id"] not in cache]
        with ThreadPoolExecutor(max_workers=POLICY["download_workers"]) as workers:
            for result in workers.map(fetch_one, missing):
                cache[result["id"]] = result
        write_json(cache_path, cache)
        register_exposure(rows, cache)
        for row in pending:
            result = cache[row["id"]]
            if result["status"] != "downloaded":
                reasons["download_or_decode_failure"] += 1
                continue
            path = ROOT / result["path"]
            signature = result["fingerprints"]
            assert sha256_file(path) == signature["sha256"]
            keys = set(row["family_keys"])
            if keys & used_keys:
                reasons["family_key_overlap"] += 1
                continue
            if is_exposed(
                signature, prior["image_fingerprints"], POLICY["dhash_threshold"]
            ):
                reasons["prior_or_reserved_image_duplicate_risk"] += 1
                continue
            if is_exposed(signature, selected_fingerprints, POLICY["dhash_threshold"]):
                reasons["within_new_catalog_image_duplicate_risk"] += 1
                continue
            chosen[row["bucket"]].append(
                {
                    **row,
                    "image_path": result["path"].removeprefix("dataset/"),
                    "fingerprints": signature,
                    "catalog_QA_status": "pending",
                    "relevance_grade": None,
                }
            )
            used_keys.update(keys)
            selected_fingerprints.append(signature)
        print(
            json.dumps(
                dict(
                    candidate_images=sum(map(len, chosen.values())),
                    target=64,
                    http_attempts=len(cache),
                )
            ),
            flush=True,
        )
    selected = [r for values in chosen.values() for r in values]
    write_rows(ROOT / "dataset/products.jsonl", selected)
    write_json(
        ROOT / "dataset/manifest.json",
        dict(
            role=POLICY["role"],
            products=len(selected),
            per_bucket={k: len(v) for k, v in chosen.items()},
            source_queue_sha256=sha256_file(ROOT / "source_candidates.jsonl"),
            registry_sha256=sha256_file(REGISTRY),
            exclusion_reasons=dict(reasons),
            human_catalog_approvals=0,
            human_relevance_labels=0,
            model_inferences=0,
            formal_metrics=None,
            goal_met=False,
            files={
                p.relative_to(ROOT / "dataset").as_posix(): sha256_file(p)
                for p in (ROOT / "dataset").rglob("*")
                if p.is_file()
            },
        ),
    )
    print(
        json.dumps(
            dict(
                candidate_images=len(selected),
                human_approved=0,
                human_relevance_labels=0,
                model_inferences=0,
            )
        )
    )


def qa_page(payload: object) -> str:
    """Return qa page for this workflow.

    Args:
        payload: Payload used by this operation.

    Returns:
        Text produced by this operation.
    """
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>开发商品质检</title><style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:16px;background:#f5f7fb;color:#18304a}header{background:white;padding:18px;border-radius:14px}button,input{font:inherit;padding:8px;margin:4px}button{cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}article{padding:14px;background:white;border:1px solid #d4deeb;border-radius:12px}img{width:100%;height:210px;object-fit:contain}article input{box-sizing:border-box;width:95%}.chosen{background:#bdd3f6}p{line-height:1.6;overflow-wrap:anywhere}#status{color:#a82e21}nav{margin:18px 0}</style><header><h1 id="heading"></h1><p>逐项检查：图片能看清商品主体，原标题与主体一致，商品符合待核实方向。元数据细类仅供核查。可用商品请填写实际主体；不适合或待核实请写理由。</p><p><strong>这里只做商品质检。“可用”不会被当成任何查询的检索相关标签。</strong>A/B独立判断，不交换答案。未处理保持空白，离开前请导出。</p><input id="reviewer" placeholder="本人姓名或唯一代号"><label><input id="human" type="checkbox">本人逐项判断</label><button id="export">导出当前进度</button><label>恢复文件<input id="restore" type="file" accept=".json"></label><p id="progress"></p><p id="status"></p></header><nav><button id="prev">上一页</button><button id="next">下一页</button></nav><main id="grid" class="grid"></main><script>window.JOB="""
        + data
        + """;</script><script src="../workbench.js"></script></html>"""
    )


def reviews() -> None:
    """Apply the reviews workflow step.

    Raises:
        FileExistsError: Preserve prepared review packets.

    Notes:
        Writes derived artifacts through (ROOT / 'index.html').write_text, (folder /
        'batches' / (batch['id'] + '.html')).write_text, (folder / 'batches').mkdir,
        (folder / 'index.html').write_text, folder.mkdir, qa.mkdir, write_json,
        write_rows.
    """
    if (ROOT / "reviews_ready.json").exists():
        raise FileExistsError("Preserve prepared review packets")
    verify_files(ORIGINAL, load(ORIGINAL / "release_receipt.json")["files"])
    verify_files(ROOT / "dataset", load(ROOT / "dataset/manifest.json")["files"])
    manifest = load(ORIGINAL / "manifest.json")
    rows = read_rows(ORIGINAL / "judgments.blank.jsonl")
    assert all(
        r["grade"] is None and r["review_origin"] is None and not r["reviewer"]
        for r in rows
    )
    conflicts = load(
        PROJECT_ROOT
        / "artifacts/module2_precision_recall/machine_conflicts_for_review.json"
    )["rows"]
    keys = {(r["query_id"], r["candidate_id"]) for r in conflicts}
    preferred = {r["id"] for r in rows if (r["query_id"], r["candidate_id"]) in keys}
    assert len(preferred) == 59
    by_id = {r["id"]: r for r in rows}
    priority = ROOT / "priority_review"
    shutil.copytree(ORIGINAL / "media", priority / "media")
    shutil.copyfile(
        PROJECT_ROOT / "assets/business_review_jobs.js", priority / "workbench.js"
    )
    counts = {b["id"]: len(set(b["row_ids"]) & preferred) for b in manifest["batches"]}
    batch_order = sorted(manifest["batches"], key=lambda b: (-counts[b["id"]], b["id"]))
    for slot in ["A", "B"]:
        folder = priority / ("rater_" + slot)
        (folder / "batches").mkdir(parents=True)
        links = []
        for batch in batch_order:
            current = priority_order(
                [by_id[key] for key in batch["row_ids"]], preferred
            )
            payload = dict(
                release_id=manifest["release_id"],
                batch_id=batch["id"],
                review_slot=slot,
                adjudication_of=None,
                rows=current,
                presentation={
                    r["id"]: {
                        k: "../../" + v if v else ""
                        for k, v in manifest["presentation"][r["id"]].items()
                    }
                    for r in current
                },
            )
            content = page(payload, "../../workbench.js")
            content = content.replace(
                '<section id="query">',
                f'<p>本批先评前{counts[batch["id"]]}项，其余条目仍保留，稍后继续补齐。导出可直接交回原收件流程。</p><section id="query">',
            )
            (folder / "batches" / (batch["id"] + ".html")).write_text(
                content, encoding="utf-8"
            )
            links.append(
                f'<li><a href="batches/{batch["id"]}.html">{batch["id"]} · {batch["query_id"]} · 优先{counts[batch["id"]]}项 · 完整批次{len(current)}项</a></li>'
            )
        (folder / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><h1>独立评审席位'
            + slot
            + "</h1><p>先完成各批前N个优先项并导出，其余保持待定；之后继续补齐全部5,143项。每批只交回选定的最新一个导出。请勿同时交回原版和本页的两份同批记录。</p><ol>"
            + "".join(links)
            + "</ol>",
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            priority / ("rater_" + slot + ".zip"), "w", zipfile.ZIP_DEFLATED
        ) as z:
            for part in [priority / "media", folder]:
                for path in part.rglob("*"):
                    if path.is_file():
                        z.write(path, path.relative_to(priority))
            z.write(priority / "workbench.js", "workbench.js")
            z.writestr(
                "START.html",
                '<meta charset="utf-8"><a href="rater_'
                + slot
                + '/index.html">进入独立优先评审</a>',
            )
    write_json(
        priority / "mapping.json",
        dict(
            original_release_id=manifest["release_id"],
            original_manifest_sha256=sha256_file(ORIGINAL / "manifest.json"),
            priority_row_ids=sorted(preferred),
            priority_counts=counts,
            full_original_rows_retained=True,
            original_pairs=5143,
            human_labels=0,
        ),
    )
    qa = ROOT / "catalog_QA"
    qa.mkdir()
    shutil.copyfile(PROJECT_ROOT / "assets/catalog_qa.js", qa / "workbench.js")
    catalog = read_rows(ROOT / "dataset/products.jsonl")
    templates = [
        dict(
            id=r["id"],
            role="development_catalog_QA_only",
            bucket=r["bucket"],
            bucket_name=BUCKETS[r["bucket"]]["name"],
            title=r["text"],
            source_label=r["label"],
            image="../../dataset/" + r["image_path"],
            verdict=None,
            product_type="",
            note="",
            reviewer="",
            review_origin=None,
        )
        for r in catalog
    ]
    release_id = canonical_hash(
        dict(
            dataset=sha256_file(ROOT / "dataset/manifest.json"),
            script=sha256_file(qa / "workbench.js"),
        )
    )
    write_rows(qa / "items.blank.jsonl", templates)
    write_json(
        qa / "manifest.json",
        dict(
            release_id=release_id,
            role="development_catalog_QA_only",
            products=len(templates),
            owners=dict(A=None, B=None),
            human_approved=0,
            human_relevance_labels=0,
        ),
    )
    for slot in ["A", "B"]:
        folder = qa / ("rater_" + slot)
        folder.mkdir()
        (folder / "index.html").write_text(
            qa_page(dict(release_id=release_id, review_slot=slot, rows=templates)),
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            qa / ("rater_" + slot + ".zip"), "w", zipfile.ZIP_DEFLATED
        ) as z:
            z.write(folder / "index.html", folder.relative_to(ROOT) / "index.html")
            z.write(qa / "workbench.js", "catalog_QA/workbench.js")
            for row in catalog:
                path = ROOT / "dataset" / row["image_path"]
                z.write(path, path.relative_to(ROOT))
            z.writestr(
                "START.html",
                '<meta charset="utf-8"><a href="catalog_QA/rater_'
                + slot
                + '/index.html">进入独立商品质检</a>',
            )
    write_json(
        qa / "release_receipt.json",
        dict(
            files={
                p.relative_to(qa).as_posix(): sha256_file(p)
                for p in qa.rglob("*")
                if p.is_file()
            }
        ),
    )
    summary = dict(
        priority_relations=59,
        original_pairs_retained=5143,
        original_batches_retained=54,
        batches_containing_priorities=sum(n > 0 for n in counts.values()),
        new_catalog_QA_items=64,
        catalog_approved=0,
        human_relevance_labels=0,
        model_predictions=0,
    )
    write_json(ROOT / "reviews_ready.json", summary)
    (ROOT / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>模块二人工评审准备</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto;padding:20px;line-height:1.8}section{background:#f3f6fb;padding:24px;border-radius:14px;margin:24px 0}a{color:#215b99}</style><h1>模块二人工评审准备</h1><p>目标：Precision@10必须≥60%，Recall@10争取≥85%。当前人工完成0，正式质量指标尚未产生。</p><section><h2>先处理59个优先关系</h2><p>54个原批次全部保留，优先项已移到各批前部。两位实际评审分别领取自己的包，独立判断。</p><a href="priority_review/rater_A.zip">评审A离线包</a> · <a href="priority_review/rater_B.zip">评审B离线包</a><p><a href="priority_review/rater_A/index.html">A任务页</a> · <a href="priority_review/rater_B/index.html">B任务页</a></p></section><section><h2>核查64件新增开发候选</h2><p>8类各8件。只质检图片与商品主体，“可用”不会被写成检索相关标签。</p><a href="catalog_QA/rater_A.zip">质检A离线包</a> · <a href="catalog_QA/rater_B.zip">质检B离线包</a><p><a href="catalog_QA/rater_A/index.html">A质检页</a> · <a href="catalog_QA/rater_B/index.html">B质检页</a></p></section><p>材料尚未向外发送，实际评审人员待指定。商品候选未经人工批准，原开发图库和验收集保持不变。</p></html>',
        encoding="utf-8",
    )
    print(json.dumps(summary))


def merge_qa(inbox: Path) -> None:
    """Merge qa.

    Args:
        inbox: Inbox used by this operation.

    Notes:
        Writes derived artifacts through out.mkdir, write_json.
    """
    qa = ROOT / "catalog_QA"
    verify_files(qa, load(qa / "release_receipt.json")["files"])
    exports = [load(p) for p in sorted(inbox.glob("*.json"))]
    report = merge_catalog_qa(
        load(qa / "manifest.json"), read_rows(qa / "items.blank.jsonl"), exports
    )
    key = canonical_hash(exports)[:24]
    out = qa / "merges" / key
    if (out / "receipt.json").exists():
        verify_files(out, load(out / "receipt.json")["files"])
        assert load(out / "status.json") == report
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "original_submissions.json", exports)
        write_json(out / "status.json", report)
        write_json(
            out / "receipt.json",
            dict(files={p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}),
        )
    write_json(
        qa / "current_status.json",
        {**report, "revision": out.relative_to(ROOT).as_posix()},
    )
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "download", "reviews", "merge-qa"]
    )
    parser.add_argument("--inbox", type=Path)
    args = parser.parse_args()
    if args.action == "merge-qa":
        if args.inbox is None or not args.inbox.is_dir():
            parser.error("--inbox must be an existing folder of actual QA exports")
        merge_qa(args.inbox)
    else:
        {"prepare": prepare, "download": download, "reviews": reviews}[args.action]()
