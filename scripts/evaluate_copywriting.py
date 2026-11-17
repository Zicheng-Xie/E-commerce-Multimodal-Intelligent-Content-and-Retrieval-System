"""创建两维人工评测包，核验冻结样本与评分，并独立汇总验收指标。"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

CATEGORIES = ["3C数码", "服饰穿搭", "家居日用"]
ELIGIBLE_LABELS = {
    "3C数码": [
        "显示器",
        "液晶显示器",
        "手机耳机",
        "耳机/耳麦",
        "耳机",
        "蓝牙耳机",
        "头戴式耳机",
        "入耳式耳机",
        "有线耳机",
        "无线鼠标",
        "有线鼠标",
        "鼠标",
        "键盘",
        "机械键盘",
        "手机充电器",
        "充电器",
        "车载充电器",
        "移动电源",
        "U盘",
        "手机U盘",
        "路由器",
        "无线路由器",
        "分布式路由器",
    ],
    "服饰穿搭": [
        "T恤",
        "运动T恤",
        "POLO衫",
        "衬衫",
        "卫衣/绒衫",
        "卫衣",
        "毛针织衫",
        "针织衫/毛衣",
        "羽绒服",
        "夹克",
        "皮衣",
        "牛仔裤",
        "休闲裤",
        "半身裙",
        "连衣裙",
    ],
    "家居日用": [
        "衣架",
        "婴童衣架",
        "水杯",
        "玻璃杯",
        "马克杯",
        "太空杯",
        "保温杯",
        "毛巾",
        "浴巾",
        "毛巾/浴巾/吸水毛巾",
        "收纳盒",
        "收纳箱",
        "桌面收纳盒",
        "内衣收纳盒",
        "置物架",
        "阳台置物架",
        "垃圾桶",
        "不锈钢垃圾桶",
        "拖把",
        "衣帽架",
        "口水巾",
    ],
}
HARD_ERRORS = ["材质编造", "功能虚标", "参数编造或矛盾", "规格混淆"]
RUBRIC = {
    "relevance": "核对原始商品属性及核心字段；核心字段命中率至少80%，没有材质、功能、参数编造或规格混淆，关键参数与成分不遗漏，才记1，否则记0。正常风格修饰不算事实编造。",
    "fluency": "标题、详情、卖点均应符合简体中文电商表达；语句通顺、逻辑清楚，无重复凑数、纯参数堆砌、说明书式片段和内部指令。整体满足才记1，否则记0。",
    "acceptance": "相关度通过条数/全部冻结样本数≥80%；通顺度通过条数/全部冻结样本数≥85%。30条分别至少24、26条。缺评不得视为通过，分品类另报10条的结果。",
}


def read_json(path: Path) -> Any:
    """读取UTF-8 JSON，保留中文和原始属性。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    """计算文件校验值，绑定评分与实际文案。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """写入任务产物，不覆盖原商品数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_acceptance(
    source: Path, history_roots: list[Path], out: Path, seed: int
) -> None:
    """排除可见使用历史后按品类蓄水池随机抽样，不按生成效果筛选。"""
    if out.exists():
        raise ValueError("冻结目录已存在，禁止重新抽样覆盖：" + str(out))
    module_path = Path(__file__).with_name("test_copywriting_batch.py")
    spec = importlib.util.spec_from_file_location(
        "copywriting_freeze_generator", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("无法加载商品解析代码")
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    used: set[tuple[str, str]] = set()
    sources = []

    def collect(value: Any) -> None:
        """同时支持生成清单、记录和官方格式的小型历史输入。"""
        if isinstance(value, dict):
            if value.get("product_id"):
                used.add(
                    (
                        str(value["product_id"]),
                        str(value.get("raw_record", {}).get("title", "")),
                    )
                )
            for key, item in value.items():
                if (
                    str(key).isdigit()
                    and isinstance(item, dict)
                    and "pv" in item
                    and "label" in item
                ):
                    used.add((str(key), str(item.get("title", ""))))
                elif isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    files: set[Path] = set()
    for root in history_roots:
        if not root.exists():
            raise ValueError("历史来源不存在：" + str(root))
        for path in [root] if root.is_file() else root.rglob("*.json"):
            if path.is_file() and (
                path.parent.name == "records"
                or path.name
                in {"manifest.json", "selection_audit.json", "selection.json"}
                or "raw_samples" in path.name
                or re.search("回归输入|针对性生成", path.name)
            ):
                files.add(path.resolve())
    for path in sorted(files):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(
                "历史文件过大，请用对应manifest或records替代：" + str(path)
            )
        collect(read_json(path))
        sources.append({"path": str(path), "sha256": digest(path)})
    if not used:
        raise ValueError("未发现历史商品记录，不能声明独立抽样")
    used_ids = {pid for pid, _ in used}
    used_titles = {generation.normalize_terms(title) for _, title in used if title}
    rng = random.Random(seed)
    reservoir: dict[str, list[dict[str, Any]]] = {key: [] for key in CATEGORIES}
    seen: Counter[str] = Counter()
    unique_titles: set[str] = set()
    label_cache: dict[str, Any] = {}
    allowed_labels = {
        label: category
        for category, labels in ELIGIBLE_LABELS.items()
        for label in labels
    }
    source_stat = source.stat()
    scanned = 0
    for pid, raw in generation.stream_products(source):
        scanned += 1
        if str(pid) in used_ids or not isinstance(raw, dict):
            continue
        title = generation.normalize_terms(raw.get("title", ""))
        if title and (title in used_titles or title in unique_titles):
            continue
        label = str(raw.get("label", ""))
        if label not in allowed_labels:
            continue
        if label not in label_cache:
            label_cache[label] = generation.detect_category({"label": label}, {})
        category = label_cache[label]
        if not category or category[2] != "label商品类型":
            continue
        sample = generation.normalize_product(str(pid), raw)
        if (
            not sample
            or sample["category_source"] != "label商品类型"
            or sample["category"] != allowed_labels[label]
        ):
            continue
        unique_titles.add(title)
        group = sample["category"]
        seen[group] += 1
        bucket = reservoir[group]
        index = len(bucket) if len(bucket) < 10 else rng.randrange(seen[group])
        if index < 10:
            if index == len(bucket):
                bucket.append(sample)
            else:
                bucket[index] = sample
    if any(len(reservoir[key]) != 10 for key in CATEGORIES):
        raise ValueError("未使用且品类明确的候选不足三个品类各10条")
    if (
        source.stat().st_size != source_stat.st_size
        or source.stat().st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise ValueError("抽样期间官方文件发生变化")
    samples = [row for key in CATEGORIES for row in reservoir[key]]
    for index, row in enumerate(samples, 1):
        row["id"] = f"S{index:03d}"
    write_json(
        out / "raw_samples_acceptance30.json",
        {row["product_id"]: row["raw_record"] for row in samples},
    )
    write_json(out / "samples.json", samples)
    write_json(
        out / "used_history.json",
        {
            "samples": [
                {"product_id": pid, "raw_record": {"title": title}}
                for pid, title in sorted(used)
            ],
            "sources": sources,
        },
    )
    write_json(
        out / "freeze.json",
        {
            "role": "acceptance",
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "selection": "按明确官方品类标签及品类蓄水池抽样，排除历史ID及规范化标题；不按规则分数、属性完整度、冲突状态或生成效果筛选",
            "eligible_labels": ELIGIBLE_LABELS,
            "source": {
                "path": str(source.resolve()),
                "size": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
            },
            "scanned": scanned,
            "eligible_by_category": dict(seen),
            "category_counts": dict(Counter(row["category"] for row in samples)),
            "history_product_count": len(used_ids),
            "parser_sha256": digest(module_path),
            "file_sha256": {
                name: digest(out / name)
                for name in [
                    "raw_samples_acceptance30.json",
                    "samples.json",
                    "used_history.json",
                ]
            },
            "restriction": "固定保留30条，不用其文案、标签或评分调整Prompt和规则；后续优化用开发集。仅能证明与列出的可见本地历史无重叠。",
        },
    )
    verify_frozen(out)


def verify_frozen(folder: Path) -> dict[str, Any]:
    """冻结清单、排除历史和样本内容必须同时匹配。"""
    if (folder / "withdrawn.json").exists():
        raise ValueError("候选清单已作废，不能用于验收")
    frozen = read_json(folder / "freeze.json")
    for name, expected in frozen["file_sha256"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError("冻结文件校验失败：" + name)
    samples = read_json(folder / "samples.json")
    ids = [row["product_id"] for row in samples]
    counts = Counter(row["category"] for row in samples)
    if len(ids) != 30 or len(set(ids)) != 30 or counts != dict.fromkeys(CATEGORIES, 10):
        raise ValueError("独立验收集必须为去重后的三个品类各10条")
    history = read_json(folder / "used_history.json")["samples"]
    used = {str(row["product_id"]) for row in history}
    titles = {
        row["raw_record"]["title"].strip().casefold()
        for row in history
        if row["raw_record"].get("title")
    }
    if used.intersection(ids) or any(
        row["raw_record"].get("title", "").strip().casefold() in titles
        for row in samples
    ):
        raise ValueError("验收集与已提供历史样本重叠")
    return frozen


def verify_development_report(path: Path) -> dict[str, Any]:
    """开发20条必须由人工完成两个维度，分数高低不影响流程就绪判断。"""
    report = read_json(path)
    if (
        report.get("role") != "development"
        or report.get("reviewer_type") != "human"
        or report.get("human_confirmed") is not True
        or report.get("overall", {}).get("total") != 20
        or any(
            report.get("overall", {}).get(key, {}).get("reviewed") != 20
            for key in ["relevance", "fluency"]
        )
    ):
        raise ValueError("开发20条的两个维度尚未完成人工评分")
    return report


def prepare(
    run_dir: Path,
    out: Path,
    role: str,
    frozen_dir: Path | None = None,
    development_report: Path | None = None,
) -> None:
    """创建不预填分数的人工标注包，绑定原始输入和最终文案。"""
    if out.exists():
        raise ValueError("评测目录已存在，避免覆盖人工评分：" + str(out))
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    samples = manifest["samples"]
    frozen = verify_frozen(frozen_dir) if frozen_dir else None
    if role == "acceptance":
        if development_report is None:
            raise ValueError("先完成20条开发样本人工评分，再提供其汇总报告")
        verify_development_report(development_report)
        if not frozen or frozen_dir is None:
            raise ValueError("正式验收必须提供冻结清单")
        expected = {row["product_id"] for row in read_json(frozen_dir / "samples.json")}
        if {row["product_id"] for row in samples} != expected:
            raise ValueError("生成批次与冻结30条不一致")
    rows = []
    for sample in samples:
        sid = sample["id"]
        if Path(sid).name != sid or sid in {".", ".."}:
            raise ValueError("非法样本编号")
        path = run_dir / "records" / (sid + ".json")
        record = read_json(path)
        if record["sample"]["product_id"] != sample["product_id"]:
            raise ValueError("文案记录与清单不匹配：" + sid)
        payload = record["payload"]
        rows.append(
            {
                "id": sid,
                "product_id": sample["product_id"],
                "category": sample["category"],
                "raw_record": sample.get("raw_record", {}),
                "facts": payload["facts"],
                "core_fields": payload["core_fields"],
                "required_fields": payload.get("required_fields", []),
                "source_conflicts": payload.get("source_conflicts", []),
                "result": record.get("result"),
                "generation_error": record.get("error"),
                "truncated": record.get("generation_completion", {}).get(
                    "truncated", False
                ),
                "record_path": str(path.resolve()),
                "record_sha256": digest(path),
                "review": {
                    "relevance": None,
                    "fluency": None,
                    "core_hits": None,
                    "hard_errors": [],
                    "missing_required_fields": [],
                    "relevance_reason": "",
                    "fluency_reason": "",
                },
            }
        )
    package = {
        "schema_version": "human_two_dimensions_v1",
        "role": role,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "manifest_sha256": digest(manifest_path),
        "code_sha256": manifest.get("code_sha256"),
        "frozen_sha256": digest(frozen_dir / "freeze.json") if frozen_dir else None,
        "frozen_dir": str(frozen_dir.resolve()) if frozen_dir else None,
        "development_report_sha256": (
            digest(development_report) if development_report else None
        ),
        "reviewer_type": "human",
        "reviewer": "",
        "human_confirmed": False,
        "rubric": RUBRIC,
        "rows": rows,
    }
    write_json(out / "review_template.json", package)
    render_form(package, out / "人工评测.html")


def render_form(package: dict[str, Any], path: Path) -> None:
    """生成离线标注表，浏览器只下载评分文件，不回写商品源文件。"""
    data = json.dumps(package, ensure_ascii=False).replace("<", "\\u003c")
    rubric = "".join("<p>" + html.escape(value) + "</p>" for value in RUBRIC.values())
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>商品文案人工评测</title>
<style>body{font:16px/1.6 sans-serif;max-width:1280px;margin:24px auto;padding:16px;color:#17212b}article{border-top:2px solid #ccd6df;padding:20px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px}textarea{width:95%;min-height:65px}label{display:inline-block;margin:5px}select,input,button{font:inherit;padding:6px}header{background:white;position:sticky;top:0;border-bottom:1px solid #ccc;padding:10px}button{cursor:pointer}h2{font-size:20px}.note{color:#704600}@media(max-width:750px){.cols{display:block}}</style>
<h1>商品文案人工评测</h1>RUBRIC<header>评审人 <input id="reviewer"> <label><input type="checkbox" id="confirmed">本人逐条核对，评分由人工确认</label> <button onclick="save()">下载人工评分JSON</button><p id="progress"></p></header><main id="rows"></main>
<script>const pack=DATA;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const select=(i,k)=>`<select id="${k}${i}"><option value="">未评</option><option value="1">通过 1</option><option value="0">不通过 0</option></select>`;
document.getElementById('rows').innerHTML=pack.rows.map((r,i)=>`<article><h2>${esc(r.id)} · ${esc(r.category)} · ${esc(r.product_id)}</h2><div class="cols"><section><h3>商品原始输入</h3><pre>${esc(JSON.stringify(r.raw_record,null,2))}</pre><h3>提取属性与核心字段（需对照原文核实）</h3><pre>${esc(JSON.stringify({facts:r.facts,core_fields:r.core_fields,required_fields:r.required_fields,source_conflicts:r.source_conflicts},null,2))}</pre></section><section><h3>生成文案</h3><pre>${esc(JSON.stringify(r.result,null,2))}</pre><p class="note">${r.truncated?'输出截断；不能记相关度通过。':''}${r.generation_error?'生成异常；两个维度均应记0。':''}</p><p>相关度 ${select(i,'relevance')} 通顺度 ${select(i,'fluency')}</p><p>核心属性命中（逐项确认）</p>${r.core_fields.map(k=>`<label><input type="checkbox" class="hit${i}" value="${esc(k)}">${esc(k)}：${esc(r.facts[k])}</label>`).join('')}<p>硬错误（有任一项，相关度不能记1）</p>ERRORS<p>相关度依据：注明命中的参数、遗漏或编造原句及输入依据</p><textarea id="rr${i}"></textarea><p>通顺度依据：注明语病、重复、堆砌或正常营销表达判断</p><textarea id="fr${i}"></textarea></section></div></article>`).join('');
function collect(){pack.reviewer=document.getElementById('reviewer').value.trim();pack.human_confirmed=document.getElementById('confirmed').checked;pack.reviewed_at=new Date().toISOString();pack.rows.forEach((r,i)=>{r.review={relevance:document.getElementById('relevance'+i).value===''?null:Number(document.getElementById('relevance'+i).value),fluency:document.getElementById('fluency'+i).value===''?null:Number(document.getElementById('fluency'+i).value),core_hits:[...document.querySelectorAll('.hit'+i+':checked')].map(n=>n.value),hard_errors:[...document.querySelectorAll('.error'+i+':checked')].map(n=>n.value),relevance_reason:document.getElementById('rr'+i).value.trim(),fluency_reason:document.getElementById('fr'+i).value.trim()}});return pack}
function save(){const p=collect();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],{type:'application/json'}));a.download='人工评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.addEventListener('change',()=>{const p=collect();let n=p.rows.filter(r=>r.review.relevance!==null&&r.review.fluency!==null).length;document.getElementById('progress').textContent=`已评分 ${n}/${p.rows.length}；下载后使用汇总命令核验，不会自动判定验收通过。`;localStorage.setItem(p.manifest_sha256,JSON.stringify(p))});
const saved=localStorage.getItem(pack.manifest_sha256);if(saved){try{const p=JSON.parse(saved);document.getElementById('reviewer').value=p.reviewer||'';document.getElementById('confirmed').checked=p.human_confirmed===true;pack.rows.forEach((r,i)=>{const q=p.rows.find(x=>x.id===r.id)?.review;if(!q)return;for(const k of ['relevance','fluency'])document.getElementById(k+i).value=q[k]===null?'':String(q[k]);for(const [k,c] of [['core_hits','hit'],['hard_errors','error']])document.querySelectorAll('.'+c+i).forEach(n=>n.checked=(q[k]||[]).includes(n.value));document.getElementById('rr'+i).value=q.relevance_reason||'';document.getElementById('fr'+i).value=q.fluency_reason||''})}catch(e){console.warn('历史草稿未能恢复',e)}}
</script></html>"""
    errors = (
        "${"
        + json.dumps(HARD_ERRORS, ensure_ascii=False)
        + '.map(k=>`<label><input type="checkbox" class="error${i}" value="${esc(k)}">${esc(k)}</label>`).join(\'\')}'
    )
    document = document.replace(
        "<p>硬错误（有任一项，相关度不能记1）</p>",
        '<p>遗漏的关键字段（没有则不勾）</p>${r.required_fields.map(k=>`<label><input type="checkbox" class="missing${i}" value="${esc(k)}">${esc(k)}</label>`).join(\'\')}<p>硬错误（有任一项，相关度不能记1）</p>',
    )
    document = document.replace(
        "hard_errors:[...document.querySelectorAll",
        "missing_required_fields:[...document.querySelectorAll('.missing'+i+':checked')].map(n=>n.value),hard_errors:[...document.querySelectorAll",
    )
    document = document.replace(
        "['hard_errors','error']",
        "['hard_errors','error'],['missing_required_fields','missing']",
    )
    path.write_text(
        document.replace("RUBRIC", rubric)
        .replace("ERRORS", errors)
        .replace("DATA", data),
        encoding="utf-8",
    )


def summarize(template_path: Path, review_path: Path, out: Path) -> dict[str, Any]:
    """仅在真实记录一致、人工身份声明完整且逐项评分后计算验收。"""
    template, submitted = read_json(template_path), read_json(review_path)
    manifest = Path(template["run_dir"]) / "manifest.json"
    if digest(manifest) != template["manifest_sha256"]:
        raise ValueError("生成清单已变化，不能复用旧评分")
    if template["frozen_dir"]:
        verify_frozen(Path(template["frozen_dir"]))
        if (
            digest(Path(template["frozen_dir"]) / "freeze.json")
            != template["frozen_sha256"]
        ):
            raise ValueError("冻结清单变化")
    for key in ["manifest_sha256", "role", "frozen_sha256"]:
        if submitted.get(key) != template[key]:
            raise ValueError("评分与评测任务不匹配：" + key)
    if (
        submitted.get("reviewer_type") != "human"
        or not submitted.get("reviewer", "").strip()
        or submitted.get("human_confirmed") is not True
    ):
        raise ValueError("须由实际人工评审填写姓名并确认；AI预审不能计入人工验收")
    incoming = submitted.get("rows", [])
    if len(incoming) != len(template["rows"]) or len(
        {row["id"] for row in incoming}
    ) != len(incoming):
        raise ValueError("评分存在缺项、重复或新增样本")
    reviews = {row["id"]: row for row in incoming}
    checked = []
    for row in template["rows"]:
        if digest(Path(row["record_path"])) != row["record_sha256"]:
            raise ValueError("文案记录已变化：" + row["id"])
        incoming_row = reviews.get(row["id"], {})
        if {key: value for key, value in incoming_row.items() if key != "review"} != {
            key: value for key, value in row.items() if key != "review"
        }:
            raise ValueError("评分文件改动了输入或输出内容：" + row["id"])
        review = incoming_row.get("review", {})
        for key in ["relevance", "fluency"]:
            if review.get(key) is not None and (
                type(review[key]) is not int or review[key] not in [0, 1]
            ):
                raise ValueError("评分只允许0、1或未评null")
        for key in ["relevance", "fluency"]:
            if (
                review.get(key) is not None
                and not review.get(key + "_reason", "").strip()
            ):
                raise ValueError("每个已评维度须填写核对依据：" + row["id"])
        hits = review.get("core_hits")
        if hits is not None and (
            not isinstance(hits, list)
            or len(set(hits)) != len(hits)
            or not set(hits).issubset(row["core_fields"])
        ):
            raise ValueError("核心字段命中记录无效")
        errors = review.get("hard_errors", [])
        if not isinstance(errors, list) or not set(errors).issubset(HARD_ERRORS):
            raise ValueError("硬错误类别无效")
        missing = review.get("missing_required_fields", [])
        if not isinstance(missing, list) or not set(missing).issubset(
            row["required_fields"]
        ):
            raise ValueError("遗漏的关键字段记录无效")
        if review.get("relevance") == 1 and (
            hits is None
            or not row["core_fields"]
            or len(hits) / len(row["core_fields"]) < 0.8
            or errors
            or missing
            or row["truncated"]
            or row["generation_error"]
            or not row["result"]
        ):
            raise ValueError("相关度通过与命中记录、硬错误或完整性矛盾：" + row["id"])
        if review.get("fluency") == 1 and (
            row["generation_error"] or not row["result"] or row["truncated"]
        ):
            raise ValueError("生成失败或截断不能记通顺度通过")
        checked.append({**row, "review": review})

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"total": len(rows)}
        for key in ["relevance", "fluency"]:
            values = [row["review"].get(key) for row in rows]
            count = sum(value is not None for value in values)
            result[key] = {
                "reviewed": count,
                "passed": values.count(1),
                "rate": (
                    values.count(1) / len(rows) if rows and count == len(rows) else None
                ),
            }
        return result

    for row in checked:
        hits = row["review"].get("core_hits")
        row["review"]["core_hit_rate"] = (
            len(hits) / len(row["core_fields"])
            if hits is not None and row["core_fields"]
            else None
        )
    overall = metrics(checked)
    complete = all(overall[key]["rate"] is not None for key in ["relevance", "fluency"])
    passed = (
        complete
        and overall["relevance"]["rate"] >= 0.8
        and overall["fluency"]["rate"] >= 0.85
    )
    report = {
        "role": template["role"],
        "reviewer": submitted["reviewer"],
        "reviewed_at": submitted.get("reviewed_at"),
        "overall": overall,
        "by_category": {
            category: metrics([row for row in checked if row["category"] == category])
            for category in CATEGORIES
        },
        "status": (
            "待完成人工评分"
            if not complete
            else (
                "独立验收通过"
                if passed and template["role"] == "acceptance"
                else (
                    "开发流程评分完成，不能作为独立验收"
                    if template["role"] == "development"
                    else "独立验收未通过"
                )
            )
        ),
        "hard_error_counts": dict(
            Counter(
                error
                for row in checked
                for error in set(row["review"].get("hard_errors", []))
            )
        ),
        "limitations": [
            "人工身份声明由评审人确认，脚本不验证现实身份。",
            "自动规则结果不参与两维人工通过率计算。",
            "未评样本保持未评，不缩小分母，也不按自动结果填分。",
        ],
    }
    report.update(
        {
            "reviewer_type": "human",
            "human_confirmed": True,
            "template_path": str(template_path.resolve()),
            "template_sha256": digest(template_path),
            "review_path": str(review_path.resolve()),
            "review_sha256": digest(review_path),
            "per_item": [
                {
                    "id": row["id"],
                    "product_id": row["product_id"],
                    "category": row["category"],
                    **row["review"],
                }
                for row in checked
            ],
        }
    )
    write_json(out, report)
    return report


def main() -> None:
    """通过命令行准备表单或核验汇总人工评分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--role", choices=["development", "acceptance"], required=True)
    prep.add_argument("--frozen-dir", type=Path)
    prep.add_argument("--development-report", type=Path)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--history-root", type=Path, action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20261104)
    ready = subs.add_parser("ready")
    ready.add_argument("--frozen-dir", type=Path, required=True)
    ready.add_argument("--development-report", type=Path, required=True)
    summary = subs.add_parser("summarize")
    summary.add_argument("--template", type=Path, required=True)
    summary.add_argument("--review", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_acceptance(args.source, args.history_root, args.out, args.seed)
        print("独立验收集已冻结：", args.out.resolve())
    elif args.command == "ready":
        verify_development_report(args.development_report)
        verify_frozen(args.frozen_dir)
        print("开发20条人工评分已完成，冻结30条校验通过，可开始独立验收。")
    elif args.command == "prepare":
        prepare(
            args.run_dir, args.out, args.role, args.frozen_dir, args.development_report
        )
        print("人工评测表已生成：", (args.out / "人工评测.html").resolve())
    else:
        print(
            json.dumps(
                summarize(args.template, args.review, args.out),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
