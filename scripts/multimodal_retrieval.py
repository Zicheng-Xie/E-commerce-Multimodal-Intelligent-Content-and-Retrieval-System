"""Qwen3-VL-Embedding-2B 商品图文、以图搜图与可复现检索评测。

用法见 README.md。库图片只需离线编码一次，线上查询重新编码。
"""

import argparse
import json
import os
import sys
import time
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen3-VL-Embedding-2B"
DATA_DIR = PROJECT_DIR / "data" / "retrieval"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "retrieval"
SOURCE_FILE = PROJECT_DIR / "data" / "ecommerce_multimodal" / "product1m_product5m_test_id_label.json"


def iter_source(path: Path):
    """流式读取官方的 {商品ID: 商品信息} JSON，不将整个文件载入内存。"""
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("prepare 需要 ijson，请运行：pip install ijson pillow") from exc
    with path.open("rb") as stream:
        yield from ijson.kvitems(stream, "")


def valid_image(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as source:
            source.verify()
        return True
    except (OSError, ValueError):
        return False


def prepare(args) -> None:
    """抽取真实商品，缓存 URL 图片，自动生成原有 build 所需清单。"""
    if args.limit <= 0 or args.timeout <= 0 or args.max_download_mb <= 0:
        raise ValueError("limit、timeout、max-download-mb 必须为正数")
    if args.max_scan < 0:
        raise ValueError("max-scan 必须大于等于零；零表示扫描到文件末尾")
    if not args.source.is_file():
        raise FileNotFoundError(f"官方数据文件不存在：{args.source}")
    # 商品图片均相对于 data-dir；元数据默认也放在这里。
    image_dir = args.data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    products, scanned, failed = [], 0, 0
    seen = set()
    error_file = args.data_dir / "prepare_errors.jsonl"
    max_bytes = int(args.max_download_mb * 1024 * 1024)
    with error_file.open("w", encoding="utf-8") as errors:
        for raw_pid, record in iter_source(args.source):
            scanned += 1
            pid = str(raw_pid)
            temporary = None
            try:
                if pid in seen:
                    raise ValueError("重复商品 ID")
                seen.add(pid)
                if not isinstance(record, dict):
                    raise ValueError("商品记录不是对象")
                title = str(record.get("title") or "").strip()
                category = str(record.get("label") or "").strip()
                url = str(record.get("url") or "").strip()
                if not title or not category or not url:
                    raise ValueError("缺少 title、label 或 url")
                if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
                    raise ValueError("图片 URL 必须使用 http/https")
                # 哈希文件名避免官方 ID 被解释为文件路径；统一保存为 RGB JPEG。
                filename = hashlib.sha256(pid.encode("utf-8")).hexdigest() + ".jpg"
                target = image_dir / filename
                if not valid_image(target):
                    temporary = target.with_suffix(".download")
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        length = response.headers.get("Content-Length")
                        if length and int(length) > max_bytes:
                            raise ValueError("图片超过下载大小限制")
                        total = 0
                        with temporary.open("wb") as output:
                            while True:
                                chunk = response.read(64 * 1024)
                                if not chunk:
                                    break
                                total += len(chunk)
                                if total > max_bytes:
                                    raise ValueError("图片超过下载大小限制")
                                output.write(chunk)
                    image = load_image(temporary)
                    try:
                        image.save(temporary, format="JPEG", quality=95)
                    finally:
                        image.close()
                    temporary.replace(target)
                products.append({"product_id": pid, "title": title,
                                 "category": category, "image": "images/" + filename,
                                 "source_url": url})
                print(f"[{len(products)}/{args.limit}] 已准备商品 {pid}", flush=True)
            except Exception as exc:
                failed += 1
                errors.write(json.dumps({"product_id": pid, "error": str(exc)},
                                        ensure_ascii=False) + "\n")
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            if len(products) >= args.limit or (args.max_scan and scanned >= args.max_scan):
                break
    if not products:
        raise RuntimeError(f"没有可用商品，请检查下载网络及错误日志：{error_file}")
    validate_products(products, args.data_dir)
    write_json(args.products, products)
    write_json(args.data_dir / "prepare_metrics.json", {
        "source": str(args.source.resolve()), "scanned": scanned,
        "prepared": len(products), "failed": failed, "requested": args.limit,
        "products_file": str(args.products.resolve()),
        "note": "按源文件顺序抽取，不代表随机样本；未生成检索相关性标注。",
    })
    print(f"准备完成：{len(products)} 个商品，失败 {failed} 条。清单：{args.products}")


def read_json(path: Path) -> object:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    """保存 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_gpu() -> None:
    """在计时时同步 CUDA，避免遗漏异步计算。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_embedder(model_dir: Path):
    """加载本地模型；与最初上传脚本的本地 scripts 路径兼容。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查本地环境")
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    scripts = model_dir / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        dtype=torch.float16,
        attn_implementation="sdpa",
        max_length=512,
        max_pixels=256 * 32 * 32,
        local_files_only=True,
    )


def load_image(path: Path):
    """使用 EXIF 校正方向并将图片复制为 RGB。"""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def encode(model, *, text: str | None = None, image: Path | None = None):
    """返回归一化 float32 向量与编码耗时（秒）。"""
    import torch
    import faiss

    if (text is None) == (image is None):
        raise ValueError("必须且只能提供 text 或 image")
    if text is not None and not text.strip():
        raise ValueError("查询文本不能为空")
    payload = {"text": text} if text is not None else {"image": load_image(image)}
    sync_gpu()
    start = time.perf_counter()
    with torch.inference_mode():
        vector = model.process([payload]).float().cpu().numpy()
    sync_gpu()
    seconds = time.perf_counter() - start
    array = np.ascontiguousarray(vector, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 1 or not np.isfinite(array).all():
        raise ValueError("模型向量形状或数值异常")
    if np.linalg.norm(array) == 0:
        raise ValueError("模型返回零向量")
    faiss.normalize_L2(array)
    return array, seconds


def validate_products(products: object, data_dir: Path) -> list[dict]:
    """验证商品元数据和图片路径，阻止重复 ID。"""
    if not isinstance(products, list) or not products:
        raise ValueError("products.json 必须是非空数组")
    seen = set()
    for product in products:
        if not isinstance(product, dict) or not all(
            product.get(k) for k in ("product_id", "title", "category", "image")
        ):
            raise ValueError("商品缺少 product_id/title/category/image")
        pid = product["product_id"]
        if pid in seen:
            raise ValueError(f"重复商品 ID：{pid}")
        seen.add(pid)
        path = (data_dir / product["image"]).resolve()
        if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
            raise FileNotFoundError(f"图片不存在或路径越界：{product['image']}")
    return products


def build(args) -> None:
    """离线生成索引和顺序对应的商品元数据。"""
    import faiss

    products = validate_products(read_json(args.products), args.data_dir)
    model = make_embedder(args.model_dir)
    vectors, seconds = [], []
    for i, product in enumerate(products, 1):
        vector, elapsed = encode(model, image=args.data_dir / product["image"])
        vectors.append(vector)
        seconds.append(elapsed)
        print(f"[{i}/{len(products)}] {product['product_id']}: {elapsed:.3f}s")
    matrix = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_dir / "products.faiss"))
    write_json(args.index_dir / "products.json", products)
    write_json(args.index_dir / "build_metrics.json", {
        "products": len(products), "dimension": int(matrix.shape[1]),
        "mean_image_encode_seconds": float(np.mean(seconds)),
        "p95_image_encode_seconds": float(np.percentile(seconds, 95)),
        "note": "离线图片编码耗时，不计入已建库热查询耗时",
    })
    print("索引已保存：", args.index_dir)


def load_index(index_dir: Path):
    """加载索引与其对应商品清单。"""
    import faiss

    index = faiss.read_index(str(index_dir / "products.faiss"))
    products = read_json(index_dir / "products.json")
    if index.ntotal != len(products):
        raise ValueError("FAISS 索引行数与商品元数据不一致")
    return index, products


def rank(index, products: list[dict], vector: np.ndarray, k: int):
    """根据余弦相似度排序，返回 Top-K。"""
    if k <= 0:
        raise ValueError("k 必须为正整数")
    t0 = time.perf_counter()
    scores, ids = index.search(vector, min(k, index.ntotal))
    elapsed = time.perf_counter() - t0
    results = []
    for position, score in zip(ids[0], scores[0]):
        if position >= 0:
            results.append({**products[int(position)], "score": float(score)})
    return results, elapsed


def query_once(model, index, products, entry: dict, data_dir: Path, k: int):
    """执行文本或图片检索，返回结果和拆分后的时延。"""
    kind = entry.get("query_type")
    if kind == "text":
        vector, encode_seconds = encode(model, text=entry["query"])
    elif kind == "image":
        path = (data_dir / entry["query_image"]).resolve()
        if not path.is_relative_to(data_dir.resolve()):
            raise ValueError("查询图片路径越界")
        vector, encode_seconds = encode(model, image=path)
    else:
        raise ValueError("query_type 只能是 text 或 image")
    results, search_seconds = rank(index, products, vector, k)
    return results, {
        "query_encode_seconds": encode_seconds,
        "faiss_search_seconds": search_seconds,
        "total_seconds": encode_seconds + search_seconds,
    }


def evaluate_ids(retrieved: list[str], relevant: set[str], k: int = 10):
    """单查询的 Recall@K、Precision@K，Precision 固定除以 K。"""
    if not relevant:
        raise ValueError("完整相关集合不可为空")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("结果中出现重复商品 ID")
    hits = len(set(retrieved[:k]) & relevant)
    return {"hits": hits, f"recall@{k}": hits / len(relevant),
            f"precision@{k}": hits / k,
            "hit@1": int(bool(set(retrieved[:1]) & relevant)),
            "hit@3": int(bool(set(retrieved[:3]) & relevant))}


def evaluate(args) -> None:
    """基于完整相关商品标注计算宏平均，同时审查困难负样本。"""
    index, products = load_index(args.index_dir)
    available = {p["product_id"] for p in products}
    cases = read_json(args.eval_file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须为非空数组")
    model = make_embedder(args.model_dir)
    rows = []
    for case in cases:
        relevant = set(case["relevant_product_ids"])
        negatives = set(case.get("hard_negative_ids", []))
        if not relevant or not relevant <= available or not negatives <= available:
            raise ValueError(f"{case['query_id']} 标注缺失或 ID 不存在于索引")
        if relevant & negatives:
            raise ValueError(f"{case['query_id']} 正负标签冲突")
        results, timing = query_once(model, index, products, case, args.data_dir, 10)
        ids = [p["product_id"] for p in results]
        metrics = evaluate_ids(ids, relevant, 10)
        negative_hits = sorted(set(ids) & negatives)
        rows.append({"query_id": case["query_id"], "query_type": case["query_type"],
                     **metrics, **timing, "hard_negative_in_top10": negative_hits,
                     "top10": results})
    summary = {
        "queries": len(rows), "products": index.ntotal,
        "hit@1": float(np.mean([r["hit@1"] for r in rows])),
        "hit@3": float(np.mean([r["hit@3"] for r in rows])),
        "recall@10": float(np.mean([r["recall@10"] for r in rows])),
        "precision@10": float(np.mean([r["precision@10"] for r in rows])),
        "avg_total_seconds": float(np.mean([r["total_seconds"] for r in rows])),
        "p95_total_seconds": float(np.percentile([r["total_seconds"] for r in rows], 95)),
        "hard_negative_top10_queries": sum(bool(r["hard_negative_in_top10"]) for r in rows),
        "prd_recall_pass": None, "prd_precision_pass": None,
        "note": "仅当人工确认相关商品标注覆盖整个评测库后，才可据此判定 PRD 达标；当前为原始测量",
    }
    summary["by_query_type"] = {}
    for kind in ("text", "image"):
        subset = [r for r in rows if r["query_type"] == kind]
        if subset:
            summary["by_query_type"][kind] = {
                "queries": len(subset),
                **{key: float(np.mean([r[key] for r in subset]))
                   for key in ("hit@1", "hit@3", "recall@10", "precision@10")},
            }
    output = {"summary": summary, "queries": rows}
    write_json(args.output, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("报告：", args.output)


def search(args) -> None:
    """单查询检索并记录全部候选相似度分布的摘要。"""
    index, products = load_index(args.index_dir)
    model = make_embedder(args.model_dir)
    case = ({"query_type": "text", "query": args.text} if args.text is not None
            else {"query_type": "image", "query_image": args.image})
    results, timing = query_once(model, index, products, case, args.data_dir, args.top_k)
    summary = {"query": case, "timing": timing, "results": results,
               "note": "分数为归一化向量余弦相似度；不应凭单次查询设统一阈值"}
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def benchmark(args) -> None:
    """测试不同图库规模的 FAISS 部分；有足够商品时才形成真实规模曲线。"""
    import faiss

    index, products = load_index(args.index_dir)
    model = make_embedder(args.model_dir)
    vector, encode_seconds = encode(model, text=args.text)
    # 精确内积索引可提取真实向量子集；不复制伪造商品填充千级。
    matrix = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype("float32")
    sizes = [n for n in (100, 1000) if n <= index.ntotal]
    sizes.append(index.ntotal)
    rows = []
    for size in sorted(set(sizes)):
        sub_index = faiss.IndexFlatIP(index.d)
        sub_index.add(np.ascontiguousarray(matrix[:size]))
        values = []
        for _ in range(args.repeats):
            _, seconds = rank(sub_index, products[:size], vector, 10)
            values.append(seconds)
        rows.append({"library_size": size, "faiss_mean_ms": 1000 * float(np.mean(values)),
                     "faiss_p95_ms": 1000 * float(np.percentile(values, 95))})
    output = {"text_encode_seconds": encode_seconds, "benchmark": rows,
              "note": "FAISS 检索耗时不含查询编码；全链路时延需另外测量"}
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def performance(args) -> None:
    """模型只加载一次，分模态预热，测量串行查询的各段耗时。"""
    if args.repeats <= 0 or args.warmup < 0 or args.top_k <= 0:
        raise ValueError("repeats、top-k 必须为正数，warmup 必须非负")
    cases = read_json(args.eval_file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("查询文件必须是非空数组")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not case.get("query_id"):
            raise ValueError("每条查询必须有 query_id")
        if case["query_id"] in seen:
            raise ValueError("query_id 不可重复")
        seen.add(case["query_id"])
        kind = case.get("query_type")
        if kind == "text":
            if not isinstance(case.get("query"), str) or not case["query"].strip():
                raise ValueError("文本查询必须有非空 query")
        elif kind == "image":
            path = (args.data_dir / case["query_image"]).resolve()
            if not path.is_relative_to(args.data_dir.resolve()) or not path.is_file():
                raise ValueError("查询图片不存在或路径越界")
        else:
            raise ValueError("query_type 只能是 text 或 image")
    start = time.perf_counter()
    index, products = load_index(args.index_dir)
    index_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = make_embedder(args.model_dir)
    sync_gpu()
    model_load_seconds = time.perf_counter() - start
    for kind in ("text", "image"):
        subset = [c for c in cases if c["query_type"] == kind]
        if subset:
            print(f"预热 {kind}: {args.warmup} 次", flush=True)
            for i in range(args.warmup):
                query_once(model, index, products, subset[i % len(subset)],
                           args.data_dir, args.top_k)
    rows = []
    for repeat in range(args.repeats):
        for case in cases:
            start = time.perf_counter()
            _, timing = query_once(model, index, products, case, args.data_dir, args.top_k)
            elapsed = time.perf_counter() - start
            rows.append({"query_id": case["query_id"], "query_type": case["query_type"],
                         "repeat": repeat + 1, **timing, "end_to_end_seconds": elapsed})
            print(f"[{repeat + 1}/{args.repeats}] {case['query_id']}: {elapsed:.3f}s", flush=True)
    groups = {}
    for kind in ("text", "image"):
        subset = [r for r in rows if r["query_type"] == kind]
        if subset:
            groups[kind] = {"samples": len(subset), "timings": {}}
            for key in ("query_encode_seconds", "faiss_search_seconds", "total_seconds", "end_to_end_seconds"):
                values = [r[key] for r in subset]
                groups[kind]["timings"][key] = {
                    "mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                }
    output = {"model_load_seconds": model_load_seconds,
              "index_load_seconds": index_load_seconds, "products": index.ntotal,
              "warmup_per_type": args.warmup, "repeats": args.repeats,
              "by_query_type": groups, "measurements": rows,
              "note": "秒为单位；预热不计入统计。end_to_end 包含图片读取、处理和结果组装，"
                      "不含模型/索引加载、网络和界面开销。小样本 P95 仅供初步参考。"}
    write_json(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "measurements"}, ensure_ascii=False, indent=2))
    print("报告：", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "search", "evaluate", "benchmark", "performance"])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--products", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=SOURCE_FILE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-scan", type=int, default=10000,
                        help="prepare 最多扫描条数，0 表示不限")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-download-mb", type=float, default=20)
    parser.add_argument("--index-dir", type=Path, default=OUTPUT_DIR / "index")
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "report.json")
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.products is None:
        args.products = args.data_dir / "products.json"
    if args.eval_file is None:
        args.eval_file = args.data_dir / "eval_queries.json"
    if args.action == "prepare":
        prepare(args)
    elif args.action == "build":
        build(args)
    elif args.action == "search":
        if (args.text is None) == (args.image is None):
            parser.error("search 必须且只能提供 --text 或 --image")
        search(args)
    elif args.action == "evaluate":
        evaluate(args)
    elif args.action == "performance":
        performance(args)
    else:
        if not args.text or args.repeats <= 0:
            parser.error("benchmark 需要 --text 和正数 --repeats")
        benchmark(args)


if __name__ == "__main__":
    main()
