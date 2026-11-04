# 模块二：多模态商品检索

本分支为独立的 `module2-dev` 工程，从 `main` 建立；模块一由 `shz-dev` 单独维护。这里精选模块二的真实实现、对应单元测试及 2026.10.19—11.04 双周报指标的回放程序与已获授权的精简证据。

## 功能

- **中文文本 / 图片检索**：完整 Qwen3-VL-Embedding-2B，整图、自动区域、标题与属性融合，ROI 裁切及结果解释。
- **完整图文精排**：Qwen3-VL-Reranker-2B、候选池精排、前排保护、整图/放大双上下文；精确输入身份缓存，不补造缺失模型分数。
- **主体与细粒度证据**：GroundingDINO、SAM 2.1、FG-CLIP 2 适配，主售/道具识别与 query-only 几何/语义核验。新掩码因质量检查未过保持停用。
- **业务学习与归因**：完整共享投影的 Multi-Similarity 学习、DINOv2 双图联合注意力与匹配头、家族隔离训练、四组角色×扩池对照。
- **可复核评测**：P@10、R@10、mAP、NDCG@10、MRR、Hit@1；完整图库分母、未知 AI 标签上下界、共享未知标签配对比较。

只保留一套公共实现。未上传模块一生成链路、旧实验快照、重复导出页面、模型权重、原图/衍生图、向量缓存、旧虚拟环境或原始大型数据集。

## 先运行不需要模型的周报复核

Python 3.12，在当前分支根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,retrieval]'
python -m pytest -q
python scripts/check_quality.py
```

默认 101 项单元测试只依赖代码和小型夹具；另有 13 项数据集成测试，逐项复核两轮盲测、双图开发、角色扩池及主售修复的六项指标。

本分支包含项目所有者明确授权公开的约 252 KB 精简回放数据。无需模型、GPU 或 Ollama，直接执行：

```bash
python scripts/replay_weekly.py
python -m pytest -q integration_tests
```

11 组结果、1,756 份完整排序已逐项回放一致。两轮盲测均已 consumed，其他组均为曝光开发；不包含新的未消耗盲测材料。

完整工程检查需要 Node.js；CI 使用 Node 22。`check_quality.py` 检查 Black、isort、flake8、mypy、Pylint、PDF 规则与全部测试，有跳过项也会失败。

## 在 Mac / CUDA 上运行真实检索

设置好 Ollama，并另行取得固定版本模型。模型不随 Git 上传：

```bash
python scripts/download_models_mac.py
# Mac Apple Silicon
export MULTIMODAL_DEVICE=mps
# CUDA 机器改用 export MULTIMODAL_DEVICE=cuda
python scripts/retrieve_regions.py --help
python scripts/retrieve_postrank.py --help
```

权重目录为 `.models/<模型仓库名称>/`，revision 与逐文件摘要位于 `configs/`。不静默回退 CPU。Qwen3.5 本地视觉服务用于主体描述/核验，检索嵌入与精排由完整独立模型执行。

使用自己的合法商品图片准备 `data/catalog/products.jsonl`：

```json
{"id":"item_001","text":"原始商品标题","category":"家居日用","attributes":{},"evaluation_eligible":true,"image_path":"images/item_001.jpg","image_sha256":"填写该图片真实的SHA256"}
```

图片路径相对 JSONL 所在目录，需位于该目录内；真实摘要必须匹配。然后构建新索引并查询：

```bash
python scripts/retrieve_regions.py build --dataset data/catalog/products.jsonl --index data/index_regions
python scripts/retrieve_regions.py search --index data/index_regions --text '双层壁挂调料架' --output artifacts/query_text
python scripts/retrieve_regions.py search --index data/index_regions --image data/catalog/images/item_001.jpg --roi 0.1 0.1 0.9 0.9 --output artifacts/query_roi
```

常驻 HTTP、整图业务学习及角色归因的历史实验驱动也保留在 `scripts/`，需要原有完整特征库、数据与冻结协议，不能仅凭代码及周报聚合指标启动。这些源码支持审查和在完整本地工程复现；公开分支的开箱入口是单元测试；历史指标回放可使用本分支已授权的精简证据。未提供新的角色方法在线部署成绩。

## 周报指标与证据

| 结果 | 总体 P@10 | 总体 R@10 | 结论 |
|---|---:|---:|---|
| 首次独立 AI 盲测候选 | 63.83–71.00% | 89.06–94.71% | 图片 P@10 仅 43.33–52.00%，未过 60% |
| 第二次独立 AI 盲测候选 | 49.58–54.17% | 89.80–94.86% | P 门槛未过，R 偏好通过 |
| 角色扩池 fresh 开发组 | 43.75–67.50% | 30.04–47.21% | P/R/mAP/NDCG 严格退步，不采用 |

所有质量范围是未知 AI 标签补全下的计算界，不是置信区间、人工真值或线上流量成绩。两轮盲测数据均已 consumed；fresh 是曝光开发组。人工标签为 0、正式验收指标为 null，默认方法没有升级。

- [周报六指标表与代码/测试映射](docs/weekly_metrics.md)
- [证据转换、来源与运行边界](docs/reproduction.md)
- `evidence/provenance.json`：原始文件 SHA256、排序计数和去重变换。
- `integration_tests/test_weekly_replay.py`：完整回放数据的独立集成测试。
- `release/source_manifest.json`：源码选择与打包改动。
- `release/verification.json`：本次精选工程的实际检查结果。

模型回放、单元测试通过不代表检索业务目标通过。OOF 拼接只作开发诊断；两个盲测与开发结果不跨版本合并为迭代增益。
