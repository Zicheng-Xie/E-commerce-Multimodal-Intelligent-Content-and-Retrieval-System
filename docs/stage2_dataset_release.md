# 第二阶段：扩展数据与正式评测准备

2026-09-23。本轮完成300件商品数据冻结、180条服饰样本预处理检查、300件商品图文索引和独立相关性标注入口。没有进行微调、RAG、对比学习、分割模型或局部定位开发。第二阶段尚未全部验收，文案事实性修复和独立人工质量评分仍待完成。

## 商品数据

源文件仅以只读方式访问。分别扫描原始train/test前50,000条记录，排除此前使用过的48个商品ID，对标题、属性、图片URL非空且能映射到三个品类的记录按固定种子20260923的SHA256排序，每个原始划分/品类选100条，共600条候选。**抽样范围是各文件的前50,000条，不代表整个数据集的随机样本。** 品类映射沿用源label优先的关键词规则，仍需人工核查。

| 原始划分 | 3C数码下载成功 | 家居日用下载成功 | 服饰穿搭下载成功 | 冻结数量 |
|---|---:|---:|---:|---:|
| train → development | 97 | 96 | 97 | 每品类50，共150 |
| test → heldout | 93 | 98 | 93 | 每品类50，共150 |

574/600张图片下载成功，26条失败保留在 `data/stage2_v1/localized/media_summary.json`。下载约52.52MiB；实际解码并检查尺寸、格式、SHA256与精确像素指纹。574件可用商品中按固定优先级保留300件，其余274件因分层配额未进入冻结集，没有删除。

冻结前同时检查候选记录及此前使用的记录：源商品ID、规范化标题、精确像素哈希，以及64位dHash汉明距离≤4的近似图片候选。本轮未发现匹配边。此结果不排除裁剪图、主体很小的图片或其他形式的语义重复；dHash也可能误报，未来发现匹配时保守排除而不直接认定商品身份。

300件中274件存在同一属性键多个不同值，标记 `multi_value_attributes_require_sku_review`。这是SKU/字段歧义风险，不表示274件数据错误；没有任取一个值覆盖原始列表。品牌与标题、参数与SKU文本间的完整语义冲突检查尚未完成。

`data/stage2_v1/release/` 内容：

- `products.jsonl`：300件商品，包含来源位置、原始记录摘要哈希、原始split、项目用途、分组ID、质量提示和图片路径。
- `images/`：从已下载副本复制的图片，复制后再次核对SHA256。
- `splits.json`：开发150/预留测试150，沿用原始划分；未训练任何模型。
- `decisions.json`：600条候选的入选/未入选理由。
- `duplicate_audit.json`：重复检查方法及证据。
- `quality_report.json`、`manifest.json`：数量、限制、来源哈希及冻结文件哈希。

旧样本、旧索引和旧生成结果继续保留。新的heldout仅用于自动质量检查、特征编码及预先固定的评测准备，尚未依照其模型成绩调整提示词或检索参数；这不构成品类标注已经正确的保证。

## 服饰预处理检查

`data/stage2_fashion_v1/` 含100张DeepFashion图片（train/validation各50，149个实例）与80条FashionAI属性样本（8种属性各10）。复制280个文件，保留COCO转换、原始标注副本和官方属性值编码。

最新检查在 `artifacts/stage2_fashion_v2/`，v1为中间检查记录，以v2的解释为准。

- 280个已复制文件的哈希一致，检查图片尺寸、边界框、分割多边形、掩码面积、属性编码及跨集合精确像素重复。
- `deepfashion/train/000006` 的一个多边形x最小值为-0.2；记录为坐标越界，在 `eligibility.json` 中将该记录标为不可用于严格标注验收。原标注与COCO副本均不静默裁剪。其他179条通过当前自动检查。
- 发现5个跨划分同名pair_id，但没有精确像素重复。不同划分中的相同编号并不能自动证明身份相同；当前用split限定分组键，并把全局编号范围记为未验证。
- 生成6张train标注覆盖图供复核。样本按文件顺序取得，类别未全部覆盖，不是独立视觉模型评测集。

[DeepFashion2官方说明](https://github.com/switchablenorms/DeepFashion2#data-organization)定义了pair_id、style、边界框和多边形的含义。官方定义有13类服饰；本轮保留源类别，没有强行转换成后续功能范围中的8类，也没有部署分割模型。

## 扩展检索索引

`data/index_stage2_v1/` 已对300件商品独立编码图片和标题，保存600条2048维向量。没有进行训练或参数调优。

- 本机建索引约57.03秒，不含模型加载；加载约3.16秒。
- PyTorch峰值分配4118.84MiB、保留4136MiB；这不是整个GPU的总占用。
- 单次建索引性能不是查询P95或压力测试成绩。

查询时显式指定新索引；原CLI默认索引仍为旧的36件基线，以免静默改变旧命令含义。

```powershell
.\.venv312\Scripts\python.exe scripts\retrieve.py search --index data\index_stage2_v1 --text "无线蓝牙耳机" --top-k 10 --output artifacts\expanded_search.json
```

若用评测查询查看并调整系统，应把该查询标为开发用途，并另建最终评测查询，不能继续当未使用的独立测试。

## 相关性标注与评分

`artifacts/evaluation_stage2_v1/` 保存12条文本需求和6条图片查询。文本需求在查看扩展库排序之前制定，三品类各4条；图片查询从预留集固定选择，每品类2条，并显式排除自身。文本查询判断300件候选，图片查询判断299件，共5,394对。

双击 `review.html` 可使用离线标注页。选择查询、填写评审人，逐件标记相关/不相关，无法判断则保留空值并写原因。可分页、导入已保存文件、导出JSONL；**关闭前必须导出，页面不自动持久化**。导出文件保存到本项目的评测目录，不覆盖 `qrels.template.jsonl`。浏览器默认下载位置可能不同，应在保存对话框中选E盘项目目录。

页面隐藏模型排序，显示原始标题、图片、属性和固定相关性规则，避免只判断Top10造成Recall分母缺失。所有候选均需判定；若缩减评测查询，必须先建立新版本并重新冻结查询清单，不能根据成绩挑选查询。本轮没有代填人工标签，也没有把自动关键词匹配当作标准答案。

评分规则：

- `grade=1`相关，`grade=0`不相关，`null`尚未判定；每条须有评审人。
- 缺失、重复、未知候选或未判定标签会阻止正式评分。
- Precision@10分母固定为10；Recall@10分母为冻结候选库中该查询的全部相关商品数。
- 无相关商品的查询单独列出，Recall不定义；主表对存在正例的查询做宏平均，分别报告文本与图片，不能隐去无正例查询的数量。
- 默认比较图片向量，无品类过滤；固定等权融合可作为额外结果，不能根据预留标签调权重后冒充独立测试。

```powershell
# 空模板会报错，这是预期防护；完成真实标注后换成reviewed文件
.\.venv312\Scripts\python.exe scripts\evaluate.py validate --qrels artifacts\evaluation_stage2_v1\qrels.reviewed.jsonl
.\.venv312\Scripts\python.exe scripts\evaluate.py run --qrels artifacts\evaluation_stage2_v1\qrels.reviewed.jsonl --output artifacts\evaluation_scored_v1.json
```

当前正式Precision/Recall成绩不可用。通用品类需求可能在小候选库中只有少量甚至零相关商品，因此Precision@10目标可能受候选库覆盖限制；应报告真实覆盖，不补造正例。

## 复现命令

新建版本，避免覆盖已冻结数据。下载可断点续跑，其他阶段拒绝覆盖已有目录。

```powershell
.\.venv312\Scripts\python.exe scripts\dataset.py sample --version stage2_v2
.\.venv312\Scripts\python.exe scripts\dataset.py download --version stage2_v2
.\.venv312\Scripts\python.exe scripts\dataset.py freeze --version stage2_v2
.\.venv312\Scripts\python.exe scripts\retrieve.py build --dataset data\stage2_v2\release\products.jsonl --index data\index_stage2_v2
.\.venv312\Scripts\python.exe scripts\evaluate.py prepare --dataset data\stage2_v2\release\products.jsonl --index data\index_stage2_v2 --kit artifacts\evaluation_stage2_v2

.\.venv312\Scripts\python.exe scripts\fashion.py prepare --dataset data\stage2_fashion_v2
.\.venv312\Scripts\python.exe scripts\fashion.py audit --dataset data\stage2_fashion_v2 --output artifacts\stage2_fashion_v3
```

已有版本可用冻结哈希验证；重新下载成功率受远程URL存活状态影响，因此重跑不保证得到相同的可用图片集合。不得覆盖当前冻结清单来掩盖变化。

## 验证与剩余工作

38项测试通过，覆盖固定抽样、旧样本排除、传递重复分组、文件复制、禁止覆盖、越界标注、pair_id作用域、不完整qrels拒绝及多正例指标分母；mypy、Black、isort、flake8通过。300件商品索引已真实GPU运行。离线标注页JavaScript语法通过，306处图片路径存在；内置浏览器安全策略拒绝打开file URL，因此未完成浏览器交互实测，不能据此宣称UI验收通过。

原始目录531,063个文件路径、大小、修改时间与基线一致，全部PDF哈希一致，无差异；非PDF没有逐个全量内容哈希。证据在 `artifacts/source_integrity.json`。检查时C盘约73.91GiB可用，E盘约83.93GiB可用；本轮无需新增依赖或模型。

第二阶段剩余：复核品类映射及SKU/字段冲突，完善否定属性和SKU的文案事实检查，在固定提示词下生成新评测文案，完成独立人工文案评分与检索相关性标注，再运行质量/延迟评测和汇总报告。当前不宣称第二阶段所有验收项已完成。
