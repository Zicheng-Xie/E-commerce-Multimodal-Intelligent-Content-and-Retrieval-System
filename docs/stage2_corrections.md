# 第二阶段：分类纠正、文案防错与视觉抽查

本次继续完成第二阶段的数据与基线工程。发现并纠正了旧大类规则中的真实问题，新增文案合同 v2；没有开展第三阶段，也没有运行服饰分割、定位或问答模型。原始目录只读，旧数据、旧索引、90条文案和5394条排序保留。

## 数据分类修订

旧版“每类100件”来自关键词启发式，并非经过语义确认的真实分布。例如，平板卫生纸被“平板”归入数码，办公电脑桌被“电脑”归入数码，摩托车气门嘴帽被“帽”归入服饰。

新增 `configs/taxonomy_v2.json` 和 `taxonomy.py`：198个明确细类采用完整名称映射，8个歧义细类使用明确的标题上下文或暂缓处理。多个上下文指向不同大类、未知细类、缺少上下文时均保留待确认，不继续猜测。家居范围包括办公家具、日用消耗品和小型家电；数码范围包括设备专用配件，排除乐器、车辆机械配件及工业电源。规则由AI辅助审阅形成，仍待人工确认。

| 修订结果 | 件数 |
|---|---:|
| 原冻结商品 | 300 |
| 保留且改类 | 29 |
| 超出项目三类范围，隔离 | 12 |
| 歧义未解决，隔离 | 2 |
| 新版本保留 | 286 |
| 其中数码／服饰／家居 | 80／92／114 |

新版本位于 `data/stage2_taxonomy_v2/`：`products.jsonl` 保存286条和独立图片副本，`decisions.jsonl` 保存全部300条决策，`quarantine.jsonl` 保存14条隔离记录及原因，`report.json` 保存变化清单。旧ID、原始划分、角色和属性不变，原大类及规则保存在 lineage 中。副本和清单有SHA256。隔离记录的图片路径属于旧来源集，不能直接当作新索引输入。

现有300件索引、18查询评审包仍属于v1，不能把新类别统计套在旧结果上。旧文案实验的“每类30件”也仅代表原分组。修订后需要补齐类别样本、再建立新的实验版本；旧预留数据已用于错误分析，不可重新宣传为未见过的盲测集。

## 文案入口升级与真实回放

新增 `content_guard.py`，由 `content_contract.py` v2 调用。默认 `run.py generate` 和 `scripts/generate_content.py` 都接入新合同；冻结v3实验的四个核心文件不变。旧 `generate-batch` 和 `copy_baseline.py` 仍是历史基线入口，不包含本次合同层新增规则。

- **生成前**：比较明确的全局容量、只数、内存、显卡与SKU文字；存在冲突即停止模型调用，等待确认来源或选定SKU。毫升/升统一单位；不把“套”当成“只”，不把SSD/显存当成内存，不猜测裸数字GPU型号。
- **生成后**：标题、详情、卖点统一检查若干高风险声明，包括纯棉程度、铝合金、全实木、木材来源、功率调节、四季适用和柔软度比较。标题不能授权核心参数，额定功率范围不能推导出调节功能。
- **返回值**：顶层 `status` 是合同最终状态；`factual_copy` 保存原草稿和原v3元数据以便追溯。新规则阻止放行时为 `blocked_by_contract_rules`，营销语句为空；旧 `_meta.release_status` 不是新的放行结论。规则无标记仍然只是待审草稿。

对原90条输出离线回放结果见 `artifacts/copy_guard_audit_v2/`：3条存在目标来源冲突，12条触发目标声明检查，合计15条互不重复；其中4条此前为规则未标记草稿。**这些数值是待审工作量，不是错误率、准确率或规则召回率。** 本次没有重新生成90条，不能据此宣称新文案质量提高了多少。

回放中还修正了一处规则误分类：“比纯棉更柔软”是比较性能声明，不表示该商品是纯棉；v2将其单独标记。中间 `copy_guard_audit_v1/` 保留为开发记录，最终回放以v2为准。

另保存8个有输出原句和结构化属性依据的AI辅助案例：`artifacts/stage2_semantic_review_v1/copy_findings.jsonl`。包括95%以上被扩写成100%全棉、只在标题出现的铝合金、全局16GB与8G内存SKU冲突、额定功率被扩写成可调等。这不是整件文案的人工相关度/通顺度评分，不进入人工成绩。

## 服饰视觉抽查

`artifacts/fashion_spotcheck_v1/` 保存9张原图与标注并排图，以及独立的 `ai_visual_review.jsonl`：训练集3张、验证集3张，FashionAI领型／领口／裙长各1张。6张可见内容与标注相容；3张因遮挡、局部裁剪或可见掩码/完整轮廓定义需要进一步确认。未据此更改原类别、几何坐标或179条可用记录。

抽样按每个源划分中首次出现的不同标签组合选取，是有目的的小样本检查，不能估计179条数据的总体准确率。FashionAI只有属性标签，所以并排图不伪造分割区域；DeepFashion叠加全部原始实例。没有进行全量逐像素复核。

## 复现与验证

以下复建命令使用新输出名，避免覆盖本次成果：

```powershell
.\.venv312\Scripts\python.exe scripts\release_taxonomy.py --output data\stage2_taxonomy_replay_v2
.\.venv312\Scripts\python.exe scripts\audit_copy_guards.py --output artifacts\copy_guard_replay_v2
.\.venv312\Scripts\python.exe scripts\prepare_fashion_spotcheck.py --output artifacts\fashion_spotcheck_replay_v1
.\.venv312\Scripts\python.exe scripts\run.py generate --product configs\example_product.json --output artifacts\my_content_v2.json
.\.venv312\Scripts\python.exe scripts\verify_stage2_corrections.py
.\.venv312\Scripts\python.exe -m pytest -q
```

校验脚本核对新旧清单、ID/角色/属性/图片保留、90条规则回放、8条引用证据、9张抽查图和旧实验哈希。AI辅助记录显式使用 `model_assisted`，没有写入真实人工评审包，也没有生成正式质量分数。

本轮92项Python测试通过；Black、isort、flake8、mypy（26个源文件）和变更文件Pylint错误级检查通过。实际产物校验结果见 `artifacts/stage2_corrections_verification.json`。原始531,063个文件的路径、大小、修改时间与基线一致，PDF额外SHA256一致；未对全部非PDF原始文件重做内容哈希。C盘剩余约72.34GiB，E盘约83.77GiB。

## 下一步顺序

1. 按v2明确品类从现有候选副本补齐样本，保留去重和分组隔离；冻结新的评测池，并另留未用于规则开发的测试样本。当前不能继续宣称三类各100件均已确认。
2. 新数据版本确定后，重建索引、查询候选清单和评审包，并用合同v2跑独立文案测试。旧评审可用于历史诊断，不混入新版本成绩。
3. 完成真实相关性标注与文案评分，汇总第二阶段四项交付物。AI抽查、规则通过和源标签同款开发指标都不能替代独立人工质量验收。
