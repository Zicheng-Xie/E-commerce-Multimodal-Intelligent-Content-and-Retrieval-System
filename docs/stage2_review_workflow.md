# 第二阶段：评审、固定排序和可用服饰数据

> 2026年10月22日独立性更新：此旧评审包已用于开发，补完人工标注也只能产生内部开发相关性评分，不能成为新一轮独立验收。以下冻结与合并能力可复用，新验收数据须重新封存。见 [模块二盲测规范](module2_blind_evaluation.md)。

版本提示：本文对应原v1的300件候选及90条文案。后续已发现品类误分并建立286件修订副本，见 [纠正记录](stage2_corrections.md)。本评审包仍可评价旧实验；新版本需重建候选与评审包，不能把两版分数或分组混合。新的AI辅助记录单独保存，没有填入此处的人工模板。

本轮把此前“准备了模板”推进为可分批填写、严格合并、离线评分的完整工程流程，同时发布排除异常标注后的服饰副本。未调整冻结文案提示词、模型参数或检索排序规则，未开展第三阶段工作。

## 现在可以直接使用的产物

| 产物 | 路径 | 本轮完成情况 |
|---|---|---|
| 评审总入口 | `artifacts/review_stage2_v1/index.html` | 21个独立批次；所有评分为空 |
| 文案评分页面 | 同目录 `copy_01.html` 至 `copy_03.html` | 每批30件，3C/服饰/家居各10件；合计全部90件，包括规则拦截和未生成条目 |
| 检索标注页面 | 同目录 `retrieval_text_01.html` 等18页 | 每批一个完整查询、300或299个候选；合计5394对关系；按固定哈希顺序展示，不显示模型排序 |
| 固定检索结果 | `artifacts/stage2_predictions_v1/` | 18条查询向量、全部5394条排序和分数、模型配置、耗时及SHA256清单 |
| 清理后的服饰副本 | `data/stage2_fashion_clean_v1/` | 179条样本，278个图片/标注副本；COCO同步重建 |
| 数据结构复检 | `artifacts/stage2_fashion_clean_audit_v1/report.json` | 179条无结构异常；跨划分数字pair_id碰撞仍明确为未确认共享身份 |
| 本轮核对证据 | `artifacts/stage2_review_verification.json` | 固定排序复算、路径、批次、分母、COCO关联与旧v3冻结代码核对通过 |

## 评审怎样填写

从总入口选择一个批次。先填写真实评审人的姓名或稳定代号，再确认“由本人逐项判断”。页面允许只查看未完成项；每个检索分页显示6件，文案每页显示1件，方便逐项对照。

检索根据固定查询规则、图片、标题和属性判断相关或不相关。没有判断的关系保留为空；无法确认时记录原因并保持待定。不能只标前10名，也不能把不同大类全部自动标为负例后冒充人工审核。页面不会提示模型预期答案。

文案对整件商品的标题、详情和三条卖点评价相关度、通顺度及事实错误。存在事实错误时，按已冻结规则相关度不能通过。不通过或有事实错误必须补充依据；来源暂停、生成失败的条目不能获得文案质量分数，只能由评审确认原因。自动规则默认折叠，先独立阅读输出。完整规则仍见 [文案评分规则](copy_scoring_rubric.md)。

点击“保存 / 导出进度”，保存至 `artifacts/review_stage2_v1/reviewed/`，保留自动建议的 `批次名.reviewed.jsonl` 文件名。可以未完成就导出，再导入**同一批次**继续。页面没有后台自动保存；离开未保存页面时会提醒。可把不同批次交给不同真实评审人，准备合并的目录内每批保留一个版本。多次导出产生的重复版本需要明确选定，合并器不会自行覆盖或猜测哪份较新。

入口中所有内容来自工程副本，所有判断初始为空。`review_origin="human"` 是评审人的明确声明，不是外部身份认证。模型辅助建议应保留在独立文件中，不能把模型输出标成真实人工意见。

## 合并与完整人工评分

以下命令在项目根目录执行；每轮合并使用新的输出目录，保留旧进度证据。

```powershell
.\.venv312\Scripts\python.exe scripts\review_packet.py merge --output artifacts\review_merged_v1
```

合并器读取评审包 `reviewed/` 中所有 `*.reviewed.jsonl`，检查批次/商品ID、字段、评分类型、署名和来源声明。错批次、重复条目、重复批次、篡改的生成状态、给不存在输出评分等都会拒绝。未收到的批次和未判断的条目原样保持null。

- `progress.json` 记录各批和各任务完成数及输入哈希。
- 未完成的任务只产生 `qrels.partial.jsonl` 或 `copy_scores.partial.jsonl`，不产生该任务的完整评分。
- 90件文案都完成后，生成 `copy_scores.reviewed.jsonl` 与 `copy_scores.json`，同时报告实际生成分母和全部预选分母。
- 5394个候选关系都完成后，生成 `qrels.reviewed.jsonl` 与 `retrieval_scores.json`，读取已固定的排序进行计算，不再调用GPU模型。
- 两类任务独立完成；可以先评完90件文案，先得到文案成绩，再继续检索标注。

已实际运行一次空目录合并，结果在 `artifacts/review_progress_initial_v1/`：检索0/5394、文案0/90，两个ready均为false，完整分数文件未生成。这证明空模板不会产生虚假“全部通过”。

也可单独对固定检索结果评分：

```powershell
.\.venv312\Scripts\python.exe scripts\freeze_retrieval.py score --judgments artifacts\review_merged_v1\qrels.reviewed.jsonl --output artifacts\retrieval_quality_final_v1.json
```

## 固定排序的范围

18条查询全部使用原300件商品索引和Qwen3-VL-Embedding-2B，主目标仍是图片向量。12条文本查询每条300个候选；6条图片查询各排除自己，剩299个候选。结果保存原始余弦、完整顺序、18×2048查询向量和配置。没有利用人工标签筛选候选、改写查询、调阈值或调融合权重。

离线核对已用保存的查询向量重新检索：5394条候选的顺序全部一致，分数在容差内一致。空相关性模板正确拒绝评分。人工评分仍按完整候选池定义Recall和Precision，零正例查询单独列出；工程冻结成功不等于检索质量达标。

此处逐查询耗时包含首次编码的预热影响，不替代已有千图库预热性能基线。旧 `scripts/evaluate.py` 仍可用；最终人工评测优先用固定排序入口，确保不同评审版本比较同一份预测。

## 服饰异常如何处理

旧180条副本中的 `deepfashion/train/000006` 存在多边形坐标越界，整条记录在新版本中排除；未直接截断坐标或改写原标注。旧副本和异常报告完整保留。

新版本包含80条FashionAI和99张DeepFashion：源train 49、validation 50；FashionAI源test_a 80仍作为源测试记录保留。278个图片/原标注副本通过哈希检查，99张服饰图对应148个实例。样本与COCO的图片、实例引用同步，排除项不会留下悬空标注。保留全部原始事实字段，另加入 `annotation_eligible` 和按官方划分限定的 `split_scoped_pair_key`。

数据保留原类别和FashionAI编码，未将已有类别强行映射到后续问答任务，也未运行分割模型。结构有效不等于所有标注语义已经人工审核；完整语义审核与代表性视觉复核仍应保留为后续验收项。

复建命令（已有版本无需重复生成）：

```powershell
.\.venv312\Scripts\python.exe scripts\review_packet.py prepare --packet artifacts\review_stage2_v2
.\.venv312\Scripts\python.exe scripts\freeze_retrieval.py predict --predictions artifacts\stage2_predictions_v2
.\.venv312\Scripts\python.exe scripts\release_fashion.py --output data\stage2_fashion_clean_v2
.\.venv312\Scripts\python.exe scripts\fashion.py audit --dataset data\stage2_fashion_clean_v2 --output artifacts\stage2_fashion_clean_audit_v2
.\.venv312\Scripts\python.exe scripts\verify_stage2_review.py
```

最后一条验证已交付的v1路径。新数据版本各自保留副本哈希及完整 `release_manifest.json`；原始 `Alibaba阿里算法实习/` 从未作为输出位置。

## 验收边界

本轮74项Python测试、16项前端校验断言通过；mypy、Black、isort、flake8和新增模块Pylint错误级检查通过。原始531,063个文件的路径、大小、修改时间与基线一致，PDF额外SHA256一致；非PDF原始全集未逐文件做内容哈希。工作结束时C盘空闲约72.38GiB，E盘约83.82GiB。

本轮完成评审操作与评分工程，并未虚构人工意见。网页已检查JavaScript语法、导入校验逻辑、内嵌数据及7188处本地图片引用；未通过浏览器进行视觉与交互验收，此前本地file URL受工具策略限制，本轮没有尝试绕过。

第二阶段仍有真实人工相关性判断、文案质量评分、商品语义与标注视觉复核待完成；这些成绩不能由测试通过或规则通过替代。第三阶段及后续服饰问答、局部定位均未开展。
