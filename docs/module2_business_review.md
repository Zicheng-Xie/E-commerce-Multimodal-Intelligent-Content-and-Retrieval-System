# 模块二业务相关性网格评审安排

最新准备见[开发候选补充与优先评审](module2_catalog_review.md)：59个优先关系已移到原批次前部，仍兼容本文收件；另有64件新候选的独立商品质检包。当前目标为Precision@10必须≥60%、Recall@10争取≥85%，本文原70%召回说明保留作历史协议，不代表最新优化目标。

更新：2026年10月22日。用户已确认：Precision@10≥60%按原PDF的商品业务相关性独立验收，原同款结果保留；目前暂无人工标注数据，安排网格评审。

后续自动推进见[机器辅助开发记录](module2_machine_assist.md)：机器弱标签已另行处理，未填写本真人模板。

当前交付的是可分发的评审材料和验收候选，真人评审尚未开始，评审人员待领取。没有替真人填写标签或署名，正式质量指标为空。

## 现在分发什么

| 工作 | 数量 | 领取人 | 材料 |
|---|---:|---|---|
| 开发独立评审A | 54批，5,143对 | 待指定实际真人A | [A离线包](../artifacts/module2_business_review/development/rater_A.zip) |
| 开发独立评审B | 54批，5,143对 | 待指定另一位真人B | [B离线包](../artifacts/module2_business_review/development/rater_B.zip) |
| 分歧仲裁 | 根据A/B分歧生成 | 第三位真人，待指定 | 合并后自动生成仲裁页 |
| 独立验收查询编写/确认 | 30个文本席位、30张查询图 | 待指定查询编写人 | [查询确认离线包](../artifacts/module2_business_acceptance/query_author.zip) |
| 验收商品类目与商品族质检 | 300件候选、30件查询商品 | 待指定独立质检人 | 封存来源在`dataset/evaluator/lineage.jsonl`，结论待人工记录 |

协调入口：[开发评审任务页](../artifacts/module2_business_review/development/index.html)。查询入口：[独立验收查询确认页](../artifacts/module2_business_acceptance/query_author/index.html)。领取表：[任务清单CSV](../artifacts/module2_business_review/work_assignments.csv)。这些材料保存在本机，尚未对外发送。

每位评审只领取自己的ZIP，解压后双击`START.html`。每批最多96项，每页12张商品图，可点开大图。按查询标准逐项选择相关/不相关；不确定就待定并写原因。先填写本人唯一代号并声明本人判断，A/B不要交换答案。每批可以分次做，浏览器缓存辅助恢复，交回前必须导出JSON；同批只交回选定的最新一个版本，旧文件移出收件文件夹。

图片和原标题已随包携带，页面隐藏模型分数、排序和另一位评审的判断。程序检查署名和声明，无法认证实际身份或证明两人没有交流。

## 收件、合并与仲裁

将A/B实际导出的JSON放入`artifacts/module2_business_review/returned_primary/`，该目录只放本轮选择的最新批次文件。运行：

```powershell
.\.venv312\Scripts\python.exe scripts\business_review_jobs.py merge --inbox artifacts/module2_business_review/returned_primary
```

输出为不可变的`development/merges/<输入摘要>/`。缺席位、缺批次、未判断项都保持null；两个真人一致才形成共识。分歧不自动裁决，生成`adjud_*.html`及`conflicts.jsonl`。本轮空收件演练结果为0/5,143项完成，没有产生质量成绩。

第三位实际评审人打开仲裁页，查看查询、商品与A/B理由，逐项判断并写仲裁依据，导出文件放入单独的`returned_adjudications/`。同一主评审输入下运行：

```powershell
.\.venv312\Scripts\python.exe scripts\business_review_jobs.py merge --inbox artifacts/module2_business_review/returned_primary --adjudications artifacts/module2_business_review/returned_adjudications
```

更换A/B输入后，旧仲裁会被拒绝，需要根据新的分歧页重新确认。合并同时保留两份原始人工判断、输入记录和共识来源。完成全部5,143对后，使用输出`judgments.jsonl`计算既有封存排序的开发指标：

```powershell
.\.venv312\Scripts\python.exe scripts\score_business_labels.py --reviewed "实际合并目录/judgments.jsonl" --status-only
.\.venv312\Scripts\python.exe scripts\score_business_labels.py --reviewed "实际合并目录/judgments.jsonl"
```

5,143对是旧18查询、286件商品的完整开发图库标签，用于训练、分析、选型，不能用于正式验收。旧780对训练包继续保留，无需优先重复做两个包。

## 新独立验收候选

已从原始商品test来源的固定20万条前缀、固定种子和30个细类抽样，排除既有开发/已消耗盲测的来源族键和图片指纹。候选含3C、服饰、家居，每类100件图库商品与10张查询图，总计300件图库、30张查询图，另留30个文本查询席位。来源ID、规范化标题、图片URL、可用品牌型号与商家货号用于保守去重，原图字节保留。

[准备核验结果](../artifacts/module2_business_acceptance/preparation_verification.json)确认：330个不同来源ID，已记录的旧/新跨角色族键重叠为0，字节、解码像素和dHash≤4的旧/新匹配为0。自动检查不能证明所有语义同款均被识别，商品类别和商品族仍需真人质检。该样本仅代表预先选定的30细类平衡闭集，不代表真实流量或全部商品类别。

查询作者只看到固定需求范围和查询图片，不看到候选图库。文本必须真人编写，图片必须检查主体、范围与判断标准；修改后要重新批准，无法确认就保留未批准并写原因，不删除查询。实际导出后运行：

```powershell
.\.venv312\Scripts\python.exe scripts\business_query_review.py import --reviewed "实际人工导出的acceptance_queries.reviewed.json"
```

导入保留全部60个查询，不把30个文本空席位当作已完成查询。当前人工批准数0、相关性标签0、模型预测0。查询全部批准、商品质检完成后，冻结查询与判断标准，再生成覆盖60×300=18,000对的独立验收相关性网格；这一数量是后续工作量，当前没有生成或完成18,000条真人标签。

验收来源已登记`data/module2_reserved_acceptance/records.jsonl`，后续开发、训练和其他新验收都应排除。原始来源只读；原同款实验与旧开发文件封存未改动。

## 接下来如何提升Precision

先完成开发网格，建立真实业务正负例，重点分析本体/配件混淆、同类不满足用途或属性、背景相似等错误。使用分组开发验证训练业务精排与分数校准，保留区域注意力、ROI裁切、局部放大的消融对照。策略选型只用开发数据，独立验收图库与标签不参与调参。

最终模型、代码、参数、对照策略和评测协议先冻结，再一次性生成独立验收预测、封存完整排序、开封评分。Precision@10固定除以10，主目标≥60%；Recall@10≥70%，mAP、NDCG@10、MRR和Hit@1对照不退化，文本/图片分别核对并报告三品类明细、零正例查询、置信区间与端到端延迟。若出现容量不足，报告原查询与完整标签下的真实上限，不改标签或删难查询凑目标。

本轮工程检查：170项测试通过，包括真人双评、缺失与分歧保留、第三人仲裁、旧仲裁失效、字段篡改和实际JS导出校验。108个便携评审页面的脚本与图片引用通过核验；旧4个封存实验目录文件哈希核验通过。测试通过不代表检索质量达标。
