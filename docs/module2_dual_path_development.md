# 模块二：双上下文与ROI同图库开发实验

更新：2026年10月24日，Asia/Shanghai。Precision@10≥60%保持硬目标，Recall@10≥85%保持偏好。

<!-- dual_path_execution_state -->
本轮已完成并封存：12种方法、480份完整排序，新增35对完整Qwen精排、严格复用973对图像上下文分数，16条文本原320对保持不变。252项测试和全量重放核验通过。主方法尚未整体通过Precision硬门槛、Recall偏好和六指标不退化检查，默认方法保持原版，正式指标为空，未启动新独立验收。
<!-- /dual_path_execution_state -->

本轮使用上一轮已曝光的两组296件图库、40条查询和11840条AI关系，标签原样保留，人工标签0。没有新采样、补有利标签、缩小候选图库或改变K；全部属于开发，独立验收数据继续隔离。上一轮源码和2478个封存文件保持原样。

## 诊断与方法

冻结AI标签下，图片检索主要损失在候选池内排序；原20池与新描述20池的并集增加的正例容量有限。候选并集共504条查询—商品关系，每条图片查询有20至24件；两种上下文需要1008份完整精排结果。973份与历史输入身份完全一致，预计仅新增35次完整Qwen精排；16条文本查询原320份结果单列复用，不能把缓存算成新模型调用。

新增四种固定方法：同并集原上下文、同并集新上下文、双上下文等权融合、双上下文加原ROI。它们使用相同候选并集和公共尾部，候选之外保持原cross20的相对顺序。两种上下文分别在同一并集内按真实CE分数归一化排名，再等权融合；CE权重保持0.65，完整图库base排名权重0.35。ROI方法沿用既有0.25辅助投票和整图/ROI各0.5的组合，没有在新结果上搜索权重。

ROI使用旧描述时期的冻结FG整图、局部和主体裁切证据，不称为重新生成的新描述FG特征或新主体ROI。该实验不增加嵌入、FG或检测调用，所有既有特征来自此前完整模型。主方法为`dual_context_roi_union`，`dual_context_union`为去ROI消融。保留原八种方法，合计12种方法、480份完整排序。全部16条文本查询逐条返回同图库原cross20。

本轮不再使用上轮类型字典扣分；两种上下文缺分时必须完整模型实算，不能用另一上下文或向量分数填补。原模型、指令、权重、dtype、像素和长度配置保持一致，GPU只在自然空闲且Ollama无任何驻留模型时启用，不卸载或打断其他作业。

## 实际观察与下一轮机制

双上下文在A组的Precision、Recall和NDCG有正增量证据，但并未在其他组持续成立；后8条存在杯子命中流失，增加原ROI辅助后又出现柜体命中流失。主方法及ROI消融均未形成跨全部组的Precision/Recall稳定净收益。宏观差值上界未出现严格为负的整体退化证据，不能忽略已发生的个别查询退化，也不能把跨零区间写成已证明增长。

新querywise配对界让部分旧ROI、旧扣分方法的增量证据变得明确；这些旧方法的排序和单方法指标未变，这属于证据范围收紧，不是本轮创造的新算法收益。

下一轮单独验证新描述引导的查询主体ROI，以及完整Qwen在查询/商品真实裁切上的交互，和整图证据融合。主要针对主商品与展示道具混淆、手机壳场景中的耳机干扰、单耳本体/套装以及柜门抽屉结构；鞋用途、材质或成人/童装边界不能仅靠裁切保证修复，原图和原标题仍保留。所有本轮方法和失败原样封存，不追改旧成绩。

## 评估改进与目标边界

单方法六项质量指标及其未知标签区间沿用原实现。新评估按查询分别处理共享未知标签：相同排序的差值精确为零；未知项不超过16时精确枚举共同补全；只有超预算且排序不同的查询使用明确披露的保守界。Recall保留零正例时未定义以及可选查询分母，不能把它算成满分。单方法指标必须与旧八种方法逐项一致；配对区间变紧属于评估改进，不能写成算法增益。这些区间不覆盖AI误判，也不是统计置信区间。

当前A、B的泛文本及总体范围，即使最有利标签补全和完美排序，Recall@10的容量也不足85%。完整图库相关数R超过10时，单查询Top10最大Recall为10/R。该约束单独披露，不能靠改分母、删查询或改名Hit@10来实现目标；图片范围仍需要实际检验。独立验收应在冻结方案后，按真实需求使用全新图库与全新查询，不用开发容量来挑选有利验收样本。

## 运行与证据

```powershell
.\.venv312\Scripts\python.exe scripts\dual_path_experiment.py run
```

入口核对源码、缓存身份和封存输入，使用并发锁并按收据接续。GPU忙时返回等待；完成后只读验证封存。工程检查不代表质量达标。

本轮材料位于`artifacts/module2_dual_path_development/`。输入诊断为[diagnosis_before_experiment.json](../artifacts/module2_dual_path_development/diagnosis_before_experiment.json)，配对计划为[pair_plan.json](../artifacts/module2_dual_path_development/pair_plan.json)。已生成[完整内部对照表](../artifacts/module2_dual_path_development/internal_comparison.ai.development.md)、[逐查询商品查看页](../artifacts/module2_dual_path_development/internal_comparison.html)、[逐查询配对评估](../artifacts/module2_dual_path_development/regression/internal_paired_querywise.json)、[结果判定](../artifacts/module2_dual_path_development/decision.json)和[封存收据](../artifacts/module2_dual_path_development/completion_receipt.json)。所有开发质量数值留在内部报告，正式指标保持为空。封存123个文件及6份源码快照；原320份排序和11840条AI关系保持原样，新增160份方法排序均可重放。

准备阶段曾发生一次重叠启动，重复冻结写入被拒绝，只有一个推理进程运行；事件及原始错误已记录在[execution_notes.json](../artifacts/module2_dual_path_development/execution_notes.json)。冻结输入未覆盖，没有重复模型作业或按结果修改源代码。

当前排除依据为[本轮继承的最终曝光清单](../artifacts/module2_dual_path_development/post_experiment_exclusion_snapshot.json)。本轮没有新增商品或图片，继承全部历史曝光登记；任何现有开发商品与查询都不能再次作为独立验收。
