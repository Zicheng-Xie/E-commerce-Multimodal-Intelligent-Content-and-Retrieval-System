# 模块二：新图库与新查询的独立AI盲测

最近同步：2026-10-25T01:29:15.084265+08:00。

主售定位修复开发对照已完成：22方法880份排序，原19方法760份排序保持；全量重放和封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，其他指标不退化尚未整体证明。

独立AI盲测已完成：96件全新图库、12条全新图片查询及12条中文文本查询，2304条冻结AI关系，单一候选与三个基线共96份排序重放封存。独立AI Precision目标未整体通过，Recall偏好通过。其他指标不退化尚未整体证明。默认原版保留，人工标签0，formal_metrics=null；开发和独立盲测分开报告，盲测材料一经打开即登记消耗，不能用于调参后重测或复用为新验收。

本轮在新图打开前冻结唯一候选的方法、模型参数和采样规则；原始源码保留，机械调用修复与指标来源schema兼容修复另有冻结协议，未改数学方法或按盲测分数挑方案。原训练源尾段采样仅得到3条候选并缺少9类，尚未打开新图片。单独恢复协议在新图打开前冻结：统一使用原只读官方test源全部1,341,932条记录，保留12类、seed 2026102502和每类48条元数据配额，继续排除全部历史商品族与图像。实际冻结576条候选，全部类型均达到48条；模型和相关性口径未变。每个原URL只尝试一次，不绕过拒绝；每类按冻结顺序取前9张可读、全局非重复图，前8张入图库、第9张为图片查询，只按技术可用性选择，不按相关性或分数替换。若数量不足，则停止并保留证据，不擅自降低样本数。

在打开新图前，使用最新开发曝光快照排除来源ID/商品族和字节、像素、dHash<=4近重复，覆盖此前失败候选、原图和衍生图。先登记消耗，再打开图库。Codex原生视觉AI先逐件写候选公开描述并冻结，再打开查询，完成全部96×24逐对关系并冻结，最后才运行检索。标签保持原PDF业务口径，主类型/用途及关键可见形态匹配；不是同SKU。配件/道具不能代替主体，证据不足保持null，不能根据排名改标。

评审为AI，人工0；评审者知道项目背景，不声称独立人工或独立标注模型。数据独立于开发，但12类平衡的小型封闭图库不代表真实流量；记录隔离也不能证明所有语义近重复都不存在。质量数字仅放内部独立报告，未知补全范围不是置信区间或AI误标界。

候选固定为grounded_whole_zoom_union，完整Qwen嵌入、完整Qwen精排及同一query-only定位/弃权流程。对照仅global、region_title、cross20；不将多个方法的最高数值拼成结果。Precision@10≥60%为硬目标，Recall@10尽量≥85%，其他指标和资源开销完整保留。无论通过与否，本盲测被消耗；后续调参只能回开发集，新的验收必须使用另一份全新数据。默认方法不自动替换，formal_metrics继续为null。

执行入口：`scripts/grounded_blind_dataset.py`负责freeze/reserve/acquire/open_gallery/freeze_profiles/open_queries/freeze_labels；`scripts/grounded_blind_runtime.py worker`在AI标签冻结后运行，`scripts/grounded_blind_report.py finish`计算与封存。GPU准入与用户模型自然释放约束保持。

[冻结发布协议](../artifacts/module2_grounded_independent_blind/release_protocol.json)、[运行状态](../artifacts/module2_grounded_independent_blind/public_status.json)。

[独立AI完整内部报告](../artifacts/module2_grounded_independent_blind/internal_comparison.ai.blind.md)、[封存收据](../artifacts/module2_grounded_independent_blind/completion_receipt.json)。

[原范围不足证据](../artifacts/module2_grounded_independent_blind/metadata_sampling_shortage_evidence.json)、[采样恢复协议](../artifacts/module2_grounded_independent_blind/source_range_execution_protocol.json)、[元数据冻结收据](../artifacts/module2_grounded_independent_blind/reservation_receipt.json)。采样恢复入口为`grounded_blind_source_recovery.py reserve`；后续采集、AI评审和推理使用原冻结入口。

完整编码初次执行因encode_views的output/configuration调用顺序错误停止，尚无图库向量或检索预测。原冻结源码保留；独立恢复入口仅按原函数签名纠正参数传递，配置、图像、模型、候选规则和排序公式不变。297项测试及隔离命名空间检查通过，全部原生查询准备和AI标签保留。[机械恢复协议](../artifacts/module2_grounded_independent_blind/argument_order_execution_protocol.json)、[失败证据](../artifacts/module2_grounded_independent_blind/runtime_argument_failure_evidence.json)。实际推理接续入口为`.venv312/Scripts/python.exe scripts/grounded_blind_execution_recovery.py worker`，已有实例不重复启动。

已直接查看全部8张原生图库联系表与12张查询图，冻结96份候选描述后再打开查询；2304条AI关系包含16条语义未决。标题采集桶与真实商品存在偏差，印油、中式盘扣服装及圆柱颈枕查询仍全部保留，不按覆盖或分数换图。[原生AI评审工作表](../artifacts/module2_grounded_independent_blind/evaluator/review_worksheet.native_AI.json)、[标签冻结收据](../artifacts/module2_grounded_independent_blind/label_receipt.json)。没有伪造人工或Ollama评审调用。

预测完成后另存[图库容量诊断](../artifacts/module2_grounded_independent_blind/internal_capacity.ai.blind.json)及[数据覆盖解释](../artifacts/module2_grounded_independent_blind/internal_dataset_coverage_notes.ai.blind.md)。按冻结AI关系，图片范围即使理想排序、未知项均按最有利情况补全，现图库容量仍不足以通过Precision硬目标；这不是模型成绩，不能用它替代实测。实际分数与容量之间仍有排序改进空间。12个采集桶是元数据分层，不能写成12类真实商品均正确、均衡。

[960张商品卡的真实排序回放](../artifacts/module2_grounded_independent_blind/internal_comparison.html)。完整只读核验入口：`.venv312/Scripts/python.exe scripts/grounded_blind_evaluation_recovery.py verify`，同时核对原发布、采样恢复、参数顺序恢复、源码快照、实际源文件、排序和指标。

旧指标入口要求review_origin固定枚举model，原始标签写明Codex原生视觉AI来源，因此首次导出被拒绝。单独冻结只读适配：仅在内存副本中映射为通用model枚举，原来源保留为annotation_tool；原2304条grade、reason、查询、商品及标签文件、预测和指标公式全部不变。298项测试通过，适配与旧枚举的指标逐项一致。实际收尾入口为`grounded_blind_evaluation_recovery.py finish`，完整核验为`grounded_blind_evaluation_recovery.py verify`。[来源兼容协议](../artifacts/module2_grounded_independent_blind/AI_provenance_execution_protocol.json)、[失败证据](../artifacts/module2_grounded_independent_blind/evaluation_provenance_failure_evidence.json)、[测试收据](../artifacts/module2_grounded_independent_blind/evaluation_recovery_checks.pre.json)。
