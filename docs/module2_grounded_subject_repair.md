# 模块二：主售定位修复与完整模型对照

最近同步：2026-10-25T01:29:15.084265+08:00。

主售定位修复开发对照已完成：22方法880份排序，原19方法760份排序保持；全量重放和封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，其他指标不退化尚未整体证明。

独立AI盲测已完成：96件全新图库、12条全新图片查询及12条中文文本查询，2304条冻结AI关系，单一候选与三个基线共96份排序重放封存。独立AI Precision目标未整体通过，Recall偏好通过。其他指标不退化尚未整体证明。默认原版保留，人工标签0，formal_metrics=null；开发和独立盲测分开报告，盲测材料一经打开即登记消耗，不能用于调参后重测或复用为新验收。

本轮使用固定query-only主售定位：原始查询图交给完整本地Qwen3.5模型与投影器，输出0..1000坐标；按统一几何规则裁切、局部放大，再用同一面板中的原图标框与裁切图复核。主售不清、拒绝、不确定或有效原生语义输出自相矛盾时均回退整图，不重试追求通过。原DINO完整检测结果保留诊断，原FG相关方法和证据保持；新策略检验Qwen定位与真实裁切精排，不声称重训神经注意力或重新运行DINO/FG。

新增整图、局部放大、双上下文融合三种方法，共享old20、新描述20和新选定查询嵌入top20的稳定并集，最多60件。所有共享候选必须有完整Qwen实际配对评分或严格身份缓存。基础排序权重固定为旧/新描述0.375/0.375、选定查询全图余弦0.25，最终0.65上下文CE排名与0.35基础排名。单上下文为共享池消融；文本保留原cross20。315件图库、40条开发查询、12600条AI关系及K10、完整Recall分母和未知null不变。

最初定位执行因模型accept与检查项矛盾而停止；原始输出、源码和failure.json保留。单独冻结恢复入口将同类语义矛盾统一标记弃权，未重写输出、重试或修改提示/模型/排序公式。恢复时缺少查询向量父目录的文件错误也保留，已补齐既定目录；不是质量结果。初始291项、恢复293项、盲测执行前295项测试分别留有收据。

开发入口：`.venv312/Scripts/python.exe scripts/grounded_subject_recovery.py worker`。已有实例时不重复启动。显存准入仍为nvidia-smi有限非负且低于1500MiB，再检查Ollama驻留为空；等待自然释放，绝不卸载或缩短用户模型keep-alive。完整模型配置和历史源码封存保持。

[开发协议](../artifacts/module2_grounded_subject_repair/protocol.json)、[恢复协议](../artifacts/module2_grounded_subject_repair/semantic_abstention_execution_protocol.json)、[运行状态](../artifacts/module2_grounded_subject_repair/public_status.json)。

已完成24条定位，接受8条、整图回退16条。AI接受不等于定位正确；全部原生判断留存，不按检索结果改框。

已冻结1196个上下文候选位置，其中需要247个新增实际配对评分，其余仅严格身份复用。

[开发内部完整对照](../artifacts/module2_grounded_subject_repair/internal_comparison.ai.development.md)、[开发封存收据](../artifacts/module2_grounded_subject_repair/completion_receipt.json)。只读核验：`scripts/grounded_subject_recovery.py verify`。

下一步独立AI盲测的隔离、评审和执行状态见[独立盲测文档](module2_grounded_independent_blind.md)。开发未过目标时，本轮用户授权的一次盲测只用于独立诊断并如实记录失败，不据此自动切换默认或宣称正式达标。
