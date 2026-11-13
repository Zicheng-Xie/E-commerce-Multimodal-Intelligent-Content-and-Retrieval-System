# 模块二：商品质量、主体掩码与业务学习开发周期

最近同步：2026-10-25T07:15:49.256343+00:00。

模块二新一轮开发已完成435张原图的数据审计、66条商品原生AI质检、完整GroundingDINO与SAM 2主体视图生成、24条图片查询的视图QA及9组家族隔离训练输入准备。原40条查询和12,600条AI关系保持不变。自动主售/结构核验已完成（435/435条）；整图业务度量学习组件对照已完成，内部组件结果不代表完整系统或独立验收。双图注意力业务适配及固定融合已完成：7方法280份排序，18个家族折模型重放通过。开发Precision硬目标尚未整体通过。Recall偏好尚未整体通过。其余指标不退化尚未整体证明。新掩码因主体结构质量检查未过全局停用。默认原版、人工标签0、formal_metrics=null；历史盲测继续封存，本轮未开启新的独立盲测。

## 当前结果与实际问题

本轮仅使用既有曝光开发数据：两组各315件图库，fresh共享tune图库；共411个唯一图库ID和24张图片查询原图，16条中文文本查询保留。共检查435张原图及完整12,600条既有AI关系，未发现与已消耗盲测ID/来源ID、字节、像素或dHash<=4冲突。检查结果不等于不存在所有语义近重复。

旧标题/主体词表的153条行级信号涉及66个唯一商品，已实际打开全部6张2400×1440联系表并逐件记录原生视觉AI观察。词表缺词不等于商品错误；例如金枕头是榴莲品种、手机壳可能只是雕刻机加工对象、杯子可能只是柜内道具。没有据此改动既有相关性标签，66条QA是候选自身描述核验，不是新增检索标签。

内部容量报告区分图库缺正例、候选池漏召回和池内排序空间。固定K10与完整相关商品分母下，部分旧开发组的Recall偏好受容量限制；补库也可能进一步扩大Recall分母。本轮保留原数据与指标，未通过删查询、改标签或缩分母追分，容量界不是实际模型成绩。质量数字继续仅放内部报告。

## 完整模型实跑与视图QA

已完成412次完整GroundingDINO调用和383次完整SAM 2.1 Base+调用。435条记录产生383组框/掩码提议，52条保留整图；这不是主体定位准确率。SAM 2使用官方固定revision、完整权重及默认预处理，float32，完整参数73,328,401；权重大小和公开LFS SHA256已核对。使用当前Transformers 4.57.1，无环境升级和新增依赖。未用精简模型或模拟分数替代。

新框策略使用同一公开主体词的全部有效实例并集，避免只取最高检测分；主体词来自图像自身旧AI描述，候选可用自身原标题，查询不读取图库。输出是提议，尚无语义正确保证。全部保留原图，不通过检测失败删除商品。生成1,532张视图及6张QA联系表，全部登记开发曝光并纳入最新排除快照。

已实际查看22组图片查询原图/框/掩码面板及2张无框回退原图，覆盖全部24条图片查询。发现掩码删除鞋跟、耳机头梁和透明杯体，内胆包仍夹带展示手袋，包装插图被当成独立主体等问题。因此原生QA没有准许将这些掩码直接作为可靠训练输入或默认策略；QA判断不转成按查询ID硬编码的运行时规则。

当前自动核验使用完整本地Qwen3.5及投影器，逐条读取原图、原生框裁切、原生掩码三张PNG，不手动缩图。候选仅附自身标题和自身旧描述；查询仅附自身图像描述。明确主售正确且裁切完整、掩码完整无道具时选掩码；只有框检查明确通过时选框；否则保留整图。不读图库关系、检索分数或盲测材料。工程错误停下并留原生输出；语义null保持弃权，不反复重试追通过。这是同一模型自审，不能写成独立标注员验证。

## 本轮执行顺序与门槛

1. 435条自动视图核验已完成，与既有原生QA对照仍发现明确漏检；新掩码未通过质量门槛，在后续主训练中全局停用。没有按具体查询ID挑选视图，所有原始图像和查询保留。自动核验的全部原生输出与错误判断不修改。
2. 业务困难负例已准备9组查询家族×图库家族隔离输入，保留全部多正例，未知不作负例。困难负例来自允许训练单元内的旧真实模型排序；不同SKU不自动为负。额外冻结的整图控制分支直接使用原完整Qwen的2048维真实向量，在CPU训练2048→256→256共享投影，以业务矩阵和Multi-Similarity困难样本挖掘监督；9个家族折加1个全开发候选各400步。它不依赖新掩码，不更改或精简Qwen骨干，不是STIR，也不是最终融合；OOF仅属曝光开发的组件诊断，全部结果留内部。[整图组件协议](../artifacts/module2_business_metric_learning/protocol.json)。
3. 双图注意力单独采用STIR启发的业务适配：两张图各按既有官方DINO默认256短边/224中心裁切，再在像素张量上沿高度连接，送入完整12层DINOv2-small；所有512个patch和CLS共同参与自注意力，提取384维联合特征。视觉骨干冻结，不声称训练注意力权重；完整业务匹配头按原家族掩码训练，9个折及1个全开发候选各400步。这是另列配置的业务适配，不是STIR的ViT-S/16论文权重或先冻结后解冻的端到端复现。双图注意力业务适配及固定融合已完成：7方法280份排序，18个家族折模型重放通过。开发Precision硬目标尚未整体通过。Recall偏好尚未整体通过。其余指标不退化尚未整体证明。
4. 最终同图库对照预先固定7种方法：global、region_title、cross20、上轮整图/放大融合，整图度量学习融合、双图注意力单独组件消融及双图注意力融合主方法。主方法仅在上轮已由完整Qwen真实评分的相同候选池内，按0.8父排名与0.2辅助排名排序；完整尾部及16条文本原cross20保持。没有新增Qwen调用、漏分补零或看新结果选权重。完整双图计算覆盖24×315共7,560个真实配对；最终分all/text/image报告六指标、共享未知标签增量、严格退化、资源和静态结果页。实际完成状态见收据，组件消融不当作完整系统成绩。
   本轮质量表使用家族隔离折模型的OOF拼接，只作内部开发诊断，不能当成单一部署模型成绩。两个全开发候选另存完整权重，不用其训练内结果替换质量表。尚未交付新方法的在线查询入口或端到端时延验收，原检索入口及完整图文功能保持。
5. 方案冻结后，新的独立验证必须另取全新查询和图库。已有两次被消耗的业务盲测及所有本轮衍生图均不得训练或充作新验收。

Precision@10≥60%为硬目标，Recall@10尽量≥85%。本轮所有新结果仍仅是曝光开发证据，不能替代独立验收；formal_metrics=null、默认原版不变。

## 入口与证据

- [数据审计协议](../artifacts/module2_business_grounding_development/audit_protocol.json)、[内部数据与容量诊断](../artifacts/module2_business_grounding_development/internal_data_quality_audit.json)。
- [66条原生AI商品QA](../artifacts/module2_business_grounding_development/QA/source_review.native_AI.json)、[24条图片视图QA](../artifacts/module2_business_grounding_development/QA/query_views.native_AI.json)。
- [完整视图协议](../artifacts/module2_business_grounding_development/views_protocol.json)、[模型运行收据](../artifacts/module2_business_grounding_development/views_receipt.json)、[主体质量门槛](../artifacts/module2_business_grounding_development/subject_quality_gate.json)。
- [自动核验冻结协议](../artifacts/module2_business_grounding_development/semantic_verifier_protocol.json)、[九组业务训练输入](../artifacts/module2_business_grounding_development/business_training_inputs.json)。
- [自动核验质量评估](../artifacts/module2_business_grounding_development/semantic_quality_assessment.json)、[双图注意力与融合协议](../artifacts/module2_business_joint_attention/protocol.json)。
- [内部完整七方法对照](../artifacts/module2_business_joint_attention/internal_comparison.ai.development.md)、[真实Top10静态回放](../artifacts/module2_business_joint_attention/internal_comparison.html)。
- [当前状态](../artifacts/module2_business_grounding_development/public_status.json)、[最新排除快照](../artifacts/module2_business_grounding_development/post_views_exclusion_snapshot.json)。

`.venv312/Scripts/python.exe scripts/business_mask_verifier.py worker` 接续未完成的原生核验，先核对源码/模型/输入身份，再检查显存低于1500MiB且Ollama驻留为空；不得并行重复启动。已有进程时持续跟进原工具session。不会卸载模型或缩短其他作业keep-alive。后续重启时自身模型也必须自然释放。

`.venv312/Scripts/python.exe scripts/business_grounding_audit.py verify`只读核验本轮审计输入。原盲测完整只读核验入口仍为`grounded_blind_evaluation_recovery.py verify`。

双图适配执行入口为`business_joint_experiment.py worker`，已有实例不重复启动。完成后的只读重放为`business_joint_experiment.py verify`；不重新运行已完成的视觉推理。主报告为`artifacts/module2_business_joint_attention/internal_comparison.ai.development.md`，结果页为同目录`internal_comparison.html`，只有完成收据存在才代表该轮完整对照完成。联合输入是已曝光原图按官方预处理连接的内存张量，不在磁盘另存7,560张图片；其源身份、规范RGB指纹及dHash全部单独登记，并纳入最终排除快照。

本轮整图组件、双图模型、完整排序、六组指标及共享未知标签增量的统一只读核验为`.venv312/Scripts/python.exe scripts/verify_business_development_cycle.py`。307项工程测试通过不等于检索质量达标。后续应根据已封存的`internal_next_cycle_diagnosis.json`区分候选覆盖与池内排序，优先补主售/道具/用途关系证据；新一轮方案仍需单独冻结，不能覆盖本轮结果或重新使用消耗盲测。

初次数据审计读取旧文本runtime的pool字段失败，尚无新检索预测。失败源代码与初版协议保存在source_history及audit_protocol.initial.json；已改从文本实际features读取原候选池，标签和排序未更改。历史封存源和68个核心源文件均未编辑。
