<!-- grounded_subject_latest -->
最近同步：2026-10-25T01:29:15.084265+08:00。

主售定位修复开发对照已完成：22方法880份排序，原19方法760份排序保持；全量重放和封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，其他指标不退化尚未整体证明。

独立AI盲测已完成：96件全新图库、12条全新图片查询及12条中文文本查询，2304条冻结AI关系，单一候选与三个基线共96份排序重放封存。独立AI Precision目标未整体通过，Recall偏好通过。其他指标不退化尚未整体证明。默认原版保留，人工标签0，formal_metrics=null；开发和独立盲测分开报告，盲测材料一经打开即登记消耗，不能用于调参后重测或复用为新验收。 见[本轮进度与证据](module2_grounded_subject_repair.md)。
<!-- /grounded_subject_latest -->

<!-- subject_learning_current -->
最近同步：2026-10-24T22:40:06.223857+08:00。

主售属性与困难负例重训已完成：11方法440份排序，原8方法320份排序及全部12600条AI关系保持不变；新主方法未整体通过Precision目标、Recall偏好和不退化检查。全图库RankNet提议与完整精排的14方法560份排序也已封存，仍未整体通过目标。类别平衡全负例损失的17方法680份排序随后完成；主类型系数转为正值，但图片Precision目标仍未整体通过。

公开主售冲突优先级对照已完成：19方法760份排序，原17方法680份排序不变，完整重放与封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，六指标不退化未整体通过。全部为已曝光AI开发实验，人工标签0；原版默认保留，正式指标null，新的独立验收未启动。

本轮归因覆盖24条图片查询：11条当前池遗漏部分已知正例，14条池内仍有已知排序空间。内胆包旧ROI裁到左下方手袋道具；华为Mate10查询的业务标签包含其他型号手机壳及部分DIY材料。标签保持冻结，型号继续作软诊断，不通过重写标签更换评价口径。属性训练使用软特征；后续公开主售冲突优先级作为单独实验披露。

属性训练对照使用20维原模型证据与38维加入公开属性、整图/裁切及FG一致性的证据；18个家族隔离折模型、2个全开发候选按固定400步、L2 0.05、每查询最多8个已确认困难负例训练。395条已知池内关系参与允许范围内的训练，109条未知排除于拟合。历史公开描述属于既有AI开发材料；本轮梯度家族掩码不能使它成为独立验证。

后续全图库提议层独立使用33维完整Qwen/FG/公开属性输入，覆盖7560个实际图库组合，其中7419条已知、141条未知；提议特征不含CE列，不填补缺失CE。9个家族排除折模型输出top20提议，稳定追加到原候选并集中，最多60件。共享池的每件商品均需完整Qwen两种原整图上下文真实评分或严格身份缓存。proposal_only为明确去掉CE的组件消融，不能替代主方法的完整精排成绩。

284项执行前测试通过；旧查询、标签、K10、完整图库Recall分母、零正例和未知null保持原口径。已确认的错误裁切坐标在这两轮保持原状，实验检验一致性融合和候选生成，不宣称主体定位已经修复。新实验不生成图片，曝光快照继承此前原图、159张衍生图和全部来源家族；全部材料排除于独立验收。

已完成属性实验：[封存收据](../artifacts/module2_subject_attribute_learning/completion_receipt.json)、[内部报告](../artifacts/module2_subject_attribute_learning/internal_comparison.ai.development.md)、[严格局部损失](../artifacts/module2_subject_attribute_learning/internal_strict_regressions.json)。全图库实验：[协议](../artifacts/module2_subject_recall_learning/protocol.json)、[候选计划收据](../artifacts/module2_subject_recall_learning/plan_receipt.json)、[运行状态](../artifacts/module2_subject_recall_learning/public_status.json)。开发质量数字只保留内部报告。

新增类别平衡损失的执行前286项测试通过，运行时使用独立模块命名空间和新输出目录，原已封存源码、模块对象与评分均不修改。全图库负例而非仅top8困难负例用于每查询类别平衡逻辑损失，偏置不正则；优化步数、学习率、L2及家族排除不变。提议层分数是标准化logit，不是业务概率。

本轮入口：`.venv312/Scripts/python.exe scripts/pointwise_recall_experiment.py worker`；[冻结协议](../artifacts/module2_pointwise_recall_learning/protocol.json)、[运行状态](../artifacts/module2_pointwise_recall_learning/public_status.json)、[负例集中诊断](../artifacts/module2_pointwise_recall_learning/negative_sampling_diagnosis.json)。当前阶段不声称已验证训练问题的因果效果。

类别平衡结果：[封存收据](../artifacts/module2_pointwise_recall_learning/completion_receipt.json)、[内部对照](../artifacts/module2_pointwise_recall_learning/internal_comparison.ai.development.md)；只读核验：`scripts/pointwise_recall_experiment.py verify`。

公开主售冲突优先级实验的288项执行前测试通过。主方法只在既有完整Qwen精排共享池内将明确主类型、部件或结构冲突稳定后移；同层原神经排序、完整尾部、所有商品ID和标签保留。未知或矛盾证据弃权，手机型号不硬过滤，手机壳材料按既有宽业务口径保留。全排序后处理是单独消融，池外商品保持原base顺序，不补造CE分数。本轮新增神经调用0、图片0；没有修复错误ROI坐标。

[冻结协议](../artifacts/module2_public_conflict_priority/protocol.json)、[完整内部对照](../artifacts/module2_public_conflict_priority/internal_comparison.ai.development.md)、[严格局部退化](../artifacts/module2_public_conflict_priority/internal_strict_regressions.json)、[封存收据](../artifacts/module2_public_conflict_priority/completion_receipt.json)。只读核验：`.venv312/Scripts/python.exe scripts/public_conflict_experiment.py verify`。主方法与消融独立完整报告，不根据事后最高数字更换主方法。

下一轮优先解决已确认的主体定位问题：先冻结适用于所有图片查询的query-only定位和不确定回退协议，再生成ROI并做完整模型评分。禁止按已知正例或检索名次选框，原整图候选必须保留；无法确认主售主体时保留整图。仍在本开发集检验Precision目标及其他指标变化，单独登记新增衍生图曝光。该定位修复尚未实现或运行，不列为已完成成果；新的独立盲测需另行冻结单一可执行候选后才启动。

<!-- /subject_learning_current -->

# 模块二：家族隔离有监督融合与平衡补库

更新：2026年10月24日，Asia/Shanghai。Precision@10≥60%为硬目标，Recall@10≥85%为偏好。

<!-- learned_fusion_state -->
有监督融合开发对照已完成封存：22种方法、880份完整排序，原19方法760份排序及16条文本原cross20不变。训练前263项测试通过，18个交叉拟合模型和2个全开发候选均可按固定配置重放；104个文件、73份源码快照封存。新主方法仍未在全部组通过目标和不退化检查，默认方法保持原版，正式指标为空。

平衡补库已冻结48条元数据。首次下载全部遭Windows沙箱ConnectError10013，原失败记录保留；允许网络的独立恢复通道仅重试相同原URL一次，取得19张非重复图片，其余站点失败不再重试。该采集阶段当时尚未加入比较图库；后续19件已进入315件开发图库，完整结果由独立收尾目录封存。
<!-- /learned_fusion_state -->

## 有监督融合的实际范围

本轮在既有完整Qwen、DINO及FG模型证据上新增融合层，没有微调或替换神经骨干，也没有用模拟分数补缺。24条图片查询使用相同504对候选；每对有13项原公开特征、整图旧/新上下文与真实裁切CE排名、旧/新全图库base排名和整图/ROI的FG融合排名，共20项输入。392条已知关系用于允许范围内的训练，112条未知保持null并排除于标准化、困难负例挖掘和梯度。

来源ID、规范标题、商家型号、字节/像素及近dHash连接实际查询和商品家族。当前为24个查询源家族、392个商品源家族。三查询家族折与三商品家族折的每个交叉单元，都排除被验证家族及被验证图片查询的来源家族；训练、标准化、挖负例和校准使用同一排除掩码。这里隔离的是保守来源家族，不声称所有商品类别互不相交。

点式逻辑融合和困难负例RankNet各训练9个折模型，沿用既有400步、学习率0.1、L2为0.05、每查询最多8个已确认困难负例的配置，不搜索验证集参数。另存2个全开发训练候选，其训练内成绩不当质量结果。逐单元OOF排序拼接多个模型，仅是全部已曝光数据上的内部开发证据，不能当成可部署单模型的独立验收成绩。

固定主方法为`learned_ranknet_blend_oof`：0.65的学习排名与0.35的原/新base投票。点式和纯RankNet作为消融。三新法保留同候选池、cross20公共尾部及全部296件图库；16条文本查询原样保留。新增神经模型前向为0，所有原模型分数均来自封存的真实推理。

## 真实结论

全部三种新方法的图片Precision门槛只在A组通过，B/C组未通过；Recall偏好仍未整体通过。主方法在B组相对父裁切方法的Precision有不退化证据，在C组相对整图双上下文的Precision有不退化证据，但其他指标和其他基线仍存在跨零范围，不能写成稳定整体增长。

明确局部损失保留：A组`image_07`相对cross20的AP下降；B组手机壳相对父裁切方法的Precision、Recall、AP和NDCG下降；C组手机壳相对cross20和父裁切方法的同四项下降，C组内胆包相对父裁切方法的AP、NDCG、MRR和Hit@1下降。没有按查询挑不同方法拼成绩，也没有根据新分数改标签或K。

六指标原定义、完整图库Recall分母、零正例查询和共享未知完成范围均保留。这些范围不是统计置信区间，也不能覆盖AI标注错误。完整开发数字仅见[内部报告](../artifacts/module2_learned_fusion/internal_comparison.ai.development.md)、[目标判定](../artifacts/module2_learned_fusion/decision.json)和各组`internal_paired_querywise.json`。

## 补库和评审状态

补库使用训练来源的新固定片段，12个公开标题层各4条元数据，涵盖主商品、配件和相似负例。分类层不是相关性真值；所有原URL均尝试，不按模型分数、标签或图片内容筛除技术成功项。全部候选、失败URL及取得的原图均登记为开发曝光，不能再次充作独立验收。

第一次48条失败是套接字权限错误，不能说成48张源图失效。独立恢复协议只允许重试ConnectError10013；HTTP拒绝、超时等不适用此恢复条件。原协议、原缓存和原收据均未覆盖。实际来源和失败类型见[首次采集收据](../artifacts/module2_balanced_supplement/acquisition_receipt.json)与[网络恢复收据](../artifacts/module2_balanced_media_recovery/acquisition_receipt.json)。

多查询评审曾在第一件商品第9批次触发上下文预留检查，部分原生输出还有图片身份混淆，未形成完整标签集。原8条有效输出和失败输出保留。新评审独立采用每次一条图片查询、候选/查询原生双图和16384上下文，未降图片分辨率或模型规模；旧部分判定不复用，所有19件统一重新评审。语义null保留，工程错误不能伪装成负例或未知标签。核验模型自然过期，不卸载、不缩短用户作业keep-alive。

<!-- balanced_review_state -->
19件商品已完成532次原生AI评审，新增760条关系冻结；完整模型编码和315件同图库8方法、320份排序对照均已完成。报告导出曾因误用macro_ranges字段停下，现已在独立收尾目录按真实ranges格式修复，278项测试与全部排序、指标重放通过；原失败证据、源码、标签、排序和参数未覆盖。主方法仍未整体通过Precision硬目标、Recall偏好及六指标不退化检查，默认原版、正式指标null，本轮未启动新的独立验收。

本轮完成与验证以[独立收尾状态](../artifacts/module2_balanced_gallery_finalization/public_status.json)、[封存收据](../artifacts/module2_balanced_gallery_finalization/completion_receipt.json)和[完整内部报告](../artifacts/module2_balanced_gallery_finalization/internal_comparison.ai.development.md)为准。旧runner的failure.json保留报告导出错误，不再重复运行旧入口；只读复核入口为`scripts/finalize_balanced_gallery.py verify`。
<!-- /balanced_review_state -->

## 隔离、复核与下一阶段

本轮没有打开或推理新的独立验收查询、图库。2026年10月23日已有AI盲测未通过的历史保留，正式指标仍为空。所有训练及新图都属于开发曝光。

训练材料见[冻结协议](../artifacts/module2_learned_fusion/protocol.json)、[来源家族及特征元数据](../artifacts/module2_learned_fusion/feature_metadata.json)、[训练家族掩码审计](../artifacts/module2_learned_fusion/fold_audits.json)、[880份排序与20个模型重放](../artifacts/module2_learned_fusion/verification.json)、[封存收据](../artifacts/module2_learned_fusion/completion_receipt.json)。补库新评审见[新单查询协议](../artifacts/module2_balanced_pair_review/protocol.json)，最新原图排除依据为[恢复后的曝光快照](../artifacts/module2_balanced_media_recovery/post_acquisition_exclusion_snapshot.json)。

```powershell
.\.venv312\Scripts\python.exe scripts\learned_fusion_experiment.py verify
```

315件图库扩库对照已完成，旧880份排序属于296件历史实验。当前主售属性和全图库候选学习各自按独立协议继续开发；候选通过同条件目标检查后，才冻结单一可执行版本，并使用全新查询和图库盲测。


<!-- balanced_gallery_execution -->
## 315件图库对照的固定执行口径

新增执行入口为`scripts/balanced_gallery_runner.py run`。它只观察现有原生评审的完整收据，不再启动第二个评审进程；接续候选专用描述、完整Qwen嵌入、完整DINO检测、完整FG特征、衍生图曝光登记、共同候选计划、完整精排/精确缓存、排序、评估和封存。用户作业与本轮原生模型均等候自然释放，不卸载或缩短keep-alive。工程失败保存原始证据并停下；未完成阶段的衍生图也登记为开发曝光。

本轮固定8个方法：整图global、区域与标题、原cross20、整图双上下文、整图加真实裁切，以及三种冻结学习融合。继续使用原18个家族隔离折模型，不用新增760条关系训练、挖负例或校准。旧商品沿用原查询折及商品家族折；新增19件通过既有来源家族与像素排除检查后，只在对应被验证查询折内对三个商品折模型作等权投票。主方法仍为0.65 RankNet排名与0.35旧/新base投票，不依据新成绩调整权重。

两组图库均从296件扩至315件，40条中文文本/图片查询不变，预计12600条关系、320份完整排序。所有方法使用同一图库与标签；315件结果只在本轮互比，不能与296件历史数字拼成成绩。相同方法仍保留完整图库Recall分母、未知null、零正例查询与K10。缓存重放时间不能当在线端到端验收时延。

执行前276项测试通过；64份原始真实特征逐条重放一致，136份旧精排文件、1734个唯一缓存身份通过一致性校验。这些是工程核验数量，不是新增检索成绩。特征协议、对照协议及接续源码清单已冻结，旧源码、标签与历史封存不变。

实际运行状态见[接续状态](../artifacts/module2_balanced_runner/public_status.json)，技术边界见[特征协议](../artifacts/module2_balanced_supplement_features/protocol.json)、[8方法对照协议](../artifacts/module2_balanced_gallery/protocol.json)与[接续源码清单](../artifacts/module2_balanced_runner/execution_source_manifest.json)。
<!-- /balanced_gallery_execution -->


<!-- balanced_gallery_finalized -->
## 扩库对照完成后的结论与后续

旧接续流水线已结束，不再等待评审或显存。独立收尾只修复报告格式，不重训模型、不重新生成神经分数、不改标签或排序。原154次新增完整Qwen嵌入、65次唯一新增完整精排调用等执行证据仍在原收据；此次收尾新增神经前向为0。159张衍生图与完整、失败阶段的曝光登记合并保留。

主方法未整体通过Precision硬目标、Recall偏好及不退化检查。B组手机壳、颈椎枕头和短靴存在AP局部损失；C组手机壳相对原cross20与整图加裁切方法的Precision、Recall、AP、NDCG存在明确损失，内胆包相对整图加裁切的AP、NDCG、MRR、Hit@1也有损失。完整范围、共享未知配对边界与其他阴性结果均保留在原内部指标文件，不能写成持续全面增长。

下一轮先做开发错例归因与覆盖检查，优先手机壳和内胆包；再冻结一条独立实验，改进主售品/道具、成品/配件与型号适配约束，并检验困难负例训练及局部特征融合。全图库、全查询、K10和六指标保持同条件比较，Precision@10≥60%为硬目标、Recall@10≥85%为偏好；未通过则继续开发，不提前打开新独立验收。最终只选择一个完整可执行方法，不能拼接各法最高数值。
<!-- /balanced_gallery_finalized -->
