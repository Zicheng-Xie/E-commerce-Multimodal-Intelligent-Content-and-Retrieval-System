# 模块二：新描述引导主体裁切与完整Qwen交互

更新：2026年10月24日，Asia/Shanghai。Precision@10≥60%为硬目标，Recall@10≥85%为偏好。

<!-- caption_roi_execution_state -->
本轮开发实验已完成并封存。255项测试以及本轮源码black、isort、flake8检查通过；15种方法、600份完整排序全部重放通过，父12种方法的480份排序和各范围单方法指标逐项不变，16条文本查询保持原cross20。

沿用两组296件图库、40条查询与11840条冻结AI关系；24条图片查询合计504对查询—商品候选，这是候选关系出现数，并非504件唯一商品。没有改旧标签、查询或K。完整DINO实际调用23次，21条查询采用ROI、3条全图回退；完整Qwen裁切精排实算498个唯一新pair，另精确复用6对输入相同的历史分数。63张native、zoom及框选衍生图已登记曝光；完成收据覆盖726个文件和76份源码快照。

主方法仍未在全部开发组通过Precision硬门槛，Recall偏好也未通过，六指标整体不退化未成立；其中C组相对上一轮双上下文方法及整图global基线的宏观MRR和Hit@1已有严格退化证据，不能概括为“仅跨零”。默认方法保持原版，正式指标为null，本轮未启动新的独立验收。

证据：[冻结协议](../artifacts/module2_caption_roi_development/protocol.json)、[推理前工程检查](../artifacts/module2_caption_roi_development/engineering_checks.pre.json)、[完成状态](../artifacts/module2_caption_roi_development/public_status.json)、[封存收据](../artifacts/module2_caption_roi_development/completion_receipt.json)。
<!-- /caption_roi_execution_state -->

## 本轮机制

前两轮的ROI只通过FG辅助分数参与排序。新实验直接让完整Qwen重排器比较查询主体裁切与商品主体裁切，再与完整整图上下文融合，验证主商品/展示道具、商品本体/配件和可见结构误排能否减少。

查询使用上一轮冻结的新主售描述，经公开粗类型映射生成检测提示。全部24张图片查询按固定流程处理，当前23条有支持的提示；商品名“女式高跟鞋”在既有类型解析中未确定family，保留全图回退，不增加针对该查询的例外。支持项用完整GroundingDINO，沿用原阈值、选盒、padding和zoom768配置。实际模型调用与接受/回退数按收据记录，不能称作固定24次检测。

图库沿用此前已封存的真实native/zoom裁切；无ROI则用原图，全部商品保留。查询新ROI保留native、zoom和检测框证据。局部放大改变模型关注范围，不声称增加原图没有的信息。Qwen使用实际选中的两张图片、原始商品标题与冻结新描述，完整模型、权重、业务指令、dtype、像素和长度配置不变。只有完整输入身份相同时才复用历史分数，其余候选真实推理，不能用整图或另一描述分数补缺。

## 固定对照

固定保留父实验12种方法，新增三种同候选方法；15种方法、600份完整排序已全部生成并重放核验：

- `crop_context_union`：主体裁切上下文。
- `whole_crop_context_union`：主方法，整图双上下文与裁切上下文各占一半CE投票。
- `whole_crop_roi_union`：再加入原描述时期的FG整图/ROI辅助投票，作为单独消融。

CE权重保持0.65，base排名0.35；辅助FG投票沿原0.25设置。所有新方法使用相同候选、相同公共尾部；16条文本查询返回原cross20。原标题与整图证据保留，裁切不能保证解决材质、用途、成人/童装等语义边界。不会根据本轮成绩改权重、删困难查询、改标签或替换默认方法。

## 开发结论与失败范围

以下只说明冻结AI标签上的开发结论，不公开开发质量数值。共享未知标签完成的差值下界大于零才表示明确改善，上界小于零才表示明确退化；含零的单侧边界与跨零边界分别保留。它们不是统计置信区间，也不能覆盖AI误标。

- A组：主方法总体、图片和文本通过Precision硬门槛，Recall偏好未通过。图片相对原cross20的Recall可证改善、Precision可证不降，AP和NDCG仍未证明不退化。相对上一轮双路方法及其ROI版，没有建立六指标整体优势。
- B组：总体和文本通过Precision硬门槛，图片未通过；Recall偏好未通过。主方法的靴子查询和杯子查询相对原cross20均有Precision、Recall严格负增量。靴子将已知相关的尖头细跟踝靴挤出前十、纳入已知不相关的低平跟系带踝靴；杯子纳入红色双耳敞口无盖杯。总体边界的不确定性不能抵消这些局部损失。
- C组：图片Precision硬门槛及Recall偏好均未通过。主方法相对上一轮`dual_context_union`及整图`global`基线的宏观MRR和Hit@1严格下降；相对原cross20则两项只可能持平或下降。杯子查询将已知不相关的红色双耳敞口无盖杯排到首位，AP、NDCG、MRR和Hit@1均明确退化。单耳耳机、内胆包等查询相对原cross20仍有局部收益，但这不能证明裁切主体语义正确，也不能与其他方法的最佳结果拼接成达标成绩。

裁切单路、整图加裁切、再加FG辅助三种消融未形成跨组稳定优势。新增FG辅助在C组救回杯子首位，同时又把内胆包首位换回道具手提包，不能称为无代价收益。纯裁切在A组厚底鞋上也出现明确的Precision、Recall损失。16条文本的原样保留不构成本轮新增收益，父轮结果与指标均未改写。

完整开发数值和配对边界仅保存在[内部完整报告](../artifacts/module2_caption_roi_development/internal_comparison.ai.development.md)；[目标决策](../artifacts/module2_caption_roi_development/decision.json)保留所有未通过项，[三方法商品及实际裁切对照](../artifacts/module2_caption_roi_development/internal_comparison.html)保留错误结果。

## 实际裁切视觉核查

完整DINO实际调用23次；24条图片查询中，21条接受ROI、3条全图回退。`accepted`仅表示通过冻结的置信度与几何规则，不表示主售主体语义正确。以下实际查看了5条查询的13张原图、框选图和zoom图，是定性抽查，不能据此计算全部查询的主体定位准确率。

- `tune/image_01`（游戏耳麦）：粗类型与`headphones.`提示正确，框基本保留耳机和麦克风；但范围接近整图，广告文字仍在，线缆末端被裁切，聚焦作用有限。证据：[原图](../artifacts/module2_structure_development/media/35ee498fc6b5adb09172398aad520dfb41b2f69da66761ea87a08708c03facf6.webp)、[框选](../artifacts/module2_caption_roi_development/query_rois/tune/image_01/box.png)、[实际zoom](../artifacts/module2_caption_roi_development/query_rois/tune/image_01/zoom.png)。
- `fresh/image_03`（内胆包）：粗类型与`bag organizer.`提示正确，但最高分框选中左下手提包展示道具，右上棕色内胆包主商品完全在框外；实际zoom只保留手提包，属于明显语义错框。证据：[原图](../artifacts/module2_primary_form_coverage/fresh/media/09f7ec2b835d9206b716eba3dcbeb2f4d6d207dece00647d4e5712d4b964454c.webp)、[框选](../artifacts/module2_caption_roi_development/query_rois/fresh/image_03/box.png)、[实际zoom](../artifacts/module2_caption_roi_development/query_rois/fresh/image_03/zoom.png)。
- `tune/image_02`（手机壳）：粗类型与框选主体正确，摄像头开孔、圆形支架和蓝色边缘完整，右侧广告文字被排除。证据：[原图](../artifacts/module2_structure_development/media/717e47337f06a6dffb30009ef836ba9107c79f1937e967edeed42f8c38f068e0.webp)、[框选](../artifacts/module2_caption_roi_development/query_rois/tune/image_02/box.png)、[实际zoom](../artifacts/module2_caption_roi_development/query_rois/tune/image_02/zoom.png)。
- `regression/image_02`（手机壳）：选中左侧粉色手机壳，粗主体正确；右侧紫色展示款被排除，裁切内仍有较多手部。单个最高分框不能覆盖多款展示的信息。证据：[原图](../artifacts/module2_structure_development/media/8cfa24efcefd16df613c8fc7b58a77b4f294adbefb8590eb5ec5102415cdfa42.webp)、[框选](../artifacts/module2_caption_roi_development/query_rois/regression/image_02/box.png)、[实际zoom](../artifacts/module2_caption_roi_development/query_rois/regression/image_02/zoom.png)。
- `fresh/image_02`（手机壳）：实际使用全图回退，没有生成框选图或zoom。检测到了几乎覆盖全图的手机壳框，但按原8% padding扩张后触及四边，超过既定`max_padded_fraction=0.98`限制而被拒绝；全图仍保留主商品和广告文字。证据：[实际回退原图](../artifacts/module2_primary_form_coverage/fresh/media/a22ee57a5a70714ffac3892398132c2ec6d6c8939dcbf14cd6971ea29c24064e.webp)、[原始检测与几何决定](../artifacts/module2_caption_roi_development/query_rois/fresh/image_02/result.json)。

本次核查没有重框、重跑、改变策略或修改标签；内胆包错框及其他局限原样保留在本轮实验中。定性核查不构成检索质量改善或独立验收结论。

## 数据隔离与证据

全部数据均已开发曝光，人工标签0。单方法六指标沿原定义，逐查询配对评估沿上一轮已核验的共享未知标签方法。Recall仍使用全图库正例分母，零正例未定义；当前泛文本和总体Recall容量不足85%的约束单独保留，不改指标来追目标。所有原方法、失败和回退均进入同图库对照，正式指标保持为空。

2026年10月23日已有独立AI盲测及其分模态验收未通过的历史事实保持不变，见[独立盲测交付](module2_independent_blind.md)。本轮未启动新的独立验收，不能将当前开发结果替代该历史结论。

GPU只在显存自然空闲且Ollama无任何驻留模型时启用，不卸载其他模型，不修改其keep-alive。素材准备源码和协议已保留；安全执行入口另行固定超时、异常与驻留列表校验，模型、提示和检测参数不因此改变。

63张新增裁切及检测框图片已登记字节/像素/dHash曝光，并继承全部历史排除清单。验收必须在最终方案冻结后使用全新图库和全新查询，不能复用本轮数据。源码与封存结果保持只读，后续改进使用新的开发目录与协议。

运行证据：[主体检测收据](../artifacts/module2_caption_roi_development/detection_receipt.json)、[完整裁切精排及缓存收据](../artifacts/module2_caption_roi_development/reranker_receipt.json)、[600份排序重放验证](../artifacts/module2_caption_roi_development/verification.json)、[页面及图片引用检查](../artifacts/module2_caption_roi_development/checks.json)、[新增衍生图曝光登记](../artifacts/module2_caption_roi_development/exposure_receipt.json)。2400张结果卡的本地图片引用和静态JS语法已检查；未声称浏览器交互或在线时延验收完成。

只读核验命令，不重新推理或修改封存文件：

```powershell
.\.venv312\Scripts\python.exe scripts\caption_roi_experiment.py verify
```

## 下一步

下一轮先固定仅依赖查询原图及查询描述的ROI语义核验与整图保护方案，检查选框是否保留主商品及必要结构，无法确认时明确保留整图。新方案不得用相关性标签、查询ID特例或候选排名选框；先冻结规则、配置和消融，再在新目录开展实验。本轮错框、局部损失和全部旧结果原样保留。该后续方案尚未实现或验证，不能提前称为已修复或已达标。
