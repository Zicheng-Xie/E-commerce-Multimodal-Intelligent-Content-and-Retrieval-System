# 模块二：查询主体语义核验与首位保护开发对照

更新：2026年10月24日，Asia/Shanghai。Precision@10≥60%为硬目标，Recall@10≥85%为偏好。

<!-- roi_semantic_gate_state -->
本轮已完成并封存。260项测试及本轮源码black、isort、flake8通过；19种方法、760份完整排序全部重放通过。父15种方法的600份排序、各范围指标及16条文本原cross20排序不变，完成收据覆盖157个文件与75份源码快照，封存后只读核验通过。

完成21条有效原生ROI语义判断，另3条继承全图回退；模型19条许可、1条拒绝、1条uncertain，共21次原生请求、0次工程失败。原11840条AI相关性关系不变，新增AI相关性标签0、人工标签0；本轮没有新检索模型调用或新像素，沿用全部开发数据和曝光排除清单。

主方法未全组通过Precision硬门槛，Recall偏好未过。A/B相对父裁切方法没有新增Precision、Recall收益；C组相对父裁切方法的Precision、Recall、AP和NDCG均严格下降。Hit@1与原cross20相同来自固定首位动作，不能视为语义识别突破。默认方法仍为原版，正式指标为null；本轮未启动新的独立验收。

证据：[冻结主协议](../artifacts/module2_roi_semantic_gate/protocol.json)、[query-only核验协议](../artifacts/module2_roi_semantic_gate/gate_review_protocol.json)、[原生语义核验收据](../artifacts/module2_roi_semantic_gate/gate_receipt.json)、[260项工程检查](../artifacts/module2_roi_semantic_gate/engineering_checks.pre.json)、[完成状态](../artifacts/module2_roi_semantic_gate/public_status.json)、[封存收据](../artifacts/module2_roi_semantic_gate/completion_receipt.json)。
<!-- /roi_semantic_gate_state -->

## 验证的问题

上一轮完整Qwen裁切交互存在主售内胆包被展示手提包替代的错框，也有杯子首位退化。本轮分别验证查询ROI语义核验与固定原首位的影响，避免将首位保护带来的变化归因于裁切识别。

21条已采用ROI的图片查询交给完整本地Qwen3.5模型，按固定顺序输入原图、实际zoom图和此前冻结的主售描述。3条原全图回退查询直接保留回退，全部24条图片与16条文本查询不变。不输入商品候选、原标题、相关性标签、排名或质量指标，也不根据核验结果重新生成描述或重选框。

模型需要判断同一主售商品、商品保留、可见关键结构保留、无主导道具四项，并给出accept、reject或uncertain。只有有效完整输出明确accept且四项均为true时，才允许裁切路线；明确reject或uncertain使用整图路线。网络、超时、截断或无效输出保留为工程未完成，不能当作模型拒绝来计算检索指标。原图证据优先，描述与图像冲突时保留不确定。

查询审核无法证明图库商品ROI正确；沿用同一模型生成的描述也可能引入自一致偏误。所有AI判断保留来源与原生输出，人工标签仍为0。

## 实际原生核验与解释错误

实际完成21条有效原生模型核验，另外3条继承全图回退；共21次请求、0次工程失败。模型给出19条裁切许可、1条拒绝、1条uncertain。19条许可只是模型路由判断，未经过人工语义验证，不能称为ROI定位准确率；本轮也没有核查图库ROI的语义正确性。数量与逐查询判断可见[核验收据](../artifacts/module2_roi_semantic_gate/gate_receipt.json)及[原始gate结果](../artifacts/module2_roi_semantic_gate/gate_results.json)。

`fresh/image_03`的原生判定为`reject`，触发冻结的整图回退，但其解释颠倒了主售角色和第2张图片中的对象。原始解释为：

> 原图主要售卖商品为LV Speedy Nano 枕头包，第2张图片裁切的是该包的专属内胆包。两者不是同一个主售商品。

实际看图可见，原图写有“专属内胆包”，右上为棕色内胆包，左下手提包用于展示适配关系；第2张真实zoom只包含左下手提包，内胆包被裁掉。证据：[真正原图](../artifacts/module2_primary_form_coverage/fresh/media/09f7ec2b835d9206b716eba3dcbeb2f4d6d207dece00647d4e5712d4b964454c.webp)、[实际zoom](../artifacts/module2_caption_roi_development/query_rois/fresh/image_03/zoom.png)、[原生输出与输入记录](../artifacts/module2_roi_semantic_gate/native_reviews/fresh/image_03.json)、[首次原生请求记录](../artifacts/module2_roi_semantic_gate/native_attempts/fresh/image_03/attempt_001.json)。

该条四项原生判断如下，保持模型输出原样：

| 核验项 | 原生值 |
|---|---|
| `same_primary_product` | `false` |
| `primary_product_retained` | `false` |
| `essential_visible_structure_retained` | `false` |
| `no_dominant_distractor` | `true` |

已重新核对实际发送顺序为原图在先、真实zoom在后，两张发送PNG的哈希均可从冻结源图重构并与原生请求记录一致，没有交换输入。此次拒绝动作使错误裁切回退，但解释以及“无主导道具”的判断不能据此视为正确，不能声称已经成功识别真正内胆包。该条只请求一次；没有重试或修改任何判定、检测框、策略或标签。

## 冻结的同图库方法

沿用两组296件图库、40条查询、11840条原AI关系；24条图片查询合计504对查询—商品候选，并非504件唯一商品。父实验15方法完整保留，新增以下4条开发路线，19方法、760份完整排序已全部完成并重放：

- `semantic_crop_gate`：通过审核使用父轮`whole_crop_context_union`；其余使用上一轮完整整图双上下文`dual_context_union`，不带FG辅助。
- `semantic_crop_gate_head1`：主方法，在语义路由上固定原cross20首位。
- `unverified_crop_head1`：不审核ROI的整图裁切路线，加相同首位保护。
- `whole_context_head1`：完整整图双上下文，加相同首位保护，作为回退控制。

语义核验与首位保护的四角对照包括父轮原`whole_crop_context_union`，该角精确复放，不称为新推理或独立算法。核验失败退路、方法名称和主方法在评审前固定，不能事后选择有利退路。

首位保护只把原cross20首件在共同候选池内稳定前移，其余顺序保持。它机械保留原Hit@1对应候选，但不能保证Precision、Recall、mAP、NDCG或MRR均不退化；所有方法按相同未知标签评估，不隐藏局部损失。展示排序值包含保护动作，不是模型原始logit或匹配概率。

16条文本查询完全保留原cross20。候选外尾部保持cross20相对顺序，K=10、完整图库Recall分母、未知null及零正例查询均不改变。当前总体Recall的容量约束仍保留，不能靠更换口径达标。

## 开发对照与明确损失

以下为共享未知标签差值界下的定性开发结论，完整数值只保存在内部报告。差值上界小于零才称明确退化，含零的单侧边界与跨零边界分别保留；这些边界不是统计置信区间，也不覆盖AI误判。

- A组：主方法通过各范围Precision门槛，但Recall偏好未通过。相对父`whole_crop_context_union`，Precision、Recall完全相同，没有新的召回或精度收益；相对原cross20的已有局部收益不能改记为本轮语义门控的成果。
- B组：总体和文本通过Precision门槛，图片未通过，Recall偏好未通过。主方法相对父裁切方法的Precision、Recall完全相同，原靴子、杯子查询相对cross20的Precision、Recall严格损失仍在。首位保护还可能把原本靠前的相关结果后移，不能保证MRR或其他指标不退化。
- C组：Precision门槛和Recall偏好均未通过。相对父裁切方法，宏观Precision、Recall、AP、NDCG四项均严格下降；门控本身已使Precision、Recall、NDCG下降，固定首位又使AP、NDCG进一步下降。单耳耳机的uncertain回退丢失相关命中；原全图回退手机壳的排序损失来自固定退路，不能归因于模型判定。

本轮三种首位控制均未改变任何查询的Top10商品集合，因此它们相对各自无首位保护路线的Precision、Recall相同。主方法Hit@1与cross20完全相同是机械保留原首件；这不表示模型识别正确，也不是可泛化的质量提升。C组内胆包在整图退路已排对首位，主方法又强制放回cross20的错误手提包，AP、NDCG、MRR、Hit@1局部严格退化；杯子首位则得到相反方向的机械修复，不能只保留有利案例。

内胆包的reject动作与解释正确性必须分开，原生解释颠倒事实的细节保留在上方核查小节。原15方法及失败均未改写，不把不同方法、不同组的最佳数值拼成达标结果。本轮语义门控或首位保护没有形成目标突破。

完整证据：[内部开发报告](../artifacts/module2_roi_semantic_gate/internal_comparison.ai.development.md)、[目标决策](../artifacts/module2_roi_semantic_gate/decision.json)、[原版/未核验裁切/主方法商品对照](../artifacts/module2_roi_semantic_gate/internal_comparison.html)、[760份完整排序重放](../artifacts/module2_roi_semantic_gate/verification.json)。

## 执行与隔离

所有新源码、模型身份、评审提示、参数、输入和比较在新评审前冻结。只在显存自然低于1500MiB、Ollama有效驻留列表明确为空时加载审核模型；不卸载或打断用户作业，不缩短用户模型的keep-alive。完整检索分数复用历史实算记录，不用模拟、缺分填充或简化模型。

本轮不下载新图片、不生成拼图或新ROI像素；直接审核封存原图和zoom。曝光排除清单继承上一轮全部63张新衍生图及更早历史。本轮所有数据已开发曝光，后续独立验收仍需要最终方案冻结后的全新查询与全新图库。2026年10月23日已有AI盲测及未通过结论保持不变。

本轮目录为`artifacts/module2_roi_semantic_gate/`，源码与产物保持只读。曝光排除继承父轮63张衍生图及所有更早记录，新增原图、衍生像素和商品源元数据均为0，见[继承曝光收据](../artifacts/module2_roi_semantic_gate/exposure_receipt.json)及[完整排除快照](../artifacts/module2_roi_semantic_gate/post_experiment_exclusion_snapshot.json)。[页面检查](../artifacts/module2_roi_semantic_gate/checks.json)覆盖2400张结果卡、图片引用和静态JS语法，不代表浏览器交互或在线时延验收。

只读核验命令，不重新请求模型或改写封存文件：

```powershell
.\.venv312\Scripts\python.exe scripts\roi_semantic_gate_experiment.py verify
```

## 下一步

CPU容量诊断已完成，证据见[内部候选容量诊断](../artifacts/module2_pool_capacity_diagnosis/capacity.json)。在当前冻结AI标签下，原/新base60候选并集已达到全部296件图库逐查询的Top10已知正例容量，保留旧pool形成严格超集后仍有同样容量。相较旧20路并集，A/B没有已知容量增量，只有C组少数查询有少量空间。B组已知正例容量接近Precision目标所需下限，仍要求排序接近池内理想上界；继续扩大同一296图库的候选不能解决其已知覆盖与精排瓶颈。未知关系仍未定，以上是已知正例在理想排序下的容量诊断，不是新模型实测成绩。

下一步依据覆盖缺口、未决关系的证据完整性与池内排序瓶颈设计新数据及排序实验。扩候选只作为有限、可检验的完整模型消融；单独的60路并集会遗漏部分旧候选，未来必须保留旧pool构成严格超集，不能用换池掩盖损失。新实验先冻结数据、候选规则、完整模型配置和比较协议，再记录实算及全部结果；当前尚未实现该新方案或新增推理，不能承诺扩到60即可达标。语义门控和首位保护保留为已完成的对照，不称为新的突破。
