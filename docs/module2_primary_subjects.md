<!-- grounded_subject_latest -->
最近同步：2026-10-25T01:29:15.084265+08:00。

主售定位修复开发对照已完成：22方法880份排序，原19方法760份排序保持；全量重放和封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，其他指标不退化尚未整体证明。

独立AI盲测已完成：96件全新图库、12条全新图片查询及12条中文文本查询，2304条冻结AI关系，单一候选与三个基线共96份排序重放封存。独立AI Precision目标未整体通过，Recall偏好通过。其他指标不退化尚未整体证明。默认原版保留，人工标签0，formal_metrics=null；开发和独立盲测分开报告，盲测材料一经打开即登记消耗，不能用于调参后重测或复用为新验收。 见[本轮进度与证据](module2_grounded_subject_repair.md)。
<!-- /grounded_subject_latest -->

<!-- latest_subject_learning_followup -->
最近同步：2026-10-24T22:40:06.223857+08:00。

主售属性与困难负例重训已完成：11方法440份排序，原8方法320份排序及全部12600条AI关系保持不变；新主方法未整体通过Precision目标、Recall偏好和不退化检查。全图库RankNet提议与完整精排的14方法560份排序也已封存，仍未整体通过目标。类别平衡全负例损失的17方法680份排序随后完成；主类型系数转为正值，但图片Precision目标仍未整体通过。

公开主售冲突优先级对照已完成：19方法760份排序，原17方法680份排序不变，完整重放与封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，六指标不退化未整体通过。全部为已曝光AI开发实验，人工标签0；原版默认保留，正式指标null，新的独立验收未启动。

本页保留最初296件图库实验的历史证据；315件图库及后续学习对照见[module2_learned_fusion](module2_learned_fusion.md)，不同图库成绩不混合。
<!-- /latest_subject_learning_followup -->

# 模块二：精确主体补库与主售商品识别对照

更新：2026年10月24日，Asia/Shanghai。目标保持Precision@10≥60%，Recall@10尽量≥85%。

<!-- primary_subject_execution_state -->
本轮已完成103件新增商品补库、40条开发查询、8种方法共320份完整排序，以及指标计算、共享未知标签增量、重放和结果页核验。主方法尚未在全部开发组通过Precision门槛，Recall偏好也未通过；六指标整体不退化尚未得到证明。当前宏观配对区间跨零，不能写成已证明整体退化。默认方法保留原版，正式指标为空，独立验收未启动。
<!-- /primary_subject_execution_state -->

## 已完成的运行与结论

完整Qwen嵌入实际858次；24张查询图生成新描述，并各自重新编码caption。完整Qwen精排新算775对，输入身份完全一致复用505对，共1280对。完整GroundingDINO调用99次，采用99个ROI、回退4件；完整FG视觉前向404次、文本前向103次。

246项测试通过；320份完整排序已重放，11840条AI关系核验通过，原7720条保留。结果页包含40条查询、3种主对照方法及2400张商品卡，JavaScript语法与本地图片路径通过静态检查；没有宣称已完成浏览器交互或在线时延验收。封存2478个文件、12份源码快照；封存后接续入口只读验证通过。本轮等待显存的心跳在完成后暂停。

主方法在A组通过开发Precision门槛；B组图片和后8条图片查询未通过，Recall偏好各范围未通过。文本输出与原方法一致。图片范围的六指标配对界跨零，尚不能证明整体不退化，也不能反过来写成已经证明宏观退化。具体分组与全部方法的真实数值仅见内部报告，不将局部最高成绩当成整体或正式验收结果。

新描述有内胆包等个例改善，也有鞋、柜、杯等查询的命中流失；整体净收益未被证明。公开冲突扣分改变部分排序，但没有增加图片查询的已知Top10命中。一个保留的失败例是户外鞋查询把儿童涉水软鞋推到首位：部分已识别类型被扣分，而未识别类型弃权，出现相对抬升。没有在看到结果后修规则、重跑追分或改标签。

ROI独立消融在后8条图片查询的Precision、Recall和NDCG上有正增量证据，MRR与Hit不变，但mAP增量仍跨零，未通过全部六指标门槛。该结论仅在本轮冻结AI标签及未知项的共同取值范围内成立，不覆盖AI误判，也不能替代独立验收。

[查询描述输入审计](../artifacts/module2_primary_subjects/query_caption_input_audit.json)核实了冻结原图、统一提示及24份完整输出，不代表描述语义全对。原始输出仍有材质/内容物推断、粗细跟冲突等局限，全部保留；证据来自冻结源码和保存的输入/输出，不声称抓取了HTTP请求报文。单方法和配对区间不覆盖AI误判，人工标签仍为0。

复核入口：[完整内部对照表](../artifacts/module2_primary_subjects/internal_comparison.ai.development.md)、[逐查询商品查看页](../artifacts/module2_primary_subjects/internal_comparison.html)、[运行审计](../artifacts/module2_primary_subjects/runtime_audit.json)、[判定](../artifacts/module2_primary_subjects/decision.json)、[最终核验](../artifacts/module2_primary_subjects/checks.json)、[封存收据](../artifacts/module2_primary_subjects/completion_receipt.json)。

## 数据与评审

从只读商品来源第2,500,001至3,500,000条，按固定主体规则、seed 2026102403及历史商品族排除清单冻结298条候选。预设14类中，门板加抽屉餐边柜仅10条、拱形竹木颈枕0条，保留真实不足；其他12类各24条。所有候选均按原URL尝试一次，103张成功、195条失败，其中191条HTTP 420、4条404。没有绕过源站拒绝、修改访问方式或按检索结果替换样本。

103张技术成功图片无历史或批内近重复，全部保留。已直接查看13张商品联系表、原查询参考图及边界商品原图，完成103份主体和可见结构描述、对全部40条既有查询的4120条新关系：195条正例、3779条负例、146条未知。原7720条关系原样保留，总11840条。全部明确标注AI来源，人工标签0；评审者知道此前开发背景，没有看本轮新排名，不称独立人工或双盲评审。

部分候选仍为后挂骨传导、双耳耳塞、数码收纳板、涉水袜鞋等，与采集词面类别不同，保留其真实主体作为干扰商品。羽绒棉、马甲与有袖服装、柜门布局、杯盖配置和跑步用途等证据不足的关系继续为null。旧标签存在的年龄或款型边界不一致保留并披露，未用年龄硬扣分制造提升。

两组开发图库已各由193件扩至296件。A、B各16条查询，后8张历史命名fresh的图片查询共享A图库；40条查询均已曝光，全部属于开发。

## 冻结实验

比较8种方法：整图、区域与标题、原Qwen、原描述全模态FG融合、原描述图片FG融合、原描述图片FG融合加ROI、新主售描述Qwen，以及新主售描述Qwen加公开主体冲突扣分。总计40×8=320份完整排序，固定K=10，不删困难或零正例查询。

新描述统一对全部24张图片查询处理，仅提供查询原图，识别主要售卖商品、可见文字及展示道具，不提供图库、标签或排名。明确不确定时弃权。描述发生变化的caption向量与完整Qwen精排配对重新计算，仅在模型、查询上下文、原标题和图片身份一致时复用分数。

新主方法与新描述消融使用完全相同的20候选、原始精排分数和完整尾部，仅增加公开主类型/配件冲突、明确互斥子型或可见结构的扣分；不叠加、不奖励、不对未知扣分。连接方式与耳机形态分开，否定结构不作为正证据，泛称外套或学饮杯不擅自推断有袖或吸管。该主方法使用Qwen，没有与ROI组合；原ROI方法作为独立参考保留。所有图片专用方法对16条文本查询保持本轮原方法输出。

分别报告新描述对原方法、冲突扣分对新描述、ROI对图片融合的增量。单方法未知标签区间按全部补全计算；配对未知过多时使用明确标注的保守界，不能称置信区间或精确差值。Recall使用完整图库正例数；正例超过10时，单查询Top-10的Recall容量为10/R。补库前后分母变化单独展示，算法收益由同一296件图库的比较判断。

## 运行与接续

复用完整固定版本Qwen3-VL-Embedding-2B、Qwen3-VL-Reranker-2B、FG-CLIP 2 base、GroundingDINO tiny及本地固定量化视觉模型。模型依次加载，不下载新权重或改依赖。用户已选择等待其作业自然释放显存；同名Qwen驻留也需要等待，不能借用后缩短其keep-alive。

接续入口会先检查源文件哈希与阶段收据，取得位于项目tmp目录的并发锁；GPU忙时安全返回等待，已经完成的阶段不重跑。只有显存低于1500 MiB且Ollama没有任何驻留模型时才尝试模型阶段，模型阶段还会再次检查。封存后只做只读核验：

```powershell
.\.venv312\Scripts\python.exe scripts\continue_primary_subjects.py run
```

模型阶段依次为：安全查询描述、完整新增图及caption编码、主体检测、FG特征、同图库原/新描述精排、8方法排序、六指标与增量、重放、结果页、衍生曝光登记和封存。实际调用数以完成后的收据和运行审计为准，不能把计划调用数当作实测。

核验入口：

```powershell
.\.venv312\Scripts\python.exe scripts\check_primary_subject_inputs.py
.\.venv312\Scripts\python.exe scripts\primary_subject_experiment.py verify
.\.venv312\Scripts\python.exe scripts\check_primary_subject_results.py
```

第二、三条已在完整模型阶段完成后执行并通过，封存后应使用接续入口做只读核验。全部等待、错误和断点保留。源配置验证和模型检查通过不等于检索质量达标；内部对照表与商品查看页已生成于`artifacts/module2_primary_subjects/`。

## 开发与独立验收

当前最新排除清单为[最终曝光快照](../artifacts/module2_primary_subjects/post_experiment_exclusion_snapshot.json)，已包含全部298条元数据、103张采集图、826张新增裁切/局部放大/ROI衍生图及历史曝光，均排除于未来独立验收。

本轮11840条关系、296件图库和40条查询均不能作为独立验收。默认方法保持原版，正式指标为空。方案及评审协议冻结后，验收必须使用全新图库和全新查询，并检查商品族、字节/像素重复和dHash近重复。未通过完整对照与独立验证，不能宣称Precision或Recall目标已实现。

资料入口：[采集收据](../artifacts/module2_primary_subjects/acquisition_receipt.json)、[原生AI评审记录](../artifacts/module2_primary_subjects/native/review_provenance.json)、[标签冻结收据](../artifacts/module2_primary_subjects/label_receipt.json)、[输入核验](../artifacts/module2_primary_subjects/pre_inference_checks.json)、[工程检查](../artifacts/module2_primary_subjects/engineering_checks.pre.json)、[运行状态](../artifacts/module2_primary_subjects/continuation_status.json)、[开发容量审计（不是实测成绩）](../artifacts/module2_primary_subjects/native/capacity_before_inference.json)。
