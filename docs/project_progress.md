<!-- p80_r90_latest -->

**2026-11-12 图库扩容已启动：** 在原 845 件电商商品基础上，按批自动编码本地 191,961 张 DeepFashion2 训练图片；已就绪的分片即时加入图搜／文搜，网页显示真实可检索数量和待处理进度。支持断点续跑，验证集、FashionAI test_a 与最终验收集保持隔离；未宣称扩容后质量指标达标。[扩容说明与实时进度](docs/module2_fashion_expansion_20261112.md)。

**可演示窗口已上线：** [打开本机商品检索平台](http://127.0.0.1:8766)，支持文字输入、图片上传、36 张图搜示例、完整 845 件开发商品浏览及原图详情。已实际验证图搜和文搜；使用原冻结排序规则，独立验收未启动。下次双击项目根目录的 `启动商品检索平台.command`。[使用说明与验证证据](docs/module2_showcase.md)。
**2026-11-10 模块二最新进展：** 已将已验证的 `primary_public_conflict_matched_v1` 工程化为独立通用 CLI（`index`、`predict`、`freeze`、`acceptance-preflight`、`acceptance-run`），保留现有服务默认实现。845 件商品、55 条完整排序的开发缓存重放全部一致；最终代码在空缓存下完成 24 件开发商品新建索引和 4 条图搜／文搜真实推理，约 203 秒建索引、64 秒推理。工程权重、Ollama digest、源码、配置与索引已绑定冻结。[工程交付、运行方法与验收流程](docs/module2_selected_runtime.md)。

| 此前完整人工开发 P@10 | 原主线 | 已验证候选 |
|---|---:|---:|
| 总体 | 95.09% | 95.64% |
| 图搜 | 93.33% | 93.89% |
| 文搜 | 98.42% | 98.95% |

以上仍是开发集成绩；新冷启动测试没有独立质量成绩。固定 Top10 的理想召回容量上限仍低于 90%，不宣称已经达标。保留 P@10≥80%、R@10≥90% 和原相关性定义。[原开发指标与容量证据](docs/module2_generalization_20261110.md)。

**独立验收未运行：** 30 条用户提交的独立文本查询保留。此前 300 件商品、60 条查询生成的 18,000 对是验收子集的完整交叉关系，不是抽检；600 批全量人工作业已撤回，当前没有新增人工任务。开发优先复用已有人审记录，自动完成模型推理、候选筛选、优化对照及回归；后续人工每轮默认不超过 60 对。抽样评测方案尚未放行执行，保留原指标目标和封存数据隔离。[当前工作量安排](artifacts/module2_acceptance_gold_20261110/人工工作量调整.md)。八个历史数据路径继续标为已消耗，18 个旧命令继续受阻。

**自动开发对照已完成：** 复用 820 条人工关系，完成家族隔离成对学习、主体描述清理和 40 对真实精排；学习排序存在确定退步，已淘汰。主体类型约束仅交换 1 对，总体 P@10 为 95.64%–95.82%，换入项尚无人审，不能宣称净增；原冻结默认继续保留。本轮没有新增人工任务、未使用最终保留集。[本轮方法、同标签结果与限制](docs/module2_automatic_development_20261110.md)。

458 个 Python 文件规范必检项通过，450 项测试全部通过；原冻结主线、候选和开发快照保留。下面是历史阶段归档，各段“当前”“最新”和待办均指当时。现在的进度以本段及工程说明为准。
<!-- /p80_r90_latest -->

<!-- simple_baseline_latest -->
**2026-11-05 当前进度：** 主图裁切与统一 Top20 完整精排的三组开发对照已完成，72份排序、480对真实精排分数及846个特征向量重放核对通过，六份索引已导出。411件唯一商品全部处理，361张主图裁切、50张保留原图；24条图片查询保持原图。Precision门槛未在全部开发组通过，fresh组MRR和首位命中率相对裁切召回明确退步，默认策略保留。用户已确认亲自完成20%人工复核（2,520条），不能沿用历史“全部由AI评审”安排替代本次人工判断。两轮已用盲测禁止调参复用；新冻结300件图库与30条图片查询，模型预测0，待人工真值和最终模型冻结后验收。383个Python文件规范必检项、349项回归测试通过。结果、证据与操作入口见[本轮说明](docs/module2_simple_baseline.md)。以下保留历史阶段记录。
<!-- /simple_baseline_latest -->

<!-- role_pool_latest -->
模块二主售角色与候选扩池开发实验。四组归因对照及4组参照已完成，共320份完整排序。开发Precision硬目标未整体通过；Recall偏好未整体通过；其他指标不退化尚未整体证明。共享未知标签成对对照确认：角色增强主方法在fresh开发组的Precision、Recall、mAP和NDCG均严格退步，本轮不采用该增强分支。单独扩池也未取得稳定收益。同图库、同40条查询、同12600条AI关系、K10；原池全部保留，扩池固定加入双图注意力前20条；主售不确定回退真实原分数。新掩码全局停用。开发数据与独立验收严格隔离，历史盲测保持封存；本轮未新增独立盲测，默认原版、人工标签0、formal_metrics=null。 见[本轮进度](docs/module2_role_pool_development.md)。
<!-- /role_pool_latest -->

<!-- business_grounding_latest -->
模块二新一轮开发已完成435张原图的数据审计、66条商品原生AI质检、完整GroundingDINO与SAM 2主体视图生成、24条图片查询的视图QA及9组家族隔离训练输入准备。原40条查询和12,600条AI关系保持不变。自动主售/结构核验已完成（435/435条）；整图业务度量学习组件对照已完成，内部组件结果不代表完整系统或独立验收。双图注意力业务适配及固定融合已完成：7方法280份排序，18个家族折模型重放通过。开发Precision硬目标尚未整体通过。Recall偏好尚未整体通过。其余指标不退化尚未整体证明。新掩码因主体结构质量检查未过全局停用。默认原版、人工标签0、formal_metrics=null；历史盲测继续封存，本轮未开启新的独立盲测。 见[本轮开发进度](docs/module2_business_grounding_development.md)。
<!-- /business_grounding_latest -->

<!-- grounded_subject_latest -->
最近同步：2026-10-25T01:29:15.084265+08:00。

主售定位修复开发对照已完成：22方法880份排序，原19方法760份排序保持；全量重放和封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，其他指标不退化尚未整体证明。

独立AI盲测已完成：96件全新图库、12条全新图片查询及12条中文文本查询，2304条冻结AI关系，单一候选与三个基线共96份排序重放封存。独立AI Precision目标未整体通过，Recall偏好通过。其他指标不退化尚未整体证明。默认原版保留，人工标签0，formal_metrics=null；开发和独立盲测分开报告，盲测材料一经打开即登记消耗，不能用于调参后重测或复用为新验收。 见[本轮进度与证据](docs/module2_grounded_subject_repair.md)。
<!-- /grounded_subject_latest -->

<!-- module2_subject_learning -->
最近同步：2026-10-24T22:40:06.223857+08:00。

主售属性与困难负例重训已完成：11方法440份排序，原8方法320份排序及全部12600条AI关系保持不变；新主方法未整体通过Precision目标、Recall偏好和不退化检查。全图库RankNet提议与完整精排的14方法560份排序也已封存，仍未整体通过目标。类别平衡全负例损失的17方法680份排序随后完成；主类型系数转为正值，但图片Precision目标仍未整体通过。

公开主售冲突优先级对照已完成：19方法760份排序，原17方法680份排序不变，完整重放与封存通过。开发Precision硬目标未整体通过，Recall偏好未整体通过，六指标不退化未整体通过。全部为已曝光AI开发实验，人工标签0；原版默认保留，正式指标null，新的独立验收未启动。 见[进度与证据](docs/module2_learned_fusion.md)。
<!-- /module2_subject_learning -->

<!-- module2_learned_fusion -->
2026年10月24日：19件商品已完成532次原生AI评审，新增760条关系冻结；完整模型编码和315件同图库8方法、320份排序对照均已完成。报告导出曾因误用macro_ranges字段停下，现已在独立收尾目录按真实ranges格式修复，278项测试与全部排序、指标重放通过；原失败证据、源码、标签、排序和参数未覆盖。主方法仍未整体通过Precision硬目标、Recall偏好及六指标不退化检查，默认原版、正式指标null，本轮未启动新的独立验收。 见[完整进度与证据](docs/module2_learned_fusion.md)。
<!-- /module2_learned_fusion -->

<!-- module2_roi_semantic_gate -->
2026年10月24日：查询ROI语义核验与原首位保护开发对照已完成封存。260项测试、19方法760份完整排序重放通过，原15方法600份排序及16条文本保留。完成21条AI核验及3条原全图回退，19许可、1拒绝、1uncertain；21次原生请求、0工程失败。原11840条相关性关系不变，无新相关性标签或检索推理、无新像素，封存157个文件与75份源码快照。

目标仍未整体达到。A/B相对父裁切方法无新增Precision/Recall收益，C组Precision、Recall、AP、NDCG严格下降；Hit@1相同仅来自机械首位保护，内胆包拒绝理由还颠倒了主售与道具。默认原版、正式指标null、人工标签0，本轮未启动新的独立验收。CPU容量诊断已完成，扩候选的已知容量增量有限；下一步针对覆盖、未知关系及排序瓶颈设计实验，扩池仅作保留旧候选的有限消融，尚未实现或新增推理。见[完成记录、失败与真实证据入口](docs/module2_roi_semantic_gate.md)。
<!-- /module2_roi_semantic_gate -->

<!-- module2_caption_roi_development -->
2026年10月24日：主体裁切与完整Qwen开发对照已完成封存。255项测试、15方法600份完整排序重放通过，父12方法480份排序原样保留。实际23次DINO检测、21条ROI与3条全图回退，新增498对完整裁切精排、复用6对；63张衍生图登记曝光，封存726个文件与76份源码快照。

主方法仍未全组通过Precision硬门槛，Recall偏好未过。C组相对上一轮双路方法及整图global基线的宏观MRR、Hit@1明确退化；B组靴子、杯子及C组杯子也有明确局部损失。正式指标为null、默认原版，本轮未启动新的独立验收。下一步拟先冻结query-only的ROI语义核验与整图保护，再做新实验，尚未实现。见[本轮完成记录、失败范围与只读核验](docs/module2_caption_roi_development.md)。
<!-- /module2_caption_roi_development -->

<!-- module2_dual_path_development -->
2026年10月24日：双上下文开发实验已完成：12方法480份完整排序，实际新增35对完整Qwen精排、精确复用973对图像上下文结果，16条文本保持原版。252项测试与重放通过；主方法仍未整体通过Precision硬门槛、Recall偏好与不退化检查，正式指标为空，默认方法保留。 继续验证新描述引导的真实主体裁切与完整Qwen交互，旧结果封存。见[本轮完成记录](docs/module2_dual_path_development.md)。

<!-- module2_primary_subjects -->
2026年10月24日：本轮已完成103件新增商品补库、40条开发查询、8种方法共320份完整排序，以及指标计算、共享未知标签增量、重放和结果页核验。主方法尚未在全部开发组通过Precision门槛，Recall偏好也未通过；六指标整体不退化尚未得到证明。当前宏观配对区间跨零，不能写成已证明整体退化。默认方法保留原版，正式指标为空，独立验收未启动。 新增826张衍生图已登记曝光；246项测试通过，封存后只读核验通过。见[本轮完成记录与内部对照入口](docs/module2_primary_subjects.md)。

<!-- module2_subject_coverage_expansion -->
80条开发候选已全部尝试采集：25张成功、55条源站HTTP 420失败；成功图片已逐图AI评审并全部补库，新增1000条相关性判断，原标签保留。完成193件同图库、40条开发查询、6种方法的240份完整排序对照。目标仍未整体达到，默认方法保留，正式验收指标为空；当前数据及衍生图已登记曝光。见[本轮处理与对照说明](docs/module2_subject_coverage_expansion.md)。

<!-- module2_subject_roi -->
已完成图片专用融合与主体ROI的同数据对照，文本保持原版，原图与裁切并行；整体未通过不退化门槛，默认方法保留。本轮为AI开发实验，未开新独立验收；243项测试通过。见[对照表、逐查询查看页与隔离说明](docs/module2_subject_roi.md)。

<!-- module2_fgclip2_comparison -->
已完成FG-CLIP 2独立辅助方法与原Qwen方法的同数据对照：40条开发查询、6方法240组完整排序，原标签保留。图片有局部收益、文本存在回退，默认策略保持原版；本轮不是独立验收。见[实验说明与完整对照表](docs/module2_fgclip2_comparison.md)。

<!-- module2_primary_form_coverage -->
模块二已完成按主体外形补库、20/60 候选对照和新查询家族隔离的开发验证。新增72件开发商品，原查询与已有标签保留；正式验收指标为空，默认策略未升级。本轮新查询使用共享图库，不能作为独立图库盲测结果。详见[按外形补库与验证](docs/module2_primary_form_coverage.md)。

<!-- module2_structure_development -->
模块二已完成主体结构开发实验和商品家族隔离验证；正式验收指标为空，默认策略保持原版。

新增两组各 96 个图库商品、16 条查询，全部 AI 评审。先在开发组固定选择，再打开家族隔离验证组；本轮未新增独立盲测。详见 [主体结构开发与验证](docs/module2_structure_development.md)。历史独立 AI 盲测结果及其图片分支未达标结论保持不变。

# 电商多模态智能内容与检索系统

最新目标：**Precision@10必须≥60%，Recall@10争取≥85%**。用户已明确全部由AI评审，开发推进不再等待真人。总体、文本和图片模式分别报告，继续检查其他指标对原基线是否退化，并保留独立heldout盲测隔离；AI评审来源单独标记，不能冒充人工验证。

最新完成：独立数据AI盲测已完成：300件图库、60条查询、18,000对预先冻结AI判断、1,200对完整模型精排。总体Precision门槛已达，图片分支未达，分模态验收未通过，默认策略保留。236项测试、66个源文件mypy通过，16个历史封存目录已核验。该heldout已consumed，不能调参后复用为未见测试。结果明确区分AI标注、开发成绩及人工验收，详见[独立盲测交付](docs/module2_independent_blind.md)。

以下为历史阶段记录，“最新”和待办以本段及当前进度文档为准；不再等待人工评审推进。

此前准备：64件候选及59个优先关系的A/B离线包保留历史记录，当前由上方AI评审流程接管，原真人表仍为空。见[原离线任务入口](artifacts/module2_catalog_review/index.html)与[准备阶段说明](docs/module2_catalog_review.md)。

最新推进：完成723对新增精排、20/60候选对照及342组完整排序重放；59对弱标签冲突直接看图后，3对改负、36对改未知，并重训候选。扩大召回未显示稳定收益，默认策略保留，202项测试通过。真实人工标签仍为0，正式质量指标为空。见[本轮结果与可运行入口](docs/module2_precision_recall.md)；最新进度为`artifacts/module2_precision_recall/public_status.json`。

最新自动推进（2026年10月22日）：完成216对疑难关系的直接看图机器复核，新增57对确定判断；重训18个分组模型，并复核含主体/属性约束候选的198组完整排序。新增[常驻检索页面](http://127.0.0.1:8765/)，支持文本、图片和ROI裁切；48次本地接口请求p50为0.138秒、p95为0.263秒，不含冷启动和浏览器渲染。194项测试通过，7个旧封存目录未变。人工标签仍为0、Precision≥60%未验证、独立验收候选预测0，默认策略保留。见[本轮技术与进度文档](docs/module2_continuation.md)；最新可同步状态为`artifacts/module2_continuation/public_status.json`。

上一阶段完成5,143对机器开发预标注、商品主体/外观语义特征及弱监督排序原型，保留全部原始证据。见[上一阶段记录](docs/module2_machine_assist.md)与[静态检索演示](artifacts/module2_machine_assist/index.html)。

既有盲测进度（2026年10月22日）：模块二已建立历史数据排除清单、策略与模型冻结流程，并完成一次独立服饰同款盲测；当前证据尚不足以确认融合策略的稳定收益。旧服饰成绩全部属于内部开发记录，正式三品类验收尚未完成。见 [盲测与验收规范](docs/module2_blind_evaluation.md)、[当前进度](docs/current_progress.md)；该次盲测的历史状态见 `artifacts/module2_blind/public_status.json`。

60%目标推进（2026年10月22日）：已完成业务相关性融合与精排原型、完整开发评审网格和严格评分入口。真实人工标签仍为0，尚未验证Precision@10≥60%；原同款集的容量不足以支撑该目标，已确认按原PDF商品业务相关性独立验收，原同款结果保留。见 [目标与评审入口](docs/module2_precision60.md)。

评审准备：5,143对开发关系已拆为54批，A/B共108个离线任务；另预留300件独立验收候选、30张查询图和30个文本查询席位。人员待领取、人工标签0、新验收候选未运行模型。见[评审材料与收件流程](docs/module2_business_review.md)；该轮准备状态保存在`artifacts/module2_business_review/public_status.json`。

Precision专项（2026年10月22日）：已实跑专用精排、边界排序、困难负例与局部视觉特征，尚未达到大幅增益和多指标不退化目标；默认策略未变。新增实验入口与空白人工开发标注包，见 [专项结论](docs/module2_precision_improvement.md)。

训练扩容（2026年10月22日）：新增150个训练组、50个开发验证组，已实训轻量相似度、区域注意力及实拍图适配，尚未通过多指标提升门槛。已补齐人工标签导入链路；详见 [训练进度](docs/module2_training_progress.md)。

2026年10月17日的历史核验见 [当日进度记录](docs/current_progress_20261017.md)：109项测试和338组排序指标复核通过，当时效果仍属于15日同款开发记录。最新评测状态以10月22日文档为准。

模块二内部策略探索记录（2026年10月15日）：[完整报告](docs/module2_strategy_exploration.md)。已实现13种固定策略及可切换重排入口 `scripts/retrieve_postrank.py`；候选组合需要独立验证，开发指标不作为对外验收依据。

当前负责范围（2026年10月14日）：**模块二多模态商品检索**。模块一和模块三保留历史成果，不作为当前待办。新入口为 [模块二增强方案](docs/module2_technical_plan.md)、[当前进度](docs/current_progress.md) 与 [增强实验报告](docs/module2_experiment_report.md)。新增区域注意力、ROI候选、裁切、局部放大及综合排序，使用 `scripts/retrieve_regions.py`，与原冻结基线分开保存。

下文保留此前全项目的历史记录；当前范围和下一步以模块二文档为准。

团队同步与汇报文档（2026年10月10日）：[环境配置与加载说明](docs/environment_setup.md)、[阶段一整体技术方案](docs/stage1_technical_design.md)、[当前进度与后续安排](docs/current_progress.md)。本次检查发现当前Ollama服务为0.40.1、同名生成模型摘要与历史冻结配置不同；下文历史环境和性能记录应结合新说明阅读。

已完成第一阶段工程验证；第二阶段已推进数据标准化、图片本地化、批量文案及事实核验，并跑通千问图文检索基线。第二阶段尚缺独立标注下的正式质量评测；第三阶段未开展。

最新进展：语义复核发现旧“每类100件”的关键词分组存在误类。新分类版本改类29件、隔离14件，保留286件；文案入口新增生成前来源一致性、生成后声明支持检查，90条旧输出回放标出15条待审风险；完成9张服饰副本的AI视觉抽查。见 [本次纠正与后续清单](docs/stage2_corrections.md)。正式人工质量成绩尚未产生。

历史基线保留300件商品（原开发/预留测试各150）、300件索引、18条查询和90条文案，属于v1启发式品类分组，不能与新分类结果混用。原扩展过程见 [扩展数据与评测准备](docs/stage2_dataset_release.md)。

文案基线v3现已加入来源冲突暂停、否定属性及SKU检查，并冻结配置完成90件预留样本运行：89件生成、1件来源暂停；42件输出被规则标记、47件仍待人工审核。见 [文案v3报告](docs/copy_baseline_v3.md) 和 [人工评分规则](docs/copy_scoring_rubric.md)。代码测试通过不等于质量验收通过。

导师建议补充已实跑：1000张真实服饰图的分项性能、5类服饰源标签同款开发检索、深色跨类负例、30个三品类文案边界输入。新增事实草稿/营销语句分离入口；缺必填信息停止生成。历史记录见 [导师建议与工程调整](docs/mentor_feedback_adoption.md)。同款开发实验不替代三品类相似款人工验收。

第二阶段最新工程入口：[分批评审与离线评分](docs/stage2_review_workflow.md)。`artifacts/review_stage2_v1/index.html` 提供18个检索批次及3个文案批次，支持保存/导入进度、严格合并；18条查询的5394条排序已固定，人工判断完成后无需模型即可评分。新增179条服饰可用副本，明确排除越界记录并同步COCO标注。人工评分仍待填写。

全部代码、依赖环境、缓存和派生数据位于 `E:\code\Multimodal_E-commerce`。`Alibaba阿里算法实习/` 是只读来源，程序拒绝向其中写入。Ollama 权重按用户要求存放在已有的 `E:\LLM_for_Ollama`，与 Llama 3 共用目录。

## 直接运行

当前模块二常驻开发界面启动命令如下；出现ready后打开`http://127.0.0.1:8765/`。服务仅在本机监听，保持模型驻留，关闭方式与API见[运行文档](docs/module2_continuation.md)。

```powershell
.\.venv312\Scripts\python.exe scripts\serve_retrieval.py --port 8765
```

以下保留原阶段一命令，当前模块二使用上方入口。

在项目根目录的 PowerShell 中执行，无需激活虚拟环境、无需修改系统执行策略：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py doctor
.\.venv312\Scripts\python.exe scripts\run.py verify-storage
.\.venv312\Scripts\python.exe scripts\run.py prepare-samples
.\.venv312\Scripts\python.exe scripts\run.py smoke
```

`smoke` 调用已下载的本地千问，输出文字回答、标题/详情/三个卖点、图片理解结果。首次冷启动耗时较长。推理结束后模型约两分钟自动卸载。

生成自己的文案：把商品 JSON 放在项目目录下，然后执行：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py generate --product configs\example_product.json
```

结果默认写入 `artifacts/generated_content.json`，已有文件则拒绝覆盖，可用 `--output` 指定新路径。顶层 `status` 给出合同最终状态，`factual_copy` 保留草稿和原规则记录。新入口检查来源冲突及部分缺少依据的声明；规则未报错不保证事实正确，文案仍需人工复核。受控营销风格入口为 `scripts/generate_content.py`。

以下为早期基线命令，`generate-batch` 保留历史生成流程，不含本次合同v2的新增检查：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py download-images
.\.venv312\Scripts\python.exe scripts\run.py generate-batch --run-name batch_v2_002 --per-category 2
```

已有 `batch_v2_001` 保存12条真实生成结果，新运行需换名字以保留证据。完整结果与已发现的问题见 [第二阶段本轮报告](docs/stage2_step1.md)。

图文检索可直接使用已有36件商品索引：

```powershell
.\.venv312\Scripts\python.exe scripts\retrieve.py search --text "红魔游戏手机" --top-k 3
.\.venv312\Scripts\python.exe scripts\retrieve.py search --text "儿童彩色小挎包" --target fused --top-k 3
```

查询输出JSON与HTML结果页。图片查询、品类过滤、复建索引等命令见 [检索部署与实测报告](docs/retrieval_baseline.md)。已保存的结果可打开 `artifacts/retrieval_v1/results.html` 查看。

## 当前环境

- Python 3.12.3，复用 `E:\Python 3.12\python.exe`，独立环境 `.venv312/`。
- Ollama 0.34.1；`qwen3.5:4b-q4_K_M`，下载约 3.39GB，实测 GPU 推理。
- 上下文限制 4096、图片最大边 768、关闭思考、固定 seed 42；不同时加载大型模型。
- 生成推理由Ollama负责；图文嵌入使用官方Qwen3-VL-Embedding-2B和Transformers/PyTorch CUDA，已实测在8GB显存GPU运行，无需额外CUDA Toolkit。
- 嵌入权重存于 `E:\LLM_for_Ollama\external\Qwen3-VL-Embedding-2B`，不由Ollama管理；代码与缓存仍在项目内。
- `requirements-retrieval.lock` 固定当前完整59包；`requirements.lock` 仅保留第一阶段35包，勿用旧锁文件覆盖当前环境。

复建环境（已安装的环境无需重复运行）：

```powershell
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
uv venv --python 'E:\Python 3.12\python.exe' .venv312
.\.venv312\Scripts\python.exe scripts\setup_retrieval.py
```

本机已下载千问，无需再次拉取。若将来重装模型，先执行 `verify-storage`，确认实际服务权重目录，再执行 `ollama pull qwen3.5:4b-q4_K_M`。仅设置客户端 `OLLAMA_MODELS` 不能改变已运行服务的存储目录。

## 文件入口

| 文件/目录 | 内容 |
|---|---|
| `src/multimodal_ecommerce/` | 数据标准化、源文件保护、Ollama 客户端、CLI |
| `data/samples/samples.jsonl` | 72 条带来源及原始 split 的标准化记录 |
| `data/samples/images/` | 24 张本地图片副本 |
| `data/samples/manifest.json` | 已复制文件 SHA256 清单 |
| `data/samples/annotations/` | 分割 COCO 转换、原始标注副本、属性类别定义 |
| `artifacts/stage1_smoke.json` | 真实推理响应、耗时和 GPU 驻留证据 |
| `artifacts/model_storage.json` | Llama 与千问实际权重路径、manifest、模型版本 |
| `artifacts/source_integrity.json` | 原始目录前后核对结果 |
| `docs/stage1_validation.md` | 环境和可用性验收报告，含真实质量问题 |
| `docs/stage_plan.md` | 第二阶段五个任务、四项产出的实施顺序 |
| `docs/data_sources.md` | 数据格式、采样限制与标签说明 |
| `data/localized/products.jsonl` | 48条商品记录，其中36张图片下载成功、12条保留失败状态 |
| `data/localized/media_manifest.json` | 图片URL、格式、大小、尺寸和哈希清单 |
| `artifacts/batch_v2_001/` | 12条批量文案、固定提示词版本、来源和事实核验记录 |
| `configs/embedding_model.json` | 官方图文嵌入模型/代码revision和运行参数 |
| `data/index_qwen_v1/` | 36组图像/标题向量、商品记录及校验清单 |
| `artifacts/retrieval_v1/results.html` | 中文查询与排除自身的图片检索结果页 |
| `docs/retrieval_baseline.md` | 检索运行方法、GPU实测、成功与失败样例 |

初始 `data/samples/` 保留原始抽样状态；图片本地化结果单独位于 `data/localized/`。36张已下载图片覆盖三个品类，另外12个URL返回HTTP 420。样本供流程验证，不能用于宣称正式评测成绩。

## 开发检查

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m black --check src tests scripts\run.py scripts\source_integrity.py
.\.venv312\Scripts\python.exe -m isort --check-only src tests scripts\run.py scripts\source_integrity.py
.\.venv312\Scripts\python.exe -m flake8 src tests scripts\run.py scripts\source_integrity.py
.\.venv312\Scripts\python.exe -m mypy src
```

原始目录核对：`.\.venv312\Scripts\python.exe scripts\source_integrity.py verify`。核对全部文件的路径、大小、修改时间，PDF 额外检查 SHA256；复制使用的图片和标注另做逐文件 SHA256。未对全部原始大数据做内容哈希。



## 编码规范检查

在项目根目录运行 `.venv-mac/bin/python scripts/check_quality.py`，或在 VS Code 的“运行任务”中选择“模块二：完整规范检查”。入口依次检查 Black、isort、flake8、mypy、Pylint、PDF 强制规则和完整 pytest；找不到 Node 或有测试跳过都会失败。检查入口不会格式化或改写源码。

结果保存到 `artifacts/coding_standard_audit/summary.json`，具体日志位于同目录 `checks/`。规范补充细则见 `configs/coding_standard_policy.json`，其中说明外部协议命名、第三方类型边界、Black 保留的长字面量及 500 行建议项。GitHub Actions 配置位于 `.github/workflows/code-quality.yml`，远程执行情况以 Actions 实际记录为准。

历史实验封存文件保留原来源摘要。源码整改后应为后续实验生成新的来源版本，旧实验的摘要与指标不能替换成当前源码的结果。
