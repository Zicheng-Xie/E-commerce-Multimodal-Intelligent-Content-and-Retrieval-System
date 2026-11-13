# 模块二开发主线

2026-11-09，用户选择继续此前注释的原图 `primary_recall` 主线。144 对图搜复核已完整导入；在这份人工结果上完成描述冲突回退和分数一致性保护，当前续版为 `primary_recall_safeguards_v2`。范围仍为 845 件商品、55 条开发查询（图搜 36 条、文搜 19 条）。本页是当前开发与复核入口；各历史实验保留原记录。

**2026-11-10 更新：** 12 批 341 对已完整导入，323 对相关、18 对不相关，无重复、缺失或未决。当前主线全部 550 个前十位置均有明确人工判断；离线主体冲突候选另存作对照，默认配置保持当前版本。

## 已落实的版本选择

- [主线配置](../configs/module2_development_mainline.json)固定原主线及续版的协议、完整排序、结果快照、144 对已完成人工答案与 341 对后续任务的 SHA256。
- 文搜沿用已验证的主体召回和 Top20 精排，保护原前两位；相对先前排序，人工已确认净增 13 个前十相关结果。
- 图搜继续使用原图召回、主售类型补召回和固定 Top20 精排。描述内部的复合商品词与外观类型冲突时，回退原图召回；原第 3 名模型得分不低于已保护两项中较低者时，增加一位保护。前十内部仍按原图相似度排列。累计相对原图基线净增 10 个前十相关结果，图搜 P@10 精确增加 2.78 个百分点，折算总体增加 1.82 个百分点。
- `simple_crop_top20` 作为裁切与直接精排的对照实验保留。此次版本选择针对离线开发主线，不表示线上默认模型已发布，也不表示完成正式验收。
- 历史模型分数和排名保存在原目录，继续使用已有完整分数。新入口检查固定输入与任务绑定，读取新收到的人工 CSV，另写主线状态。

## 2026-11-10 已完成四对，转入验收隔离

四对新增判断已合并：2 对相关、2 对不相关。已验证的 `primary_public_conflict_matched_v1` 候选在全部 55 条查询上人工 P@10 为总体 **95.64%**、图搜 **93.89%**、文搜 **98.95%**；同标签下净增 3 个相关结果，六指标不退化。原 `primary_recall_safeguards_v2` 冻结排序与默认入口保留作基线，新候选与源码另行封存，不改旧归档。[完整开发结果与验收隔离](module2_generalization_20261110.md)。

新验收集已技术封存 300 件商品、30 条图片查询，预留 30 条独立文本查询；当前相关性标签与模型预测均为 0。八个历史路径明确登记消耗，18 个旧 CLI 启动受阻；新集禁止调参与推理。现在没有新的开发人工任务。以下的“四对待判”说明属于准备阶段，已由本段完成记录替代。

## 当前人工评测与容量审计

[人工精度、召回容量和错误清单](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/evaluation_latest.md)已接入 `status` 与 `score-review`。当前前十已全部明确作答：总体 **550/550**、图搜 **360/360**、文搜 **190/190**。完整人工开发 P@10 分别为 **95.09%、93.33%、98.42%**。评测分开输出三类结果：

| 输出 | 口径与限制 |
|---|---|
| 人工 P@10 | 仅用明确人工正负标签，固定分母为总体 550、图搜 360、文搜 190；缺失或证据不足保留补全边界，明确判断齐全后才给点值 |
| 人工 R@10 与容量 | AI 标签视为未标注；相关集未完整时不报真实召回点值；逐查询列出人工正例数、未决数及理想 R@10 上限 |
| 历史混合标签诊断 | 人工覆盖与既有 AI 标签共同计算，单独标明未计入 AI 错标风险；不当作完整人工成绩 |

按当前人工正例，总体理想 R@10 的乐观上限为 **79.03%**，图搜为 **81.62%**，文搜为 **74.14%**。三个范围均已标记“与当前人工标签及固定 Top10 不兼容”；这些上限不等于当前实际召回率。完整相关集仍缺人工真值，真实召回点值保持未确定。目标、相关性定义、全部查询及 K=10 保持原样。若一条查询已人工确认 16 件相关商品，理想 R@10 上限为 10/16=62.5%。

快照位于 `mainline/evaluation_snapshots/<输入与源码哈希>/`，包含 `report.md`、`metrics.json`、`capacity.json`、`errors.json` 和文件校验清单 `manifest.json`。相同输入与源码复用原快照；标签更新另存快照，记录标签与排序是否分别改变，不能把补标引起的分数变化算作模型收益。正式独立验收始终单列。

错误清单保留全部 55 条查询，按前十人工误召回数排序，并按人工理由整理主体／配件、属性冲突及待归因项；列出已确认正例在第 11–20 名还是 Top20 外。Top20/50/100 只报告已确认人工正例的覆盖诊断，不能替代 R@10 达标。前十错误已由人工全部覆盖；前十之外的正例位置与覆盖率仍是部分证据。当前共 27 个前十误召回，主体／配件关键词归因为 25 项，属性冲突 1 项，待归因 2 项，分类有重叠不可相加。

当前固定排序、同一标签快照下的总体**混合标签开发诊断**区间：

| 指标 | 当前主线 |
|---|---:|
| P@10 | 95.09%–95.09% |
| R@10 | 2.23%–55.31% |
| mAP | 20.85%–89.38% |
| NDCG@10 | 95.64%–96.09% |
| MRR | 98.18%–98.30% |
| Hit@1 | 98.18%–98.18% |

这些是人工、AI 与未决关系的补全区间，不是完整人工成绩或置信区间。新收到的人工判断会更新主线状态中的评测区间；固定预测的身份由 SHA256 核验。新增判断使区间变化时，需先区分标签补齐与排序变化。

完成 144 对人工复核后、排序尚未修改的原主线总体 P@10 为 70.91%–91.09%。新修复在相同标签下额外增加 1 个相关结果，总体 P@10 精确增加 0.18 个百分点。截至 144 对阶段，旧快照约 61% 到当时约 71% 的变化主要来自人工补标；本次补齐 341 对后得到的 95.09% 同样属于标签纠正后的评测，排序增益另按配对证据报告。[完整对照与消融](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/primary_safeguards/report.md)

## 已完成复核与当前四对新任务

已完成的五批 144 对图搜复核已核验：127 对相关、17 对不相关。它们全部导入，文件内容保持原样，已有人工换位证明原主线净增 9 个相关结果；本次修复又净增 1 个。

12 批 [341 对总集前十完整复核](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/primary_safeguards/human_review/index.html)也已完成，全部 341 个任务身份、批次、审核者、理由和时间通过校验。用户将 CSV 导出到了早期复核目录，现已原样归入当前任务的 `answers`，内容哈希保持一致。[导入与路径记录](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/primary_safeguards/human_review/import_receipt_20261110.json)。

本次更高的完整人工 P@10 来自人工补标纠正先前的 AI 标签，原主线排序没有变化，不能算作模型提升。

下一轮已对照宽泛过滤、明确冲突过滤、保留前两位及明确匹配替换。宽泛过滤会损害精度，不采用。保守 `explicit_matched` 候选只在原 Top20 内做 5 次互换，每次换出项已被人工判为不相关，其余位置不动；其中 1 件换入项已有人工相关判断，其余 4 件待判。总体人工 P@10 边界 **95.27%–96.00%**，相对主线增加 **0.18 至 0.91 个百分点**。根据这些互换的人工标签，可证明六项指标在同一标签补全下不退化；这仍是开发证据，未替换默认配置。[完整对照与容量诊断](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/public_conflicts_20261110/report.md)。

本轮已完成 [4 对换入候选](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/public_conflicts_20261110/human_review/index.html)，图搜 3 对、文搜 1 对，按原规则判断并写理由，CSV 保存到该页同目录 [answers](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/public_conflicts_20261110/human_review/answers/)。它们此前没有明确人工标签，已确认的换出项不重复提问。原来的 341 对不需重做，整套 372 对 Top20 标注没有追加。

在项目根目录的 Mac 终端查看主线状态：

```bash
.venv-mac/bin/python scripts/module2_mainline.py status
```

需要重新生成同一批复核页面时：

```bash
.venv-mac/bin/python scripts/module2_mainline.py review
```

导出 CSV 后导入并重算：

```bash
.venv-mac/bin/python scripts/module2_mainline.py score-review
```

状态保存在 [mainline/status.json](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/status.json)。三个入口均使用已有预测，不触发新模型推理。它们校验完整图库、全部查询、原前两位保护、文搜排序及任务绑定，拒绝缺失或错配数据。人工 CSV 校验固定任务身份、批次、重复项、审核者、时间和理由；部分完成会明确列出缺失任务，证据不足不会强转正负标签。`human_evaluation` 为人工视图，`candidate` 继续保留混合标签诊断；`review.batches` 给出 12 批完成情况，`evaluation_snapshot` 指向本次不可覆盖的快照。

## 下一轮比较命令

```bash
.venv-mac/bin/python scripts/module2_mainline_conflicts.py evaluate
.venv-mac/bin/python scripts/module2_mainline_conflicts.py score-review
```

新入口固定原版与候选任务身份，读取四对答案后在同一标签快照下重算五组对照，保留人工未决项；不修改默认配置。

## 继续开发的顺序

1. **换位收益已确认并完成修复。** 原图搜改动净增 9 件；本次修复再增 1 件，并消除了新增标签揭示的 MRR 下界小幅回退。新规则是基于已曝光开发判断提出的，独立泛化效果仍需验证。
2. **总 P@10 的人工依据已补齐。** 550 个前十位置均明确作答，全部 55 条查询纳入固定分母；总体、图搜与文搜人工精度均超过 80%。
3. **验证保守主体冲突替换。** 四个实际换入候选已全部明确作答，同标签总体精度提升 0.55 个百分点。选择规则与代码封存后，再进行独立验收；不依据验收结果回头调参。
4. **召回按容量证据推进。** 当前 Top10 的总体、图搜与文搜乐观理想上限均低于 90%。保留目标和相关性规则；较深候选的覆盖与乐观容量只作诊断。固定查询和 Top10 下，继续修改排序无法消除这个容量限制。后续若讨论返回数量或业务目标，需另行确认，不能改成另一个指标宣称达标。最终版本仍需冻结后的独立人工验收。

定向换位复核用于比较排序，不能估计整套 AI 标签的总体准确率。原 20% 随机人工复核、开发调参与冻结最终验收分别管理。

## 版本与记录

- [固定预测与原结果](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/primary_recall/results.json)
- [当前续版与完整对照](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/primary_safeguards/report.md)
- [用户选择记录](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/selection.json)
- [裁切对照报告](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/simple_crop_top20/report.md)
- [80/90 目标与历次进展](module2_current_target.md)

原续版与入口调整通过了当时的 366 项回归测试，详见[主线切换核验](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/verification.json)。本次人工评测完善通过 427 个 Python 文件的规范必检与全部 **391 项测试**，其中新增 25 项覆盖人工／AI 隔离、未决边界、容量、12 批导入及快照校验；真实项目的 `status`、`score-review` 核对通过，固定数据和实现的 27 项哈希全部保持一致。[本阶段核验记录](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/human_reporting_verification.json)

2026-11-10 本轮导入与优化通过 **430 个 Python 文件**的规范必检及全部 **398 项回归测试**；新增 7 项验证复合商品名、明确匹配、原位互换与六指标非退化结论。原主线 27 项冻结数据和实现哈希保持不变。 [完整人工复核完成记录](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/mainline/human_review_completion_20261110.json) · [新候选核验](../artifacts/module2_p80_r90_20261106/development_benchmark/subject_experiment/public_conflicts_20261110/verification.json)。

2026-11-10 四对导入与验收隔离已完成：已验证候选总体人工 P@10 95.64%，历史数据消耗和新 300 件／30 图片查询封存记录见[本轮说明](module2_generalization_20261110.md)。最终全量检查为 433 个 Python 文件、405 项测试通过；当前没有新的开发人工任务。


## 2026-11-10 工程化交付更新

选中策略已实现独立通用推理 CLI，完成 55 条完整开发排序缓存重放与最终代码的 24 商品／4 查询真实空缓存测试，运行包已冻结。447 个 Python 文件必检通过，431 项测试通过。独立标注与一次性验收工具已交付，真实保留集仍封存、未运行验收。操作入口、证据范围及下一阶段标注流程见 [通用引擎说明](module2_selected_runtime.md)。


## 2026-11-10 独立人工标注准备完成

30 条新提交的独立文本查询已原样导入并固定。600 批、18,000 对完整标注任务已生成并通过结构核验。当前只等待人工相关性判断；查询和模型不再根据验收内容调参，真实验收尚未运行。入口见 [标注目录](../artifacts/module2_acceptance_gold_20261110/review/index.html)，CSV 保存到该目录的 answers。


## 2026-11-10 人工工作量调整

600 批、18,000 对是验收子集的完整交叉关系，原全量人工作业已撤回，当前没有新增人工任务。30 条查询和冻结运行包保留，开发不以全量人工标注为前提；后续确需人审时每轮默认不超过 60 对。现有正式真值条件没有放宽，抽样评测方案仍需另行固定，不把抽样或 AI 标签直接当完整人工真值。见 [工作量调整](../artifacts/module2_acceptance_gold_20261110/人工工作量调整.md)。
