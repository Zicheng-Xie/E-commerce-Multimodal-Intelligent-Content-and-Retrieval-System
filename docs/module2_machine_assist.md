# 模块二机器辅助开发推进

后续进展已写入[像素复核与常驻检索记录](module2_continuation.md)。本文和对应封存目录保留上一阶段的实际数值，最新状态见`artifacts/module2_continuation/public_status.json`。

更新：2026年10月22日。用户将后续工作交由本任务推进。本轮完成可自动执行的开发预标注、语义特征、排序候选训练、检索演示、性能修正与证据复核；真实人工评审尚未发生，Precision@10≥60%尚未验证。

## 本轮实际产物

- 处理既有286件开发商品与6张图片查询，形成292份机器商品/外观描述。
- 覆盖全部18条旧开发查询的5,143对允许关系：机器判断相关145对、不相关1,683对、无法判断3,315对。空值不转换为负例。
- 文本相关性由本地视觉模型读取原标题、商品图片和原查询标准推断；图片相关性根据机器提取的查询/商品描述推断，未进行全部图片对的直接视觉复核。两种结果均为机器弱标签。
- 增加商品主体、可见外观语义描述的2,048维嵌入，结合整图、区域注意力、最匹配局部、原标题、字面词和颜色，形成13个排序特征。模型特征不读取人工标签、源类别或查询答案。
- 训练平衡逻辑回归与困难负例RankNet两个候选。3个查询分组×3个商品族分组，每种训练9个交叉验证模型；另在全部已知机器标签上训练2个最终开发候选。
- 复核18个分组模型、144组完整排序，保留原5条排序策略和新增3条候选策略。内部机器一致性观察不足以支持替换默认策略，弱标签训练的候选也未显示可靠优势。

[自动开发结果入口](../artifacts/module2_machine_assist/index.html) · [当前机器/人工状态](../artifacts/module2_machine_assist/public_status.json) · [完整复核证据](../artifacts/module2_machine_assist/verification.json)。

## 标签和验证边界

`weak_labels.jsonl`每条明确声明`review_origin=model`、`human_label=false`、`role=machine_development_only`，没有写入原真人评审模板。模型不确定项为null，训练只使用有明确机器判断的关系。机器建议与协调用疑难项清单单独保存，原A/B盲评页面和原评分入口未改。

标准化、困难负例挖掘和训练梯度均排除对应的查询分组及商品族分组；被留出的图片查询来源族在训练的查询和图库两种角色中均排除。保存完整分组记录并重新训练复算权重。该流程仍基于已暴露开发素材和机器判断；机器提取阶段看到了开发查询标准，内部交叉验证不能视为独立质量证据。

机器判断及其一致性上下界保存在内部文件`semantic_ranker/machine_agreement_bounds.json`，它们不是实际业务Precision或正式成绩。3,315对关系无法判断，当前也无法给出完整人工相关性下的Recall、mAP等分数。

300件新验收图库、30张查询图与30个文本查询席位继续封存。新开发输入与预留来源/图片哈希隔离；新检索入口拒绝预留图片及其重复风险，实测在模型加载前拒绝了一张预留图，未产生验收预测。旧4个实验目录和前轮2个准备目录共6套封存文件哈希核验通过。

## 真实检索与耗时

已实际运行一次文本检索、一次图片ROI检索，并检查预留图拒绝路径。查询与排序功能跑通；这不等于人工质量验收。

发现图片查询原本为排除自身，每次重复解码全部286件商品。现复用既有暴露审计中相同算法的字节/解码像素指纹，并核验源文件哈希，保留排除逻辑。相同原图、ROI和策略下，Top10 ID保持完全一致，源商品仍被排除。

| 单次开发探针 | 查询处理，不含模型/索引加载 | 完整加载与查询 |
|---|---:|---:|
| 文本查询 | 0.445秒 | 18.422秒 |
| ROI查询，重复解码图库 | 10.025秒 | 19.273秒 |
| 相同ROI查询，复用审计指纹 | 0.837秒 | 9.618秒 |

这些是单例实测，不是p50/p95或正式端到端性能验收，完整冷启动仍超过5秒。[前后对照](../artifacts/module2_machine_assist/runtime_cache_comparison.json)保留原执行源码与两次输出，未覆盖旧耗时。

## 新入口

原检索入口与默认策略未改。新候选入口默认固定语义融合，弱标签训练模型须显式指定。输出目录必须是尚不存在的新路径：

```powershell
.\.venv312\Scripts\python.exe scripts\retrieve_semantic.py --text "无线蓝牙耳机" --method semantic_profile_blend --output artifacts/semantic_search_001
.\.venv312\Scripts\python.exe scripts\retrieve_semantic.py --image "实际开发图片路径" --roi 0.1 0.1 0.9 0.9 --method hard_negative_ranknet --output artifacts/semantic_search_002
```

已有演示：[文本结果](../artifacts/module2_machine_assist/demo_text/results.html)、[ROI结果](../artifacts/module2_machine_assist/demo_image_roi_cached/results.html)。

执行链路为`machine_development.py`准备/图片预标注/描述关系判断，`train_semantic_ranker.py`冻结/语义编码/训练，`prepare_semantic_runtime.py`准备审计指纹缓存，`verify_machine_assist.py`复核。已完成的协议、预测、标签和训练产物均保留，不原位重复生成。

## 环境与检查

本轮使用既有本地`qwen3.5:4b-q4_K_M`，实际Ollama服务0.40.6、runner为llamacpp，平台子清单摘要`91fa08b9fedfbbda924cd71febc5f6adbf31503cf7811df817dcf93bd56b08ff`。按服务实际选中的平台子清单核验模型与视觉projector全部文件，解决单层清单检查不能识别平台权重的兼容问题；未修改旧生成链路。嵌入模型继续使用原本地Qwen3-VL-Embedding-2B。

178项测试通过，49个源文件mypy通过，新改Python文件的black/isort/flake8检查通过。工程检查和机器一致性均不代表业务质量达标。

## 尚待真实证据

真人开发标签到位后，使用保留的双评合并、分歧仲裁与完整评分入口检验候选；不得把机器记录导成人工记录。正式查询批准、商品族/类目质检和独立验收相关性标签尚缺。最终模型、参数、对照和协议冻结后才能一次性盲测，确认Precision≥60%、Recall≥70%、其他指标不退化及正式响应时间。
