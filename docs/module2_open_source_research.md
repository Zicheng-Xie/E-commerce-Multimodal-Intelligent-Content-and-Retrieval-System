# 模块二多模态商品检索开源参考与迁移方案

核验日期：2026年10月23日。通过 GitHub 连接器检索，并阅读官方 README、实际源码、默认分支提交及许可证信息；模型信息补充核对官方模型卡和论文页面。

优先试验**语义主体 ROI、FG-CLIP 2 细粒度辅助通道、已确认困难负例学习**。三者分别针对主体定位错误、外形与属性区分不足、相似商品误排序。ColPali 的局部匹配方法可作为后续候选；GCL 的查询与图库双维度隔离应纳入下一轮验证设计。这些是基于代码和本项目错误分析提出的迁移假设，尚无本项目新增效果数字。

本轮只完成调研和迁移设计，没有下载新权重、安装依赖、训练或运行新检索实验，默认策略保持不变。当前目标仍为 Precision@10≥60%，Recall@10争取≥85%；分文本、图片及总体检查其余指标不退化。

## 先明确本项目已有能力

| 现有位置 | 已实现 | 本轮参考要补的能力 |
| --- | --- | --- |
| [regional_retrieval.py](../src/multimodal_ecommerce/regional_retrieval.py) | 整图、边缘背景对比 ROI、中心及四角裁切，放大，多视图余弦注意力 | 按商品语义定位主体；检测框与可选前景掩码，而非继续增加固定裁切 |
| [visual_query.py](../src/multimodal_ecommerce/visual_query.py)、[query_fusion.py](../src/multimodal_ecommerce/query_fusion.py) | 查询图自动描述、无法识别时停用描述通道、多路融合 | 对局部主体和属性增加独立证据；主体识别不确定时保留原图分支 |
| [intent_retrieval.py](../src/multimodal_ecommerce/intent_retrieval.py)、[semantic_ranker.py](../src/multimodal_ecommerce/semantic_ranker.py) | 多通道候选、完整 Qwen 精排、类型约束、最佳区域相似度、轻量学习排序 | 基于区域与属性的困难负例；多局部对应关系，而非单个最大相似度 |

完整 Qwen3-VL-Embedding-2B 和 Qwen3-VL-Reranker-2B 已经在使用。泛泛地换成 CLIP、增加裁切、增加描述或把精排候选从20扩大至60，均不能算本轮新策略。此前结构提示和候选扩张的负结果继续保留，见[最近一次开发验证](module2_primary_form_coverage.md)。

## 最值得迁移的实现

### 1. GroundingDINO 与 SAM 2：先找到实际商品主体

[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 的推理接口接收图片和文本提示，输出框、分数及对应短语，代码见 [`predict`](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py#L53)。[SAM 2](https://github.com/facebookresearch/sam2) 的图片预测器接收框或点提示生成掩码，见 [`SAM2ImagePredictor.predict`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L237)。SAM 2 的职责是分割，不能独自判断哪个物体是出售商品。

**拟迁移：**先只试检测框裁切，将查询和图库的主体图接入独立的 Qwen 区域编码分支；检测词来自查询图片描述或商品公开字段。若框裁切有效，再单独比较掩码裁切。保留整图，主体不清、检测失败或多主体冲突时回退；不按“面积最大”直接认定商品，避免把模特、包袋道具或附属家具当主体。

**预期作用与限制：**有望降低背景与道具造成的图片误检，但这是待验证假设。耳罩套、内胆包等配件仍需要商品语义判断；检测置信度不是业务相关概率。检测和掩码必须分开消融，才能判断收益来自哪里。

**工程适配：**新增独立主体视图生成器，不修改已封存的区域实现。官方 GroundingDINO 安装涉及扩展编译，SAM 2 对 Windows 推荐 WSL；先做兼容性检查，在隔离环境运行。8GB 显存下分阶段处理、缓存图库框和视图，不预设多个视觉模型可以同时驻留。

### 2. FG-CLIP 2：补足中文细粒度与局部属性辨别

[官方仓库](https://github.com/360CVGroup/FG-CLIP)提供中英双语对齐、区域与文本学习、属性扰动困难负例。实际模型代码具有 [`get_image_box_roi_features`](https://github.com/360CVGroup/FG-CLIP/blob/b1af1cda8ae62089664251670c7819b25e07a0bc/fgclip2/model/strcs/fgclip2.py#L182)，使用密集特征及 RoIAlign；[`hard_contrastive_total_loss`](https://github.com/360CVGroup/FG-CLIP/blob/b1af1cda8ae62089664251670c7819b25e07a0bc/fgclip2/model/strcs/fgclip2.py#L587)体现正例与属性扰动描述的区分。方法背景见[论文](https://arxiv.org/abs/2510.10921)，可选起点为[官方 Base 模型](https://huggingface.co/qihoo360/fg-clip2-base)。

**拟迁移：**先冻结 Base 权重，新增独立的整图/主体细粒度评分通道，用来检验“厚底与细跟”“柜门与抽屉”“有袖与无袖”等可见差异。Qwen 继续承担主检索与精排。不同模型分别建立特征库，融合排名或经开发集校准的分数，不直接相加不同维度的向量，也不把原始 logit 当概率。

**实际限制：**官方大规模训练脚本采用分布式和很大的批量，不能作为当前8GB设备的直接训练方案；先验证推理成本及冻结特征。源码局部损失计算存在 [`try/except: pass`](https://github.com/360CVGroup/FG-CLIP/blob/b1af1cda8ae62089664251670c7819b25e07a0bc/fgclip2/model/strcs/fgclip2.py#L485)路径，后续若适配训练，必须让异常显式失败并验证各损失实际参与反向传播。模型卡使用自定义代码加载，实施时还需固定权重及代码 revision。论文和模型卡成绩不能转作本项目成绩。

### 3. ColPali / ColQwen：从整体相似走向局部对应

[官方仓库](https://github.com/illuin-tech/colpali)主要服务视觉文档检索。其 [`MaxSim`](https://github.com/illuin-tech/colpali/blob/97487f8871ff4d5d2284411fe61bdcd2cfe99894/colpali_engine/utils/maxsim.py#L29)对每个查询 token 寻找最相似的文档 token 后求和；[`HierarchicalTokenPooler`](https://github.com/illuin-tech/colpali/blob/97487f8871ff4d5d2284411fe61bdcd2cfe99894/colpali_engine/compression/token_pooling/hierarchical_token_pooling.py)压缩局部向量。

**拟迁移：**借鉴延迟交互评分，让鞋底、鞋面、袖口、柜门等多个局部分别找到对应证据；优先用于小候选池的辅助评分，并比较主体前景约束和压缩前后的成本。现有“最佳区域相似度”只有一类聚合信号，不能称为 token 级 ColPali。

**实际限制：**文档模型对商品图的适用性未经本项目验证；当前2048维池化向量不能还原 token。真正的 token 匹配需要新的特征提取及缓存，或采用有对应训练的多向量模型。MaxSim 可能让多个查询局部反复匹配同一背景，因此不能单独替换整图及主体语义判断。代码 MIT 与各模型权重许可证分开核对。

### 4. Marqo GCL：商品相关性排序与双维度隔离

[官方仓库](https://github.com/marqo-ai/GCL)围绕商品多模态相关性排名，说明如何用分级排名信息监督对比学习；其[划分说明](https://github.com/marqo-ai/GCL/blob/e4a918b9c94c50e5f82086421a6902408d009782/README.md#2-dataset-and-benchmarks)区分新查询、新图库和两者都新的场景。已检查目录和 [`eval_gs_v1.py`](https://github.com/marqo-ai/GCL/blob/e4a918b9c94c50e5f82086421a6902408d009782/evals/eval_gs_v1.py)：公开实现主要是数据加载、特征组合及评测，未在本次默认分支目录中找到 GCL 训练循环，不能当作完整训练框架直接接入。

**拟迁移：**新查询开发验证、图库扩张实验、全新查询加全新图库的下一轮 AI 盲测分别报告。后续若引入分级排序训练，只使用预先定义含义的开发标注；不重写现有二元标签，正式 Precision/Recall 仍按原合同计算。商用搜索排名也不能直接作为本项目商品相关性真值。

### 5. MODA：困难负例与对“选最佳结果”的审计

[hopit-ai/Moda](https://github.com/hopit-ai/Moda)是较新的服饰检索项目，公开了较多实验脚本。[困难负例挖掘](https://github.com/hopit-ai/Moda/blob/b6513ca77e076b090427ffe13f4959446a15662e/scripts/mine_hard_negatives.py)包含近邻采样及假负例启发式过滤；[结果选择审计](https://github.com/hopit-ai/Moda/blob/b6513ca77e076b090427ffe13f4959446a15662e/benchmark/lookbench_honest_headline.py)明确讨论在同一基准上选最佳配置带来的偏差，并披露无法复测的配置。

**拟迁移：**从开发检索靠前结果中提取“类型相近但用途不同、主体与配件混淆、属性冲突”的已确认负例，先训练小型排序头。相似度只用于发现候选，不能证明其为负例；所有未判断及不确定关系仍保持未知。多正例查询不能将除一个正例外的其他商品自动视为负例。

**使用边界：**将其作为实验设计参考，优先级低于官方模型实现。其服饰数据、LLM判断和公开基准选择方式均不等价于我们的独立验收；不能复制其标签或宣称其指标可复现到本项目。

## 可作对照或后续工程参考

| 仓库 | 借鉴点 | 在本项目中的位置 |
| --- | --- | --- |
| [Marqo-FashionCLIP](https://github.com/marqo-ai/marqo-FashionCLIP) | 官方服饰模型及七个公开数据集评测；[`eval.py`](https://github.com/marqo-ai/marqo-FashionCLIP/blob/d0b3bdfb62fb4964537cb582e76d6ddab0be8450/eval.py)、[`utils/metrics.py`](https://github.com/marqo-ai/marqo-FashionCLIP/blob/d0b3bdfb62fb4964537cb582e76d6ddab0be8450/utils/metrics.py)可核对任务与指标 | 服饰子域的辅助基线。类别到商品与描述到图片的标签定义不同；其 P@10不能移用为我们的60%验收成绩。中文和跨品类效果需另测 |
| [Faiss](https://github.com/facebookresearch/faiss) | 精确和近似向量检索；[精确索引示例](https://github.com/facebookresearch/faiss/blob/bce5e24dd753296e241a67dbd0914cb652382fc5/tutorial/python/1-Flat.py) | 大图库工程储备。先保留精确内积对照，再评估近似索引召回损失与时延；不能指望换索引自动提高业务Precision |

## 下一轮实施顺序

以下是拟执行的开发路线，不是已冻结实验协议，也没有新增运行结果。

1. **先按现有错误清单判断瓶颈。**沿用最新开发工作清单，不删除容量不足或零正例查询。分别记录图库缺相关商品、候选池漏召回、池内排序错误；图库覆盖建设与算法比较分开进行。
2. **主体定位小步验证。**固定同一图库、查询、标签、模型与精排预算，只增加检测框主体视图。记录定位失败、框数量、回退原因、额外编码量、显存与时延。框方案通过开发门槛后，才单独试掩码。
3. **冻结 FG-CLIP 2 辅助通道。**先比较固定候选池内的细粒度评分，隔离排序收益；如需研究新增通道召回，再独立比较相同精排预算下的候选分配。保留不加新通道的原版。
4. **确认负例后再学习。**仅在开发部分挖掘并由 AI 看图评审，保留 `review_origin=model`、`human_label=false`。先做按商品家族隔离的小排序头；数据规模与收益足够后，再考虑编码器适配或蒸馏。未知标签不进入负例损失。
5. **局部多向量作为后续消融。**先确认主体 ROI 和细粒度通道是否有效，再决定 token 特征、压缩及延迟交互是否值得增加资源成本。
6. **方法固定后再开新盲测。**开发阶段固定选择规则，记录所有尝试及失败。冻结模型 revision、特征、融合、候选预算、采样、相关性协议及指标，再使用全新查询与全新图库开展下一轮独立数据 AI 盲测。

每一步均在相同数据上配对比较 P@10、Recall@10、mAP、NDCG@10、MRR、Hit@1，分别报告文本、图片和总体。新方法未证明不退化时保持默认版本；有限开发集通过也不能保证未来所有查询都不退化。正式判定沿用现有指标和未知关系处理规则，不能在看到结果后换口径。

图库里没有足够相关商品时，排序无法补出正例；固定 Top-10 对 Recall 也存在 `min(10,R)/R` 的容量限制。这些容量诊断与原始指标一起保留，不通过缩小分母、减少返回数或改标签实现达标。

原独立 AI 盲测已消耗；后续曝光的查询及共享图库开发验证也均不能重新当作未见验收。数据隔离不等于独立标注员验证：当前评审由 AI 完成，仍需披露评审上下文和潜在模型偏差。

## 本次核验的版本与许可证

以下是核验时默认分支的提交身份，日期为 API 返回的该提交日期，不以星标更新时间或仓库 `pushed_at` 代替。所有仓库当时均未标为 archived；这不构成未来维护保证。许可证列为仓库代码信息，具体模型与数据仍需检查各自条款。

| 仓库 | 核验提交 | 提交日期 UTC | 代码许可证 |
| --- | --- | --- | --- |
| IDEA-Research/GroundingDINO | [856dde20aee6](https://github.com/IDEA-Research/GroundingDINO/commit/856dde20aee659246248e20734ef9ba5214f5e44) | 2024-08-12 | Apache-2.0 |
| facebookresearch/sam2 | [2b90b9f5ceec](https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4) | 2024-12-16 | Apache-2.0 |
| 360CVGroup/FG-CLIP | [b1af1cda8ae6](https://github.com/360CVGroup/FG-CLIP/commit/b1af1cda8ae62089664251670c7819b25e07a0bc) | 2026-06-16 | Apache-2.0 |
| illuin-tech/colpali | [97487f8871ff](https://github.com/illuin-tech/colpali/commit/97487f8871ff4d5d2284411fe61bdcd2cfe99894) | 2026-08-24 | MIT |
| marqo-ai/GCL | [e4a918b9c94c](https://github.com/marqo-ai/GCL/commit/e4a918b9c94c50e5f82086421a6902408d009782) | 2025-01-28 | Apache-2.0 |
| hopit-ai/Moda | [b6513ca77e07](https://github.com/hopit-ai/Moda/commit/b6513ca77e076b090427ffe13f4959446a15662e) | 2026-08-28 | MIT |
| marqo-ai/marqo-FashionCLIP | [d0b3bdfb62fb](https://github.com/marqo-ai/marqo-FashionCLIP/commit/d0b3bdfb62fb4964537cb582e76d6ddab0be8450) | 2024-09-20 | Apache-2.0 |
| facebookresearch/faiss | [bce5e24dd753](https://github.com/facebookresearch/faiss/commit/bce5e24dd753296e241a67dbd0914cb652382fc5) | 2026-10-23 | MIT |

迁移时新建开发实现和索引，保留上游来源及许可证，不修改既有封存源文件、评审关系和盲测结果。本次没有承诺任何开源方法能使本项目直接达到60%；是否采用由后续开发对照和新独立数据验证决定。
