# 模块二：查询主体描述与多视角精排

更新：2026年10月23日。目标保持Precision@10≥60%，Recall@10尽量≥85%。当前只推进模块二。

本轮已完成开发编码、16种原策略实跑、5种前缀保护跟进，以及新查询的自动描述、ROI、联合精排和缓存护头运行验证。正式业务验收尚未进行；独立heldout继续封存，开发成绩不能对外作为验收成绩。

## 本轮新增策略

图片检索新增“查询图主体描述＋多视角匹配＋图文联合精排”。本地固定Qwen3.5只看查询图片，描述可见主体、形状、层数、部件和安装结构；不接收商品候选、评审标签或历史排名。随后使用完整Qwen3-VL-Embedding-2B计算查询多视角和描述向量，与商品整图、区域、原标题及主体描述共同召回。

新增方法包括查询多视角、主体描述融合，以及20/60候选的图文联合精排。完整Qwen3-VL-Reranker-2B同时读取查询图、查询描述和候选商品图文，沿用原业务相关性指令与本地完整权重。文本模式沿用固定原策略锚点，便于识别图像分支的实际影响。

描述是辅助特征，不是相关性标签。模型可能写出未核实的材质或部件数量，原始输出全部保留，不据此改标签。不能因一条描述看起来合理，就宣布新方法有效。

## 开发数据与隔离

原383件开发商品保留，新增35件：初批21件来自上轮冻结候选；另按事先冻结的外形条件扫描训练来源第200001至1000000条，登记30条元数据，下载14件补充商品。补充原计划16件，但符合条件的调料架元数据只有6条，短缺2件如实记录。全部新增商品逐图AI质检并保留，没有按相关性替换困难样本。

当前开发图库418件，2,546个图像视图；18条原查询及相关性规则保持不变。新增630对AI判断，完整关系7,519对。质检、相关性和推理描述分别保存，未判断项保留未知，原人工模板没有混入AI结果。

本轮完整嵌入模型实际调用326次：商品编码283次，查询多视角及描述43次。6张开发查询图的描述由本地模型实际生成。所有输入、模型、代码、参数和标签在新对照前冻结；仅在推理前修正一次整数类型声明，保留原冻结版本与差异证据，算法参数未改。

这些补库已经接触开发查询，属于有针对性的开发覆盖，不能称为独立验证。验收预留来源ID、商品族及原图指纹在推理前检查，heldout图片未用于本轮开发，新增验收推理0。

## 如何判断是否改善

16种固定方法在同一418件图库、同一18条查询上比较。与上一轮383件的变化还包含图库覆盖与Recall分母变化，不能混为算法增益。

未知标签采用完整取值范围。比较新旧方法时使用同一套未知标签补全，再计算配对差值，避免把不同补全的上下界混用。另检查20/60候选池召回，区分“召回漏掉相关商品”和“精排把相关商品排错”。

固定Top-10对每个查询的Recall上限是`min(10,R)/R`，R为图库内全部相关商品数。不删正例、删困难查询或改变K来追逐85%。零正例查询单独记录，不伪造Recall。所有质量数字、配对差值、数学上限和错误商品只存于[内部开发报告](../artifacts/module2_focus_coverage/internal_report.md)。

原始联合精排有Precision/Recall收益，也有MRR、Hit@1回退，因此没有替换默认策略。观察到回退后，另行冻结并比较4种交叉编码基线前缀保护，以及1种原区域基线前两位保护，共21种方法、378组完整排序。新增5种只是对真实排序的后处理，没有新增模型推理；原16种结果及负结果全部保留。

保护前缀保证对应基线的Hit@1不变；MRR还要求原首个相关结果落在被保护前缀内，不能无条件保证。当前仍不能证明所有指标对原基线同时不退化，不升级默认策略。跟进属于自适应开发探索，不能当作独立验证；详细成绩与模态路由修复记录见[内部护头报告](../artifacts/module2_focus_coverage/region_guard/internal_report.md)。

## 环境与运行入口

沿用`.venv312`及完整本地嵌入、精排权重，无新增依赖或模型下载。查询描述模型固定为既有`qwen3.5:4b-q4_K_M`，在本地Ollama运行。描述生成、嵌入和精排按顺序执行；其他驻留模型会影响显存和耗时，不能把本轮批处理时长当作正式性能成绩。

```powershell
# 新文本请求；输出目录必须是新的
.\.venv312\Scripts\python.exe scripts/retrieve_focus_development.py `
  --text "入耳式无线蓝牙耳机" --method region_title `
  --output artifacts/module2_focus_coverage/another_text_request

# 自动描述、ROI、多视角及60候选完整图文精排
$queryRecipe = Get-Content artifacts/module2_focus_coverage/query_runtime_protocol.json -Raw -Encoding UTF8 | ConvertFrom-Json
.\.venv312\Scripts\python.exe scripts/retrieve_focus_development.py `
  --image $queryRecipe.fresh_image_query --roi 0.48 0.13 0.98 0.80 `
  --method described_cross_pool60 `
  --output artifacts/module2_focus_coverage/another_image_request

# 从这次真实新请求的向量和原始精排分数继续做区域基线前两位保护
# 只复用本次请求的缓存，不查历史评测查询表，不再调用模型
.\.venv312\Scripts\python.exe scripts/compare_region_prefix.py query `
  --query-run artifacts/module2_focus_coverage/another_image_request `
  --output artifacts/module2_focus_coverage/another_protected_request

# 无GPU模型推理的结果重放与完整性核验
.\.venv312\Scripts\python.exe scripts/verify_focus_coverage.py
.\.venv312\Scripts\python.exe scripts/verify_focus_delivery.py
.\.venv312\Scripts\python.exe scripts/verify_focus_guards.py
```

新查询入口读取实际输入，自动生成描述和裁切视图，不查历史查询排名或把评审表作为运行特征。整张原图用于排除自身，ROI用于实际查询；预留验收路径在打开像素或加载模型前拒绝。结果分数为模型或排序分数，不是经校准的相关概率。

228项测试、64个源文件mypy及本轮格式检查通过。418件索引、全部AI关系、原始精排与缓存来源、分组重训及14个历史封存目录已重放核验。图片需求的空正例缺口已补齐，但这不等于指标达标。

当前默认策略和旧索引保留。下一步先处理尚未通过的不退化项，完成最终候选及盲评协议冻结，再进行一次独立heldout验证。AI评审来源持续显式记录，不再等待人工推进。
