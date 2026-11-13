<!-- role_pool_runtime -->
主售角色/扩池实验复用完整本地Qwen3.5及Qwen3-VL-Reranker-2B，无新增依赖安装。原生RGB画像采用16384上下文、1024输出上限、温度0；精排保持原完整模型配置与无静默截断检查。九折双图注意力OOF仅提供开发候选，不能替代单一部署模型。显存准入及自然释放规则保持。
<!-- /role_pool_runtime -->

<!-- business_grounding_runtime -->
2026年10月25日：商品质量、主体掩码、业务度量学习、双图注意力适配和固定融合开发周期已完成。继续使用`.venv312`、PyTorch 2.8.0+cu128与Transformers 4.57.1，未安装或升级依赖。新增官方完整SAM 2.1 Base+权重位于项目`.cache/open_source_models/sam2.1-hiera-base-plus/b7320756a13354e7530a63935656d35b2f91a290`，公开大小和LFS SHA256已核对；通过现有Transformers的Sam2Model加载，完整模型float32，无缺失参数。原外部DINOv2-small和Qwen模型只读保持。

完整GroundingDINO检测412次、SAM 2分割383次；完整本地Qwen3.5原生视图核验383次，另52条无提议回退，保留全部435条记录。语义及结构复核仍发现漏检，新掩码全局停用。新请求仅维持本作业模型1分钟并等待自然到期，不卸载或缩短其他作业keep-alive。

整图业务投影及双图匹配头在CPU按家族隔离各训练9折和1个全开发候选，每模型400步。双图特征使用完整12层DINOv2-small，float32，按原官方预处理对两图各取224×224，再连接为448×224像素张量；7,560个配对共1,890次实际批前向，每对全部512个patch共同参与注意力。骨干冻结，仅训练业务头；这是STIR思路的单独业务适配，不是论文端到端模型复现。原Qwen精排完整分数复用，没有新精排调用或补零。

本轮已完成7方法280份排序和2,800张Top10卡的静态引用检查；307项测试、模型权重/排序/指标重放通过。完整只读核验：`.venv312/Scripts/python.exe scripts/verify_business_development_cycle.py`。三种模型阶段均在显存低于1500MiB且Ollama明确空驻留后启用GPU，未打断用户作业。记录的模型前向和CPU训练耗时不代表在线端到端时延。开发目标尚未整体通过，正式指标仍为空；新一轮独立盲测未启动。见[完整进度与证据](module2_business_grounding_development.md)。
<!-- /business_grounding_runtime -->

<!-- grounded_subject_runtime -->
本轮开发恢复入口`.venv312/Scripts/python.exe scripts/grounded_subject_recovery.py worker`；完成后只读核验使用`verify`。独立AI盲测的实际推理入口为`grounded_blind_execution_recovery.py worker`，须先冻结原生AI标签；指标来源适配与封存使用`grounded_blind_evaluation_recovery.py finish`，完整只读核验使用`.venv312/Scripts/python.exe scripts/grounded_blind_evaluation_recovery.py verify`。已完成时只核验，已有实例不重复启动。完整模型与GPU自然释放规则保持。见[module2_grounded_subject_repair](module2_grounded_subject_repair.md)。
<!-- /grounded_subject_runtime -->

<!-- subject_learning_runtime -->
新增主售学习入口：`.venv312/Scripts/python.exe scripts/subject_attribute_experiment.py verify`。全图库候选实验使用`.venv312/Scripts/python.exe scripts/subject_recall_experiment.py worker`串行接续，已有进程时不要重复启动。该入口先校验冻结清单与并发锁；完整GPU精排前执行nvidia-smi低于1500MiB及Ollama明确空驻留列表检查，等待用户模型自然释放，不卸载或缩短keep-alive。完成后可执行`verify`只读核验。状态见`artifacts/module2_subject_recall_learning/public_status.json`。 当前类别平衡损失对照入口为`.venv312/Scripts/python.exe scripts/pointwise_recall_experiment.py worker`，已有本轮实例时不重复启动；同一完整模型、参数与GPU准入规则继续适用，使用新命名空间和输出目录，保留旧源码及模块对象。 最新公开主售冲突优先级实验已完成，入口`.venv312/Scripts/python.exe scripts/public_conflict_experiment.py verify`只读核验；该阶段不调用新模型，无需加载GPU。已有模型、原始标签、曝光排除和历史收据均保持。
<!-- /subject_learning_runtime -->

<!-- balanced_gallery_runtime -->
模块二315件扩库对照已完成。本轮原接续入口在报告导出时结束，失败证据保留；独立收尾已修复导出并封存。只读复核命令：`.venv312/Scripts/python.exe scripts/finalize_balanced_gallery.py verify`，当前完成状态见`artifacts/module2_balanced_gallery_finalization/public_status.json`。勿重复运行保留失败证据的旧runner入口。本轮无待补的评审或GPU推理；后续新实验单独冻结协议。原生评审采用完整原图、16384上下文与1024输出预留；所有GPU任务均遵守先nvidia-smi、后空Ollama驻留列表的准入，不卸载或缩短用户模型keep-alive。原完整模型检查点、策略及历史收据保持。
<!-- /balanced_gallery_runtime -->

# 电商多模态项目环境配置与加载说明

<!-- module2_learned_fusion -->
2026年10月24日：有监督融合使用既有完整Qwen/DINO/FG实算特征，新增20维融合层的家族隔离训练在CPU运行，无新增依赖、权重下载或神经前向。训练前263项测试通过，22方法880份排序及20个模型重放封存，复核命令为`scripts/learned_fusion_experiment.py verify`。新补库网络首次遇WinError10013，保留原失败并用独立恢复脚本在允许网络的环境重试相同URL，取得19图；没有绕过HTTP拒绝。新评审沿用完整本地Qwen3.5及视觉projector，采用单图片查询双图和16384上下文、1024输出上限，不人工缩图，不降低模型规模。268项当前工程测试通过，模型仅在显存自然低于1500MiB且Ollama明确无驻留时加载，已有10分钟keep-alive自然过期，不卸载或中断用户模型。详见[本轮执行与隔离说明](module2_learned_fusion.md)。

<!-- module2_roi_semantic_gate -->
2026年10月24日：查询ROI语义核验与首位保护开发对照已完成，继续使用现有`.venv312`，无新增依赖或模型下载。实际复用完整本地`qwen3.5:4b-q4_K_M`及视觉projector，模型、平台子清单与全部blob核验后，以原图和真实zoom双图输入完成21次原生评审，3条继承全图回退；19条许可、1条拒绝、1条不确定是模型决定数，不能当定位准确率。显存低于1500MiB且Ollama驻留明确为空才加载；未卸载用户模型，审核模型按本轮固定10分钟keep-alive自然过期。原检索模型分数全部复用实算记录，没有新嵌入、重排、DINO或FG调用。19方法760份完整排序重放及260项测试通过；157个文件和75份源码快照封存，无新增图片像素，完整继承历史曝光排除清单。只读复核入口为`scripts/roi_semantic_gate_experiment.py verify`。主方法仍未全部通过目标并出现严格退化，默认方法保留原版，正式指标为空。详见[本轮语义核验实验](module2_roi_semantic_gate.md)。

<!-- module2_caption_roi_development -->
2026年10月24日：新主体裁切开发实验使用现有`.venv312`及完整固定权重完成实跑，无新增依赖或模型下载。完整GroundingDINO实际检测23次，21条查询采用裁切、3条保留全图（含1条未决类型），图库复用旧裁切。完整Qwen实际新增498对裁切评分，另6对严格复用相同输入；原整图与文本结果保留。15方法600份完整排序、255项测试与重放通过，新增63张衍生图登记曝光，726个文件及76份源码快照封存。只读复核用`scripts/caption_roi_experiment.py verify`；完成后的`run`也仅验证封存。显存低于1500MiB且Ollama明确无驻留模型才加载，未卸载或修改用户作业keep-alive。新主方法尚未整体通过Precision门槛，并存在明确退化，正式指标为空，默认方法保持原版。详见[主体裁切实验与核查](module2_caption_roi_development.md)。

<!-- module2_primary_subjects -->
296件主体覆盖开发实验已复用现有`.venv312`及完整固定版本模型完成实跑，没有新增依赖或下载权重。等待其他作业自然释放显存后，顺序运行query-only描述、Qwen编码/精排、GroundingDINO和FG特征；未卸载或中断用户模型。接续入口`scripts/continue_primary_subjects.py run`现仅做封存只读核验；本轮显存等待心跳完成后暂停。默认检索入口保持原版，工程通过不等于质量达标。实际调用收据、核验与局限见[本轮完成说明](module2_primary_subjects.md)。

<!-- module2_subject_coverage_expansion -->
193件同图库开发对照使用现有`.venv312`和完整固定版本Qwen嵌入/精排、FG-CLIP 2、GroundingDINO，模型顺序加载，不增加依赖和权重。新采集尊重配置代理，仅访问原公开图片URL，80条全部尝试、55条HTTP 420复试失败。运行材料位于`artifacts/module2_subject_coverage_expansion`；复核用`scripts/verify_subject_coverage.py`，详细隔离规则与入口见[本轮说明](module2_subject_coverage_expansion.md)。没有启动新的独立验收或替换默认索引。

<!-- module2_primary_form_coverage -->
按主体外形补库仍使用既有完整本地模型和依赖，无新模型或依赖下载。新入口为 `scripts/primary_form_coverage.py`；索引引用已核验的历史向量和裁切图，仅对新增商品实际编码。重放入口为 `scripts/evaluate_primary_form_coverage.py verify`。详见[补库与新查询开发验证](module2_primary_form_coverage.md)。

<!-- module2_structure_development_environment -->
主体结构开发与隔离验证补充：两组新数据均使用既有完整本地 Qwen 嵌入、重排和 Qwen3.5 查询描述模型；无新增依赖或权重下载。新入口为 `scripts/structure_development.py`，重放入口为 `scripts/verify_structure_development.py`。原方案与结构提示/移除描述对照仅用于开发，模型和全部源码摘要绑定在 `artifacts/module2_structure_development/protocol.json`。默认检索入口保持原版，详见[主体结构开发与验证](module2_structure_development.md)。

2026年10月23日独立盲测补充：固定版本已使用预留独立数据完成300件图库、60条查询的AI盲测，复用既有完整本地模型和依赖，无新增下载。模型、参数、AI协议及输入绑定见release.json；重放复核入口为scripts/score_independent_business_blind.py。该heldout现已consumed，结果及运行说明见[独立盲测交付](module2_independent_blind.md)。

2026年10月23日本轮补充：有限重排候选和6张新来源图片查询已运行，仍使用既有完整本地嵌入、精排及查询描述模型，无新增依赖或权重下载。新实验图片入口支持自动描述、多视角、ROI、20候选联合精排和尾部恢复；文本沿用原入口。实际新增42次完整嵌入编码、120对精排，运行与核验命令见[本轮说明](module2_rank_stability.md)。正式heldout仍封存，默认策略保留，开发质量数字只存内部。

2026年10月23日最新补充：418件开发索引已运行，新增本地查询图主体描述、多视角、ROI、完整图文联合精排与缓存前缀保护。复用既有完整Qwen3-VL-Embedding-2B、Qwen3-VL-Reranker-2B、`qwen3.5:4b-q4_K_M`和`.venv312`，无新增依赖或权重下载。运行命令见[主体描述与精排说明](module2_focus_coverage.md)。此前383件、344件及原索引保留；所有开发成绩留内部，正式指标仍为空。

最新AI评审环境：已复用本地Ollama 0.40.6及经清单、模型和视觉projector哈希核验的Qwen3.5:4b-q4_K_M运行79件自动图文评审；发现批量需求判断噪声后，最终350件由Codex直接看图完成AI裁决，保留原始自动输出。排序重放及AI指标计算使用既有`.venv312`和NumPy，在CPU执行，无新依赖。当前不等待真人回传，执行命令和评审来源见[全量AI评审说明](module2_ai_review.md)。

上一轮开发补充与评审准备无需加载模型：复用现有httpx/Pillow下载并核验原数据图片，历史A/B离线包在浏览器打开`START.html`即可，收件合并统一使用`.venv312\Scripts\python.exe`。该准备阶段未启动Ollama或嵌入模型，未新增依赖。64件新候选及59项优先评审入口见[准备与运行说明](module2_catalog_review.md)。

2026年10月22日常驻服务补充：使用既有`.venv312`、Qwen3-VL-Embedding-2B和286件开发商品索引，启动一次后复用模型与索引，每次查询重新编码。启动命令为`.\.venv312\Scripts\python.exe scripts\serve_retrieval.py --port 8765`，页面地址`http://127.0.0.1:8765/`；仅监听本机，不安装服务或设置开机启动。当前完整冷启动8.640秒；48次顺序HTTP请求p50为0.138秒、p95为0.263秒，不含冷启动及浏览器渲染，尚非正式性能验收。机器复核用的生成模型已卸载；检索服务运行期间嵌入模型持续占用显存。停止命令、API及本轮输出见[常驻服务与技术记录](module2_continuation.md)。

2026年10月22日机器开发实测补充：当前本地Ollama为0.40.6，`qwen3.5:4b-q4_K_M`由llamacpp平台子清单绑定模型与视觉projector，完整摘要与核验实现在`src/multimodal_ecommerce/machine_teacher.py`及本轮协议。下方旧版本表保留历史快照；原单层存储检查未改。新增语义候选仍使用原Qwen3-VL-Embedding-2B，见[环境与执行记录](module2_machine_assist.md)。

2026年10月22日评审环境补充：新增离线HTML/JS评审包，浏览器打开解压后的`START.html`即可，不依赖Ollama或GPU；Python收件、合并与查询确认导入统一使用`.venv312\Scripts\python.exe`。验收候选仅准备数据，未加载模型。入口与完整命令见[网格评审安排](module2_business_review.md)。

2026年10月14日范围更新：当前仅负责模块二多模态商品检索。本文保留历史配置或整体规划；当前实施与验收以 [模块二增强方案](module2_technical_plan.md) 和 [当前进度](current_progress.md) 为准。检索使用本地Transformers嵌入模型，Ollama生成链路的历史版本差异不作为模块二质量提升的前置任务。

整理日期：2026年10月10日，Asia/Shanghai。用途：团队同步开发环境、核对模型与数据路径，以及运行现有基线。项目采用 Windows、Python 独立环境、Ollama 本地生成和 Transformers 图文嵌入两条推理链路。

当前 Python 与 CUDA 可用，Ollama 本地接口可访问。Ollama 服务与模型摘要已偏离历史冻结配置，新实验前须重新确认版本与实际模型。本文的加载性能引用历史运行，不表示本次重新完成了推理验收。

## 本机配置与版本

| 项目 | 配置 | 核验范围 |
|---|---|---|
| 项目根目录 | `E:\code\Multimodal_E-commerce` | 当前工作目录 |
| 操作系统 | Windows 10 | 历史环境记录 |
| CPU与内存 | i7-12700H，约32GB内存 | 阶段一记录 |
| GPU | RTX 3070 Laptop，8192MiB显存 | 本次查询 |
| NVIDIA驱动 | 610.88 | 本次查询 |
| Python | 3.12.3，`.venv312\Scripts\python.exe` | 本次查询 |
| PyTorch与Torchvision | `2.8.0+cu128`／`0.23.0+cu128` | 本次查询 |
| CUDA运行时 | PyTorch报告12.8，`cuda_available=true` | 本次查询 |
| Transformers | 4.57.1 | 本次查询 |
| Accelerate与Qwen工具 | 1.12.0／qwen-vl-utils 0.0.14 | 本次查询 |
| Ollama服务 | 0.40.1，`http://127.0.0.1:11434` | 本次接口查询 |
| 完整依赖锁 | `requirements-retrieval.lock`，59包 | 当前锁文件 |

`requirements.lock` 是阶段一的35包旧锁；同步当前完整环境应使用 `requirements-retrieval.lock`。项目另有 `.venv/`，团队命令统一使用 `.venv312/`，避免调用错解释器。本机 CUDA PyTorch 已可用，现有运行链路无需额外安装 CUDA Toolkit；其他机器仍应单独检查驱动和 CUDA 可用性。

## 路径与配置文件

| 路径 | 用途 |
|---|---|
| `configs/local.json` | 原始数据目录、本地接口、生成模型名、上下文和图像配置 |
| `configs/embedding_model.json` | 嵌入权重路径、模型与代码revision、代码哈希和预处理参数 |
| `configs/copy_recipe_v3.json` | 历史文案实验冻结配置 |
| `Alibaba阿里算法实习/` | 原始商品、DeepFashion2、FashionAI和需求资料，只读 |
| `data/` | 标准化记录、图片副本、转换标注及向量索引 |
| `artifacts/` | 实验输出、清单、核验记录和评审页面 |
| `vendor/qwen3_vl_embedding/` | 固定版本的官方嵌入参考代码 |
| `.cache/`、`tmp/` | 项目缓存与临时文件 |
| `E:\LLM_for_Ollama` | 共用的Ollama模型根目录 |
| `E:\LLM_for_Ollama\external\Qwen3-VL-Embedding-2B` | Transformers加载的嵌入权重 |

同步到其他电脑时，应保持原始数据与派生目录分开。生成权重根目录由 `configs/local.json` 指定，嵌入路径由 `configs/embedding_model.json` 指定；下载脚本还包含固定E盘路径，需要一并检查。当前脚本和路径规则按 Windows 工程编写。

## 两条模型链路

生成使用 `qwen3.5:4b-q4_K_M`，通过本地Ollama接口请求；历史下载约3.39GB。代码固定 `num_ctx=4096`、`num_predict=512`、`temperature=0`、`seed=42`、`think=false`、`keep_alive=2m`，图像最大边768。`configs/local.json` 同时记录部分参数，但当前请求选项也写在 `ollama_client.py` 中；修改时需核对实际请求和冻结文件，不能只改JSON。

检索使用 `Qwen/Qwen3-VL-Embedding-2B`，经本地Transformers与PyTorch运行。配置为BF16、batch=1、最大262144像素、最大2048 token，输出2048维向量。权重revision为 `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`，参考代码revision为 `393e2978d27852b0d0230d6994f37f9c15bed73c`。适配器检查参考代码哈希、只读本地权重，CUDA不可用或输入超长时明确失败。

两条链路顺序使用8GB显存。检索入口会先停止本项目生成用千问，再加载嵌入模型；生成模型请求后约两分钟自动卸载。历史嵌入峰值PyTorch分配约4.02GiB，不能当作整卡总显存。已有其他大型模型不应同时驻留。

## 已有环境的日常检查

在项目根目录打开PowerShell，直接调用解释器，不需要激活脚本或修改系统执行策略：

```powershell
Set-Location 'E:\code\Multimodal_E-commerce'
.\.venv312\Scripts\python.exe --version
nvidia-smi
Invoke-RestMethod http://127.0.0.1:11434/api/version
Invoke-RestMethod http://127.0.0.1:11434/api/tags
.\.venv312\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv312\Scripts\python.exe scripts\setup_retrieval.py --dry-run
```

若本地接口不通，先启动Ollama应用；使用命令行部署时，在专用终端运行 `ollama serve`，并避免重复启动已有服务。设置客户端 `OLLAMA_MODELS` 不会改变已运行服务的存储目录，服务进程必须使用预期设置。

项目诊断命令如下，它们会写入对应的 `artifacts/environment.json` 和 `artifacts/model_storage.json`。保留历史证据时先备份这两个文件。

```powershell
.\.venv312\Scripts\python.exe scripts\run.py doctor
.\.venv312\Scripts\python.exe scripts\run.py verify-storage
```

`verify-storage` 结合服务返回路径、磁盘manifest与blob核对实际存储目录；不要仅凭环境变量认定权重位置。

## 新电脑重建环境

前置条件是可用的Python 3.12、uv、Ollama和NVIDIA驱动。以下以本机路径为例；已有正常环境无需重建。

```powershell
Set-Location 'E:\code\Multimodal_E-commerce'
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
uv venv --python 'E:\Python 3.12\python.exe' .venv312
.\.venv312\Scripts\python.exe scripts\setup_retrieval.py
```

安装脚本按完整锁文件同步，使用PyTorch cu128索引，缓存和临时目录放在项目内。`scripts/run.py` 还会设置HF、Torch、CUDA等项目缓存路径并加载 `src/`；按本文入口运行即可。机器能够访问缓存时，可用 `setup_retrieval.py --offline --dry-run` 检查离线同步条件。

先配置Ollama服务使用预期权重根目录并核验，再按需下载生成模型：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py verify-storage
ollama pull qwen3.5:4b-q4_K_M
```

模型标签可能对应不同摘要，拉取后必须记录摘要。嵌入模型可按已有配置下载：

```powershell
.\.venv312\Scripts\python.exe scripts\download_embedding.py
```

下载脚本要求E盘至少15GiB可用空间，会下载权重、参考代码并写回嵌入配置。已有权重无需重复执行。迁移时保留revision与哈希，确保索引与编码配置一致。

## 数据和功能加载

已有数据直接复用。初始样本包含72条记录与24张本地图像；`prepare-samples` 用于初次建立样本副本。以下初始化和冒烟命令会写入固定数据或证据路径，重跑前检查并保留已有产物：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py prepare-samples
.\.venv312\Scripts\python.exe scripts\run.py smoke
```

`smoke` 测试文字回答、结构化文案与图片理解，需要真正加载生成模型。首次加载较慢。历史三项请求已完成；本次仅检查环境与接口。

单件文案使用当前合同v2入口，输出选择新文件名：

```powershell
.\.venv312\Scripts\python.exe scripts\run.py generate --product configs\example_product.json --output artifacts\handoff_content_001.json
```

查看顶层 `status`，结合来源与规则进行人工复核。历史 `generate-batch` 和冻结文案运行入口不包含全部合同v2新增检查，不能互换使用。

检索示例显式选择原300件索引，避免误用默认36件小索引：

```powershell
.\.venv312\Scripts\python.exe scripts\retrieve.py search --index data\index_stage2_v1 --text "红魔游戏手机" --top-k 3 --output artifacts\handoff_search_001.json
```

输出JSON和同名HTML。图片查询用 `--image` 替换 `--text`，并按需指定 `--exclude-id` 排除自身。该索引仍对应旧分类。新分类数据需使用独立目录重建索引，模型或预处理配置变化时也必须重建。

## 当前与历史配置差异

| 项目 | 历史冻结文案实验 | 本次接口检查 |
|---|---|---|
| Ollama服务版本 | 0.34.1 | 0.40.1 |
| 生成模型名 | `qwen3.5:4b-q4_K_M` | 列表中有两个同名条目 |
| 模型摘要 | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` | `3c3adf637187c95955232acd3ed05b9ea45aeb2a5f953e631c38174a46b5b56a` 与 `91fa08b9fedfbbda924cd71febc5f6adbf31503cf7811df817dcf93bd56b08ff` |

列表信息不足以确认同名请求实际选中哪个条目，也不足以认定权重内容变化的原因。新实验前核对 `/api/show`、manifest与实际blob，记录选中模型，并重新做加载验证。当前配置不满足原v3冻结版本条件，不能直接宣称复现了原实验；历史冻结文件与输出应原样保留，新的配置和结果另建版本。

本次核验快照见 [environment_handoff_20261010.json](../artifacts/environment_handoff_20261010.json)。历史完整加载记录见 [阶段一验证报告](stage1_validation.md)；整体设计见 [阶段一整体技术方案](stage1_technical_design.md)。


## 2026年10月22日独立评测环境

模块二盲测统一使用 `.venv312\Scripts\python.exe`；项目中的 `.venv/` 缺少当前图像检索依赖，不用于此流程。新入口 `scripts/blind_retrieval.py` 使用项目内缓存与临时目录，从本地权重加载Transformers嵌入模型，GPU编码与只读源数据准备分开执行。

本轮在数据推理前将模型实际权重、处理器与分词器文件哈希，Python及关键依赖版本，策略代码、配置和历史暴露清单写入 `artifacts/module2_blind/recipe.json`。验证命令如下，它只校验冻结状态，不重新运行推理或开封答案：

```powershell
.\.venv312\Scripts\python.exe scripts\blind_retrieval.py verify
```

冻结期间更改模型文件、依赖、相关代码或已登记数据会导致预测或评分拒绝继续。同一盲测不能通过更换输出目录重复执行。完整运行顺序、数据角色和评分约束见 [模块二独立盲测与正式验收](module2_blind_evaluation.md)。


## Precision开发实验新增模型

2026年10月22日新增 `configs/reranker_model.json` 与 `configs/dino_model.json`，分别固定完整Qwen3-VL-Reranker-2B和DINOv2-small的revision、实际文件哈希与推理配置。权重位于既有 `E:\LLM_for_Ollama\external` 下各自的新目录，未覆盖原嵌入模型；原Python依赖未升级。

当前精排原型顺序加载和释放嵌入、重排模型，加载与校验成本单独记录。30对配对精排过慢，十对边界精排也尚未通过完整服务验收；默认入口不变。原型仅面向服饰图搜同款，不能直接用于三品类正式验收。参数、完整正负结果及人工开发标注入口见 [Precision专项结论](module2_precision_improvement.md)。


## 模块二训练环境与新增入口

2026年10月22日已使用既有`.venv312`和本地Qwen3-VL-Embedding-2B完成训练数据编码，无新增包或权重下载。整图与ROI/中心放大裁切由GPU编码，轻量度量、下游注意力头和实拍域适配在CPU训练；大模型参数冻结。

新入口为`scripts/prepare_metric_training.py`、`scripts/train_retrieval_metric.py`、`scripts/train_region_attention.py`；它们执行本次固定训练协议，已存在阶段会拒绝覆盖。人工导出文件用`scripts/import_precision_labels.py --reviewed "实际文件完整路径.jsonl"`校验和训练，未判断项不参与损失。

所有缓存、训练副本和模型仍放在项目内；原始数据只读，既有外部大模型路径不变。训练数据角色和已使用组清单见[训练进度](module2_training_progress.md)，完整环境与权重哈希见`artifacts/module2_training/environment_verification.json`。离线总编码时间不能作为5秒端到端请求验收证据。


## Precision 60%目标开发原型

2026年10月22日沿用现有本地嵌入缓存和完整重排权重，无新增依赖或权重下载。`scripts/business_precision.py`封存并运行原标题/区域/关键词融合和业务重排，已有结果拒绝覆盖；业务指令只写入新实例，旧同款配置不改。

两个离线网格可用本地浏览器打开，文件入口见[60%目标文档](module2_precision60.md)。人工导出后使用`scripts/score_business_labels.py --reviewed "实际导出文件.jsonl" --status-only`只检查进度；移除`--status-only`才会导入、在完整标签时评分。缺失标签不算负例；正式质量指标仍为空。

本轮360对精排约165.61秒是离线阶段耗时，复用既有查询向量，不属于单次请求的端到端性能验收。网格本身不需要Node；Node仅用于本轮导出函数回归检查。

## 2026年10月23日：383件开发图库与完整精排入口

使用既有`.venv312`、完整Qwen3-VL-Embedding-2B与Qwen3-VL-Reranker-2B，本轮未新增依赖或权重。模型仍固定原revision与文件哈希、使用CUDA/BF16，顺序加载并释放。新增商品图片仅从公开alicdn源限量下载，原始数据和外部模型目录继续只读。

`scripts/retrieve_visual_coverage.py`使用单独383件开发索引，支持新文本、图片、ROI和60候选完整交叉编码精排。命令见[本轮运行说明](module2_visual_coverage.md)。`scripts/verify_visual_coverage.py`可在不运行GPU推理、不读取heldout像素的条件下重放开发排序与模型训练。

新开发索引在`artifacts/module2_visual_coverage/index/`；旧344件索引及原286件索引分别保留。开发来源、预留验收来源、AI评审与旧人工模板分开保存。启动完整模型的校验、加载及离线批处理时长不能直接当作正式端到端性能成绩。
# FG-CLIP 2实验分支补充

2026年10月23日新增独立FG-CLIP 2开发对照分支，使用完整官方Base权重、CUDA、float32、单样本推理；现有依赖直接运行，无新增安装。权重与代码保存在本项目 `.cache/open_source_models/fg-clip2-base/430fbc8a912c86fd4de601381b6245a0edab22f0`，下载和校验入口为 `scripts/prepare_fgclip2_model.py download`。该分支按阶段运行并保留原Qwen默认方法，运行配置、对照范围与复核入口见[FG-CLIP 2实验说明](module2_fgclip2_comparison.md)。以下历史环境记录继续保留。

# 主体ROI实验环境补充

2026年10月23日新增完整GroundingDINO tiny模型，固定revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`，权重位于本项目 `.cache/open_source_models/grounding-dino-tiny/<revision>`。下载及校验入口为 `scripts/prepare_grounding_model.py`；现有transformers直接运行官方模型，采用float32及PyTorch算子后端，无新增安装。检测结束释放模型，再加载已有完整FG-CLIP 2编码裁切图，按阶段使用8GB显存。原Qwen配置及默认服务不变，详见[本轮环境与对照说明](module2_subject_roi.md)。

