# 第二阶段：本地图文检索基线

2026-09-23。已完成模型部署、36件商品图文特征提取、文本与图片查询、余弦排序、可选图文融合与结果页面。小样本工程链路可运行，但没有正式相关性标注，不能宣称PDF中的检索质量目标已达标。

## 模型与存储

生成模型继续使用Ollama的 `qwen3.5:4b-q4_K_M`。真正图文嵌入使用官方 **Qwen/Qwen3-VL-Embedding-2B**，通过本地Transformers后端运行。

这是对“Ollama优先”的兼容性处理：[Ollama公开嵌入接口](https://docs.ollama.com/api/embed)目前定义的是文本输入；找到的[社区Qwen图文嵌入模型页](https://ollama.com/batiai/qwen3-vl-embed-2b)明确指出图像需要其他后端。没有把文字嵌入或生成的图片描述冒充图像特征。

- 权重目录：`E:\LLM_for_Ollama\external\Qwen3-VL-Embedding-2B`。与Llama权重同属E盘模型根目录，但由Transformers读取，**不会出现在 `ollama list` 中**。
- 模型revision：`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`。
- 权重大小：4,255,140,312 bytes，SHA256与Hugging Face下载元数据一致，证据在 `artifacts/embedding_download.json`。
- 官方代码revision：`393e2978d27852b0d0230d6994f37f9c15bed73c`；未改动的参考实现与Apache-2.0许可在 `vendor/qwen3_vl_embedding/`，项目适配代码在 `embedding.py`。
- Python 3.12.3，torch 2.8.0+cu128，torchvision 0.23.0+cu128，transformers 4.57.1，qwen-vl-utils 0.0.14。
- BF16、SDPA、batch=1、最大262,144像素、最多2,048输入token、2048维归一化向量。

项目适配层遇到坏图或预处理异常会报错，禁止沿用上游代码中“失败后编码NULL文本”的兜底。模型和tokenizer从已下载的本地目录读取。检索前只卸载本项目Ollama生成模型，检索进程退出后释放嵌入模型显存，不自动停止其他模型。

## 运行命令

在项目根目录执行。索引已建立，可直接查询：

```powershell
# 中文文本直接匹配真实图片向量
.\.venv312\Scripts\python.exe scripts\retrieve.py search --text "红魔游戏手机" --top-k 3

# 图文等权融合；不是微调、RAG或学习排序
.\.venv312\Scripts\python.exe scripts\retrieve.py search --text "儿童彩色小挎包" --target fused --top-k 3 --output artifacts\bag_search.json

# 图片搜商品；示例显式排除输入图片对应的商品
.\.venv312\Scripts\python.exe scripts\retrieve.py search --image data\localized\images\ef5eb9d6a8d9ef9f5aee2539.webp --exclude-id product/train/623181777103 --top-k 3

# 可选品类过滤
.\.venv312\Scripts\python.exe scripts\retrieve.py search --text "黑色鞋靴" --category "服饰穿搭"

# 复建索引和诊断时用新目录，保留旧结果
.\.venv312\Scripts\python.exe scripts\retrieve.py build --index data\index_qwen_v2
.\.venv312\Scripts\python.exe scripts\retrieve.py check --index data\index_qwen_v2 --output artifacts\retrieval_v2
```

单次查询保存JSON和同名HTML；默认 `artifacts/search_latest.json` / `.html` 会被下次查询覆盖，需要留档时指定新 `--output`。图片查询默认允许找回自身；需要相似商品时显式传 `--exclude-id`，本轮图片示例均已排除自身。

`--target image` 是默认纯图片余弦分数；`text` 对比标题向量；`fused` 为 `0.5×图片分数 + 0.5×标题分数`。品类是显式过滤条件，没有利用测试答案人为加分。36条规模使用NumPy精确排序，无需近似索引；后续规模增大时可替换为FAISS。分数不是概率或置信度，没有强行将不相关的低分结果判为相关。

## 实测结果

索引包括36条（train 19、test 17），保留原始split；每条独立编码图片与商品原始标题，共72次编码。图片编码不输入标题或标签，使用统一语义空间实现真正的跨模态匹配。

| 工程检查 | 实际结果 |
|---|---|
| 模型设备/精度 | cuda:0，torch.bfloat16 |
| 建索引耗时 | 约7.18s，不含模型加载 |
| 模型加载 | 约2.65s；当时系统文件缓存状态未控制 |
| PyTorch峰值分配/保留 | 4118.84 / 4130MiB，约4.02 / 4.03GiB |
| 同向量索引自检 | 36/36找回自身；仅验证索引连线正确 |
| 文本示例查询 | 预加载后约0.024–0.373s，包含编码与排序 |

上述是单次小样本实测，不是P95、压力测试或端到端冷启动SLA。峰值只统计PyTorch分配器，不包括驱动和其他程序占用。检查时C盘剩余约74.21GiB、E盘约84.04GiB；新增依赖/模型/缓存均配置在E盘。

标题→配对图片诊断（候选池为全部36张）：

| 查询来自 | 数量 | 配对Hit@1 | 配对Hit@5 | 配对Hit@10 |
|---|---:|---:|---:|---:|
| train标题 | 19 | 18/19（94.74%） | 19/19 | 19/19 |
| test标题 | 17 | 16/17（94.12%） | 17/17 | 17/17 |

两个未排第一的配对图片均排第二。这里每条查询只检查“原始商品对应图片是否找回”，**不是PDF定义的Recall@10或Precision@10**：尚未标注全部相关商品，不能把其他相似商品一概当负例。模型未在这些样本上训练，但样本已被查看用于工程调试，正式评测仍应另建未使用的查询与相关性标注集。

## 成功与失败示例

- “儿童绘本收纳书架”“黑色马丁靴”“红魔游戏手机”“苹果平板键盘”“木质水果盘展示架”在这批示例中均把预期商品排第一；这些是已查看候选集后的演示查询，不是独立评分集。
- “儿童彩色小挎包”纯图片排序未在前三返回对应包；固定等权融合后该商品排第一（融合0.3987，图片0.1764，标题0.6211）。这是单例对比，不是整体效果提升证明。
- 对照原图，儿童包仅占模特穿搭照的一部分，整图特征可能更强调衣服；区域裁剪可以作为后续实验方向，本次未增加定位/分割模型。
- 排除自身的红魔手机图片查询，前三返回鼠标、防尘塞和充电宝。小库里没有另一台同款手机，返回结果不等于真正相关商品，后续需要相关性标注和拒绝低相关结果的策略。

可直接打开 `artifacts/retrieval_v1/results.html` 查看六条文字查询和三条排除自身的图片查询；完整数值保存在同目录 `report.json`。融合结果单独在 `artifacts/search_bag_fused.html` / `.json`。

## 复现与完整性

`configs/embedding_model.json` 固定模型、代码revision及预处理参数。`data/index_qwen_v1/manifest.json`记录数据哈希、模型配置、向量和记录文件哈希。查询发现模型配置不一致时要求重建，索引读取时检查文件哈希并禁止pickle加载。

完整59包锁定在 `requirements-retrieval.lock`，旧 `requirements.lock` 仅保留第一阶段35包。已有环境不要用旧锁文件sync，否则会移除检索依赖。复建虚拟环境后执行：

```powershell
.\.venv312\Scripts\python.exe scripts\setup_retrieval.py
# 离线检查当前环境与完整锁文件一致
.\.venv312\Scripts\python.exe scripts\setup_retrieval.py --dry-run --offline
# 模型已有，无需重复下载；重装时需要联网并写入E盘模型根目录
.\.venv312\Scripts\python.exe scripts\download_embedding.py
```

下载脚本沿用本机现有代理访问官方站点，不修改系统代理。CUDA wheel从PyTorch官方索引取得，无需额外安装CUDA Toolkit。

本轮完成31项测试及格式、静态检查，覆盖归一化、排序、过滤、自身排除、非法向量、索引文件修改、指标分母和HTML转义。真实GPU运行另有索引和诊断报告证明。第三阶段仍未开始。

原始资料目录核对通过：531,063个文件的路径、大小、纳秒修改时间与基线一致，全部PDF的SHA256一致，差异列表为空，证据在 `artifacts/source_integrity.json`。非PDF文件未逐个进行内容哈希，因此此项不等同于全量内容哈希校验。四份HTML结果页引用的40处本地图片均存在。

下一步：定义独立的多相关商品查询标注、扩展候选库，完成真正的Recall@10/Precision@10和人工文案质量评分，形成第二阶段正式评测报告。

参考：[Qwen官方实现](https://github.com/QwenLM/Qwen3-VL-Embedding)、[官方模型](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)。
