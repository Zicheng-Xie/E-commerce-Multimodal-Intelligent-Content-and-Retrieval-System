# MacBook M4 Pro 迁移运行说明

本次主机：Apple Silicon ARM64、48 GiB 统一内存。原 Windows 虚拟环境保留，新环境为 `.venv-mac`，Python 3.12；uv 和 Node.js 安装在项目 `.tools` 内，系统 Python 不变。

## 日常启动

在 VS Code 打开本项目后新建终端。解释器应为 `.venv-mac/bin/python`。如旧窗口仍选中 Windows 解释器，执行 Python: Select Interpreter 并选择该路径。

```bash
source .venv-mac/bin/activate
export PATH="$PWD/.tools/bin:$PWD/.tools:$PATH"
bash scripts/start_mac.sh --port 8765
```

也可在 VS Code 的“运行和调试”中选择“Mac：启动多模态检索页面”并按 F5。

服务出现启动信息后打开 http://127.0.0.1:8765/ 。按 Ctrl+C 停止。

Ollama 应先在应用程序中打开；服务地址为 `http://127.0.0.1:11434`。

```bash
ollama list
curl http://127.0.0.1:11434/api/version
python scripts/run.py doctor
```

## 模型与复建

- Ollama：`qwen3.5:4b-q4_K_M`，保留项目模型标签与 4096 上下文设置。此标签可能被上游更新，不保证与旧电脑的模型摘要一致。
- `.models/Qwen3-VL-Embedding-2B`：项目固定 revision 的嵌入模型。
- `.models/Qwen3-VL-Reranker-2B`：项目固定 revision 的精排模型。
- `.models/dinov2-small`：项目固定 revision 的局部视觉模型。
- `.cache/open_source_models`：随项目迁入的 GroundingDINO、SAM 2、FG-CLIP 权重。

```bash
bash scripts/setup_mac.sh
.venv-mac/bin/python scripts/download_models_mac.py
ollama pull qwen3.5:4b-q4_K_M
PATH="$PWD/.tools/bin:$PATH" .venv-mac/bin/python -m pytest -q
```

依赖以 `requirements-macos.lock` 为准。原 `requirements-retrieval.lock` 含 Windows CUDA 包，不应在 Mac 上直接同步。下载脚本可以续传，校验失败会报错。

## 迁移边界

核心模型适配器支持 CUDA/MPS，Mac 自动选择 MPS；未启用静默 CPU 回退。上游 vendor 源码和原始模型配置保留，Mac 运行时解析本机模型路径；已有区域索引在校验后转换 Windows 绝对路径，不改写冻结数据。

原 `.venv`、`.venv312` 和历史实验产物保留。部分封存实验、报告生成和历史验收脚本仍记录 Windows 路径、CUDA 设备名或旧硬件性能；这些历史记录不可当作 Mac 实测结果。新硬件上的浮点差异可能影响排序，测试通过不代表业务质量验收通过，也不应重写旧指标或收据。

模型尚未下载完成时不能运行实时推理。最终完成状态以本次实测记录为准，不能从文件目录存在推断权重完整。

## 2026-11-01 实测结果

- 317 项测试全部通过，61 个包的依赖一致性检查通过。
- Ollama 0.40.13 与 qwen3.5:4b-q4_K_M 已安装，文本、图片、正式结构化文案入口实跑通过。
- Qwen 嵌入、精排、DINOv2 的固定版本权重全部下载完成，LFS SHA256 与配置逐文件校验通过。
- Qwen 图文向量、Qwen 精排、DINOv2、GroundingDINO、SAM 2、FG-CLIP 均已在 Apple MPS 完成真实推理。
- 常驻检索服务启动成功；286 件商品索引校验通过，文本、图片查询与页面访问均返回 HTTP 200。
- 详细响应：`artifacts/mac_migration/smoke_1793541251383127000.json`；示例文案：`artifacts/mac_migration/generated_copy_mac.json`。
- 本次验证说明工程可以运行，不代表已完成业务质量验收；历史实验与盲测没有重新执行。
