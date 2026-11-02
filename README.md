# E-commerce Multimodal Intelligent Content and Retrieval System

## Overview
An AI project for e-commerce content generation, multimodal product retrieval, and fine-grained fashion understanding.

## Current Stage
Phase 1: Business Research and Technical Solution Design

## Modules
1. Intelligent Product Content Generation
2. Multimodal Product Retrieval
3. Fine-grained Fashion Semantic Understanding

## Project Structure
- docs/: research and technical documentation
- src/: source code
- notebooks/: experiments
- scripts/: utility scripts
- configs/: model and experiment configurations

## 商品内容智能生成子系统

### 功能与算法

使用 Qwen3.5-4B 预训练生成模型，结合商品属性解析、
品类知识规则和提示词约束，生成三类简体中文电商内容：

- 商品标题：呈现商品类型、品牌型号和关键特征。
- 详情文案：连贯介绍商品属性、设计及已知功能。
- 核心卖点：提炼商品主要特征。

支持3C数码、服饰穿搭、家居日用品类的表达风格适配。
服饰商品还支持从已有属性中提取结构化标签。

处理流程：

官方格式商品数据 → 属性解析与事实整理 → 品类提示词构建
→ 模型生成 → 格式、事实和属性覆盖检查 → 必要时局部修复
→ 保存文案及评测文件

当前实现属于“预训练模型推理 + 规则约束”，未进行专门的模型微调。
服饰标签来自商品属性映射，不是图像识别结果。

### 输入格式

保持官方数据集格式：顶层以商品ID为键，每个商品包含
title、label、pv、url、video字段。

```json
{
  "demo_001": {
    "title": "示例直筒休闲裤",
    "label": "休闲裤",
    "pv": "品牌#:#示例品牌#;#服装版型#:#直筒#;#裤长#:#长裤#;#材质成分#:#聚酯纤维65% 棉35%",
    "url": "",
    "video": ""
  }
}
```

pv使用 `#:#` 分隔属性名与属性值，使用 `#;#` 分隔属性项。
本子系统主要使用文本属性，url和video保留在输入中，
不代表当前生成流程已经使用图片或视频。

### 输出格式

每条商品保存为独立JSON文件。生成文案位于其中的result字段：

```json
{
  "title": "示例品牌直筒休闲长裤",
  "description": "这款休闲裤采用直筒版型，面料由65%聚酯纤维与35%棉混纺而成，裤长为长裤。",
  "selling_points": [
    "直筒版型，呈现直线裤腿轮廓",
    "65%聚酯纤维与35%棉混纺",
    "长裤款式"
  ]
}
```

以上为格式示例，不是模型效果评测结果。

结果目录：

```text
outputs/copywriting_v3/<运行时间>/
├── records/          # 每条商品的文案、输入、校验结果和生成耗时
├── manifest.json     # 本次样本与生成配置
├── loading.json      # 模型加载时间与设备信息
├── review.json       # 待填写的质量评分
└── summary.json      # 汇总与验收状态
```

records中的generation_seconds为该条文案的生成耗时，
包含修复重试，不包含模型加载时间。
attempts记录每次生成的耗时与输出token数量。

### 环境准备

已验证的本地环境：

- Python 3.11
- PyTorch 2.9.1+cu128
- Transformers 5.3.0
- Accelerate 1.15.0
- BitsAndBytes 0.50.2
- NVIDIA RTX 4060 Laptop GPU，8GB显存

生成和检索建议使用不同虚拟环境，避免依赖版本冲突。

在项目根目录创建生成环境：

```powershell
py -3.11 -m venv .venv-generation
.\.venv-generation\Scripts\python.exe -m pip install --upgrade pip
.\.venv-generation\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-generation\Scripts\python.exe -m pip install transformers==5.3.0 accelerate==1.15.0 bitsandbytes==0.50.2
```

将Qwen3.5-4B模型文件放入：

```text
models/Qwen3.5-4B/
```

模型目录需要包含config.json、模型权重和tokenizer相关文件。
模型权重不包含在本仓库中，需另行获取。

### 使用方法

以下命令均在项目根目录执行。
输入文件示例路径为 `data/test data/raw_samples.json`，
也可以通过--samples指定其他官方格式JSON文件。

1. 检查输入，不加载模型：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py run --samples '.\data\test data\raw_samples.json' --dry-run
```

2. 生成一条商品文案：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py run --samples '.\data\test data\raw_samples.json' --model-dir '.\models\Qwen3.5-4B' --quantization 4bit --compact-prompt --use-cache --max-tokens 384 --limit 1
```

3. 批量生成：

删除上述命令末尾的 `--limit 1`。

4. 自定义表达风格与长度：

在生成命令后添加：

```text
--style "专业简洁" --length short
```

length支持short、medium、long。
style控制文案语气，不应作为商品本身的属性。

5. 断点续跑：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py run --resume-dir '.\outputs\copywriting_v3\<实际运行目录>' --model-dir '.\models\Qwen3.5-4B' --quantization 4bit --compact-prompt --use-cache --max-tokens 384
```

请将 `<实际运行目录>` 替换为已生成目录名。
断点续跑使用该目录保存的样本与配置；
修改生成逻辑后进行对比测试，应创建新的运行目录。

6. 查看全部可用参数：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py --help
```

### 评测与限制

目标验收指标：

- 文案相关度：≥80%。
- 文案通顺度：≥85%。

规则检查和核心属性覆盖率不能直接替代相关度、通顺度评分。
需要逐条审核文案，填写review.json，并使用独立样本验证效果。

当前尚未完成正式验收。规则检查可能存在同义表达漏判、
数值匹配误报和无依据效果表述漏检。
官方商品属性也可能存在缺失或冲突，因此生成内容需复核后使用。