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

## 多模态商品检索基线

基于本地 Qwen3-VL-Embedding-2B 模型与 FAISS，实现中文文本搜商品
和图片搜商品。当前图库只编码商品图片，商品标题、类别用于展示，
尚未参与向量匹配或综合排序。

### 环境与资源

已验证环境：
- Python 3.11.7
- PyTorch 2.9.1+cu128
- Transformers 4.57.6
- NVIDIA RTX 4060 Laptop GPU，8GB 显存

其他依赖包括 numpy、faiss、Pillow、ijson、qwen-vl-utils，
具体版本见 requirements-retrieval.txt。

需要准备：
- 模型目录：models/Qwen3-VL-Embedding-2B/
- 模型辅助代码：模型目录下 scripts/qwen3_vl_embedding.py
- 官方数据：data/ecommerce_multimodal/product1m_product5m_test_id_label.json

数据准备需要联网下载图片；建库和查询使用本地模型与图片，
需要可用的 CUDA 环境。

### 运行方法

以下命令均在项目根目录、检索 Python 环境中执行。

安装依赖：

    python -m pip install -r requirements-retrieval.txt

准备100个商品：

    python scripts/multimodal_retrieval.py prepare --limit 100

构建索引：

    python scripts/multimodal_retrieval.py build

文字查询：

    python scripts/multimodal_retrieval.py search --text "木制水果盘" --top-k 3

图片查询：

    python scripts/multimodal_retrieval.py search --image "queries/example.jpg" --top-k 3

图片查询路径相对于 data/retrieval，查询图片需先放入对应目录。
更改商品清单或图片编码设置后，应重新构建索引。

### 初步评测

当前索引包含100个真实商品。

6条文字查询、每条一个指定目标的测试结果：
- Top-1命中率：66.7%
- Top-3命中率：100%

2条图库图片自身检索均排名第一。
自身检索仅验证编码与索引对应关系，不代表图库外图片检索效果。

模型单次加载、分模态预热后的性能测试：
- 文字查询平均耗时约45毫秒
- 图片查询平均耗时约11秒
- 上述耗时不包含模型加载、网络与界面开销

### 已知限制

- 查询及相关性标注样本较少，尚未完成正式业务指标验收。
- 图片查询尚未达到项目要求的5秒响应时间。
- 当前采用纯图片向量召回，尚未加入标题特征、品类属性综合排序。
- 结果以终端输出和JSON报告为主。