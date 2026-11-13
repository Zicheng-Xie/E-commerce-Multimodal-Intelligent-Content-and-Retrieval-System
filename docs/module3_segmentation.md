# 模块 3 第一阶段：服饰分割数据与基线

本阶段先交付可复现的服饰分割数据契约和评测基线。语言引导定位、FashionAI 属性和事实约束问答在此契约之上逐步接入。

## 数据契约

`src/multimodal_ecommerce/segmentation.py` 将原始 `服饰分割/{split}/annos/*.json` 转成一行一个图像的 JSONL。每个对象保留 `category_id/category_name`、`bbox`、`polygons`、`occlusion`、`viewpoint`、`scale`、`zoom_in` 和 `annotation_quality`，记录图像与标注 SHA-256。test split 允许生成记录，但明确标记为 `annotated: false`，不参与有监督评测。

## 生成数据契约

```bash
.venv-mac/bin/python scripts/module3_segmentation_data.py \
  --root 'Alibaba阿里算法实习/服饰分割' \
  --output data/module3_segmentation_v1
```

开发调试可增加 `--limit 32`。输出包含每个 split 的 JSONL、`scan_report.json` 和 `manifest.json`。扫描会拒绝缺失配对图像、越界 bbox，并将无效多边形记录为质量问题而不是静默修正。

## 当前基线边界

- 数据和 mask 指标计算不依赖 Torch，CPU 环境即可运行。
- 监督模型配置见 `configs/module3_segmentation.json`；模型训练需安装 retrieval/视觉依赖后单独执行。
- `data/stage2_fashion_clean_v1` 用于固定回归样例；官方 train/validation 用于正式训练和评测；官方 test 仅用于无标注推理。
- 后续 GroundingDINO + SAM2、FashionAI 多任务属性和证据约束问答必须复用同一 `image_id`、区域和来源字段。
