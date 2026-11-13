# 模块 3：服饰细粒度语义与分割

模块 3 面向服饰商品的细粒度视觉理解，当前主线聚焦于官方标注数据整理、分割格式转换、数据质量扫描和可复现发布，为主体定位、ROI 检索、细粒度属性分析提供数据基础。

## 当前能力

- 整理 FashionAI 与 DeepFashion 风格的服饰标注数据。
- 统一 JSON / JSONL / COCO 兼容结构。
- 检查图片、标注、类别、路径和多边形一致性。
- 生成 manifest、scan report 与文件哈希，保留排除原因。
- 为模块 2 的主体区域分析和 ROI 检索提供可复用数据接口。

## 数据流程

```mermaid
flowchart LR
 A[官方图片与标注] --> B[路径和格式扫描] --> C[类别/多边形/完整性检查]
 C --> D{质量通过?}
 D -- 是 --> E[统一发布目录\nmanifest + scan report]
 D -- 否 --> F[记录排除项与原因]
 E --> G[主体分割 / ROI 检索 / 属性分析]
```

## 工程入口

| 路径 | 作用 |
| :--- | :--- |
| `src/multimodal_ecommerce/segmentation.py` | 分割数据结构与处理逻辑 |
| `scripts/module3_segmentation_data.py` | 数据准备、转换与检查 |
| `scripts/publish_module3_official.py` | 发布通过扫描的数据版本 |
| `configs/module3_segmentation.json` | 数据版本与处理策略配置 |
| `data/module3_segmentation_official_v1/manifest.json` | 已发布数据清单 |
| `data/module3_segmentation_official_v1/scan_report.json` | 数据扫描结果 |
| `tests/test_segmentation.py` | 模块 3 单元测试 |

## 技术栈

Python 3.12+ · NumPy · Pillow · JSON/JSONL · COCO 标注结构 · pytest · Black · isort · Flake8 · SHA-256 完整性校验

## 运行示例

```bash
python scripts/module3_segmentation_data.py --help
python scripts/publish_module3_official.py --help
python -m pytest -q tests/test_segmentation.py
```

## 当前边界

当前完成的是数据工程与质量门禁，不把结构有效性直接当成语义准确率。正式的服饰属性准确率、主体分割质量和独立视觉验收，需要在冻结评测集后单独报告。

本开发分支：`module-3-dev`。
