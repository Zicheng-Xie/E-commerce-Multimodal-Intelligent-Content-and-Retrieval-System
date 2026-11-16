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

## 2026-11-16：v2 数据修复与 CPU 真实训练测试

正式使用 `data/module3_segmentation_official_v2`，保留旧 v1，不覆盖历史哈希。

- 官方 train 191,961、validation 32,153、test 62,629 张的归属不变。
- v2 使用 xywh bbox、保留原始 bbox 与全部 polygon；严格检查有限数值、边界与非零面积，部分坏 polygon 不再静默删除。
- train 有 2,688 个 polygon 异常、3 个 bbox 异常（问题计数可重叠），涉及 2,580 张图；训练可用 189,381 张。默认整图排除异常，避免未标注服饰被当背景。validation 全部可用。
- test 无标签且不参与训练或验证。缺失训练/验证标注会失败。异常记录见 issues.jsonl。
- 修复 mask 像素求和单位，补充 mask 尺寸检查和 bbox IoU。10 项相关测试通过。

复现发布：

```bash
.venv-mac/bin/python scripts/publish_module3_official.py --root 'Alibaba阿里算法实习/服饰分割' --output data/module3_segmentation_official_v2
```

目录已存在时拒绝覆盖。训练测试：

```bash
.venv-mac/bin/python scripts/module3_smoke.py --data data/module3_segmentation_official_v2 --output artifacts/module3_cpu_smoke_v1
```

当前设备 torch 2.8.0，CUDA/MPS 均不可用。使用随机初始化三层 CNN，32 张 train、16 张 validation，96×96 二值服饰并集，5 轮真实反向传播。报告、checkpoint、16 份预测和三联图已保存。重载权重及单图/批量输出一致性检查通过。

训练 loss 从 0.6680 降到 0.5245；验证 mean IoU=0.08389，低于全图前景基线 0.22306，空预测基线 0。当前为工程 smoke run，未达到效果验收。分组指标为含该类/属性的图像整体并集 IoU，不能解释为该实例的 AP 或类别 IoU。三联图从左到右为原图、GT、预测。

尚未完成：实例匹配/AP、正式监督式预训练模型适配、完整 validation 验收、训练恢复（不仅权重重载）、独立预测 CLI、test 正式推理。下一步应先完成这些接口与算力准备，再进行完整训练。GitHub module-3 分支尚未同步本次 v2 变更。

## 2026-11-16：实例分割模型、AP 与训练恢复

新增 `scripts/module3_instances.py`，使用 torchvision 0.23 的 Mask R-CNN ResNet50-FPN，官方 COCO 初始化权重已下载到 `.cache/module3/`，SHA256 为 `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`。加载完整预训练模型后替换为 13 类服饰加背景的 box/mask heads；新头随机初始化，不能把原 COCO 模型分数视为服饰成绩。

当前 CPU 配置：输入缩放 min=128/max=192，冻结 backbone，SGD lr=.001/momentum=.9，固定种子42。RPN候选上限及每图20实例输出限制固定于代码。这是低分辨率开发配置，尚未运行标准分辨率或全量训练。

实例 AP：`instance_metrics.py`，按类别、图片内一对一贪心匹配，按全局置信度排序，IoU=.50:.05:.95，101点插值；保留漏检、误检和重复预测。无GT类显示 null。没有 COCO crowd/面积区间规则，指标为项目内 mask AP，不是官方 COCO AP。测试覆盖完美预测、漏检、错类、提前误检和重复预测。

已实际运行：

- `artifacts/module3_maskrcnn_smoke_v1`：4张train训练1轮、2张validation预测；分类/框/掩码/RPN损失均真实计算。
- `artifacts/module3_maskrcnn_resume_v1`：从 checkpoint 恢复模型、优化器动量、epoch和Torch RNG，再训练1轮。
- `artifacts/module3_maskrcnn_continuous_v1`：独立连续训练2轮，与续训模型全部参数逐项完全相等。验证收据在 resume_verification.json。
- `artifacts/module3_maskrcnn_predict_v1`：独立 predict CLI，batch=2，与续训时batch=1的全部预测数组完全一致。
- 两张验证图项目内 mask AP50/AP均为0。仅工程验证，样本很少，不代表全量验证集效果。test未访问。
- 13项相关测试通过。

训练命令（输出目录必须不存在）：

```bash
.venv-mac/bin/python scripts/module3_instances.py train --data data/module3_segmentation_official_v2 --output artifacts/module3_maskrcnn_next --weights .cache/module3/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --limit 32 --eval-limit 16 --epochs 2
```

续训用 `--checkpoint artifacts/module3_maskrcnn_smoke_v1/checkpoint.pt` 替代 `--weights`。独立推理用 `predict --checkpoint ... --split validation --batch-size 2`；每图输出 NPZ（原图尺寸 masks、xyxy boxes、labels、scores），report.json记录来源与输出哈希。数据内部仍为xywh，仅模型边界转为xyxy。测试集推理入口已提供但未执行正式test推理。`--limit 0`/`--eval-limit 0`表示完整split；CPU上不建议无预算启动全量实验。

下一阶段：确定有GPU的训练环境；在固定、覆盖13类的开发子集上建立效果基线，再运行完整官方validation，并补充bbox AP、IoU/Dice及遮挡/视角分组、可视化失败案例和checkpoint选择规则。当前没有完成这些正式验收项，也没有向GitHub推送本轮代码。

官方接口依据：https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html

## 2026-11-16：100步学习门槛实验

入口：`scripts/module3_learning.py`。配置已更新为v2数据契约与Mask R-CNN。固定哈希覆盖清单为211张train、106张validation，每类至少20/10实例；GT回放13类AP全部为1。选择的四图含单/多目标，100步过拟合最高AP50=0.04208、最终AP50=0，未通过0.8门槛，按协议未执行开发子集5轮训练。test未使用，GitHub未更新。

每20步的checkpoint、优化器/随机状态、配置/子集哈希、逐步损失与分类梯度、预测NPZ和GT/预测三联图已保存在 `artifacts/module3_learning_v1`。17项测试通过，独立CLI从80步续训到100步与连续运行全部参数完全一致。

详细诊断见 `artifacts/module3_learning_v1/progress.md`。当前结论为低分辨率配置尚未证明可学习，主要失败为漏检；根因仍需候选框召回和分类分数分阶段诊断。不得把该失败隐去后直接扩大数据量。

## 2026-11-16：分辨率/训练步数对照与MPS确认

诊断入口 `scripts/module3_diagnose.py`，产物 `artifacts/module3_diagnosis_v1`。同四图、同种子与初始化：原128/192@100步AP50=0；仅提高到256/384@100步AP50=0.62624、AP=0.19022；原低分辨率延长到500步AP50仍为0。候选框Recall@0.5由3/6变为4/6，低分辨率正确类别概率低、背景概率高，是漏检的直接环节。高分辨率有效但仍未过0.8门槛，不扩大训练。

此前“MPS不可用”仅限沙箱内；沙箱外检测MPS available=True，单元素张量、Mask R-CNN推理与合成单图前向/反向均成功。不能再将本机描述为硬件只有CPU。当前尚无真实服饰MPS训练或速度/内存容量实测，下一步应适配显式设备选择再做小规模验证，无需立即租用服务器。

详见该目录 progress.md、mps_receipt.json 和各组 report.json。17项测试通过，test保持隔离，未同步GitHub。

## 2026-11-16：显式CPU/MPS接入与真实四图对照

`module3_instances.py` 新增 `--device cpu/mps`，默认cpu；不可用的MPS会明确报错。模型、输入和target统一放置到选定设备，预测转回CPU后评测/保存。checkpoint保存CPU/MPS随机状态；跨设备加载先映射CPU，跨设备结果不承诺逐位一致。

`module3_device_check.py` 使用固定四图、256/384分辨率训练100步。CPU耗时21.59秒、AP50=0.62624；沙箱外真实MPS训练耗时150.94秒、AP50=0.37624。MPS参数位于mps:0，未启用CPU fallback，checkpoint重载预测检查通过。逐步采样最高live/driver内存约350.85/1234.73 MiB，不是完整进程峰值。19项测试通过。

详细报告见 `artifacts/module3_device_comparison_v1/progress.md`。该单次四图实验中MPS较慢，不能外推正式规模性能。两组均未通过0.8门槛，暂不扩容。设备访问问题已确认解决，后续四图诊断优先CPU，继续处理候选召回与ROI背景偏置。未使用test或validation作本轮训练诊断，未同步GitHub。

通用训练入口示例（需在能访问GPU的进程执行）：

```bash
.venv-mac/bin/python scripts/module3_instances.py --device mps train --data data/module3_segmentation_official_v2 --output artifacts/module3_mps_next --weights .cache/module3/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --limit 4 --eval-limit 2
```

此通用入口默认分辨率仍为128/192；上述256/384的设备对照由 `module3_device_check.py` 单独实现，不能混为同一配置结果。
