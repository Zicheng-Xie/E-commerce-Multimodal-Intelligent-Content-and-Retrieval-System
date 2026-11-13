# 模块二全量AI评审与当前推进

用户已明确：没有真人评审资源，全部由AI承担。当前开发流程不再等待真人领取或回传；原人工模板保留用于历史兼容，AI判断单独存放，不导入人工评分入口。

本轮直接查看350件商品图片与原标题，完成6,295对业务相关性判断，包括原图库全部5,143对、59个优先冲突关系和64件新商品质检。结果为154对相关、6,122对不相关、19对未知。新增64件中，58件可用、3件剔除、3件待定；生成344件可用开发目录，尚未更新运行中的286件索引。

## 开发成绩及当前阶段

本轮AI评审成绩仅供开发诊断，完整数值留在[内部实验记录](../artifacts/module2_ai_review/internal_report.md)，不作为正式验收结果对外输出。区间只反映未知标签的影响，不是统计置信区间。

后续已完成344件开发索引的实际构建与同池策略对照，见[最新开发扩容与验收隔离说明](module2_expanded_development.md)。原286件索引和默认策略保留；独立heldout仍未推理。

## 评审依据及限制

逐件主体、图文一致性、相关需求与未知项记录在`artifacts/module2_ai_review/native_audit/`，保留44张网格图及所有原标题。3件剔除项为相机专用电池、无袖背心和散装枕芯填料。3件待定项为图文无线属性冲突的耳机、影视道具键盘和POLO/T恤边界。

本地Qwen3.5批量评审完成79件后发现背景道具/配件误判，停止该批并保留原始输出，最终350件由Codex直接看图裁决。两种AI过程不等于真人双评；确定标签仍可能存在AI误判。模糊的专业舞蹈鞋、单人沙发、图文冲突等保持未知。

## 后续执行

1. 将58件AI质检可用商品构建为新的开发索引，原286件对照继续保留，在相同344件候选池实测各方法。
2. 补充缺失图片需求的开发商品，保留商品族/图像去重及暴露登记，再推进主体与配件识别、属性一致性、ROI多视图和精排。
3. 冻结最终候选、查询和评审规则后，执行独立heldout的AI盲测，继续明确标记AI评审来源。当前heldout未推理，正式指标为null，60%目标尚未达成。

## 运行及交付

```powershell
.\.venv312\Scripts\python.exe scripts\finalize_module2_ai_review.py
.\.venv312\Scripts\python.exe scripts\verify_module2_ai_review.py
.\.venv312\Scripts\python.exe scripts\report_module2_ai_review.py
```

上述命令只使用本地已保存证据和CPU训练，不下载模型、不查看验收图片、不填充人工表。原批量模型评审入口已关闭，以防覆盖最终AI裁决。模型及环境沿用`.venv312`，无需新增依赖。

- [完整指标、分模式结果和依据](../artifacts/module2_ai_review/internal_report.md)
- [当前机器可读状态](../artifacts/module2_ai_review/status.json)
- [完整AI相关性判断](../artifacts/module2_ai_review/judgments.ai.jsonl)
- [344件开发目录](../artifacts/module2_ai_review/catalog.reviewed.jsonl)
- [原64件准备阶段记录](module2_catalog_review.md)
