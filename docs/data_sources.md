# 数据源与本轮样本

本文主要描述第一阶段保留的 `data/samples/`。第二阶段已在独立的 `data/localized/` 下载36张商品图，12条失败；详见 `stage2_step1.md`。下文“pending_download”指原始抽样版本，不是最新本地化状态。

三个来源均在 `Alibaba阿里算法实习/`，只读。图片、标注副本与派生文件位于 `data/samples/`。

## 商品元数据

`多模态电商数据/` 下是大型JSON对象，顶层商品ID映射 `title,label,url,video,pv`。全量约5.79GB，train约3.98GB，test约1.27GB，无本地商品图片。使用 `ijson.kvitems` 流式消费，最多扫描每split的50,000条，不整体载入内存，不合并全量与train/test。

本轮在train/test内各选3类×8条，共48条。类别规则保存在 `summary.json`：优先原始label，仅label缺失时匹配title，属启发式映射，需人工审核。`pv` 按 `#;#` 分条、`#:#` 分键值，多值及重复值保留；不把同名键覆盖。图片URL保留，`image_path=null`、`media_status=pending_download`。

## DeepFashion2服饰分割

`服饰分割/train` 有191,961张图，`validation`有32,153张图，均按同名JSON配对。validation标注目录额外的快捷方式忽略、不跟随。test有62,629张图，但相关信息文件的annotations为空，不能用来计算监督分割指标。

本轮train/validation各复制前4张，共8张、12个实例。源 `[x1,y1,x2,y2]` 转为COCO `[x,y,width,height]`；多边形保留，area按栅格化多边形并集像素数计算。COCO的 `file_name` 相对于 `data/samples/`，不是annotations目录。类别ID保留原始体系，未擅自映射成PDF后续要求的8类。pair_id可能跨split重复，记录ID使用数据集/split/源ID命名空间。

## FashionAI服饰属性

`服饰细分属性/round1_fashionAI_attributes_test_a` 有10,080张图片，8类属性。真实标签来自 `Tests/round1_fashionAI_attributes_answer_a.csv`，类别顺序来自同目录README。

本轮每属性复制2张，共16张，全部保留 `test_a` 与 held-out用途，不转成训练数据。`y`表示选中类别；保留完整 `n/y/m` 原始编码，`m`的语义未从本地资料确认，不当作负类。`answer_mock.csv`是预测分数示例，不能作真值；question文件中的问号不是标签。

## 可复现性与限制

`samples.jsonl` 保存源路径、源ID、split、属性及图像相对路径；`manifest.json`保存32份已复制图片/标注文件的SHA256。复制后重新计算SHA256，图片通过Pillow解码检查。目标路径经过resolve并拒绝写入保护目录、父级或项目外；源路径不允许软链接/junction逃逸。

本轮抽样按文件/源顺序，便于复现，存在选择偏差。72条记录只有24张本地图像，视觉数据未提供自然语言caption，`text=null`。没有伪造图文对、没有训练或正式检索评测。下一步应解决商品图片本地化及真实标注，不能仅扩大当前样本数就宣称第二阶段完成。
