"""Write factual training progress documents from preserved measured results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/module2_training"


def load(relative: str) -> Any:
    """Read the serialized input from the supplied path.

    Args:
        relative: Relative used by this operation.

    Returns:
        Decoded JSON value; the caller must validate its expected schema.
    """
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the completed report and its documentation update.

    Notes:
        Writes derived artifacts through (OUT / 'internal_summary.md').write_text,
        (ROOT / 'docs/module2_training_progress.md').write_text, p.write_text.
    """
    if (OUT / "internal_summary.md").exists():
        raise FileExistsError(
            "Preserve the completed report and its documentation update"
        )
    metric = load("analysis/report.json")
    attention = load("region_attention/report.json")
    residual = load("rank_correction/report.json")
    model = load("region_attention/model.json")
    verification = load("verification_full.json")
    names = {
        "precision@10": "Precision@10",
        "recall@10": "Recall@10",
        "ap": "mAP",
        "ndcg@10": "NDCG@10",
        "mrr": "MRR",
        "hit@1": "Hit@1",
    }

    def table(summary: dict, methods: list[tuple[str, str]]) -> str:
        """Return table for this workflow.

        Args:
            summary: Summary used by this operation.
            methods: Methods used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        keys = list(names)
        lines = [
            "| 方法 | " + " | ".join(names.values()) + " |",
            "|---|" + "---:|" * len(keys),
        ]
        for label, name in methods:
            lines.append(
                "| "
                + label
                + " | "
                + " | ".join(f"{summary[name][k]*100:.2f}" for k in keys)
                + " |"
            )
        return "\n".join(lines)

    old = attention["old"]["summary"]
    new = attention["new"]["summary"]
    old_table = table(
        old,
        [
            ("原融合基线", "baseline"),
            ("未训练的三视图注意力", "frozen_three_view_attention"),
            ("可训练三视图注意力", "learned_three_view_attention"),
            ("注意力与原融合混合", "learned_attention_blend25_anchor2"),
            ("实拍图适配+三视图注意力", "ridge_query_attention"),
            ("适配与原融合混合", "ridge_attention_blend25_anchor2"),
        ],
    )
    new_table = table(
        new,
        [
            ("新组整图基线", "baseline"),
            ("未训练三视图注意力", "frozen_three_view_attention"),
            ("可训练三视图注意力", "learned_three_view_attention"),
            ("实拍图适配+三视图注意力", "ridge_query_attention"),
        ],
    )
    content = f"""# 模块二训练扩容与可训练注意力：内部开发记录

更新日期：2026年10月22日。仅供开发分析，全部成绩不能作为正式验收或对外质量宣传。

本轮实际增加150个训练商品组、50个开发验证商品组，完成轻量度量学习、可训练区域注意力、实拍图域适配及排序修正。新开发组有局部收益；旧开发集上仍未超过已有融合策略，所有候选均未通过多指标门槛。默认策略保持原样，正式验收指标仍为null。

## 数据与隔离

- 官方DeepFashion2 train前40,000份标注，按固定哈希选择5类服饰，各30组训练、10组开发验证；每组一张单件用户图及最多3张去重商品图。
- 实际434张训练图片（150查询、284商品图），146张开发验证图片（50查询、96商品图），共580张。新组仅用于训练/开发，已经暴露，后续不能转作独立验收。
- 排除历史已暴露及已开封盲测共344个原始pair组；200个新组与其无交集，训练与开发的pair也无交集。
- 逐图校验文件哈希、解码像素和dHash；跨新商品组以及旧暴露图片均拒绝dHash距离≤4的风险样本。这一筛查不能证明消除了所有语义近重复。
- 仅使用官方split/pair/style/category确定严格同款身份。并未把它伪装成人工业务相关性标签。
- 编码器输入只含匿名ID、图片路径及查询/商品角色；裁切来自像素边缘对比和几何中心，不读取源标注框、分割掩码或类别答案。

## 实际训练内容

1. 冻结Qwen3-VL-Embedding-2B，基于580张图片提取2048维真实特征。
2. 用150个训练组学习有界对角相似度，按每条查询挖掘20个困难负例，训练300步。
3. 对380张商品图补充ROI或中心放大视图，共新增455次真实裁切编码。使用整图、像素ROI、中心裁切三类视图；查询使用整图。
4. 在训练组上训练200步下游注意力头，学习2048维通道权重、区域softmax温度和整图/局部融合权重。大模型参数保持冻结；这属于下游可训练注意力，并非微调Qwen内部注意力。
5. 独立对照使用岭回归残差映射，把用户实拍特征适配到商品展示图特征。锚点和目标只来自150个训练组。
6. 后续固定权重0.5、保留前2名、候选池30，分别测试三种“训练后减训练前”的排序修正。没有修改K、候选总数、正例答案或Precision分母。

模型学习到的整图权重为{model['global_weight']:.4f}，区域温度为{model['temperature']:.4f}。完整模型、训练损失、商品组列表和固定配置见对应JSON。

## 旧开发集结果

26查询、334候选，与原融合基线在同一图库内比较。下表为百分数。

{old_table}

对角度量直接相对整图基线的Precision为{metric['old']['summary']['learned_global']['precision@10']*100:.2f}%，整图为{metric['old']['summary']['original_global']['precision@10']*100:.2f}%；但它弱于已有融合基线。三种排序修正的Precision分别为{residual['old_development']['summary']['metric_delta']['precision@10']*100:.2f}%、{residual['old_development']['summary']['attention_delta']['precision@10']*100:.2f}%、{residual['old_development']['summary']['query_adapter_delta']['precision@10']*100:.2f}%，未取得所需收益。正负结果均完整保存，没有选一项高分替代整体结论。

## 新开发验证组结果

50查询、96候选，商品组与训练组隔离，但仍是内部开发数据。每条查询只有1—3张同款正例，平均Precision@10上限为19.20%；图库规模、正例数与旧开发集不同，因此只比较本表内差值，不能横向比较两张表的绝对值。

{new_table}

实拍图适配在这个小图库上六项均值均改善，但Precision只增加0.40个百分点（500个Top10位置新增2个正例），不是大幅提升，也不能证明在其他图库稳定有效。按商品组的探索性bootstrap区间已保留，不作多重比较修正后的显著性声明。

## 门槛与结论

初版对角度量协议要求新旧两组均Precision增加2个百分点。检查发现新组基线18.40%、上限19.20%，最多只剩0.80个百分点，这个新组门槛不可达，原协议和原报告保留。

后续区域实验在区域特征编码前单独冻结门槛：旧开发组Precision至少增加2个百分点，其他五项均值不退化；新开发组六项均值不退化。排序修正是区域结果之后登记的开发探索，仍使用该门槛。全部方法均未通过，未因改门槛选出上线版本。

目前证据支持“训练链路已跑通、有局部开发改善”，不支持“Precision已经大幅提高”。后续要扩充更多训练组和更完整的同款视图，并获得真实业务相关性标签；最终候选仍需新的独立数据一次盲测。现有50个新开发组和已开封的旧盲测均不能再次充当未见测试。

## 人工业务标签链路

780对开发标签仍全部为空，真实人工标签完成数为0。已有入口校验条目ID、packet、图片字段、0/1/null类型及人工判断声明；未判断项直接从训练损失排除，绝不补成负例。每条训练查询必须同时有人工正例和负例；至少3个可用查询商品组才启动训练。导入后自动留一商品组训练，并同时排除该组候选，最终另存训练模型。

入口只是校验评审声明，不能独立认证评审人身份。这个Top30包不能计算全图库Recall，也不能作为正式验收；同款源标签与人工业务标签分开保存。

## 核验和环境

580张图片的源标注、原图及副本哈希核对通过；{verification['complete_ranking_metric_sets_replayed']}组完整排序的全部指标复算一致；两个训练模型的组隔离和域适配锚点核对通过。旧盲测和上一轮Precision封存文件保持原样，未重跑旧盲测评分。

143项测试通过，本轮新增12项；41个源文件mypy通过，本轮文件Black、isort、flake8通过。使用既有.venv312、既有本地Qwen权重；未新增依赖和模型下载。580次整图编码约82.22秒，属于离线特征准备；不能据此声称单请求端到端响应达标。

数据准备的原执行源码保存在source_preparation/。数据准备后、特征推理前仅补类型注解并清理未使用导入，两个阶段协议分别保留，训练协议在特征编码前重新冻结。详见protocol.json、data_preparation_protocol.json和training_protocol.json。

## 运行入口

当前目录已存在完整证据，以下已完成阶段会拒绝覆盖；不需要为查看进度重新执行。

```powershell
.\\.venv312\\Scripts\\python.exe scripts\\prepare_metric_training.py prepare
.\\.venv312\\Scripts\\python.exe scripts\\train_retrieval_metric.py freeze
.\\.venv312\\Scripts\\python.exe scripts\\prepare_metric_training.py encode
.\\.venv312\\Scripts\\python.exe scripts\\train_retrieval_metric.py fit
.\\.venv312\\Scripts\\python.exe scripts\\train_region_attention.py encode
.\\.venv312\\Scripts\\python.exe scripts\\train_region_attention.py fit
.\\.venv312\\Scripts\\python.exe scripts\\analyze_rank_correction.py
```

取得真实人工导出文件后，可执行下面的导入；它支持未完成的部分进度，同一内容只导入一次。文件必须是原780行模板，未判断项保留null。

```powershell
.\\.venv312\\Scripts\\python.exe scripts\\import_precision_labels.py --reviewed "实际导出文件的完整路径.jsonl"
```

结果集中保存至artifacts/module2_training，训练副本集中保存至data/module2_training。数据准备入口为固定本次协议；扩容下一批应先扩展为新release参数，不能覆盖本批或把已使用数据重新命名成盲测。
"""
    (OUT / "internal_summary.md").write_text(content, encoding="utf-8")
    doc = """# 模块二训练扩容当前进度

更新日期：2026年10月22日。仅推进模块二。

已新增150个训练商品组、50个开发验证商品组，共580张服饰图片，排除旧开发数据与已开封盲测的344个商品组。所有新数据均属于训练/开发，后续不能转为独立验收。

已实际训练轻量相似度、可训练区域注意力和实拍图域适配，并完成固定排序修正。新开发组上有局部改善，但旧开发集上仍未超过现有融合策略，未满足Precision明显提高且其他指标不退化的门槛，因此默认策略保持原样，正式验收指标为空。

人工相关性导入与分组训练入口已就绪：校验原始条目和人工声明，未标项不作为负例。780对仍全部待真实人工判断；源同款标签与人工业务标签独立保存。

本轮143项测试通过；复算1,216组完整排序指标，核验580张图片、训练/开发隔离和旧盲测封存文件。

- [完整内部报告、结果与命令](../artifacts/module2_training/internal_summary.md)
- [训练与隔离核验](../artifacts/module2_training/verification_full.json)
- [人工开发标注入口](../artifacts/module2_precision/review_packet/index.html)
- [环境加载配置](environment_setup.md)

下一步补齐真实人工业务相关性标签，继续扩充独立商品组和同款视图。任何最终候选都需要新的未曝光数据盲测；新开发组的小幅收益不能作为正式三品类验收结果。
"""
    (ROOT / "docs/module2_training_progress.md").write_text(doc, encoding="utf-8")
    p = ROOT / "docs/current_progress.md"
    s = p.read_text(encoding="utf-8")
    a = s.index("## 本轮Precision专项推进")
    b = s.index("## 评测隔离与盲测已完成", a)
    s = s[:a] + """## 本轮训练扩容与Precision推进

已新增150个训练商品组和50个开发验证商品组，共580张服饰图片；训练/开发按商品组隔离，并排除旧开发和已开封盲测的344个pair组。完成轻量度量学习、可训练ROI/中心视图注意力、实拍图域适配与排序修正。

新开发组上有局部改善，但旧开发集上仍未超过原融合策略，所有候选未通过Precision明显增长且其他指标不退化的门槛，默认策略继续保留。已完成780对人工开发标签的严格导入与分组训练链路，当前真实人工标签仍为0。

详见 [训练扩容当前进度](module2_training_progress.md)、[本轮完整内部报告](../artifacts/module2_training/internal_summary.md)；上一轮专用精排和负结果见 [Precision专项结论](module2_precision_improvement.md)。

""" + s[b:]
    s = s.replace(
        "131项测试通过；本轮新增9项精排输入、边界重排与训练隔离测试；Black、isort、flake8、mypy通过",
        "143项测试通过；本轮新增12项训练隔离、注意力、人工标签与排序修正检查；41个源文件mypy及本轮格式检查通过",
    )
    s = s.replace(
        "完成780对开发关系的真实人工标注，扩充训练商品组，再训练业务相关性精排；本轮退化的学生模型暂不使用。",
        "完成780对开发关系的真实人工标注；现已扩容150个训练组并跑通可训练注意力，继续补充商品组及同款视图，训练业务相关性精排。",
    )
    p.write_text(s, encoding="utf-8")
    p = ROOT / "README.md"
    s = p.read_text(encoding="utf-8")
    point = "2026年10月17日的历史核验"
    a = s.index(point)
    s = (
        s[:a]
        + """训练扩容（2026年10月22日）：新增150个训练组、50个开发验证组，已实训轻量相似度、区域注意力及实拍图适配，尚未通过多指标提升门槛。已补齐人工标签导入链路；详见 [训练进度](docs/module2_training_progress.md)。

"""
        + s[a:]
    )
    p.write_text(s, encoding="utf-8")
    p = ROOT / "docs/environment_setup.md"
    with p.open("a", encoding="utf-8") as f:
        f.write("""

## 模块二训练环境与新增入口

2026年10月22日已使用既有`.venv312`和本地Qwen3-VL-Embedding-2B完成训练数据编码，无新增包或权重下载。整图与ROI/中心放大裁切由GPU编码，轻量度量、下游注意力头和实拍域适配在CPU训练；大模型参数冻结。

新入口为`scripts/prepare_metric_training.py`、`scripts/train_retrieval_metric.py`、`scripts/train_region_attention.py`；它们执行本次固定训练协议，已存在阶段会拒绝覆盖。人工导出文件用`scripts/import_precision_labels.py --reviewed "实际文件完整路径.jsonl"`校验和训练，未判断项不参与损失。

所有缓存、训练副本和模型仍放在项目内；原始数据只读，既有外部大模型路径不变。训练数据角色和已使用组清单见[训练进度](module2_training_progress.md)，完整环境与权重哈希见`artifacts/module2_training/environment_verification.json`。离线总编码时间不能作为5秒端到端请求验收证据。
""")
    p = ROOT / "docs/module2_precision_improvement.md"
    s = p.read_text(encoding="utf-8")
    s = s.replace(
        "## 本轮交付",
        "新增后续训练扩容：150个训练组、50个开发验证组；已跑通可训练注意力、实拍图适配和人工标签导入，仍未通过提升门槛。见 [后续训练进度](module2_training_progress.md)。\n\n## 本轮交付",
    )
    p.write_text(s, encoding="utf-8")
    p = ROOT / "configs/evaluation_registry.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["rules"].extend(
        [
            {
                "path_prefix": "data/module2_training/",
                "role": "exposed_same_item_training_and_group_disjoint_development",
                "external_quality_claim_allowed": False,
                "reuse_as_unseen_test_allowed": False,
            },
            {
                "path_prefix": "artifacts/module2_training/",
                "role": "internal_training_and_development_experiments",
                "formal_acceptance_claim_allowed": False,
                "new_independent_validation_required": True,
            },
        ]
    )
    p.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
