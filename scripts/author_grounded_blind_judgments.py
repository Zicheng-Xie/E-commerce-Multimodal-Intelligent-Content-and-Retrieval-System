"""Persist Codex's direct native-image review, never infer grades from ranks."""

from __future__ import annotations

import grounded_blind_dataset as dataset
import run  # noqa: F401
from learned_fusion_experiment import load, once, rel
from multimodal_ecommerce.safety import sha256_file

ROOT = dataset.ROOT
# These are the AI assessor's explicit judgments after viewing all eight native
# gallery sheets and the query sheet. Strata/classifier outputs are not labels.
TEXT_POSITIVE = {
    1: [1, 2, 3, 4, 5, 6, 7, 8],
    2: [9, 10, 11, 12, 13, 14],
    3: [18, 19, 20, 21, 22, 23, 24],
    4: [27, 28, 29, 31],
    5: [25, 32, 33, 34, 35, 36, 37, 38, 39],
    6: [42, 44, 47],
    7: [49, 50, 53, 54, 55, 56],
    8: [57, 58, 59, 60, 61, 62, 64],
    9: [65, 66, 67, 68, 69, 70, 71, 72],
    10: [73, 74, 75, 77, 78, 79, 80],
    11: [81, 82, 83, 84, 85, 86, 87],
    12: [89, 90, 91, 92, 93, 95, 96],
}
IMAGE_POSITIVE = {
    1: [1, 2, 3, 4, 6, 7, 8],
    2: [],
    3: [],
    4: [27, 28, 31],
    5: [25, 32, 33, 34, 35, 36, 37, 38, 39],
    6: [41, 42, 43, 44, 45, 46, 47, 48],
    7: [],
    8: [58, 59, 60, 61, 62, 64],
    9: [65, 66, 68, 70, 71, 72],
    10: [73, 75, 77, 78],
    11: [81, 82, 83, 84, 85],
    12: [],
}
TEXT_UNKNOWN = {
    6: {
        i: "可见为无袖保暖马甲，但原标题/图示中棉、羽绒棉或填充描述不能明确确认查询要求的羽绒；保留材质未知"
        for i in [41, 43, 45, 46, 48]
    },
    8: {63: "标题称短靴，图像为白色包头的高帮系带童鞋；短靴与高帮帆布鞋边界不明确"},
    9: {63: "高帮系带童鞋图像与马丁靴式原标题存在类别边界，无法确认运动鞋用途"},
    11: {88: "原标题同时列直饮/吸管两种款，当前包装图未展开饮嘴，不能确认所示为吸管款"},
}
IMAGE_UNKNOWN = {
    2: {
        i: "查询为带充电盒的真无线入耳式耳塞，候选为带线小耳塞；同为双耳入耳式但线缆/充电盒形态差异在本业务相似性边界内不确定，不强判相关"
        for i in [11, 12, 13, 14]
    },
    5: {
        40: "候选标题明确为iPad平板套，查询图是电脑携带包；缺少尺寸与兼容证据，无法确认可用于同类电脑"
    },
    8: {63: "候选为高帮包头系带童鞋，原标题称短靴；与查询系带靴的类别边界不确定"},
    11: {
        86: "候选展示奶瓶/鸭嘴/吸管多阶段杯盖，无法确认当前可售配置与查询长吸管饮水瓶相符",
        88: "候选同时称直饮/吸管款，图中未展开饮嘴，无法确认与查询吸管配置相符",
    },
}
OBSERVATIONS = {
    1: "银黑色双耳头戴耳机，弧形头梁、两侧包耳圆罩，无需同品牌或SKU；单耳话务结构不相符",
    2: "橙色真无线入耳式耳塞放在开盖充电盒内；颈挂环、开放式耳挂和半入耳柄式结构不同",
    3: "四瓶尖嘴印章印油，瓶身可见Premium Ink及印章印油文字；不是手机保护壳",
    4: "红色长方形手袋内胆整理包，有短提手和多格外围收纳袋；电脑套、轮滑包和圆筒包结构不同",
    5: "黑色长方形电脑携带保护包，双提手、外层拉链袋与肩带；不要求同品牌或颜色",
    6: "棕黑色立领无袖绗缝保暖马甲，前中拉链、竖口袋；内搭长袖不是主售马甲袖子",
    7: "红色中式盘扣立领上衣配灰色裤子，主图为唐装式服装；不是普通鼓格羽绒外套",
    8: "酒红色圆头系带时装靴，筒口高于脚踝、后提环、粗底；摩托护具靴和登山鞋用途/结构不同",
    9: "粉蓝色低帮网面运动鞋，有厚圆弧鞋底、后提环和多条搭扣；高帮登山鞋与高帮帆布鞋结构不同",
    10: "棕色低矮餐边收纳柜，左侧开放搁格和小抽屉，右侧不透明门；高大上下柜、转角高柜和水槽灶台柜形态不同",
    11: "透明细长大容量饮水瓶，蓝色翻盖、长吸管和提环；环形甜甜圈瓶和婴儿奶嘴配置不同",
    12: "多只长圆柱形护颈枕，条纹外套、端部束绳；并非矩形床枕、U形颈枕、汽车头枕或方形抱枕被",
}


def author() -> None:
    """Apply the author workflow step.

    Notes:
        Writes derived artifacts through once.
    """
    dataset.guard()
    dataset.checked("query_open_receipt.json")
    assert (
        not (ROOT / "prediction_receipt.json").exists()
        and not (ROOT / "pair_plan.json").exists()
    )
    gallery = load(ROOT / "evaluator/gallery_mapping.json")
    profiles = load(ROOT / "evaluator/profiles.native_AI.json")
    sheet_paths = [ROOT / "review_gallery" / f"{i:02d}.png" for i in range(1, 9)] + [
        ROOT / "review_queries/all.png"
    ]
    worksheet = dict(
        reviewed_native_gallery_sheets=list(range(1, 9)),
        reviewed_all_gallery_display_ids=[d["display_id"] for d in gallery],
        reviewed_native_image_queries=list(range(1, 13)),
        query_observations=OBSERVATIONS,
        text_positive=TEXT_POSITIVE,
        image_positive=IMAGE_POSITIVE,
        text_unknown=TEXT_UNKNOWN,
        image_unknown=IMAGE_UNKNOWN,
        reviewed_files={rel(p): sha256_file(p) for p in sheet_paths},
        review_origin="Codex direct native-image AI review; no Ollama response or human review fabricated",
        human_labels=0,
        reviewer_knows_development_background=True,
        rankings_or_scores_seen=False,
        all_cross_primary_type_pairs_reviewed=True,
        negatives="Every unlisted cell was reviewed as incompatible main type/use or visible key form; do not infer positives from sampled bucket, title substring, broad category or background alone.",
        empty_and_uncertain_queries="Ink, Tang-style outfit and cylindrical neck-pillow images are preserved. No matching gallery item is fabricated or substituted. Ambiguous same-type form/material/configuration remains null.",
    )
    once(ROOT / "evaluator/review_worksheet.native_AI.json", worksheet)
    authored = {}
    for d in gallery:
        number = int(d["display_id"][1:])
        p = profiles[d["display_id"]]
        value = dict(
            evidence="直接查看该候选原图与原标题："
            + p["product_type"]
            + "；"
            + p["appearance"]
            + "。逐项比较12条中文需求和12张查询原图，跨主体/用途/关键形态不符判负；具体正例与未决边界见原生AI评审工作表。"
        )
        for mode, positive, unknown in [
            ("text", TEXT_POSITIVE, TEXT_UNKNOWN),
            ("image", IMAGE_POSITIVE, IMAGE_UNKNOWN),
        ]:
            value[mode + "_positive"] = [
                i for i in range(1, 13) if number in positive[i]
            ]
            value[mode + "_unknown"] = {
                str(i): cells[number] for i, cells in unknown.items() if number in cells
            }
        authored[d["display_id"]] = value
    once(ROOT / "evaluator/judgments.native_AI.json", authored)
    dataset.freeze_labels()


if __name__ == "__main__":
    author()
