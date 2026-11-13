"""Serialize direct visual judgments made before heldout query inference."""

from __future__ import annotations

import run  # noqa: F401
from independent_business_blind import ROOT, check
from multimodal_ecommerce.safety import write_json


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve native blind review.

    Notes:
        Writes derived artifacts through write_json.
    """
    check()
    if (ROOT / "native/judgments.json").exists():
        raise FileExistsError("Preserve native blind review")
    # These positions identify images inspected in native/gallery_01..30.jpg.
    # Neither source category labels nor retrieval predictions generate grades.
    text_positive = {
        1: [8, 9, 10],
        2: list(range(11, 21)),
        3: [21, 22, 23, 24, 25, 26, 29, 30],
        4: [31, 32, 33, 34, 35, 36, 38, 39, 40],
        5: [42, 43, 44, 47, 48, 49, 50],
        6: list(range(51, 61)),
        7: list(range(61, 71)),
        8: list(range(71, 81)),
        9: list(range(81, 90)),
        10: [91, 92, 93, 94, 99, 100],
        11: [101, 102, 103, 104, 105, 107, 108, 109, 110],
        12: [111, 112, 113, 114, 115, 116, 117, 119, 120],
        13: list(range(121, 131)),
        14: list(range(131, 141)),
        15: list(range(141, 151)),
        16: list(range(151, 161)),
        17: [161, 164, 165, 166, 167, 169, 170],
        18: [171, 172, 173, 174, 175, 176, 177, 178, 180, 187],
        19: [183, 184, 185, 186, 189, 190],
        20: list(range(191, 201)),
        21: list(range(202, 210)),
        22: [212, 213, 215, 216, 217, 218, 219, 220, 229],
        23: list(range(221, 231)),
        24: list(range(231, 241)),
        25: list(range(241, 251)),
        26: [251, 252, 254, 255, 256, 257, 258, 259, 260],
        27: [261, 263, 264, 265, 266, 268],
        28: list(range(271, 281)),
        29: [281, 282, 283, 284, 285, 286, 287, 289, 290],
        30: [291, 292, 294, 295, 296, 299, 300],
    }
    text_unknown = {
        3: {
            27: "原标题兼称智能计步，但画面仅见指针盘，不能确定是智能腕表还是普通指针表。"
        },
        7: {
            52: "背夹充电宝与手机壳功能重叠，但主要出售主体与保护壳的关系不明确。",
            60: "背夹电池带保护背壳，无法确认是否满足普通保护壳购买需求。",
        },
        9: {90: "收纳盒上有接口和线，原标题主要是座椅收纳，无法确认充电器主体。"},
        10: {
            96: "开放硬盘底座与封闭移动硬盘盒具有相同连接用途，但产品形态边界不明确。",
            97: "多盘位开放底座与移动硬盘盒边界不明确。",
            98: "卧式开放硬盘底座，不确定是否满足移动硬盘盒需求。",
        },
        11: {106: "主要是翻领POLO，原标题同时称T恤；维持POLO与T恤语义边界的不确定。"},
        17: {
            162: "实际是保暖棉鞋/雪地靴，与时装靴边界不明确。",
            163: "作战作训与工装马丁靴用途并列，时装用途不确定。",
            168: "球鞋改装工装靴，实际图像为高帮球鞋，类别与成品形态边界不清。",
        },
        18: {
            179: "原标题为增高休闲老爹鞋，没有足够证据确认跑步用途。",
            182: "老爹休闲运动鞋，原标题未明确跑步用途。",
        },
        19: {181: "幼儿园小白鞋未说明帆布面料，单图不能确认。"},
        21: {
            201: "主体更像办公/泡茶长桌，原标题兼称茶几，家用茶几用途不确定。",
            210: "带椅泡茶桌套装，与客厅茶几的主体用途边界不明确。",
        },
        22: {211: "厨房灶台组合碗柜，是否满足餐边收纳柜需求不明确。"},
        23: {
            217: "茶水柜上有开放书格与书本，但原标题主要是餐边/茶柜，书柜主体边界不明确。"
        },
        26: {
            253: "画面只见蚕茧，标题同时有蚕丝被和翻新加工服务，无法确定出售被芯、原料还是服务。"
        },
        27: {
            262: "标题称新疆棉花被，图中却同时提供棉花与羽丝棉选项，具体填充款不明。",
            267: "水洗棉夏被未明确填充物，不能把面料材质当棉花填充。",
            269: "水洗棉薄被未明确棉花填充。",
            270: "多层纱布纯棉被与棉花填充被并非同一结构，购买需求边界不明。",
        },
        29: {288: "普通高脚翼背沙发椅，是否属于懒人沙发购买需求不明确。"},
        30: {297: "标题与图像混合枕套和枕芯，无法确定主要出售部件。"},
    }
    quilts = [n for n in range(251, 271) if n not in [253, 255]]
    image_positive = {
        1: [8, 9, 10],
        2: list(range(11, 21)),
        3: [29],
        4: [],
        5: [42, 43, 44, 45, 47, 48, 49, 50],
        6: [],
        7: [],
        8: [71, 72, 74, 75, 76, 77, 78, 79, 80],
        9: [83, 85, 86, 89],
        10: [91, 92, 93, 94, 99],
        11: [106],
        12: [112, 113, 114, 115, 116, 117, 119, 120],
        13: [121, 122, 123, 124, 125, 127],
        14: [131, 138],
        15: list(range(141, 151)),
        16: [],
        17: [],
        18: list(range(171, 181)) + [182, 187, 188],
        19: [190],
        20: [192, 193, 194, 195, 197, 198, 199, 200],
        21: [208],
        22: [217, 220, 221, 223, 226, 229],
        23: [217, 220, 221, 229],
        24: [231, 238],
        25: list(range(241, 251)),
        26: quilts,
        27: quilts,
        28: [271, 278],
        29: [282],
        30: [291, 292, 294, 295, 296, 299, 300],
    }
    image_unknown = {
        3: {27: "圆形黑色指针表外形接近，但普通表与智能表主体功能不明。"},
        4: {
            32: "同属有线入耳式耳机，但缺少查询图的显著挂耳和大角形腔体，外形相近程度不确定。",
            34: "有线入耳耳机主体匹配，挂耳和腔体外形差异明显，保留外形相关性未知。",
        },
        11: {
            102: "普通圆领短袖与翻领POLO主体用途接近，但关键领口结构不同。",
            104: "圆领短袖缺少翻领和前襟，外形相关性不明确。",
            105: "上衣属于无领短袖套装，翻领结构不同且出售组合不明。",
            108: "儿童插肩短袖与翻领POLO领袖结构不同，外形边界不明确。",
            109: "圆领拼色短袖与翻领短袖外形边界不明确。",
            110: "宽圆领短袖缺少翻领前襟，外形边界不明确。",
        },
        13: {
            126: "短款薄外套与查询图中长毛领外套在长度和帽部结构上不同。",
            129: "连帽短外套与中长毛领外套在长度、帽沿结构上不同。",
        },
        14: {
            132: "连帽涂鸦卫衣与查询圆领套头卫衣在可见领帽结构上不同。",
            134: "连帽卫衣与圆领套头款的领帽结构不同。",
            135: "连帽款与圆领套头款的领帽结构不同。",
            139: "连帽印花款与圆领套头款的领帽结构不同。",
        },
        16: {
            157: "宽袖开襟男式长袍与查询收腰圆领长袍在袖形和开襟结构上不同。",
            158: "多件宽袖长袍组合与单件圆领袍的关键结构不明确。",
        },
        17: {
            164: "踝靴主体一致，但平底尖头与查询高粗跟厚底结构差异较大。",
            169: "高跟踝靴主体一致，但细跟尖头与粗跟厚底差异较大。",
        },
        18: {
            181: "多SKU小白鞋图片混合系带、魔术贴和套脚款，是否包含与查询相近运动鞋形态不明确。"
        },
        19: {181: "多SKU白鞋含套脚款，但具体鞋面与查询编织一脚蹬的相近程度不明。"},
        20: {
            191: "卷口长筒防水包与方形拉链防水包主体功能近，但包体和闭合结构差异较大。",
            196: "长筒卷口包与方形拉链包的主体形态差异较大。",
        },
        21: {
            202: "椭圆雕花茶几与简洁圆形套几主体一致，轮廓与支撑结构相近程度不明。",
            206: "单个圆形网篮边几与低圆套几用途和支撑结构边界不明确。",
        },
        22: {
            213: "高餐边柜带上柜门，与开放格架加底柜的查询结构不同。",
            222: "细高书柜的开放上格和大闭合中段，与查询柜体比例和开闭结构相近程度不明。",
        },
        23: {223: "高大多格书柜与低宽两层书柜有相同用途，整体比例与层数相近程度不明。"},
        24: {237: "长条凳用途接近，但软座加鞋架与户外条板长凳的结构差异较大。"},
        26: {255: "画面混合蚕茧加工与小幅折叠被芯，无法确认主要出售主体和被体外形。"},
        27: {255: "加工服务与折叠被芯混合展示，无法确认主要主体。"},
        28: {280: "多SKU灯具组合中是否有与查询相近的矮圆柱灯体不明确。"},
        29: {281: "可折叠地板靠背椅与豆袋沙发在结构和轮廓上不同，相关性边界不明。"},
        30: {297: "枕头与枕套混合展示，主要出售部件不明。"},
    }
    entries: dict = {
        str(i): dict(
            text_positive=[], text_unknown={}, image_positive=[], image_unknown={}
        )
        for i in range(1, 301)
    }
    for mode, positives, unknowns in [
        ("text", text_positive, text_unknown),
        ("image", image_positive, image_unknown),
    ]:
        assert set(positives) == set(range(1, 31))
        for q, positions in positives.items():
            for p in positions:
                entries[str(p)][mode + "_positive"].append(q)
        for q, unknown_positions in unknowns.items():
            for p, reason in unknown_positions.items():
                assert q not in entries[str(p)][mode + "_positive"]
                entries[str(p)][mode + "_unknown"][str(q)] = reason
    write_json(
        ROOT / "native/judgments.json",
        dict(
            review_origin="model",
            human_label=False,
            gallery_sheets_viewed=list(range(1, 31)),
            query_sheets_viewed=[1, 2, 3],
            gallery_profiles_frozen_before_query_inspection=True,
            query_model_predictions_seen=False,
            ranking_or_score_inputs_used=False,
            all_queries_and_products_retained=True,
            important_observations=[
                "Phone source group contains feature phones; the frozen text request is smart phone, so only actual smart-phone bodies are positive.",
                "Mouse macro software, earphone replacement cable, denim skirt, disk storage suitcase, pillow cover and baby anti-fall backpad remain in the gallery and are not mislabeled as requested bodies.",
                "Image query 06 is an open UPS battery box; image 07 is a flip phone cover; image 11 is a POLO; image 19 is a woven slip-on; image 24 is a park bench. Keep actual visual demand rather than inheriting source fine types.",
                "Image quilt requests 26/27 do not impose unobservable filling or certification from source labels. All visibly similar quilt bodies remain relevant, so Recall denominators are not reduced.",
                "Gallery positions 148/150 are visually near-identical photographs despite different registered families. Both retained under frozen sampling; this is a residual within-heldout semantic-duplicate limitation, not development overlap proof.",
                "Image query 30 contains a baby, onesie and head pillow; the head-shape claim plus visible pillow support identifies the pillow demand. Background clothing alone is not a positive.",
            ],
            product_count=300,
            query_count=60,
            products=entries,
        ),
    )
    print(
        "Direct visual AI judgments serialized before query inference; all 300 products and 60 queries retained"
    )


if __name__ == "__main__":
    main()
