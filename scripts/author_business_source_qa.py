"""Persist Codex's actual six-sheet visual QA; not a relevance label generator."""

from __future__ import annotations

import json
from pathlib import Path

import run  # noqa: F401
from business_grounding_audit import ROOT, load, once, relative
from multimodal_ecommerce.safety import sha256_file

# Authored after viewing all native 2400x1440 source sheets 01..06. These
# candidate-only observations do not change the 12,600 frozen query relations.


OBSERVATIONS: dict[str, tuple[str, str, str]] = {
    "Q01": (
        "游戏机收纳保护包",
        "accessory",
        "画面有Switch专用收纳字样及游戏机；标题硬盘/耳机词不能覆盖实际用途",
    ),
    "Q02": (
        "耳机充电盒保护套",
        "accessory",
        "六款卡通AirPods外套，不是耳机整机；原主体描述有依据",
    ),
    "Q03": (
        "无线讲解器套装",
        "set",
        "发射接收设备、麦克风及耳挂共同展示；不将单个耳挂视为唯一主体",
    ),
    "Q04": (
        "耳机与点烟器组合设备",
        "set",
        "图文共同展示点烟器及耳机；保持多功能组合，不强制单物体定位",
    ),
    "Q05": (
        "小型拉链收纳盒",
        "accessory",
        "多款圆形与方形拉链盒；用途众多，不能仅凭标题认定只适用耳机",
    ),
    "Q06": (
        "耳机充电盒编织保护套",
        "accessory",
        "三个卡通编织套，不能因耳机字样识别为耳机整机",
    ),
    "Q07": ("运动手机臂包", "accessory", "展示手机臂包及使用模特；模特不是主售品"),
    "Q08": ("激光雕刻机", "main", "明确雕刻机本体；手机壳是可加工对象，不是出售主体"),
    "Q09": ("手机配件展示柜", "main", "出售整排柜体；柜内手机配件不是主体"),
    "Q10": ("手机防水袋", "accessory", "透明防水袋包住演示手机；不将手机当主售品"),
    "Q11": (
        "腰挂翻盖手机包",
        "accessory",
        "皮质腰包与插卡结构，不是贴合手机的保护背壳",
    ),
    "Q12": ("手提圆桶包", "main", "手提包在桌面展示；枕头包是包型词，不是寝具"),
    "Q13": ("猫薄荷宠物玩具", "main", "人物图案玉米形宠物玩具；标题枕头不是普通睡眠枕"),
    "Q14": ("衣物修补贴", "material", "衣袖补丁贴及贴用演示；羽绒服只是修补对象"),
    "Q15": ("工业烘干机", "main", "大型滚筒设备；羽绒服是适用衣物，不是主体"),
    "Q16": ("茶水杯收纳柜", "main", "柜内杯瓶是展示内容；主体是分格金属柜"),
    "Q17": ("带托盘小茶几", "main", "桌面与开放支架结构，不是带柜门餐边柜"),
    "Q18": ("一脚蹬低帮布鞋", "main", "无系带平底休闲鞋；旧词表未覆盖不代表主体不清"),
    "Q19": ("高帮篮球运动鞋", "main", "可见系带高帮运动鞋；具体运动性能不能由图保证"),
    "Q20": (
        "带音频功能的智能眼镜",
        "main",
        "可见眼镜本体，图文说明音乐通话；不能只截取耳机关键词",
    ),
    "Q21": ("耳机替换耳垫", "accessory", "两只环形海绵耳垫，不是头戴耳机整机"),
    "Q22": ("组合餐边柜", "set", "高低柜及电视共同展示；柜体主售可辨但不强迫单实例"),
    "Q23": ("包装榴莲果肉", "main", "透明包装黄色果肉；金枕头是品种词，不是枕头"),
    "Q24": ("包装榴莲果肉", "main", "包装果肉及开袋演示；与寝具无关"),
    "Q25": ("盒装榴莲果肉", "main", "黑色托盘果肉；花束为场景道具"),
    "Q26": ("耳机充电盒皮质保护套", "accessory", "套内耳机是使用演示；主售为保护套"),
    "Q27": ("耳机充电盒印花保护套", "accessory", "两款黑色BOY保护套；不是两副耳机"),
    "Q28": ("手机挂绳", "accessory", "多色布带及扣件；标题手机壳不能改写主售类型"),
    "Q29": ("盒装榴莲果肉", "main", "托盘与黄色果肉；金枕头词不构成枕头商品"),
    "Q30": ("袋装榴莲果肉", "main", "透明袋黄色果肉；主售为食品"),
    "Q31": ("袋装榴莲果肉", "main", "透明袋黄色果肉；主售为食品"),
    "Q32": (
        "弹性数码收纳板",
        "accessory",
        "交叉弹性带固定演示数码配件；不是独立封闭内胆包",
    ),
    "Q33": ("女士尖头低跟及踝鞋", "main", "两色低跟尖头及踝鞋；不是运动鞋"),
    "Q34": ("男士毛领长大衣", "main", "长款外套可见，羊毛与羽绒成分仅属标题主张"),
    "Q35": ("儿童带手柄吸管水壶", "main", "两只带手柄及肩带的杯壶；保留杯体完整轮廓"),
    "Q36": ("户外溯溪鞋", "main", "深色低帮系带排水鞋；水景不能替代鞋体特征"),
    "Q37": ("低帮户外徒步鞋", "main", "灰橙系带低帮鞋，鞋底及鞋面清晰"),
    "Q38": ("低帮户外休闲鞋", "main", "灰色系带低帮鞋；实际品牌和性能不另作保证"),
    "Q39": ("儿童涉水软鞋", "main", "多款贴脚薄鞋及花色；不等同普通厚底跑鞋"),
    "Q40": ("户外徒步鞋", "main", "模特脚上两只系带户外鞋；不截腿作为主体"),
    "Q41": ("低帮战术休闲鞋", "main", "绿色低帮厚底系带鞋"),
    "Q42": ("户外溯溪运动鞋", "main", "两只鞋及鞋底演示；保持多视图主体"),
    "Q43": ("洞洞半拖鞋", "main", "无后跟包覆的洞洞半拖；不因标题运动词认定为系带跑鞋"),
    "Q44": ("儿童包头凉鞋", "main", "有开放镂空和搭扣结构的包头凉鞋"),
    "Q45": ("户外徒步鞋", "main", "黑色低帮系带鞋；原主体有依据"),
    "Q46": ("彩色镭射亮片材料", "material", "多色散装亮片；手机壳是DIY用途，不是成品"),
    "Q47": ("儿童网面搭扣鞋", "main", "蓝色单只儿童网面鞋；婴儿年龄适配不能由图确认"),
    "Q48": ("护肤精华液", "main", "滴管瓶及护肤品包装；与奶瓶无关"),
    "Q49": (
        "多件彩妆展示",
        "set",
        "口红与粉盒等共同出现，不能确认只有口红是唯一主售；保持未决",
    ),
    "Q50": (
        "干燥植物叶片",
        "main",
        "主体为袋装或盘装干叶；红毛榴莲字样不能认作可食榴莲果肉",
    ),
    "Q51": ("袋装榴莲果肉", "main", "透明袋黄色食品；无枕头寝具"),
    "Q52": ("保温饭盒手提袋", "accessory", "黑色手提饭盒袋；杯与水果是展示道具"),
    "Q53": ("带盖马克杯套装", "set", "两只带盖带柄杯及勺，作为套装展示"),
    "Q54": ("耳机充电盒布艺保护套", "accessory", "印花布艺保护套及内装充电盒演示"),
    "Q55": ("耳机手环组合设备", "set", "手环与可拆耳机共同展示，保持多功能设备"),
    "Q56": ("圆形耳机充电盒保护套", "accessory", "多款图案圆形套及挂扣；不是耳机整机"),
    "Q57": ("装饰贴钻链材料", "material", "珠钻链条及装饰件；手机壳是可用场景"),
    "Q58": ("自动泡沫洗手机", "main", "洗手液出泡设备及手部演示；不是移动电话"),
    "Q59": (
        "婴儿枕席",
        "accessory",
        "薄片编织枕席，不是填充枕芯；英文pillow case提示不够准确",
    ),
    "Q60": (
        "手表展示托盘道具",
        "accessory",
        "出售陈列托盘及腕表支撑道具，表本体为使用演示",
    ),
    "Q61": ("高帮帆布休闲鞋", "main", "黑色高帮帆布及白底，不能凭标题短靴词改成皮靴"),
    "Q62": ("衣物去污剂", "main", "桶装去污粉或剂；羽绒服是清洗对象"),
    "Q63": ("茶水杯收纳柜", "main", "大尺寸多格柜体；格内瓶杯为使用演示"),
    "Q64": ("折叠餐桌", "main", "折叠桌及凳，开放支腿结构；不是餐边柜"),
    "Q65": ("低帮板鞋", "main", "两只白色系带平底板鞋"),
    "Q66": ("系带跑步鞋", "main", "黄绿色低帮系带运动鞋"),
}


def main() -> None:
    """Run the command-line workflow defined by this module.

    Notes:
        Writes derived artifacts through once.
    """
    inventory = load(ROOT / "QA/source_inventory.json")
    assert {r["opaque"] for r in inventory} == set(OBSERVATIONS)
    observations = []
    for r in inventory:
        product, role, note = OBSERVATIONS[r["opaque"]]
        observations.append(
            {
                "opaque": r["opaque"],
                "candidate_id": r["id"],
                "source_image_sha256": r["fingerprints"]["sha256"],
                "observed_primary": product,
                "sale_role": role,
                "primary_unclear": r["opaque"] == "Q49",
                "native_visual_evidence": note,
                "review_origin": "model",
                "annotation_tool": "Codex native image view",
                "human_label": False,
                "existing_relevance_labels_changed": False,
                "new_relevance_grade": None,
            }
        )
    result = {
        "reviewed_sheets": [f"QA/source_sheets/{i:02d}.png" for i in range(1, 7)],
        "reviewed_source_records": 66,
        "observations": observations,
        "scope": "candidate-only development QA of parser flags; not retrieval labels or independent annotator verification",
        "existing_12600_relations_unchanged": True,
        "human_labels": 0,
        "formal_metrics": None,
    }
    once(ROOT / "QA/source_review.native_AI.json", result)
    paths = [
        Path(__file__),
        ROOT / "QA/source_inventory.json",
        ROOT / "QA/source_review.native_AI.json",
    ]
    paths += sorted((ROOT / "QA/source_sheets").glob("*.png"))
    once(
        ROOT / "QA/source_review_receipt.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "completed": True,
            "human_labels": 0,
        },
    )
    print(
        json.dumps(
            {
                "native_source_QA_completed": True,
                "records": 66,
                "relevance_labels_changed": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
