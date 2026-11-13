"""Native AI judgments recorded after inspecting the ten original contact sheets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import verify_files
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import sha256_file
from primary_form_coverage import receipt
from subject_coverage_acquisition import ROOT, check
from subject_roi_experiment import COVERAGE, load, once

# These descriptions are direct pixel observations, not predictions from retrieval.
PROFILES = [
    ("手机防水袋", "黑色长方形透明软袋，带颈带，图中手机与耳机线为使用展示"),
    ("无线蓝牙耳机", "白色短柄双耳耳塞与白色翻盖充电盒"),
    ("有线入耳耳机", "黑色金属圆柱耳塞，硅胶耳帽，双耳连接线"),
    ("耳机盒保护套", "棕色皮纹圆角盒套，扣合盖，图中露出的白色耳机为使用展示"),
    ("耳机盒保护套", "两只黑色圆角翻盖保护套，白色文字和图案"),
    ("手机挂绳", "多色扁平织带，金属扣和细绳连接端，主体为挂绳"),
    ("手机背壳", "黑色与红色长方形背壳，龙纹和圆形装饰，竖排摄像头开孔"),
    ("鞋收纳盒", "透明塑料叠放抽拉鞋盒，鞋子和手为使用场景"),
    ("芭蕾舞鞋", "黄色软底舞鞋，低帮包脚，交叉弹力带"),
    ("时装中筒靴", "黑色圆头厚底靴，直筒靴帮，侧面弹力片或拼接，低粗底"),
    ("劳保鞋", "黑色低帮系带网面鞋，厚橡胶底及侧面支撑片，图中另有配色"),
    ("儿童牛仔马甲", "蓝色无袖翻领牛仔背心，背面卡通刺绣与磨白纹理"),
    ("儿童羽绒服", "黑色与银色拼接中长款有袖外套，连帽毛领，儿童模特"),
    ("抽屉斗柜", "棕色木纹低柜，多排抽屉与黑色把手，短柜脚，无门板"),
    ("推拉门衣柜", "灰白色高大落地衣柜，推拉门、顶柜和底部抽屉"),
    ("窗口对讲机", "灰色底座带长鹅颈麦克风，另有有线圆形副机"),
    ("台面置物架", "银色金属开放双层长架，四根立柱，没有封闭柜门"),
    ("婴儿定型枕", "浅黄色长方形软枕，婴儿头部枕在中央区域"),
    ("床头靠背板", "灰色长方形软包床头板与黑色床架，枕头为床上场景配物"),
    ("笔记本电脑包", "浅蓝色长方形手提软包，双提手与前侧拉链袋，电脑为使用展示"),
    ("笔记本电脑内胆包", "灰色扁平长方形拉链软套，图中笔记本电脑为使用展示"),
    ("相机内胆包", "灰色立体方盒软包，翻盖与前开口，内部多个厚垫隔仓，摄影器材为展示"),
    ("榴莲果肉", "黑色盒装黄色果肉，可见完整肉瓣和外盒文字"),
    ("榴莲果肉", "透明包装内多瓣黄色果肉，红色促销字覆盖部分画面"),
    ("榴莲果肉", "红色外袋透明窗内多瓣黄色果肉，外袋白色标签"),
]


def entry(
    positive: Sequence[int] = (),
    unknown: dict | None = None,
    reason: str = "本批新增主体与查询要求的主要类型、用途或关键形态不匹配。",
) -> dict:
    """Return entry for this workflow.

    Args:
        positive: Positive used by this operation.
        unknown: Unknown used by this operation.
        reason: Reason used by this operation.

    Returns:
        Mapping with fields: positive, unknown, reason.
    """
    return dict(positive=list(positive), unknown=unknown or {}, reason=reason)


def native() -> dict:
    """Return native for this workflow.

    Returns:
        Mapping with fields: tune, regression, fresh.
    """
    text = {
        "text_01": entry(
            [2, 3], reason="可见双耳耳机主体；保护套、挂绳和对讲机不当作耳机。"
        ),
        "text_02": entry([7], reason="可见手机背壳主体；防水袋与挂绳是不同附件。"),
        "text_03": entry(
            [18],
            reason="可见婴儿枕主体，未限定年龄的枕头需求可以匹配；不认可医疗功效。",
        ),
        "text_04": entry([10], reason="可见黑色时装靴主体；舞鞋和劳保低帮鞋不是靴。"),
        "text_05": entry(
            [13], reason="可见有袖蓬松连帽外套，原标题明确儿童羽绒服；查询未限定成人。"
        ),
        "text_06": entry(
            unknown={
                "14": "纯抽屉卧室斗柜，是否满足餐边收纳用途不明确；沿用既有斗柜边界。"
            }
        ),
        "text_07": entry(),
        "text_08": entry(
            reason="新增低帮鞋为明确劳保用途或芭蕾舞鞋，不据网面外观推断跑步性能。"
        ),
    }
    return {
        "tune": dict(
            text,
            **{
                "image_01": entry(),
                "image_02": entry(
                    [7],
                    reason="同为带相机开孔的长方形手机背壳，装饰和颜色不要求完全一致。",
                ),
                "image_03": entry(
                    [23, 24, 25],
                    reason="查询主体为盒装榴莲果肉，新增为同用途的包装果肉。",
                ),
                "image_04": entry(),
                "image_05": entry(),
                "image_06": entry(),
                "image_07": entry(),
                "image_08": entry(),
            },
        ),
        "regression": dict(
            text,
            **{
                "image_01": entry(),
                "image_02": entry(
                    [7],
                    reason="同为长方形手机背壳，图案和颜色不限定；挂绳与防水袋不算背壳。",
                ),
                "image_03": entry(),
                "image_04": entry(),
                "image_05": entry(),
                "image_06": entry(),
                "image_07": entry(),
                "image_08": entry(),
            },
        ),
        "fresh": {
            "image_01": entry(
                reason="查询为单耳挂耳通话耳机，新增双耳无线耳塞及有线耳塞形态不同。"
            ),
            "image_02": entry(
                [7],
                reason="同为贴合手机机背的保护壳，颜色图案不同不影响主要类型与用途。",
            ),
            "image_03": entry(
                reason="查询是手袋整理内胆；电脑内胆为扁平保护套，相机内胆为厚垫器材隔仓，不直接当作手袋整理内胆。"
            ),
            "image_04": entry(
                unknown={
                    "10": "同为黑色厚底靴，但中筒无系带与查询系带踝靴结构有差异，是否足够相似不确定。"
                }
            ),
            "image_05": entry(
                unknown={
                    "13": "连帽毛领中长羽绒形态相近，但明确童装与查询成人男装是否可替代不确定。"
                }
            ),
            "image_06": entry(
                unknown={
                    "14": "同为低矮木质储物柜，但全抽屉与查询两侧柜门加中央抽屉的结构差异较大。"
                }
            ),
            "image_07": entry(),
            "image_08": entry(
                unknown={
                    "11": "低帮网面厚底轮廓接近，但原标题明确劳保鞋，与查询户外运动用途是否兼容不确定。"
                }
            ),
        },
    }


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve pre-inference labels.

    Notes:
        Writes derived artifacts through once, receipt, write_rows.
    """
    check()
    verify_files(ROOT / "retry", load(ROOT / "retry/receipt.json")["files"])
    verify_files(ROOT, load(ROOT / "native_image_receipt.json")["files"])
    if (ROOT / "label_receipt.json").exists():
        raise FileExistsError("Preserve pre-inference labels")
    docs = load(ROOT / "additions/catalog.final.public.json")
    assert len(docs) == len(PROFILES) == 25
    profiles = [
        dict(id=d["id"], product_type=p[0], appearance=p[1])
        for d, p in zip(docs, PROFILES)
    ]
    write_rows(ROOT / "additions/profiles.public.jsonl", profiles)
    once(ROOT / "native/judgments.json", native())
    once(
        ROOT / "native/review_provenance.json",
        dict(
            review_origin="model",
            human_label=False,
            human_labels=0,
            native_images_reviewed=25,
            failed_images_unreviewable=55,
            viewed_contact_sheets=[
                f"native/candidates_{i:02d}.jpg" for i in range(1, 11)
            ],
            query_visual_reference="artifacts/module2_subject_roi/query_roi_contact_sheet.png",
            original_source_titles_used=True,
            expanded_predictions_seen=False,
            reviewer_knows_prior_development_results=True,
            independent_reviewer=False,
            negative_basis="Direct visible primary type, use and structural incompatibility; displayed props not primary products.",
            scope="Business relevance, frozen earlier definition; unresolved relations remain null.",
        ),
    )
    files = [
        ROOT / "additions/profiles.public.jsonl",
        ROOT / "native/judgments.json",
        ROOT / "native/review_provenance.json",
        Path(__file__),
    ]
    for cohort, entries in native().items():
        folder = ROOT / cohort
        queries = load(COVERAGE / cohort / "queries.public.json")
        assert set(entries) == {q["id"] for q in queries}
        original = read_rows(COVERAGE / cohort / "judgments.ai.jsonl")
        added = []
        for query in queries:
            v = entries[query["id"]]
            assert not set(v["positive"]) & {int(k) for k in v["unknown"]}
            for i, doc in enumerate(docs, 1):
                grade = (
                    1 if i in v["positive"] else None if str(i) in v["unknown"] else 0
                )
                added.append(
                    dict(
                        query_id=query["id"],
                        candidate_id=doc["id"],
                        grade=grade,
                        reason=(
                            v["reason"]
                            if grade == 1
                            else (
                                v["unknown"][str(i)]
                                if grade is None
                                else f"主体为{PROFILES[i-1][0]}；{v['reason'] if not v['positive'] else '主要类型、用途或关键形态不满足此查询。'}"
                            )
                        ),
                        review_origin="model",
                        human_label=False,
                        evidence="native/judgments.json",
                        addition_position=i,
                    )
                )
        write_rows(folder / "judgments.ai.jsonl", original + added)
        once(folder / "queries.public.json", queries)
        index_cohort = "tune" if cohort == "fresh" else cohort
        once(
            folder / "catalog.public.json",
            load(COVERAGE / index_cohort / "catalog.public.json") + docs,
        )
        write_rows(
            folder / "profiles.public.jsonl",
            read_rows(COVERAGE / index_cohort / "profiles.public.jsonl") + profiles,
        )
        files += [
            folder / n
            for n in [
                "judgments.ai.jsonl",
                "queries.public.json",
                "catalog.public.json",
                "profiles.public.jsonl",
            ]
        ]
    receipt(
        ROOT,
        "label_receipt.json",
        [p for p in files if p.is_relative_to(ROOT)],
        reviewer_source_sha256=sha256_file(Path(__file__)),
        original_relations=6720,
        new_AI_relations=1000,
        total_relations=7720,
        native_images_reviewed=25,
        failed_images_not_fabricated=True,
        human_labels=0,
        frozen_before_expanded_predictions=True,
        original_labels_unchanged=True,
    )
    print(
        dict(reviewed_images=25, unreviewable_images=55, added_relations=1000),
        flush=True,
    )


if __name__ == "__main__":
    main()
