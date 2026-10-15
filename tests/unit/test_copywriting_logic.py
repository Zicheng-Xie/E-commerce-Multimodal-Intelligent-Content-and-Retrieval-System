"""Test copywriting facts, validation and model-free execution."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

path = Path(__file__).resolve().parents[2] / "scripts/test_copywriting_batch.py"
spec = importlib.util.spec_from_file_location("generation_v3", path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load generation module: {path}")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


def record(label: str, pv: str, title: str | None = None) -> Any:
    """Provide a deterministic fixture for the unit test."""
    return dict(label=label, pv=pv, title=title or label, url="", video="")


def args() -> Any:
    """Provide a deterministic fixture for the unit test."""
    return SimpleNamespace(
        style="", title_max=30, description_max=150, point_max=40, points=3
    )


class LogicTests(unittest.TestCase):
    """Verify copywriting rules with independent temporary fixtures."""

    def setUp(self) -> None:
        """Capture CLI output and restore the caller's offline environment."""
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.dict("os.environ"))

    def test_manifest_path_traversal_rejected(self) -> None:
        """Reject a record ID that could escape the run directory."""
        samples = [{"id": "../outside", "product_id": "1"}]
        with self.assertRaisesRegex(ValueError, "非法样本ID"):
            generation.choose_pending(Path("unused"), samples)

    def test_valid_sample_id_is_accepted(self) -> None:
        """Accept a normal sample ID in an otherwise empty run directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            samples = [{"id": "S001", "product_id": "1"}]
            pending = generation.choose_pending(Path(temporary_directory), samples)
            self.assertEqual(pending, [0])

    def test_composition_order_alias_and_contradiction(self) -> None:
        """Verify composition order alias and contradiction."""
        sample = generation.normalize_product(
            "1",
            record(
                "休闲裤", "服装版型#:#直筒#;#裤长#:#长裤#;#材质成分#:#聚酯纤维65% 棉35%"
            ),
        )
        payload = generation.make_payload(sample, args())
        output = dict(
            title="直筒休闲长裤",
            description="65%涤纶与35%棉混纺。",
            selling_points=["直筒设计，呈现直线裤腿轮廓。"],
        )
        result, check = generation.validate(
            json.dumps(output, ensure_ascii=False), payload
        )
        self.assertEqual(check["core_coverage"], 1)
        self.assertTrue(check["fact_rule_pass"])
        output["description"] = "35%涤纶与65%棉混纺。"
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )
        output["description"] = "100%纯棉，亲肤舒适。"
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )

    def test_equal_units_dimensions_and_capacity_bound(self) -> None:
        """Verify equal units dimensions and capacity bound."""
        facts = {"商品类型": "水杯", "尺寸": "直径46*高55mm", "容量": "0.25L及以下"}
        payload = dict(
            facts=facts,
            core_fields=list(facts),
            length_limits={"title": 60, "description": 150, "selling_point": 40},
            selling_point_count=3,
        )
        output = dict(
            title="水杯容量≤250毫升",
            description="直径46毫米，高55毫米，容量不超过250毫升。",
            selling_points=["46毫米直径，核对摆放空间。"],
        )
        check = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertEqual(check["core_coverage"], 1)
        self.assertTrue(check["fact_rule_pass"], check)
        output["title"] = "水杯容量250毫升"
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )
        output["title"] = "水杯容量至少250毫升"
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )

    def test_subcategory_color_and_extra_parameters(self) -> None:
        """Verify subcategory color and extra parameters."""
        sample = generation.normalize_product(
            "1", record("车载垃圾桶", "颜色分类#:#黑色#;#颜色分类#:#白色")
        )
        self.assertEqual(sample["facts"]["商品类型"], "车载垃圾桶")
        self.assertEqual(sample["options"]["颜色"], ["黑色", "白色"])
        payload = generation.make_payload(sample, args())
        output = dict(
            title="车载垃圾桶",
            description="不同容量可选。",
            selling_points=["用于车内垃圾收纳。"],
        )
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )
        sample = generation.normalize_product(
            "2", record("显示器", "型号#:#M1#;#刷新率#:#144HZ#;#分辨率#:#2560x1440")
        )
        self.assertEqual(sample["facts"]["刷新率"], "144HZ")
        self.assertIn("分辨率", sample["core_fields"])

    def test_effect_and_style_are_located(self) -> None:
        """Verify effect and style are located."""
        sample = generation.normalize_product("1", record("水杯", "材质#:#不锈钢"))
        payload = generation.make_payload(sample, args())
        output = dict(
            title="不锈钢水杯",
            description="采用简约设计，坚固耐用。",
            selling_points=["不锈钢材质，易清洁。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(checks["fact_rule_pass"])
        self.assertIn("description", checks["field_issues"])
        self.assertIn("selling_points", checks["field_issues"])

    def test_partial_repair_keeps_untargeted_fields(self) -> None:
        """Verify partial repair keeps untargeted fields."""
        previous = dict(title="原标题", description="旧详情", selling_points=["原卖点"])
        merged = json.loads(
            generation.merge_partial(
                json.dumps({"description": "新详情"}), previous, ["description"]
            )
        )
        self.assertEqual(merged["title"], "原标题")
        self.assertEqual(merged["selling_points"], ["原卖点"])
        self.assertEqual(merged["description"], "新详情")
        self.assertEqual(
            generation.merge_partial(
                json.dumps({"title": "偷改标题"}), previous, ["description"]
            ),
            "",
        )

    def test_format_failure_does_not_change_fluency_rating(self) -> None:
        """Verify format failure does not change fluency rating."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            payload = Path(temporary_directory)
            generation.write_json(
                payload / "manifest.json",
                {"samples": [dict(id="S001", category="3C数码")]},
            )
            generation.write_json(
                payload / "review.json",
                {
                    "items": [
                        dict(id="S001", **{key: 1 for key in generation.REVIEW_FIELDS})
                    ]
                },
            )
            generation.write_json(
                payload / "records/S001.json",
                {
                    "result": {"title": "可读但超长的标题"},
                    "validation": {
                        "format_pass": False,
                        "fact_rule_pass": True,
                        "core_coverage": 1,
                    },
                },
            )
            report = generation.summarize(payload)
            self.assertEqual(report["overall"]["fluency"]["final_pass_rate"], 1)
            self.assertEqual(report["delivery_pass_rate"], 0)

    def test_resume_pending_preserves_existing_and_retry_is_explicit(self) -> None:
        """Verify resume pending preserves existing and retry is explicit."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            payload = Path(temporary_directory)
            samples = [
                dict(id="S001", product_id="1"),
                dict(id="S002", product_id="2"),
                dict(id="S003", product_id="3"),
            ]
            generation.write_json(
                payload / "records/S001.json",
                {"sample": samples[0], "validation": {"quality_gate_pass": True}},
            )
            generation.write_json(
                payload / "records/S002.json",
                {"sample": samples[1], "validation": {"quality_gate_pass": False}},
            )
            before = (payload / "records/S001.json").read_bytes()
            self.assertEqual(generation.choose_pending(payload, samples), [2])
            self.assertEqual(generation.choose_pending(payload, samples, True), [1, 2])
            self.assertEqual((payload / "records/S001.json").read_bytes(), before)
            generation.write_json(
                payload / "records/S001.json", {"sample": dict(product_id="wrong")}
            )
            with self.assertRaises(ValueError):
                generation.choose_pending(payload, samples)

    def test_resume_completed_skips_model_and_keeps_review(self) -> None:
        """Verify resume completed skips model and keeps review."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            payload = Path(temporary_directory)
            sample = generation.normalize_product("1", record("T恤", "袖长#:#短袖"))
            sample["id"] = "S001"
            a = args()
            a.run_dir = None
            a.resume_dir = payload
            a.limit = 0
            generation.write_json(
                payload / "manifest.json",
                {
                    "samples": [sample],
                    "payloads": [generation.make_payload(sample, a)],
                    "dry_run": False,
                },
            )
            generation.write_json(
                payload / "review.json",
                {
                    "items": [
                        dict(
                            id="S001", **{key: None for key in generation.REVIEW_FIELDS}
                        )
                    ]
                },
            )
            r = {
                "sample": sample,
                "validation": {
                    "format_pass": True,
                    "fact_rule_pass": True,
                    "core_coverage": 1,
                    "quality_gate_pass": True,
                },
            }
            generation.write_json(payload / "records/S001.json", r)
            before = (payload / "records/S001.json").read_bytes()
            review = (payload / "review.json").read_bytes()
            generation.run(a)
            self.assertEqual((payload / "records/S001.json").read_bytes(), before)
            self.assertEqual((payload / "review.json").read_bytes(), review)
            self.assertTrue((payload / "summary.json").exists())

    def test_atomic_record_leaves_no_partial_final_file(self) -> None:
        """Verify atomic record leaves no partial final file."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            payload = Path(temporary_directory) / "records/S001.json"
            generation.atomic_record(
                payload, {"sample": {"product_id": "1"}, "result": {"title": "T恤"}}
            )
            self.assertEqual(generation.read_json(payload)["result"]["title"], "T恤")
            self.assertFalse(payload.with_suffix(".json.tmp").exists())

    def test_parameter_list_and_option_claim_regression(self) -> None:
        """Verify parameter list and option claim regression."""
        sample = generation.normalize_product(
            "1",
            record(
                "显示器",
                "品牌#:#Asus/华硕#;#型号#:#XG32VQ#;#接口#:#DP#;#尺寸#:#32英寸#;#颜色分类#:#黑色",
            ),
        )
        payload = generation.make_payload(sample, args())
        self.assertFalse(payload["has_options"])
        output = dict(
            title="Asus/华硕XG32VQ显示器",
            description="32英寸DP接口显示器，支持多种规格可选。",
            selling_points=[
                "品牌型号：Asus/华硕XG32VQ",
                "屏幕尺寸：32英寸",
                "接口类型：DP",
            ],
        )
        _, checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)
        self.assertTrue(checks["format_pass"])
        self.assertFalse(checks["copywriting_rule_pass"])
        self.assertFalse(checks["quality_gate_pass"])
        self.assertGreaterEqual(len(checks["copywriting_issues"]), 3)
        output["description"] = "Asus/华硕XG32VQ显示器采用32英寸屏幕，配备DP接口。"
        output["selling_points"] = [
            "32英寸屏幕，展开呈现画面内容。",
            "DP接口，为设备视频连接提供接口。",
        ]
        self.assertTrue(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "quality_gate_pass"
            ]
        )

    def test_run_three_categories_and_repair_without_model_weights(self) -> None:
        """Verify run three categories and repair without model weights."""

        class Inputs(dict):
            """Minimal model-interface test double; does not load weights."""

            def __init__(self, prompt: Any) -> None:
                """Provide a deterministic fixture for the unit test."""
                super().__init__(input_ids=SimpleNamespace(shape=(1, 1)), prompt=prompt)

            def to(self, device: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return self

        class Tensor:
            """Minimal model-interface test double; does not load weights."""

            def __init__(self, text: Any) -> None:
                """Provide a deterministic fixture for the unit test."""
                self.text = text

            def __getitem__(self, key: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return self

            def detach(self) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return self

            def cpu(self) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return self

            def tolist(self) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return [ord(checks) for checks in self.text]

        class Tokenizer:
            """Minimal model-interface test double; does not load weights."""

            eos_token_id = 0

            def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return messages[-1]["content"]

            def __call__(self, prompt: Any, **kwargs: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return Inputs(prompt)

            def decode(self, tokens: Any, **kwargs: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return "".join((chr(tensor) for tensor in tokens))

        class Model:
            """Minimal model-interface test double; does not load weights."""

            hf_device_map: dict[str, str] = {}

            def eval(self) -> Any:
                """Provide a deterministic fixture for the unit test."""
                pass

            def get_input_embeddings(self) -> Any:
                """Provide a deterministic fixture for the unit test."""
                return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

            def generate(self, **kwargs: Any) -> Any:
                """Provide a deterministic fixture for the unit test."""
                request = json.loads(kwargs["prompt"])
                kind = request["facts"]["商品类型"]
                output = dict(
                    title=kind,
                    description="；".join(request["facts"].values()),
                    selling_points=[kind],
                )
                if "repair" not in request:
                    output["description"] += "防晒"
                if request.get("repair", {}).get("target_fields"):
                    output = {
                        key: output[key] for key in request["repair"]["target_fields"]
                    }
                return Tensor(json.dumps(output, ensure_ascii=False))

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _: "fake",
                synchronize=lambda: None,
                empty_cache=lambda: None,
            ),
            bfloat16="fake",
            inference_mode=nullcontext,
            __version__="test",
        )
        fake_transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *a, **key: SimpleNamespace(tokenizer=Tokenizer())
            ),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda *a, **key: Model()
            ),
            __version__="test",
        )
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            payload = Path(temporary_directory)
            (payload / "model").mkdir()
            (payload / "model/config.json").write_text("{}")
            generation.write_json(
                payload / "raw.json",
                {
                    "1": record("T恤", "袖长#:#短袖"),
                    "2": record("键盘", "接口#:#USB"),
                    "3": record("衣架", "材质#:#塑料"),
                },
            )
            a = args()
            a.samples = payload / "raw.json"
            a.limit = 0
            a.model_dir = payload / "model"
            a.run_dir = payload / "run"
            a.dry_run = False
            a.repair_attempts = 1
            a.gpu_memory = "5GiB"
            a.cpu_memory = "10GiB"
            a.max_tokens = 512
            a.use_cache = False
            with patch.dict(
                sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
            ):
                generation.run(a)
            for recpath in (payload / "run/records").glob("*.json"):
                rec = generation.read_json(recpath)
                self.assertEqual(len(rec["attempts"]), 2)
                self.assertTrue(rec["validation"]["quality_gate_pass"])
            self.assertEqual(
                generation.read_json(payload / "run/summary.json")["acceptance"][
                    "status"
                ],
                "待评测",
            )

    def test_option_not_color_or_conflict(self) -> None:
        """Verify option not color or conflict."""
        sample = generation.normalize_product(
            "1",
            record(
                "T恤",
                "颜色分类#:#黑色M#;#颜色分类#:#白色L#;#尺码#:#M#;#尺码#:#L#;#材质#:#棉",
            ),
        )
        self.assertNotIn("颜色", sample["facts"])
        self.assertEqual(sample["options"]["颜色分类"], ["黑色M", "白色L"])
        self.assertEqual(sample["conflicts"], [])
        self.assertEqual(sample["fashion_tags"]["尺码"]["value"], ["M", "L"])

    def test_brand_label_not_category(self) -> None:
        """Verify brand label not category."""
        sample = generation.normalize_product(
            "1", record("Dior/迪奥", "领型#:#圆领", "迪奥圆领T恤")
        )
        self.assertEqual(sample["category"], "服饰穿搭")
        self.assertTrue(sample["needs_review"])
        self.assertIsNone(
            generation.normalize_product("2", record("资生堂", "", "资生堂精华"))
        )

    def test_hub_not_keyboard(self) -> None:
        """Verify hub not keyboard."""
        sample = generation.normalize_product(
            "1", record("USB HUB/转换器", "接口#:#USB", "USB转换器支持键盘鼠标")
        )
        self.assertEqual(sample["facts"]["商品类型"], "USB HUB/转换器")

    def test_accessories_rejected(self) -> None:
        """Verify accessories rejected."""
        self.assertIsNone(
            generation.normalize_product(
                "1", record("平板电脑零部件", "", "平板维修键盘")
            )
        )
        self.assertIsNone(
            generation.normalize_product("2", record("鼠标/垫/贴", "", "鼠标垫"))
        )

    def test_sparse_preserved(self) -> None:
        """Verify sparse preserved."""
        sample = generation.normalize_product("1", record("T恤", "品牌#:#other/其他"))
        self.assertEqual(sample["facts"], {"商品类型": "T恤"})
        self.assertEqual(sample["core_fields"], ["商品类型"])

    def test_conflicting_material_quarantined(self) -> None:
        """Verify conflicting material quarantined."""
        sample = generation.normalize_product(
            "1", record("衬衫", "材质#:#棉混纺布#;#面料成分#:#棉100%")
        )
        self.assertNotIn("材质", sample["facts"])
        self.assertNotIn("面料成分", sample["facts"])
        self.assertEqual(sample["fashion_tags"]["材质"]["status"], "待确认")

    def test_capacity_count_and_inconsistent_volume(self) -> None:
        """Verify capacity count and inconsistent volume."""
        sample = generation.normalize_product("1", record("衣架", "容量#:#1个"))
        self.assertEqual(sample["facts"]["数量"], "1个")
        self.assertNotIn("容量", sample["facts"])
        sample = generation.normalize_product(
            "2", record("收纳箱", "尺寸#:#32cmx22cmx16cm#;#容量#:#50L")
        )
        self.assertNotIn("容量", sample["facts"])
        self.assertNotIn("尺寸", sample["facts"])

    def test_pv_malformed_and_repeated(self) -> None:
        """Verify pv malformed and repeated."""
        ap, issues = generation.parse_pv("材质#:#棉#;#材质#:#棉#;#坏片段#;#品牌#:#")
        self.assertEqual(ap, {"材质": ["棉"]})
        self.assertEqual(len(issues), 2)

    def test_style_override_and_duplicate_products(self) -> None:
        """Verify style override and duplicate products."""
        sample = generation.normalize_product("1", record("T恤", "袖长#:#短袖"))
        sample["style"] = "复古"
        a = args()
        a.style = "专业简洁"
        self.assertEqual(generation.make_payload(sample, a)["style"], "专业简洁")
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            path = Path(temporary_directory) / "samples.json"
            generation.write_json(
                path, [dict(sample, id="S001"), dict(sample, id="S002")]
            )
            with self.assertRaises(ValueError):
                generation.load_samples(path)

    def test_model_validate_claims_coverage_and_lengths(self) -> None:
        """Verify model validate claims coverage and lengths."""
        sample = generation.normalize_product(
            "1", record("移动电源", "型号#:#P20#;#容量#:#20000mAh")
        )
        payload = generation.make_payload(sample, args())
        good = dict(
            title="P20移动电源",
            description="容量20000mAh。",
            selling_points=["20000mAh容量"],
        )
        _, checks = generation.validate(json.dumps(good), payload)
        self.assertTrue(checks["quality_gate_pass"])
        good["description"] = "容量30000mAh，支持快充。"
        _, checks = generation.validate(json.dumps(good), payload)
        self.assertFalse(checks["quality_gate_pass"])
        self.assertTrue(checks["fact_risks"])
        good["title"] = "长" * 40
        _, checks = generation.validate(json.dumps(good), payload)
        self.assertFalse(checks["format_pass"])
        self.assertIsNone(generation.validate("not json", payload)[0])

    def test_acceptance_missing_or_failed_not_pass(self) -> None:
        """Verify acceptance missing or failed not pass."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            folder = Path(temporary_directory)
            samples = [
                dict(id="S001", category="服饰穿搭"),
                dict(id="S002", category="3C数码"),
            ]
            generation.write_json(folder / "manifest.json", {"samples": samples})
            generation.write_json(
                folder / "review.json",
                {
                    "items": [
                        dict(
                            id=sample["id"],
                            **{key: 1 for key in generation.REVIEW_FIELDS},
                        )
                        for sample in samples
                    ]
                },
            )
            r: dict[str, Any] = dict(
                validation=dict(
                    format_pass=True,
                    fact_rule_pass=True,
                    quality_gate_pass=True,
                    core_coverage=1,
                )
            )
            generation.write_json(folder / "records/S001.json", r)
            self.assertEqual(
                generation.summarize(folder)["acceptance"]["status"], "待评测"
            )
            r["error"] = "OOM"
            generation.write_json(folder / "records/S002.json", r)
            report = generation.summarize(folder)
            self.assertEqual(report["acceptance"]["relevance_rate"], 0.5)
            self.assertEqual(report["acceptance"]["status"], "未通过")

    def test_unreviewed_and_factuality_in_relevance(self) -> None:
        """Verify unreviewed and factuality in relevance."""
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).parent
        ) as temporary_directory:
            folder = Path(temporary_directory)
            generation.write_json(
                folder / "manifest.json",
                {"samples": [dict(id="S001", category="3C数码")]},
            )
            generation.write_json(
                folder / "records/S001.json",
                {
                    "validation": {
                        "format_pass": True,
                        "fact_rule_pass": True,
                        "core_coverage": 1,
                    }
                },
            )
            review: dict[str, Any] = dict(
                id="S001", **{key: None for key in generation.REVIEW_FIELDS}
            )
            generation.write_json(folder / "review.json", {"items": [review]})
            self.assertEqual(
                generation.summarize(folder)["acceptance"]["status"], "待评测"
            )
            review.update({key: 1 for key in generation.REVIEW_FIELDS})
            review["factuality"] = 0
            generation.write_json(folder / "review.json", {"items": [review]})
            self.assertEqual(
                generation.summarize(folder)["acceptance"]["relevance_rate"], 0
            )


if __name__ == "__main__":
    unittest.main()
