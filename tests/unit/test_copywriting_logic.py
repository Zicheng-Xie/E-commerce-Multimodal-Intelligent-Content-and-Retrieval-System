"""测试商品事实校验与无需加载模型权重的生成流程。"""

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
    """为单元测试提供确定性的样本或模拟行为。"""
    return dict(label=label, pv=pv, title=title or label, url="", video="")


def args() -> Any:
    """为单元测试提供确定性的样本或模拟行为。"""
    return SimpleNamespace(
        style="", title_max=30, description_max=150, point_max=40, points=3
    )


class LogicTests(unittest.TestCase):
    """使用独立临时数据验证文案规则。"""

    def setUp(self) -> None:
        """捕获命令行输出，并在测试后恢复调用方的环境变量。"""
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.dict("os.environ"))

    def test_first_token_probe_synchronizes_once(self) -> None:
        """只记录一次首词元生成完成时间，不中断后续解码。"""
        calls = []
        probe = generation.FirstTokenTimer(lambda: calls.append("sync"))
        with patch.object(generation.time, "perf_counter", side_effect=[10.0, 12.0]):
            probe.start()
            self.assertFalse(probe(None, None))
            self.assertFalse(probe(None, None))
        self.assertEqual(probe.first_token_at - probe.started, 2.0)
        self.assertEqual(calls, ["sync"])

    def test_timing_bucket_boundaries(self) -> None:
        """验证相邻输入长度落入确定且包含边界的分组。"""
        self.assertEqual(generation.timing_bucket(512), "0000-0512")
        self.assertEqual(generation.timing_bucket(513), "0513-1024")

    def test_timing_report_preserves_missing_ttft(self) -> None:
        """区分修复请求与冷启动请求，首词元耗时缺失时保持缺失。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            generation.write_json(
                folder / "manifest.json",
                {"samples": [{"id": "S001", "category": "3C数码"}]},
            )
            timings = {
                "session_id": "session",
                "request_kind": "initial",
                "cold_start_request": True,
                "input_bucket": "0000-0512",
                "input_tokens": 100,
                "ttft_seconds": None,
            }
            generation.write_json(
                folder / "records/S001.json",
                {
                    "generation_seconds": 3.0,
                    "attempts": [{"attempt": 1, "timing": timings}],
                },
            )
            report = generation.summarize_timing(folder)
            self.assertEqual(report["profiled_attempts"], 1)
            self.assertEqual(report["warm_initial_attempts"]["attempts"], 0)
            self.assertIsNone(report["all_successful_attempts"]["ttft_seconds"]["mean"])

    def test_manifest_path_traversal_rejected(self) -> None:
        """拒绝可能越出运行目录的记录编号。"""
        samples = [{"id": "../outside", "product_id": "1"}]
        with self.assertRaisesRegex(ValueError, "非法样本ID"):
            generation.choose_pending(Path("unused"), samples)

    def test_valid_sample_id_is_accepted(self) -> None:
        """空运行目录中允许使用正常样本编号。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            samples = [{"id": "S001", "product_id": "1"}]
            pending = generation.choose_pending(Path(temporary_directory), samples)
            self.assertEqual(pending, [0])

    def test_composition_order_alias_and_contradiction(self) -> None:
        """验证面料成分倒装、别名和比例矛盾检查。"""
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
        """验证等价单位、尺寸及容量范围限定。"""
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
        """验证商品子类型、颜色选项和额外参数。"""
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
        """验证效果承诺和风格问题能定位到对应字段。"""
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
        """验证局部修复保留未指定字段。"""
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
        """验证格式失败不会篡改独立通顺度评分。"""
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
        """验证断点续跑保留已保存记录，失败重试必须显式指定。"""
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
        """验证全部完成时跳过模型加载，并保留原评分。"""
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
        """验证原子保存不会留下不完整的最终记录。"""
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
        """回归验证参数清单及选项声明检查。"""
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
        self.assertEqual(payload["selling_point_count"], 2)
        self.assertFalse(checks["format_pass"])
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
        """不加载模型权重，验证三个品类的生成与修复流程。"""

        class Inputs(dict):
            """最小模型接口模拟对象，不加载模型权重。"""

            def __init__(self, prompt: Any) -> None:
                """为单元测试提供确定性的样本或模拟行为。"""
                super().__init__(input_ids=SimpleNamespace(shape=(1, 1)), prompt=prompt)

            def to(self, device: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return self

        class Tensor:
            """最小模型接口模拟对象，不加载模型权重。"""

            def __init__(self, text: Any) -> None:
                """为单元测试提供确定性的样本或模拟行为。"""
                self.text = text

            def __getitem__(self, key: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return self

            def detach(self) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return self

            def cpu(self) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return self

            def tolist(self) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return [ord(checks) for checks in self.text]

        class Tokenizer:
            """最小模型接口模拟对象，不加载模型权重。"""

            eos_token_id = 0

            def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return messages[-1]["content"]

            def __call__(self, prompt: Any, **kwargs: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return Inputs(prompt)

            def decode(self, tokens: Any, **kwargs: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return "".join((chr(tensor) for tensor in tokens))

        class Model:
            """最小模型接口模拟对象，不加载模型权重。"""

            hf_device_map: dict[str, str] = {}

            def eval(self) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                pass

            def get_input_embeddings(self) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

            def generate(self, **kwargs: Any) -> Any:
                """为单元测试提供确定性的样本或模拟行为。"""
                for callback in kwargs.get("stopping_criteria", []):
                    callback(None, None)
                request = json.loads(kwargs["prompt"])
                kind = request["facts"]["商品类型"]
                if kind == "键盘":
                    if (
                        request["facts"].get("接口") != "USB"
                        or request.get("sku_context", {}).get("sku_id") != "测试USB款"
                    ):
                        raise AssertionError(
                            "已核验的接口必须实际进入初始和修复模型请求"
                        )
                output = dict(
                    title=kind,
                    description="；".join(request["facts"].values()),
                    selling_points=[kind + "，用于日常使用。"],
                )
                if "repair" not in request:
                    output["description"] += "防晒"
                if request.get("repair", {}).get("target_fields"):
                    output = {
                        key: output[key] for key in request["repair"]["target_fields"]
                    }
                return Tensor(json.dumps(output, ensure_ascii=False))

        cleanup_calls = []
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _: "fake",
                synchronize=lambda: None,
                empty_cache=lambda: cleanup_calls.append("empty_cache"),
                memory_allocated=lambda: 3 * 1024**3,
                memory_reserved=lambda: (8 if not cleanup_calls else 5) * 1024**3,
            ),
            bfloat16="fake",
            inference_mode=nullcontext,
            __version__="test",
        )
        fake_transformers = SimpleNamespace(
            StoppingCriteriaList=list,
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
                    "2": record(
                        "键盘",
                        "接口类型#:#USB#;#接口类型#:#蓝牙#;#颜色分类#:#【USB】黑色#;#颜色分类#:#【双模】白色",
                    ),
                    "3": record("衣架", "材质#:#塑料"),
                },
            )
            generation.write_json(
                payload / "attributes.json",
                {
                    "2": {
                        "selected_sku": {
                            "sku_id": "测试USB款",
                            "source": "测试夹具SKU表",
                            "attributes": {"接口": "USB"},
                        }
                    }
                },
            )
            a = args()
            a.attribute_review = payload / "attributes.json"
            a.samples = payload / "raw.json"
            a.limit = 0
            a.model_dir = payload / "model"
            a.run_dir = payload / "run"
            a.dry_run = False
            a.repair_attempts = 1
            a.gpu_memory = "5GiB"
            a.cpu_memory = "10GiB"
            a.max_tokens = 512
            a.profile_timing = True
            a.use_cache = False
            with patch.dict(
                sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
            ):
                generation.run(a)
            self.assertEqual(cleanup_calls, ["empty_cache"])
            loading = generation.read_json(payload / "run/loading.json")
            self.assertEqual(loading["startup_cache_cleanup"]["empty_cache_calls"], 1)
            self.assertEqual(
                loading["startup_cache_cleanup"]["before"]["reserved_bytes"],
                8 * 1024**3,
            )
            self.assertEqual(
                loading["startup_cache_cleanup"]["after"]["reserved_bytes"], 5 * 1024**3
            )
            self.assertFalse(loading["use_cache"])
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
        """验证混合规格不能当作单一颜色或公共属性。"""
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
        """验证品牌标签不能直接用作商品品类。"""
        sample = generation.normalize_product(
            "1", record("Dior/迪奥", "领型#:#圆领", "迪奥圆领T恤")
        )
        self.assertEqual(sample["category"], "服饰穿搭")
        self.assertTrue(sample["needs_review"])
        self.assertIsNone(
            generation.normalize_product("2", record("资生堂", "", "资生堂精华"))
        )

    def test_hub_not_keyboard(self) -> None:
        """验证转换器不会因标题提到键盘而被识别成键盘。"""
        sample = generation.normalize_product(
            "1", record("USB HUB/转换器", "接口#:#USB", "USB转换器支持键盘鼠标")
        )
        self.assertEqual(sample["facts"]["商品类型"], "USB HUB/转换器")

    def test_accessories_rejected(self) -> None:
        """验证不支持的配件样本会被拒绝。"""
        self.assertIsNone(
            generation.normalize_product(
                "1", record("平板电脑零部件", "", "平板维修键盘")
            )
        )
        self.assertIsNone(
            generation.normalize_product("2", record("鼠标/垫/贴", "", "鼠标垫"))
        )

    def test_sparse_preserved(self) -> None:
        """验证稀疏属性保持原样，不补充未知事实。"""
        sample = generation.normalize_product("1", record("T恤", "品牌#:#other/其他"))
        self.assertEqual(sample["facts"], {"商品类型": "T恤"})
        self.assertEqual(sample["core_fields"], ["商品类型"])

    def test_conflicting_material_quarantined(self) -> None:
        """验证冲突材质被隔离，并标记待确认。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "材质#:#棉混纺布#;#面料成分#:#棉100%")
        )
        self.assertNotIn("材质", sample["facts"])
        self.assertNotIn("面料成分", sample["facts"])
        self.assertEqual(sample["fashion_tags"]["材质"]["status"], "待确认")

    def test_capacity_count_and_inconsistent_volume(self) -> None:
        """验证数量与容量区分，以及矛盾体积数据的隔离。"""
        sample = generation.normalize_product("1", record("衣架", "容量#:#1个"))
        self.assertEqual(sample["facts"]["数量"], "1个")
        self.assertNotIn("容量", sample["facts"])
        sample = generation.normalize_product(
            "2", record("收纳箱", "尺寸#:#32cmx22cmx16cm#;#容量#:#50L")
        )
        self.assertNotIn("容量", sample["facts"])
        self.assertNotIn("尺寸", sample["facts"])

    def test_pv_malformed_and_repeated(self) -> None:
        """验证属性串格式错误和重复属性的处理。"""
        ap, issues = generation.parse_pv("材质#:#棉#;#材质#:#棉#;#坏片段#;#品牌#:#")
        self.assertEqual(ap, {"材质": ["棉"]})
        self.assertEqual(len(issues), 2)

    def test_style_override_and_duplicate_products(self) -> None:
        """验证风格覆盖和重复商品检查。"""
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
        """验证型号保留、属性承诺、核心覆盖及长度限制。"""
        sample = generation.normalize_product(
            "1", record("移动电源", "型号#:#P20#;#容量#:#20000mAh")
        )
        payload = generation.make_payload(sample, args())
        good = dict(
            title="P20移动电源",
            description="容量20000mAh。",
            selling_points=["20000mAh容量，适合日常供电使用。"],
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
        """验证缺失或生成失败的记录不能通过验收。"""
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
        """验证未评审状态和相关度中的事实性门槛。"""
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

    def test_adjacent_numeric_parameters_keep_boundaries(self) -> None:
        """验证分辨率数字不会与其后的刷新率拼接。"""
        from decimal import Decimal

        values = generation.quantities("2560x1440 144Hz")
        self.assertIn((Decimal(144), "hz"), values)
        self.assertNotIn((Decimal(1440144), "hz"), values)

    def test_each_capacity_occurrence_keeps_its_bound(self) -> None:
        """另一卖点保留限定词，不能掩盖当前卖点的容量限定丢失。"""
        payload = dict(
            facts={"商品类型": "水杯", "容量": "0.25L及以下"},
            core_fields=["商品类型", "容量"],
            selling_point_count=3,
            length_limits={"title": 60, "description": 150, "selling_point": 60},
        )
        output: dict[str, Any] = dict(
            title="水杯容量≤250毫升",
            description="容量不超过250毫升。",
            selling_points=["容量0.25L及以下，便于核对。", "容量0.25L，便于核对。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertIn("selling_points", checks["field_issues"])
        self.assertTrue(
            any("范围限定" in x for x in checks["field_issues"]["selling_points"])
        )
        output["selling_points"][1] = "容量250毫升以内，便于核对。"
        # 范围判断同时支持以下、不超过、≤和以内等上界写法。
        self.assertTrue(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )

    def test_approximate_capacity_must_remain_approximate(self) -> None:
        """保留近似限定，不将约数变成精确承诺。"""
        self.assertTrue(generation.bounded_quantity_issues("容量250毫升", "约0.25L"))
        self.assertFalse(generation.bounded_quantity_issues("容量约250毫升", "约0.25L"))

    def test_unsupported_claims_require_relevant_evidence(self) -> None:
        """验证版本、材质、数量和摆放方式不能推出额外功能。"""
        cases = [
            ("USB接口即插即用", {"接口": "USB"}),
            ("实现快速数据传输", {"接口": "OTG"}),
            ("兼容主流电脑设备", {"接口": "USB"}),
            ("纸材质环保", {"材质": "纸"}),
            ("无需组装", {"安装方式": "置地式"}),
            ("1个独立包装", {"数量": "1个"}),
            ("表面处理增加摩擦力", {"功能": "防滑"}),
            ("裙身印有卡通图案", {"图案": "卡通动漫"}),
            ("采用IPS面板", {"面板类型": "VA"}),
        ]
        for text, facts in cases:
            with self.subTest(text=text):
                self.assertTrue(generation.claim_issues(text, facts))

    def test_provided_functions_are_not_blanket_banned(self) -> None:
        """验证输入有明确依据时允许对应声明。"""
        for text, facts in [
            ("支持即插即用", {"功能": "即插即用"}),
            ("采用IPS面板", {"面板类型": "IPS"}),
            ("独立包装", {"包装": "独立包装"}),
            ("卡通图案印制", {"工艺": "印花"}),
        ]:
            with self.subTest(text=text):
                self.assertFalse(generation.claim_issues(text, facts))

    def test_normal_marketing_is_accepted_without_new_functions(self) -> None:
        """保留主观电商修饰，同时维持事实检查。"""
        sample = generation.normalize_product("1", record("T恤", "材质成分#:#棉100%"))
        payload = generation.make_payload(sample, args())
        output: dict[str, Any] = dict(
            title="纯棉T恤",
            description="100%纯棉面料，柔软舒适，简约风格便于日常搭配。",
            selling_points=["纯棉材质，呈现自然质感。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertTrue(checks["fact_rule_pass"], checks)
        output["description"] += "支持防晒。"
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(checks["fact_rule_pass"])
        self.assertIn("description", checks["field_issues"])

    def test_mixed_fabric_cannot_be_advertised_as_pure_cotton(self) -> None:
        """其他位置比例正确，不能抵消混纺被写成纯棉的错误。"""
        sample = generation.normalize_product(
            "1", record("卫衣", "材质成分#:#棉90% 氨纶10%")
        )
        payload = generation.make_payload(sample, args())
        output = dict(
            title="棉氨混纺卫衣",
            description="90%棉与10%氨纶混纺。",
            selling_points=["纯棉材质，适合日常搭配。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(checks["fact_rule_pass"])
        self.assertIn("selling_points", checks["field_issues"])

    def test_color_options_do_not_imply_size_options(self) -> None:
        """即使存在部分选项，也必须区分颜色、尺寸等规格类型。"""
        sample = generation.normalize_product(
            "1", record("收纳箱", "颜色分类#:#白色#;#颜色分类#:#黑色")
        )
        payload = generation.make_payload(sample, args())
        output = dict(
            title="收纳箱",
            description="提供不同尺寸选择。",
            selling_points=["用于整理杂物。"],
        )
        self.assertFalse(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )

    def test_known_function_omission_is_a_repair_target(self) -> None:
        """验证旧核心列表之外的已知功能不会被悄悄遗漏。"""
        sample = generation.normalize_product("1", record("T恤", "功能#:#透气"))
        sample["core_fields"] = ["商品类型"]
        payload = generation.make_payload(sample, args())
        self.assertIn("功能", payload["core_fields"])
        output: dict[str, Any] = dict(
            title="T恤",
            description="简约风格，便于日常搭配。",
            selling_points=["经典单品，适合日常搭配。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(checks["quality_gate_pass"])
        self.assertIn("description", checks["field_issues"])
        output["description"] += "具备透气功能。"
        self.assertTrue(
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "fact_rule_pass"
            ]
        )

    def test_function_aliases_keep_all_provided_functions(self) -> None:
        """允许功能同义表达，但不能遗漏其他已知功能。"""
        self.assertTrue(
            generation.fact_covered("功能", "可壁挂、升降", "支持壁挂及升降")
        )
        self.assertFalse(generation.fact_covered("功能", "可壁挂、升降", "支持壁挂"))
        self.assertTrue(
            generation.fact_covered(
                "功能", "主动降噪功能(Active Noise Canceling)", "支持主动降噪"
            )
        )

    def test_specification_only_points_are_not_ecommerce_copy(self) -> None:
        """识别重复陈述规格的卖点，不能因此认定文案通顺合格。"""
        payload = dict(
            facts={"商品类型": "水杯", "材质": "不锈钢", "容量": "0.35L"},
            core_fields=["商品类型", "材质", "容量"],
            selling_point_count=3,
            length_limits={"title": 30, "description": 150, "selling_point": 60},
        )
        output = dict(
            title="不锈钢水杯",
            description="不锈钢材质，容量0.35L。",
            selling_points=[
                "不锈钢材质：选用不锈钢材料打造水杯主体。",
                "0.35L容量：单杯容量为0.35L。",
            ],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(checks["copywriting_rule_pass"])

    def test_car_charger_type_is_not_reduced_to_generic_charger(self) -> None:
        """保留来源明确的商品子类型，不凭常识补类型。"""
        sample = generation.normalize_product("1", record("车载充电器", "品牌#:#YX"))
        self.assertEqual(sample["facts"]["商品类型"], "车载充电器")

    def test_long_title_rebuild_preserves_other_fields(self) -> None:
        """超长标题按已知属性重组，保留完整型号和其余字段。"""
        sample = generation.normalize_product(
            "1",
            record(
                "显示器",
                "品牌#:#Asus/华硕#;#型号#:#XG32VQ#;#尺寸#:#32英寸#;#分辨率#:#2560x1440#;#刷新率#:#144HZ",
            ),
        )
        payload = generation.make_payload(sample, args())
        output = dict(
            title="Asus/华硕 XG32VQ 32英寸显示器 2560x1440 144HZ 多参数长标题",
            description="32英寸显示器，分辨率2560x1440，刷新率144HZ。",
            selling_points=["配备32英寸屏幕。"],
        )
        answer, changes = generation.repair_title(
            json.dumps(output, ensure_ascii=False), payload
        )
        result = json.loads(answer)
        self.assertLessEqual(len(result["title"]), 30)
        self.assertIn("XG32VQ", result["title"])
        self.assertEqual(result["description"], output["description"])
        self.assertEqual(result["selling_points"], output["selling_points"])
        self.assertEqual(changes[0]["before"], output["title"])
        self.assertEqual(changes[0]["after"], result["title"])

    def test_short_title_is_not_rewritten(self) -> None:
        """已合规的标题保持原文，不随意改成模板。"""
        sample = generation.normalize_product("1", record("卫衣", "材质#:#棉"))
        payload = generation.make_payload(sample, args())
        answer = json.dumps(
            dict(
                title="棉质卫衣",
                description="棉质面料。",
                selling_points=["棉质面料，适合日常搭配。"],
            ),
            ensure_ascii=False,
        )
        self.assertEqual(generation.repair_title(answer, payload), (answer, []))

    def test_title_rebuild_does_not_escape_repair_targets(self) -> None:
        """局部修复未指定标题时，不额外修改标题。"""
        payload = dict(facts={"商品类型": "水杯"}, length_limits={"title": 10})
        answer = json.dumps(
            dict(title="水杯" * 20, description="原详情", selling_points=["原卖点"]),
            ensure_ascii=False,
        )
        self.assertEqual(
            generation.repair_title(answer, payload, ["selling_points"]), (answer, [])
        )

    def test_unfittable_model_is_not_truncated(self) -> None:
        """完整型号无法容纳时保留失败状态，不能切断型号冒充合规。"""
        payload = dict(
            facts={"商品类型": "显示器", "型号": "MODEL-" + "A" * 40},
            length_limits={"title": 30},
        )
        self.assertIsNone(generation.build_short_title(payload))

    def test_title_capacity_bound_rebuild_is_traceable(self) -> None:
        """短标题中的容量上界丢失也需修复并保留修改记录。"""
        payload = dict(
            facts={"商品类型": "水杯", "容量": "0.25L及以下"},
            length_limits={"title": 30},
        )
        output = dict(
            title="0.25L水杯",
            description="容量0.25L及以下。",
            selling_points=["小容量，便于携带。"],
        )
        answer, changes = generation.repair_title(
            json.dumps(output, ensure_ascii=False), payload
        )
        self.assertTrue(changes)
        self.assertIn("及以下", json.loads(answer)["title"])
        self.assertEqual(json.loads(answer)["description"], output["description"])

    def test_repair_request_has_explicit_title_limits(self) -> None:
        """修复请求给出实际字数和上限，并避免要求标题罗列全部参数。"""
        sample = generation.normalize_product("1", record("显示器", "型号#:#M1"))
        payload = generation.make_payload(sample, args())
        answer = json.dumps(
            dict(
                title="长" * 40,
                description="M1显示器",
                selling_points=["日常显示使用。"],
            ),
            ensure_ascii=False,
        )
        checks = generation.validate(answer, payload)[1]
        request = generation.build_repair_request(payload, answer, checks)
        self.assertEqual(request["repair"]["title_constraints"]["current_chars"], 40)
        self.assertEqual(request["repair"]["title_constraints"]["max_chars"], 30)
        self.assertIn("标题无需列全参数", generation.generation_system(request))

    def test_bluetooth_version_wording_covers_connection(self) -> None:
        """蓝牙版本插在词间时仍能识别连接方式。"""
        self.assertTrue(
            generation.fact_covered("连接方式", "蓝牙连接", "支持蓝牙5.0连接")
        )
        self.assertTrue(
            generation.fact_covered("连接方式", "蓝牙连接", "支持蓝牙 5.0 连接")
        )
        self.assertFalse(
            generation.fact_covered("连接方式", "蓝牙连接", "采用有线连接")
        )

    def test_generic_labels_are_not_valid_selling_points(self) -> None:
        """纸制材质、结构合理、家居日用不能凑成三条卖点。"""
        payload = dict(facts={"商品类型": "收纳箱", "品牌": "沐春", "材质": "纸"})
        output = dict(
            title="沐春纸制收纳箱",
            description="纸质收纳箱，便于整理物品。",
            selling_points=["纸制材质", "结构合理", "家居日用"],
        )
        self.assertTrue(generation.copywriting_issues(output, payload))

    def test_reordered_same_attributes_are_duplicate_points(self) -> None:
        """改变词序不能把同一组属性变成不同核心卖点。"""
        payload = dict(
            facts={
                "商品类型": "衬衫",
                "颜色": "橘色",
                "袖长": "长袖",
                "领型": "翻领尖领",
                "图案": "格子",
                "风格": "休闲",
            }
        )
        output = dict(
            title="橘色格子衬衫",
            description="橘色格子衬衫，适合日常搭配。",
            selling_points=[
                "橘色长袖翻领尖领衬衫，格子图案，休闲风格",
                "格子图案橘色衬衫，长袖翻领尖领，休闲百搭",
            ],
        )
        self.assertTrue(
            any(
                "重复同一组属性" in issue
                for issue in generation.copywriting_issues(output, payload)
            )
        )

    def test_fashion_points_need_not_include_purpose_keywords(self) -> None:
        """面料、领袖、图案分开介绍时，不因缺少用途词误触发修复。"""
        payload = dict(
            facts={
                "商品类型": "T恤",
                "材质": "纯棉",
                "领型": "圆领",
                "袖长": "短袖",
                "图案": "卡通动漫",
                "风格": "欧美",
                "面料成分": "棉100%",
                "功能": "吸汗",
            }
        )
        output = dict(
            title="欧美风纯棉T恤",
            description="纯棉圆领短袖T恤，适合日常穿着。",
            selling_points=[
                "100%纯棉面料，吸汗舒适",
                "圆领短袖设计，简约百搭",
                "卡通动漫图案，欧美风格",
            ],
        )
        self.assertEqual(generation.copywriting_issues(output, payload), [])

    def test_serial_converter_is_not_assumed_to_be_hub(self) -> None:
        """明确的串口转换关系保留来源，上位分类不证明USB多接口功能。"""
        sample = generation.normalize_product(
            "1", record("USB HUB/转换器", "", "USB转RS485隔离模块")
        )
        self.assertEqual(sample["facts"]["商品类型"], "串口转换器")
        self.assertTrue(sample["needs_review"])
        self.assertTrue(
            generation.claim_issues(
                "提供多接口扩展能力，支持连接多种USB设备", sample["facts"]
            )
        )
        self.assertTrue(
            generation.claim_issues(
                "提供多接口扩展能力", {"商品类型": "USB HUB/转换器"}
            )
        )
        self.assertFalse(
            generation.claim_issues(
                "提供多接口扩展能力", {"商品类型": "USB集线器", "功能": "多接口扩展"}
            )
        )

    def test_local_title_repair_does_not_hide_invalid_output(self) -> None:
        """非法输出仍由校验器报错，不把修复失败伪装成成功。"""
        payload = dict(facts={"商品类型": "水杯"}, length_limits={"title": 30})
        for answer in ["not json", "[]", '{"title":12}']:
            with self.subTest(answer=answer):
                self.assertEqual(generation.repair_title(answer, payload), (answer, []))

    def test_model_with_product_type_is_not_duplicated(self) -> None:
        """型号已经含商品类型时不重复追加，避免产生本可避免的超长。"""
        payload: dict[str, Any] = dict(
            facts={"商品类型": "蓝牙耳机", "型号": "Redmi AirDots 真无线蓝牙耳机"},
            length_limits={"title": 24},
        )
        self.assertEqual(
            generation.build_short_title(payload), payload["facts"]["型号"]
        )

    def test_unsupported_effect_guarantees_are_detected(self) -> None:
        """刷新率和商品类型不能证明流畅度、准确性与效率保证。"""
        for text, facts in [
            (
                "60Hz刷新率，确保流畅的视觉体验",
                {"商品类型": "显示器", "刷新率": "60Hz"},
            ),
            ("确保数据交互的准确性与效率", {"商品类型": "串口转换器"}),
            ("保障画面始终流畅", {"商品类型": "显示器"}),
            ("保证信号传输不丢包", {"商品类型": "串口转换器"}),
        ]:
            with self.subTest(text=text):
                self.assertTrue(generation.effect_guarantee_issues(text, facts))
                self.assertTrue(generation.claim_issues(text, facts))

    def test_plain_function_is_not_a_stronger_guarantee(self) -> None:
        """输入描述流畅显示，不代表允许追加确保流畅的承诺。"""
        self.assertTrue(
            generation.effect_guarantee_issues("确保流畅显示", {"功能": "流畅显示"})
        )

    def test_matching_explicit_commitment_is_allowed(self) -> None:
        """输入有对应明确承诺时，允许按其原文引用。"""
        self.assertFalse(
            generation.effect_guarantee_issues(
                "确保流畅显示", {"厂家承诺": "确保流畅显示"}
            )
        )

    def test_negated_evidence_cannot_support_guarantee(self) -> None:
        """不保证、未证实等描述不能当作正向保证的依据。"""
        self.assertTrue(
            generation.effect_guarantee_issues(
                "保证数据传输准确性", {"厂家承诺": "不保证数据传输准确性"}
            )
        )

    def test_normal_marketing_and_function_explanation_are_kept(self) -> None:
        """正常修饰和已有功能解释不会因保证检查被一概否决。"""
        for text in [
            "线条流畅，经典百搭，柔软舒适",
            "支持主动降噪，减少环境噪音",
            "核对型号，确保型号原样保留",
        ]:
            with self.subTest(text=text):
                self.assertFalse(
                    generation.effect_guarantee_issues(text, {"功能": "主动降噪"})
                )
        self.assertFalse(
            generation.effect_guarantee_issues("无法保证数据传输准确性", {})
        )

    def test_long_point_repack_preserves_all_text_and_other_fields(self) -> None:
        """完整分句重排解决超长卖点，不丢属性、品牌、限定词和其余字段。"""
        payload = dict(
            facts={
                "商品类型": "水杯",
                "品牌": "Fire－Maple/火枫",
                "材质": "钛合金",
                "尺寸": "134*134*60MM",
                "容量": "0.25L及以下",
                "用途": "成人",
            },
            length_limits={"title": 30, "description": 150, "selling_point": 40},
            selling_point_count=3,
        )
        output: dict[str, Any] = dict(
            title="火枫水杯",
            description="原详情保持不变。",
            selling_points=[
                "Fire－Maple/火枫成人水杯，采用钛合金材质，尺寸134*134*60MM，容量0.25L及以下",
                "专为成人设计，提供简约自然的日常饮水体验",
                "结构紧凑，适合随身携带",
            ],
        )
        answer, changes = generation.repair_selling_points(
            json.dumps(output, ensure_ascii=False), payload
        )
        result = json.loads(answer)
        self.assertTrue(changes)
        self.assertLessEqual(len(result["selling_points"]), 3)
        self.assertTrue(all(len(p) <= 40 for p in result["selling_points"]))
        self.assertEqual(result["title"], output["title"])
        self.assertEqual(result["description"], output["description"])

        def content_key(value: list[str]) -> str:
            """忽略分句标点后比较完整文字。"""
            return generation.re.sub(r"[，,；;。\s]", "", " ".join(value))

        self.assertEqual(
            content_key(result["selling_points"]), content_key(output["selling_points"])
        )
        self.assertIn("0.25L及以下", " ".join(result["selling_points"]))

    def test_unbreakable_long_model_is_not_truncated(self) -> None:
        """无法沿完整分句压缩时保留失败，不切断型号充当成功。"""
        model = "MODEL-" + "A" * 50
        payload = dict(
            facts={"商品类型": "键盘", "型号": model},
            length_limits={"selling_point": 40},
            selling_point_count=3,
        )
        answer = json.dumps(
            dict(title="键盘", description=model, selling_points=[model]),
            ensure_ascii=False,
        )
        self.assertEqual(
            generation.repair_selling_points(answer, payload), (answer, [])
        )

    def test_short_points_are_not_repacked(self) -> None:
        """合规短卖点保持原样，避免无必要改写。"""
        payload = dict(
            facts={"商品类型": "水杯"},
            length_limits={"selling_point": 40},
            selling_point_count=3,
        )
        answer = json.dumps(
            dict(
                title="水杯",
                description="日常饮水使用。",
                selling_points=["小巧便携，适合日常使用。"],
            ),
            ensure_ascii=False,
        )
        self.assertEqual(
            generation.repair_selling_points(answer, payload), (answer, [])
        )

    def test_point_repack_respects_partial_repair_targets(self) -> None:
        """未指定卖点修复时，不额外改写卖点。"""
        payload = dict(
            facts={"商品类型": "水杯"},
            length_limits={"selling_point": 10},
            selling_point_count=3,
        )
        answer = json.dumps(
            dict(
                title="水杯",
                description="原详情",
                selling_points=["水杯采用已知材质，适合日常饮水，便于整理摆放"],
            ),
            ensure_ascii=False,
        )
        self.assertEqual(
            generation.repair_selling_points(answer, payload, ["title"]), (answer, [])
        )

    def test_clause_split_keeps_parentheses_numbers_and_known_models(self) -> None:
        """括号内分隔、千分位和型号中的逗号不能作为切断位置。"""
        self.assertEqual(
            generation.split_complete_clauses(
                "面料（棉90%，氨纶10%），容量1,000mAh，便于携带", {}
            ),
            ["面料（棉90%，氨纶10%）", "容量1,000mAh，便于携带"],
        )
        self.assertEqual(
            generation.split_complete_clauses(
                "型号MODEL,A100，适合日常使用", {"型号": "MODEL,A100"}
            ),
            ["型号MODEL,A100，适合日常使用"],
        )

    def test_clause_split_keeps_negative_scope(self) -> None:
        """保留但不支持等限定语与前文的关系。"""
        self.assertEqual(
            generation.split_complete_clauses(
                "支持有线连接，但不支持蓝牙，便于核对", {}
            ),
            ["支持有线连接，但不支持蓝牙，便于核对"],
        )

    def test_point_repack_does_not_exceed_count_or_drop_content(self) -> None:
        """内容容纳不下时保留错误，不通过删句或增加条数规避。"""
        payload = dict(
            facts={"商品类型": "水杯"},
            length_limits={"selling_point": 8},
            selling_point_count=1,
        )
        answer = json.dumps(
            dict(
                title="水杯",
                description="原详情",
                selling_points=["采用钛合金材质，便于日常摆放，容量范围已知"],
            ),
            ensure_ascii=False,
        )
        self.assertEqual(
            generation.repair_selling_points(answer, payload), (answer, [])
        )

    def test_invalid_point_structure_is_not_silently_fixed(self) -> None:
        """非法卖点类型保留给校验器，不能悄悄丢掉错误元素。"""
        payload = dict(
            facts={"商品类型": "水杯"},
            length_limits={"selling_point": 40},
            selling_point_count=3,
        )
        for answer in ["not json", "[]", '{"selling_points":[12]}']:
            with self.subTest(answer=answer):
                self.assertEqual(
                    generation.repair_selling_points(answer, payload), (answer, [])
                )

    def test_repair_request_lists_each_point_length(self) -> None:
        """失败诊断及修复请求明确列出超长卖点序号和字符上限。"""
        sample = generation.normalize_product("1", record("水杯", "材质#:#不锈钢"))
        payload = generation.make_payload(sample, args())
        answer = json.dumps(
            dict(
                title="不锈钢水杯",
                description="不锈钢水杯。",
                selling_points=["长" * 51],
            ),
            ensure_ascii=False,
        )
        checks = generation.validate(answer, payload)[1]
        self.assertTrue(
            any(
                "卖点1长度51字，超过40字" in issue
                for issue in checks["field_issues"]["selling_points"]
            )
        )
        request = generation.build_repair_request(payload, answer, checks)
        self.assertEqual(
            request["repair"]["selling_point_constraints"]["items"][0]["current_chars"],
            51,
        )
        self.assertEqual(
            request["repair"]["selling_point_constraints"]["max_chars"], 40
        )

    def test_local_repack_cannot_whitewash_effect_guarantee(self) -> None:
        """长度修复不会删除承诺或把存在事实风险的文案直接判合格。"""
        payload = dict(
            facts={"商品类型": "显示器", "刷新率": "60Hz"},
            core_fields=["商品类型", "刷新率"],
            length_limits={"title": 30, "description": 150, "selling_point": 24},
            selling_point_count=3,
        )
        output = dict(
            title="60Hz显示器",
            description="刷新率60Hz。",
            selling_points=["采用60Hz刷新率，适合日常显示，确保流畅的视觉体验"],
        )
        answer, _ = generation.repair_selling_points(
            json.dumps(output, ensure_ascii=False), payload
        )
        self.assertIn("确保流畅", answer)
        self.assertFalse(generation.validate(answer, payload)[1]["fact_rule_pass"])

    def test_startup_cleanup_releases_unused_cache_once(self) -> None:
        """启动清理仅调用一次，不减少张量占用，并记录前后显存。"""
        events = []
        state = {"reserved": 8 * 1024**3}

        def clear() -> None:
            """模拟仅释放未使用的分配器缓存。"""
            events.append("empty_cache")
            state["reserved"] = 5 * 1024**3

        cuda = SimpleNamespace(
            synchronize=lambda: events.append("sync"),
            empty_cache=clear,
            memory_allocated=lambda: 3 * 1024**3,
            memory_reserved=lambda: state["reserved"],
        )
        with patch.object(
            generation.gc, "collect", side_effect=lambda: events.append("gc")
        ):
            report = generation.release_startup_cuda_cache(SimpleNamespace(cuda=cuda))
        self.assertEqual(events, ["gc", "sync", "empty_cache", "sync"])
        self.assertTrue(report["allocated_bytes_unchanged"])
        self.assertEqual(report["after"]["reserved_bytes"], 5 * 1024**3)
        self.assertEqual(report["empty_cache_calls"], 1)

    def test_startup_cleanup_does_not_hide_cuda_failure(self) -> None:
        """清理接口失败时保留异常，不能伪造成功日志。"""
        cuda = SimpleNamespace(
            synchronize=lambda: None,
            memory_allocated=lambda: 1,
            memory_reserved=lambda: 2,
            empty_cache=lambda: (_ for _ in ()).throw(RuntimeError("CUDA异常")),
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA异常"):
            generation.release_startup_cuda_cache(SimpleNamespace(cuda=cuda))

    def test_wireless_entailment_is_scoped_to_connection(self) -> None:
        """蓝牙只支持无线通信表述，不扩大为无线充电或真无线结构。"""
        facts = {"商品类型": "蓝牙耳机", "连接方式": "蓝牙连接", "蓝牙版本": "5.0"}
        self.assertFalse(generation.claim_issues("采用蓝牙5.0无线连接", facts))
        for text in ["支持无线充电", "真无线耳机", "无线连接并支持无线充电"]:
            with self.subTest(text=text):
                self.assertTrue(generation.claim_issues(text, facts))

    def test_wired_connection_does_not_imply_bluetooth(self) -> None:
        """同属连接方式也不能把有线设备描述为蓝牙或无线设备。"""
        facts = {"商品类型": "耳机", "连接方式": "有线连接"}
        for text in ["支持蓝牙连接", "支持无线连接"]:
            with self.subTest(text=text):
                self.assertTrue(generation.claim_issues(text, facts))

    def test_property_support_requires_relevant_fields(self) -> None:
        """品牌、颜色、风格、材质名称不能充当性能或环保依据。"""
        for facts, text in [
            ({"品牌": "耐用科技", "材质": "塑料"}, "产品耐用"),
            ({"颜色": "环保绿", "材质": "纸"}, "环保材质"),
            ({"材质": "稳固合金"}, "结构稳固"),
            ({"风格": "防水风格"}, "具备防水功能"),
        ]:
            with self.subTest(facts=facts):
                self.assertTrue(generation.claim_issues(text, facts))

    def test_explicit_properties_are_allowed_across_categories(self) -> None:
        """相同声明机制支持不同品类的明确正向依据。"""
        for facts, text in [
            ({"功能": "耐用", "商品类型": "水杯"}, "耐用水杯"),
            ({"功能": "透气", "商品类型": "T恤"}, "具备透气功能"),
            ({"是否防水": "是", "商品类型": "蓝牙耳机"}, "支持防水"),
            ({"环保": "环保", "商品类型": "收纳箱"}, "环保收纳箱"),
        ]:
            with self.subTest(facts=facts):
                self.assertFalse(generation.claim_issues(text, facts))

    def test_negative_source_does_not_prove_positive_property(self) -> None:
        """不支持、无认证和不耐用不能通过字面包含成为正向证据。"""
        for facts, text in [
            ({"功能": "不支持防水"}, "支持防水"),
            ({"认证": "无认证"}, "已获认证"),
            ({"性能": "不耐用"}, "耐用设计"),
            ({"是否防水": "否"}, "支持防水"),
        ]:
            with self.subTest(facts=facts):
                self.assertTrue(generation.claim_issues(text, facts))

    def test_output_negation_is_checked_separately(self) -> None:
        """已知支持与已知不支持分别核验，未知也不能写成不支持。"""
        self.assertTrue(generation.claim_issues("不支持防水", {"功能": "防水"}))
        self.assertTrue(generation.claim_issues("不支持防水", {"商品类型": "水杯"}))
        self.assertFalse(generation.claim_issues("不支持防水", {"功能": "不支持防水"}))
        self.assertFalse(generation.fact_covered("功能", "防水", "不支持防水"))
        self.assertTrue(
            generation.fact_covered("功能", "不支持防水", "该商品不支持防水")
        )

    def test_negation_does_not_consume_another_positive_property(self) -> None:
        """一个否定声明不会污染另一个正向声明，无毒也不是否定防水。"""
        facts = {"功能": "不支持快充，支持防水"}
        self.assertFalse(generation.claim_issues("不支持快充，支持防水", facts))
        self.assertFalse(generation.claim_issues("无毒且防水", {"功能": "无毒，防水"}))

    def test_interface_claims_are_checked_by_interface_field(self) -> None:
        """接口枚举按对应属性检查，不能借面板或型号词证明其他接口。"""
        self.assertTrue(generation.claim_issues("配备HDMI接口", {"接口": "VGA"}))
        self.assertFalse(generation.claim_issues("配备HDMI接口", {"接口": "HDMI"}))
        model = "FRM-101HB-HDMI-RT"
        self.assertFalse(generation.claim_issues(model + "显示器", {"型号": model}))
        self.assertTrue(
            generation.claim_issues(model + "显示器，配备HDMI接口", {"型号": model})
        )

    def test_compatibility_scope_does_not_expand(self) -> None:
        """兼容多种设备不等于所有设备，接口存在也不证明兼容范围。"""
        for facts, text in [
            ({"接口": "USB"}, "兼容所有设备"),
            ({"功能": "兼容多种设备"}, "兼容所有设备"),
            ({"商品类型": "收纳盒"}, "适用于各类电池"),
            ({"接口": "USB"}, "适用于各类电脑设备"),
        ]:
            with self.subTest(text=text):
                self.assertTrue(generation.claim_issues(text, facts))
        self.assertFalse(
            generation.claim_issues("兼容所有设备", {"兼容范围": "兼容所有设备"})
        )

    def test_multi_device_capability_is_evidence_bound(self) -> None:
        """多设备能力用同一机制检查，不限定某个商品编号或某个具体型号。"""
        for product_type in ["串口转换器", "USB HUB/转换器", "键盘", "显示器"]:
            with self.subTest(product_type=product_type):
                self.assertTrue(
                    generation.claim_issues(
                        "支持多设备连接", {"商品类型": product_type, "接口": "USB"}
                    )
                )
        self.assertFalse(
            generation.claim_issues(
                "支持多设备连接", {"商品类型": "键盘", "功能": "多设备连接"}
            )
        )
        self.assertFalse(
            generation.claim_issues("支持多接口扩展", {"商品类型": "USB集线器"})
        )

    def test_brand_positioning_is_not_single_domain_special_case(self) -> None:
        """品牌定位统一依赖介绍证据，覆盖家居、数码与服饰等不同领域。"""
        for domain in ["家居日用", "数码配件", "运动服饰"]:
            claim = "专注" + domain
            with self.subTest(domain=domain):
                self.assertTrue(generation.claim_issues(claim, {"品牌": "示例品牌"}))
                self.assertFalse(generation.claim_issues(claim, {"品牌定位": claim}))

    def test_required_quantity_cannot_hide_behind_high_coverage(self) -> None:
        """核心覆盖率达到八成也不能掩盖已知数量遗漏。"""
        payload = dict(
            facts={
                "商品类型": "衣架",
                "材质": "塑料",
                "颜色": "白色",
                "风格": "简约",
                "数量": "1个",
            },
            core_fields=["商品类型", "材质", "颜色", "风格", "数量"],
            length_limits={"title": 30, "description": 150, "selling_point": 40},
            selling_point_count=3,
        )
        output = dict(
            title="白色塑料衣架",
            description="简约风格，适合日常衣物悬挂。",
            selling_points=["塑料材质，便于搭配家居空间。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertEqual(checks["core_coverage"], 0.8)
        self.assertIn("数量", checks["missing_required_fields"])
        self.assertFalse(checks["quality_gate_pass"])
        self.assertIn("关键遗漏", checks["issue_category_counts"])

    def test_required_omission_is_found_even_outside_legacy_core_list(self) -> None:
        """旧核心列表漏选关键参数时仍能检查已知数量和容量。"""
        sample = generation.normalize_product(
            "1", record("衣架", "容量#:#2个#;#材质#:#塑料")
        )
        sample["core_fields"] = ["商品类型"]
        payload = generation.make_payload(sample, args())
        output = dict(
            title="塑料衣架",
            description="适合日常悬挂衣物。",
            selling_points=["塑料材质，便于日常使用。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertEqual(checks["core_coverage"], 1)
        self.assertFalse(checks["quality_gate_pass"])
        self.assertIn("数量", payload["required_fields"])

    def test_noncritical_details_can_be_omitted(self) -> None:
        """关键字段完整且达到覆盖率时，不强迫复述全部非关键细节。"""
        sample = generation.normalize_product(
            "1", record("T恤", "袖长#:#短袖#;#风格#:#休闲")
        )
        payload = generation.make_payload(sample, args())
        output = dict(
            title="短袖T恤",
            description="适合日常搭配。",
            selling_points=["短袖设计，便于日常穿搭。"],
        )
        self.assertNotIn("风格", payload["required_fields"])
        self.assertNotIn(
            "风格",
            generation.validate(json.dumps(output, ensure_ascii=False), payload)[1][
                "missing_required_fields"
            ],
        )

    def test_quantity_normalization_keeps_count_semantics(self) -> None:
        """允许明确数量语境的中文计数，不把一个需求当作商品数量。"""
        self.assertTrue(generation.fact_covered("数量", "1个", "包装内含一个衣架"))
        self.assertTrue(generation.fact_covered("数量", "1个", "数量为1件"))
        self.assertFalse(generation.fact_covered("数量", "1个", "是一个实用选择"))
        self.assertFalse(generation.fact_covered("数量", "1双", "数量为1只"))

    def test_structured_issues_include_category_location_and_action(self) -> None:
        """分类诊断保留字段位置、原文和通用修复策略。"""
        sample = generation.normalize_product("1", record("收纳箱", "材质#:#纸"))
        payload = generation.make_payload(sample, args())
        output = dict(
            title="纸制收纳箱",
            description="纸质材质，环保且耐用。",
            selling_points=["用于整理杂物。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        selected = [
            i
            for i in checks["quality_issues"]
            if i["field"] == "description" and i["category"] == "无依据声明"
        ]
        self.assertTrue(selected)
        self.assertEqual(selected[0]["quote"], output["description"])
        self.assertTrue(selected[0]["code"])
        self.assertTrue(selected[0]["repair_action"])
        self.assertEqual(checks["rules_version"], generation.QUALITY_RULES_VERSION)

    def test_format_and_structure_claim_have_different_categories(self) -> None:
        """长度不合法和真实结构声明不能因为同含结构一词而混淆。"""
        self.assertEqual(
            generation.classify_quality_issue("卖点结构或长度不符合要求")[0], "格式约束"
        )
        self.assertEqual(
            generation.classify_quality_issue("具体结构无依据")[0], "无依据声明"
        )
        self.assertEqual(
            generation.classify_quality_issue("否定声明与输入矛盾：防水")[0], "属性矛盾"
        )

    def test_source_conflict_is_not_repaired_by_guessing(self) -> None:
        """已标记输入冲突单列待核，模型输出合法也不能自行解除。"""
        sample = generation.normalize_product("1", record("T恤", "袖长#:#短袖"))
        sample["conflicts"] = [{"field": "材质", "values": ["棉", "涤纶"]}]
        payload = generation.make_payload(sample, args())
        output = dict(
            title="短袖T恤",
            description="适合日常搭配。",
            selling_points=["短袖设计，便于日常穿搭。"],
        )
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertTrue(checks["input_review_required"])
        self.assertFalse(checks["quality_gate_pass"])
        self.assertIn("输入冲突", checks["issue_category_counts"])

    def test_repair_request_restores_exact_missing_values_by_category(self) -> None:
        """修复请求携带问题类别和缺失原值，不只给出笼统重写指令。"""
        sample = generation.normalize_product(
            "1", record("衣架", "容量#:#1个#;#材质#:#塑料")
        )
        payload = generation.make_payload(sample, args())
        answer = json.dumps(
            dict(
                title="塑料衣架",
                description="适合悬挂衣物。",
                selling_points=["塑料材质，便于日常使用。"],
            ),
            ensure_ascii=False,
        )
        checks = generation.validate(answer, payload)[1]
        request = generation.build_repair_request(payload, answer, checks)
        self.assertIn("关键遗漏", request["repair"]["category_actions"])
        self.assertEqual(request["repair"]["required_facts_to_restore"]["数量"], "1个")
        self.assertIn("description", request["repair"]["target_fields"])

    def test_category_statistics_count_products_not_repeated_findings(self) -> None:
        """同条多处同类错误在影响商品数统计中只计一次。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sample = dict(id="S001", category="家居日用")
            generation.write_json(folder / "manifest.json", {"samples": [sample]})
            generation.write_json(
                folder / "review.json",
                {
                    "items": [
                        dict(id="S001", **{k: None for k in generation.REVIEW_FIELDS})
                    ]
                },
            )
            generation.write_json(
                folder / "records/S001.json",
                dict(
                    result={"title": "收纳箱"},
                    validation=dict(
                        format_pass=True,
                        fact_rule_pass=False,
                        quality_gate_pass=False,
                        core_coverage=1,
                        quality_issues=[
                            dict(category="无依据声明"),
                            dict(category="无依据声明"),
                        ],
                    ),
                ),
            )
            report = generation.summarize(folder)
            self.assertEqual(
                report["overall"]["rule_issue_categories"]["affected_counts"][
                    "无依据声明"
                ],
                1,
            )
            self.assertEqual(
                report["by_category"]["家居日用"]["rule_issue_categories"][
                    "available_records"
                ],
                1,
            )

    def test_invalid_json_also_has_structured_diagnostics(self) -> None:
        """非法JSON也按格式类别记录，不丢失统一诊断结构。"""
        sample = generation.normalize_product("1", record("T恤", ""))
        checks = generation.validate(
            "not json", generation.make_payload(sample, args())
        )[1]
        self.assertIn("格式约束", checks["issue_category_counts"])
        self.assertFalse(checks["quality_gate_pass"])

    def test_wireless_audio_variants_keep_evidence_boundary(self) -> None:
        """蓝牙允许音频传输与使用，不允许扩展为充电或免驱。"""
        facts = {"商品类型": "蓝牙耳机", "连接方式": "蓝牙连接"}
        for text in ["无线音频传输", "无线音频体验", "无线使用"]:
            self.assertEqual(generation.claim_issues(text, facts), [])
        for text in ["无线充电", "真无线耳机", "免驱安装"]:
            self.assertTrue(generation.claim_issues(text, facts))

    def test_fiber_parentheses_punctuation_and_ratio_direction(self) -> None:
        """材料别名括号及标点不影响比例，比例改变仍被发现。"""
        source = "粘胶纤维(粘纤)54.6% 聚酯纤维29.2% 聚酰胺纤维(锦纶)16.2%"
        output = "面料成分：粘胶纤维 （粘纤）54.6%、聚酯纤维 29.2%、聚酰胺纤维 （锦纶）16.2%。"
        self.assertTrue(generation.fact_covered("面料成分", source, output))
        self.assertFalse(
            generation.fact_covered(
                "面料成分", source, output.replace("54.6%", "45.6%")
            )
        )
        self.assertEqual(
            generation.composition("65%聚酯纤维、35%棉"),
            generation.composition("棉35% 涤纶65%"),
        )

    def test_interface_alias_and_combined_description(self) -> None:
        """复合接口逐项核验，不要求复述整段原始字段。"""
        source = "苹果Lighting+安卓OTG+电脑 三合一手机U盘"
        self.assertTrue(
            generation.fact_covered(
                "接口", source, "苹果Lightning、安卓OTG及电脑三合一接口"
            )
        )
        self.assertFalse(generation.fact_covered("接口", source, "安卓OTG接口"))

    def test_weave_and_fiber_are_different_dimensions(self) -> None:
        """斜纹织法与棉纤维可以共存，纯棉与混纺仍需隔离。"""
        sample = generation.normalize_product(
            "1", record("夹克", "材质#:#棉#;#面料#:#斜纹布")
        )
        self.assertEqual(sample["facts"]["材质"], "棉")
        self.assertEqual(sample["facts"]["织法"], "斜纹布")
        self.assertFalse(sample["conflicts"])
        bad = generation.normalize_product(
            "2", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        self.assertTrue(bad["conflicts"])

    def test_known_multiple_interfaces_and_variant_ambiguity(self) -> None:
        """公共多接口共存可用，单模双模混合SKU保留冲突。"""
        good = generation.normalize_product(
            "1", record("显示器", "接口类型#:#HDMI#;#接口类型#:#VGA")
        )
        self.assertIn("HDMI", good["facts"]["接口"])
        self.assertIn("VGA", good["facts"]["接口"])
        bad = generation.normalize_product(
            "2",
            record(
                "键盘",
                "接口类型#:#USB#;#接口类型#:#蓝牙#;#颜色分类#:#【USB】黑色#;#颜色分类#:#【双模】白色",
            ),
        )
        self.assertNotIn("接口", bad["facts"])
        self.assertTrue(bad["conflicts"])

    def test_conflict_values_never_enter_initial_or_repair_prompt(self) -> None:
        """初始请求与修复上下文都不携带争议原值及原始审计信息。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        payload = generation.make_payload(sample, args())
        payload["repair"] = {
            "previous_output": "棉混纺布，棉100%",
            "field_issues": {"description": ["删除棉100%"]},
        }
        payload["raw_record"] = {"pv": "绝不发送"}
        safe = generation.generation_payload(payload, sample["conflicts"])
        serialized = json.dumps(safe, ensure_ascii=False)
        for value in ["棉混纺布", "棉100%", "source_conflicts", "绝不发送"]:
            self.assertNotIn(value, serialized)
        self.assertIn("材质", safe["unavailable_fields"])
        self.assertTrue(sample["conflicts"][0]["values"])

    def test_known_option_numbers_not_false_independent_values(self) -> None:
        """已知尺码数字可用于尺码表述，新增数字仍失败且定位字段。"""
        sample = generation.normalize_product(
            "1", record("针织衫", "尺码#:#165/M#;#尺码#:#170/L")
        )
        payload = generation.make_payload(sample, args())
        output: dict[str, Any] = {
            "title": "针织衫",
            "description": "针织衫，提供165/M、170/L尺码可选。",
            "selling_points": ["针织衫，尺码可选"],
        }
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(any("独立数值" in risk for risk in checks["fact_risks"]))
        output["description"] += "身高199适用。"
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertTrue(any("199" in risk for risk in checks["fact_risks"]))
        self.assertTrue(checks["field_issues"].get("description"))

    def test_discrete_size_options_do_not_become_age_range(self) -> None:
        """离散尺码不能概括为未给出的年龄或连续范围。"""
        payload = {
            "option_types": {"尺码": ["1-2M（儿童）", "3-4M（儿童）", "7-8M（儿童）"]}
        }
        for text in ["提供1-8M儿童尺码", "满足不同年龄段需求"]:
            self.assertTrue(
                generation.specification_issues(text, "description", payload)
            )
        self.assertEqual(
            generation.specification_issues("1-2M（儿童）可选", "description", payload),
            [],
        )

    def test_selected_sku_checked_and_does_not_mutate_source(self) -> None:
        """显式选择只影响新请求，未知SKU不能被任意指定。"""
        sample = generation.normalize_product(
            "1", record("T恤", "颜色分类#:#黑色#;#颜色分类#:#白色")
        )
        sample["selected_sku"] = {"颜色": "黑色"}
        payload = generation.make_payload(sample, args())
        self.assertEqual(payload["facts"]["颜色"], "黑色")
        self.assertNotIn("颜色", sample["facts"])
        self.assertTrue(generation.specification_issues("白色T恤", "title", payload))
        sample["selected_sku"] = {"颜色": "红色"}
        with self.assertRaises(ValueError):
            generation.make_payload(sample, args())

    def test_multidevice_claim_variants_require_functional_proof(self) -> None:
        """不同语序的多设备连接由同类证据约束。"""
        for text in ["多设备同时连接", "便于连接多个设备", "支持多设备连接"]:
            self.assertTrue(generation.claim_issues(text, {"商品类型": "串口转换器"}))
            self.assertFalse(
                generation.claim_issues(
                    text, {"商品类型": "移动电源", "功能": "多U口输出"}
                )
            )

    def test_multiple_storage_specifications_require_explicit_proof(self) -> None:
        """商品类型不能证明适配不同规格，明确功能证据可以支持。"""
        text = "便于分类整理不同规格的电池"
        self.assertTrue(generation.claim_issues(text, {"商品类型": "电池收纳盒"}))
        self.assertFalse(
            generation.claim_issues(
                text, {"商品类型": "电池收纳盒", "功能": "收纳不同规格的电池"}
            )
        )

    def test_single_quantity_does_not_imply_packaging(self) -> None:
        """数量一件与独立包装是两个事实。"""
        self.assertTrue(generation.claim_issues("单件包装", {"数量": "1个"}))
        self.assertFalse(generation.claim_issues("单件包装", {"包装方式": "单件包装"}))

    def test_long_color_enumeration_can_be_safely_compressed(self) -> None:
        """完整颜色枚举可概括为多色可选，保留原文修改记录。"""
        payload = {
            "facts": {"商品类型": "垃圾桶"},
            "option_types": {"颜色": ["黑色", "白色", "绿色", "黄色"]},
            "selling_point_count": 3,
            "length_limits": {"selling_point": 15},
        }
        output = {
            "title": "垃圾桶",
            "description": "垃圾桶，颜色可选",
            "selling_points": ["颜色：提供黑色、白色、绿色、黄色等多种颜色可选。"],
        }
        answer, changes = generation.repair_selling_points(
            json.dumps(output, ensure_ascii=False), payload
        )
        self.assertTrue(changes)
        self.assertEqual(json.loads(answer)["selling_points"], ["提供多种颜色可选"])

    def test_compression_must_not_drop_unique_parameters(self) -> None:
        """独有型号和容量超长时不能通过删信息假装合格。"""
        payload = {
            "facts": {
                "商品类型": "水杯",
                "型号": "MODEL-VERY-LONG-123456",
                "容量": "0.35L",
            },
            "selling_point_count": 3,
            "length_limits": {"selling_point": 10},
        }
        output = {
            "title": "水杯",
            "description": "水杯",
            "selling_points": ["型号MODEL-VERY-LONG-123456，容量0.35L"],
        }
        before = json.dumps(output, ensure_ascii=False)
        answer, changes = generation.repair_selling_points(before, payload)
        self.assertEqual(answer, before)
        self.assertEqual(changes, [])

    def test_duplicate_merge_keeps_fact_coverage_and_target_scope(self) -> None:
        """去重保留三项事实覆盖，局部修复不能越权改卖点。"""
        payload = {"facts": {"商品类型": "水杯", "材质": "不锈钢", "容量": "0.6L"}}
        output = {
            "title": "水杯",
            "description": "水杯",
            "selling_points": ["不锈钢水杯，容量0.6L", "容量0.6L的不锈钢水杯"],
        }
        before = json.dumps(output, ensure_ascii=False)
        after, changes = generation.deduplicate_selling_points(before, payload, [])
        self.assertEqual(len(json.loads(after)["selling_points"]), 1)
        self.assertTrue(changes)
        self.assertEqual(
            generation.deduplicate_selling_points(before, payload, ["title"])[0], before
        )

    def test_point_plan_does_not_force_three_points_from_two_facts(self) -> None:
        """只有商品类型和品牌时只规划有证据的主题。"""
        payload = {
            "facts": {"商品类型": "收纳盒", "品牌": "测试品牌"},
            "selling_point_count": 3,
        }
        plan = generation.build_point_plan(payload)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["fields"], ["商品类型"])

    def test_launcher_rejects_zero_discovered_tests(self) -> None:
        """兼容启动器在零项测试时必须失败。"""
        launcher_path = (
            Path(__file__).resolve().parents[2] / "scripts/test_copywriting_logic.py"
        )
        spec = importlib.util.spec_from_file_location(
            "launcher_for_test", launcher_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法载入测试启动器：{launcher_path}")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with patch.object(
            unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()
        ):
            with self.assertRaises(SystemExit) as caught:
                launcher.main()
        self.assertNotEqual(caught.exception.code, 0)

    def test_size_letters_do_not_match_brand_names(self) -> None:
        """短尺码标记不能误命中Columbia等品牌。"""
        payload = {"option_types": {"尺码": ["S", "M", "L", "XL"]}}
        self.assertEqual(
            generation.specification_issues("Columbia宽松T恤", "title", payload), []
        )
        self.assertTrue(
            generation.specification_issues("Columbia M码T恤", "title", payload)
        )

    def test_known_waist_range_is_not_discrete_size_rewrite(self) -> None:
        """均码里的真实腰围范围不应被当作离散儿童尺码。"""
        payload = {"option_types": {"尺码": ["均码（松紧腰围66-90cm）"]}}
        self.assertEqual(
            generation.specification_issues(
                "松紧腰围66至90厘米", "description", payload
            ),
            [],
        )

    def test_sparse_product_uses_one_point_budget(self) -> None:
        """只有类型和品牌时最多一条，既不凑三条也不补造功能。"""
        sample = generation.normalize_product(
            "1", record("电池收纳盒", "品牌#:#测试品牌")
        )
        payload = generation.make_payload(sample, args())
        self.assertEqual(payload["selling_point_count"], 1)
        self.assertEqual(len(payload["selling_point_plan"]), 1)

    def test_options_add_one_topic_not_one_per_option(self) -> None:
        """颜色尺码选项只形成一个可选规格主题。"""
        sample = generation.normalize_product(
            "1",
            record("T恤", "尺码#:#M#;#尺码#:#L#;#颜色分类#:#黑色#;#颜色分类#:#白色"),
        )
        payload = generation.make_payload(sample, args())
        self.assertEqual(payload["selling_point_count"], 2)

    def test_known_apparel_size_is_not_electrical_current(self) -> None:
        """尺码整体来自选项时，104A不进入电流解析。"""
        payload = {"option_types": {"尺码": ["190/104A/XXL"]}}
        cleaned = generation.text_without_known_options("尺码190/104A/XXL可选", payload)
        self.assertEqual(generation.quantities(cleaned), [])
        self.assertIn(
            (generation.Decimal("105"), "a"),
            generation.quantities(
                generation.text_without_known_options("电流105A", payload)
            ),
        )
        self.assertIn(
            "190/105A/XXL",
            generation.text_without_known_options("190/105A/XXL", payload),
        )

    def test_color_enumeration_and_unique_color_are_distinct(self) -> None:
        """列举两种颜色不是唯一SKU，单独确定一种仍失败。"""
        sample = generation.normalize_product(
            "1", record("T恤", "颜色分类#:#军绿色#;#颜色分类#:#天蓝色")
        )
        payload = generation.make_payload(sample, args())
        output = {
            "title": "T恤",
            "description": "T恤，提供军绿色、天蓝色两种颜色。",
            "selling_points": ["两种颜色选择，便于搭配"],
        }
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertFalse(any("唯一商品颜色" in risk for risk in checks["fact_risks"]))
        output["description"] = "这款T恤为军绿色。"
        self.assertTrue(
            any(
                "唯一商品颜色" in risk
                for risk in generation.validate(
                    json.dumps(output, ensure_ascii=False), payload
                )[1]["fact_risks"]
            )
        )

    def test_installation_claim_needs_installation_evidence(self) -> None:
        """塑料和防滑功能不能证明置地式结构。"""
        self.assertTrue(
            generation.claim_issues("结构为置地式", {"材质": "塑料", "功能": "防滑"})
        )
        self.assertFalse(
            generation.claim_issues("采用置地式安装", {"安装方式": "置地式"})
        )
        self.assertTrue(
            generation.claim_issues("采用壁挂式安装", {"安装方式": "置地式"})
        )

    def test_wired_connection_is_not_proof_of_no_charging(self) -> None:
        """有线连接不能证明免充电，必须有同向供电证据。"""
        for phrase in ["无需额外充电", "不用充电", "免充电"]:
            self.assertTrue(generation.claim_issues(phrase, {"连接方式": "有线连接"}))
        self.assertFalse(
            generation.claim_issues("无需额外充电", {"供电方式": "无需充电"})
        )
        self.assertTrue(generation.claim_issues("免充电", {"功能": "不支持免充电"}))

    def test_latency_claims_do_not_follow_bluetooth_version(self) -> None:
        """无线版本不证明零延迟，明确性能证据只支持对应表述。"""
        for phrase in ["无延迟", "零延迟", "低延迟"]:
            self.assertTrue(generation.claim_issues(phrase, {"蓝牙版本": "5.0"}))
        self.assertFalse(generation.claim_issues("低延迟", {"性能": "低延迟"}))
        self.assertTrue(generation.claim_issues("零延迟", {"性能": "低延迟"}))

    def test_natural_language_counts_require_quantity_field(self) -> None:
        """单个商品和两件装需要数量，接口数量不能替代销售数量。"""
        self.assertTrue(generation.claim_issues("单个商品", {"品牌": "测试品牌"}))
        self.assertTrue(generation.claim_issues("单个商品", {"接口数量": "1个"}))
        self.assertFalse(generation.claim_issues("单个商品", {"数量": "1个"}))
        self.assertTrue(generation.claim_issues("两件装", {"数量": "1个"}))
        self.assertFalse(generation.claim_issues("两件装", {"数量": "2件"}))

    def test_normal_subjective_marketing_is_allowed(self) -> None:
        """正常舒适百搭与质感修饰保持允许，不等同效果保证。"""
        self.assertEqual(
            generation.claim_issues(
                "简约百搭，穿着舒适，呈现自然质感", {"商品类型": "T恤"}
            ),
            [],
        )

    def test_keyword_description_is_not_fluent_copy(self) -> None:
        """详情关键词串需要改为完整句子，正常叙述不误报。"""
        payload = {"facts": {"商品类型": "连衣裙"}}
        result = {
            "title": "连衣裙",
            "description": "品牌 连衣裙 纯棉 动漫 公主风 100%棉 领口袖长设计 搭配建议：日常穿着",
            "selling_points": ["连衣裙，适合日常搭配"],
        }
        self.assertTrue(
            any(
                "关键词堆砌" in issue
                for issue in generation.copywriting_issues(result, payload)
            )
        )
        result["description"] = "这款连衣裙采用纯棉面料，搭配动漫图案，适合日常穿着。"
        self.assertFalse(
            any(
                "关键词堆砌" in issue
                for issue in generation.copywriting_issues(result, payload)
            )
        )

    def test_awkward_option_phrase_has_expression_category(self) -> None:
        """表达生硬应归入表达质量，不计为事实编造。"""
        self.assertEqual(
            generation.classify_quality_issue("支持多种规格可选表达生硬，按事实省略"),
            ("表达质量", "COPY_ORGANIZATION"),
        )

    def test_claim_edit_feedback_points_to_clause(self) -> None:
        """修复反馈包含具体分句和声明原因。"""
        result = {
            "title": "耳机",
            "description": "有线耳机，无需额外充电。",
            "selling_points": ["支持有线连接"],
        }
        payload = {"facts": {"商品类型": "耳机", "连接方式": "有线连接"}}
        edits = generation.build_claim_edits(result, payload)
        self.assertTrue(any(edit["quote"] == "无需额外充电" for edit in edits))

    def test_local_grounding_deletion_preserves_known_facts(self) -> None:
        """删除无依据完整分句，已知材质和容量仍在输出中。"""
        sample = generation.normalize_product(
            "1", record("水杯", "材质#:#不锈钢#;#容量#:#0.35L")
        )
        payload = generation.make_payload(sample, args())
        result = {
            "title": "不锈钢水杯",
            "description": "不锈钢水杯，容量0.35L。",
            "selling_points": ["不锈钢材质，稳固耐用。", "0.35L容量，便于日常使用。"],
        }
        answer, changes = generation.repair_grounded_copy(
            json.dumps(result, ensure_ascii=False), payload
        )
        self.assertTrue(changes)
        self.assertNotIn("耐用", answer)
        self.assertIn("0.35L", answer)
        self.assertIn("不锈钢", answer)
        self.assertTrue(any(change["before"] != change["after"] for change in changes))

    def test_local_grounding_does_not_drop_unique_fact_in_bad_clause(self) -> None:
        """无依据承诺与独有参数粘连时不删掉唯一参数假装通过。"""
        sample = generation.normalize_product("1", record("水杯", "容量#:#0.35L"))
        payload = generation.make_payload(sample, args())
        result = {
            "title": "水杯",
            "description": "这款水杯适合日常使用。",
            "selling_points": ["稳固耐用的0.35L水杯", "日常使用方便"],
        }
        original = json.dumps(result, ensure_ascii=False)
        self.assertEqual(
            generation.repair_grounded_copy(original, payload), (original, [])
        )

    def test_local_grounding_respects_partial_repair_targets(self) -> None:
        """只修详情时不得暗改卖点；没有安全替代内容时仍失败。"""
        sample = generation.normalize_product("1", record("耳机", ""))
        payload = generation.make_payload(sample, args())
        result = {
            "title": "耳机",
            "description": "耳机，适合听音。",
            "selling_points": ["无延迟"],
        }
        original = json.dumps(result, ensure_ascii=False)
        answer, changes = generation.repair_grounded_copy(
            original, payload, ["description"]
        )
        self.assertEqual(json.loads(answer)["selling_points"], ["无延迟"])
        self.assertFalse(generation.validate(answer, payload)[1]["quality_gate_pass"])

    def test_sparse_local_budget_converges_without_fake_features(self) -> None:
        """稀疏商品只保留一条用途，保留前后修改轨迹。"""
        sample = generation.normalize_product(
            "1", record("电池收纳盒", "品牌#:#测试品牌")
        )
        payload = generation.make_payload(sample, args())
        result = {
            "title": "测试品牌电池收纳盒",
            "description": "测试品牌电池收纳盒，用于整理电池。",
            "selling_points": ["用于电池收纳", "帮助整理电池", "集中存放电池"],
        }
        answer, changes = generation.repair_grounded_copy(
            json.dumps(result, ensure_ascii=False), payload
        )
        self.assertEqual(len(json.loads(answer)["selling_points"]), 1)
        self.assertTrue(changes)
        self.assertIn("测试品牌", answer)

    def test_numeric_sale_quantity_cannot_borrow_model_number(self) -> None:
        """销售数量需要数量字段，不能借用型号中相同数字绕过检查。"""
        facts = {"型号": "A1", "商品类型": "收纳盒"}
        self.assertTrue(generation.claim_issues("数量为1个", facts))
        self.assertTrue(generation.claim_issues("1个装", facts))
        self.assertFalse(generation.claim_issues("数量为1个", {"数量": "1个"}))
        self.assertFalse(generation.claim_issues("2个接口", {"接口数量": "2个"}))

    def test_single_point_budget_does_not_excuse_parameter_only_copy(self) -> None:
        """只写一条时仍需正常表达，不能只剩材质参数清单。"""
        payload = {"facts": {"商品类型": "收纳箱", "材质": "塑料"}}
        result = {
            "title": "塑料收纳箱",
            "description": "塑料收纳箱用于整理日常物品。",
            "selling_points": ["材质为塑料。"],
        }
        self.assertTrue(
            any(
                "参数复述" in reason
                for reason in generation.copywriting_issues(result, payload)
            )
        )
        result["selling_points"] = ["塑料材质收纳箱，便于整理日常物品。"]
        self.assertFalse(
            any(
                "参数复述" in reason
                for reason in generation.copywriting_issues(result, payload)
            )
        )

    def test_partial_repair_preserves_function_only_present_in_points(self) -> None:
        """卖点独有的正确功能必须明确告知模型保留，不能只反馈删除话术。"""
        result = {
            "title": "T恤",
            "description": "棉涤T恤，宽松设计。",
            "selling_points": ["功能：透气", "材质：棉涤"],
        }
        payload = {"facts": {"商品类型": "T恤", "材质": "棉涤", "功能": "透气"}}
        preserved = generation.repair_preservation_facts(
            result, payload, ["selling_points"]
        )
        self.assertEqual(preserved["selling_points"], {"功能": "透气"})

    def test_dimensions_compare_states_units_and_approximation(self) -> None:
        """普通和拉伸后尺寸分别匹配，不依赖原文中的约为写法。"""
        value = "约3*6.5cm，拉伸后约6*7cm"
        self.assertTrue(
            generation.fact_covered(
                "尺寸", value, "尺寸约为30×65毫米，拉伸后约60×70毫米。"
            )
        )
        for output in [
            "尺寸约为3*6.5cm",
            "约6*7cm，拉伸后约3*6.5cm",
            "尺寸3*6.5cm，拉伸后6*7cm",
            "约3*6.5cm，拉伸后约6*8cm",
        ]:
            self.assertFalse(generation.fact_covered("尺寸", value, output))

    def test_dimension_coordinate_roles_and_unit_expansion(self) -> None:
        """末尾共用单位与逐坐标单位等价，直径和高度顺序保持。"""
        self.assertTrue(
            generation.fact_covered("尺寸", "直径46*高55mm", "直径46毫米，高55毫米。")
        )
        self.assertTrue(generation.fact_covered("尺寸", "3cm*65mm", "30mm*6.5cm"))
        self.assertFalse(generation.fact_covered("尺寸", "3*6.5cm", "6.5*3cm"))

    def test_dimension_each_claim_preserves_approximate_qualifier(self) -> None:
        """另一句正确的约数不能掩盖本句改成精确数值。"""
        value = "约3*6.5cm"
        self.assertTrue(generation.dimension_claim_issues("尺寸3*6.5cm", value))
        self.assertFalse(generation.dimension_claim_issues("尺寸约为30*65mm", value))
        self.assertTrue(generation.dimension_claim_issues("尺寸约3*7cm", value))

    def test_strict_and_inclusive_parameter_bounds_are_not_equal(self) -> None:
        """以下含边界，低于为严格小于，不能混同。"""
        self.assertNotEqual(
            generation.bound_sign("100g以下"), generation.bound_sign("低于100g")
        )
        self.assertTrue(
            generation.fact_covered("充绒量", "100g以下", "充绒量不超过100g")
        )
        self.assertFalse(
            generation.fact_covered("充绒量", "100g以下", "充绒量低于100g")
        )
        self.assertTrue(
            generation.bounded_quantity_issues("充绒量低于100g", "100g以下")
        )

    def test_intact_usage_sentence_not_misread_as_keyword_list(self) -> None:
        """带空格数字单位与用途谓语的正常介绍不误判。"""
        sentence = "MUXIN CAMP 不锈钢水杯，直径 46 毫米，高 55 毫米，容量 0.6 升，适用于成人日常饮水。"
        self.assertFalse(generation.description_is_keyword_list(sentence))
        self.assertTrue(
            generation.description_is_keyword_list(
                "品牌 水杯 不锈钢 直径46 高55 0.6L 成人 日常饮水"
            )
        )

    def test_product_name_is_not_storage_action(self) -> None:
        """收纳盒加规格选择不能跨分句误匹配为多规格收纳能力。"""
        for text in [
            "收纳盒，支持多种规格选择",
            "收纳盒支持多种规格选择",
            "收纳箱。多种规格可选",
        ]:
            self.assertFalse(
                any(
                    "不同规格收纳能力" in item
                    for item in generation.claim_issues(text, {"商品类型": "收纳盒"})
                )
            )
        self.assertTrue(
            any(
                "不同规格收纳能力" in item
                for item in generation.claim_issues(
                    "可以收纳不同规格的电池", {"商品类型": "电池收纳盒"}
                )
            )
        )

    def test_ergonomics_is_not_fatigue_effect_evidence(self) -> None:
        """人体工学设计不证明缓解疲劳，健康效果必须有对应声明证据。"""
        phrase = "有助于缓解长时间使用带来的疲劳"
        self.assertTrue(generation.claim_issues(phrase, {"人体工学": "支持"}))
        self.assertFalse(generation.claim_issues(phrase, {"功能": phrase}))
        self.assertTrue(generation.claim_issues("消除疲劳", {"功能": "缓解疲劳"}))

    def test_multiple_interfaces_do_not_prove_seamless_connection(self) -> None:
        """接口存在只能证明连接选项，无缝切换等效果需单独证据。"""
        self.assertTrue(
            generation.claim_issues("多设备无缝连接", {"接口": "Lightning、OTG、USB"})
        )
        self.assertFalse(
            generation.claim_issues("多设备无缝连接", {"连接能力": "无缝连接"})
        )

    def test_effect_degree_cannot_borrow_an_unrelated_effect(self) -> None:
        """显著减少噪音需要同向程度与效果依据。"""
        self.assertTrue(
            generation.claim_issues("显著减少环境噪音", {"功能": "主动降噪"})
        )
        self.assertFalse(
            generation.claim_issues(
                "显著减少环境噪音", {"检测结果": "显著减少环境噪音"}
            )
        )
        self.assertTrue(
            generation.claim_issues("显著降低噪音", {"检测结果": "显著降低延迟"})
        )

    def test_repeated_pure_material_is_not_a_complete_selling_point(self) -> None:
        """纯棉前加100%不能掩盖单条中重复材质的说明书表达。"""
        payload = {
            "facts": {"商品类型": "连衣裙", "材质": "纯棉", "面料成分": "棉100%"}
        }
        result = {
            "title": "纯棉连衣裙",
            "description": "这款连衣裙采用纯棉面料。",
            "selling_points": ["100%纯棉面料，材质为纯棉。"],
        }
        self.assertTrue(
            any(
                "重复" in item or "参数复述" in item
                for item in generation.copywriting_issues(result, payload)
            )
        )
        result["selling_points"] = ["纯棉面料，适合日常穿搭。"]
        self.assertFalse(
            any(
                "重复" in item or "参数复述" in item
                for item in generation.copywriting_issues(result, payload)
            )
        )

    def test_new_effect_claims_generate_actionable_repair_feedback(self) -> None:
        """新增效果声明也能定位分句，正常事实和用途不被全删。"""
        sample = generation.normalize_product(
            "1", record("键盘", "接口类型#:#USB#;#是否支持人体工程学#:#支持")
        )
        payload = generation.make_payload(sample, args())
        result = {
            "title": "USB键盘",
            "description": "键盘配备USB接口，支持人体工学设计。",
            "selling_points": [
                "支持人体工学设计，缓解长时间使用疲劳。",
                "USB接口键盘，方便连接输入设备。",
            ],
        }
        edits = generation.build_claim_edits(result, payload)
        self.assertTrue(any("疲劳" in edit["quote"] for edit in edits))
        changed, trace = generation.repair_grounded_copy(
            json.dumps(result, ensure_ascii=False), payload
        )
        self.assertTrue(trace)
        self.assertNotIn("疲劳", changed)
        self.assertIn("人体工学", changed)

    def test_negative_comparative_words_preserve_inclusive_bounds(self) -> None:
        """不多于与不少于是含边界的限定，不能误识别为严格比较。"""
        self.assertEqual(
            generation.bound_sign("不多于100g"), generation.bound_sign("100g以下")
        )
        self.assertEqual(
            generation.bound_sign("不少于100g"), generation.bound_sign("100g以上")
        )
        self.assertNotEqual(
            generation.bound_sign("多于100g"), generation.bound_sign("100g以上")
        )
        self.assertFalse(generation.bounded_quantity_issues("不多于100g", "100g以下"))
        self.assertFalse(generation.bounded_quantity_issues("不少于100g", "100g以上"))

    def test_unsupported_dimension_units_do_not_allow_partial_coverage(self) -> None:
        """混合未知单位时不能只命中已解析坐标便宣称完整覆盖。"""
        self.assertFalse(generation.fact_covered("尺寸", "2cm*3英寸", "尺寸2cm"))
        self.assertTrue(generation.fact_covered("尺寸", "2cm*3英寸", "尺寸2cm*3英寸"))

    def test_attribute_review_no_confirmation_keeps_conflict(self) -> None:
        """空核验和待填模板都不能解除真实矛盾。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        template = generation.attribute_review_template([sample])
        for reviews in [{}, template]:
            after = generation.apply_attribute_review(sample, reviews)
            self.assertEqual(len(generation.unresolved_attribute_issues(after)), 2)
            self.assertNotIn("材质", after["facts"])
            self.assertTrue(generation.make_payload(after, args())["source_conflicts"])

    def test_confirmed_fields_restore_facts_tags_and_leave_raw_unchanged(self) -> None:
        """有依据的字段恢复后刷新标签，历史冲突与原样本仍完整保留。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        original = json.dumps(sample, ensure_ascii=False, sort_keys=True)
        reviews = {
            "1": {
                "fields": {
                    "材质": {
                        "status": "confirmed",
                        "value": "纯棉",
                        "source": "测试夹具规格表",
                    },
                    "面料成分": {
                        "status": "confirmed",
                        "value": "棉100%",
                        "source": "测试夹具规格表",
                    },
                }
            }
        }
        after = generation.apply_attribute_review(sample, reviews)
        self.assertFalse(generation.unresolved_attribute_issues(after))
        self.assertEqual(after["facts"]["材质"], "纯棉")
        self.assertEqual(after["fashion_tags"]["面料成分"]["status"], "已提供")
        self.assertEqual(
            after["fashion_tags"]["面料成分"]["source"]["source"], "人工属性核验"
        )
        self.assertIn("面料成分", after["core_fields"])
        self.assertTrue(after["conflicts"])
        self.assertEqual(len(after["attribute_review_log"]), 2)
        self.assertEqual(
            json.dumps(sample, ensure_ascii=False, sort_keys=True), original
        )
        self.assertNotIn("source_conflicts", generation.make_payload(after, args()))

    def test_partial_field_confirmation_does_not_resolve_other_fields(self) -> None:
        """只确认材质不能顺带解除成分矛盾。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        reviews = {
            "1": {
                "fields": {
                    "材质": {
                        "status": "confirmed",
                        "value": "纯棉",
                        "source": "测试夹具",
                    }
                }
            }
        }
        after = generation.apply_attribute_review(sample, reviews)
        self.assertEqual(
            [issue["field"] for issue in generation.unresolved_attribute_issues(after)],
            ["面料成分"],
        )
        self.assertNotIn("面料成分", after["facts"])

    def test_confirmed_value_requires_nonempty_evidence(self) -> None:
        """不能用空确认值或空来源解除冲突。"""
        sample = generation.normalize_product("1", record("T恤", "材质#:#棉"))
        for value, source in [(" ", "测试夹具"), ("棉", ""), ("棉", "  ")]:
            with self.assertRaises(ValueError):
                generation.apply_attribute_review(
                    sample,
                    {
                        "1": {
                            "fields": {
                                "材质": {
                                    "status": "confirmed",
                                    "value": value,
                                    "source": source,
                                }
                            }
                        }
                    },
                )

    def test_review_product_and_field_mismatch_are_rejected(self) -> None:
        """错误商品ID和拼错的属性不静默忽略。"""
        sample = generation.normalize_product("1", record("T恤", "材质#:#棉"))
        item = {
            "fields": {
                "材质": {"status": "confirmed", "value": "棉", "source": "测试夹具"}
            }
        }
        with self.assertRaises(ValueError):
            generation.apply_attribute_reviews([sample], {"2": item})
        with self.assertRaises(ValueError):
            generation.apply_attribute_review(
                sample, {"1": {**item, "product_id": "2"}}
            )
        with self.assertRaises(ValueError):
            generation.apply_attribute_review(
                sample,
                {
                    "1": {
                        "fields": {
                            "面科": {
                                "status": "confirmed",
                                "value": "棉",
                                "source": "测试夹具",
                            }
                        }
                    }
                },
            )

    def test_sku_dependent_interface_requires_explicit_sku(self) -> None:
        """SKU接口不能仅靠全局字段确认解除。"""
        sample = generation.normalize_product(
            "1",
            record(
                "键盘",
                "接口类型#:#USB#;#接口类型#:#蓝牙#;#颜色分类#:#【USB】黑色#;#颜色分类#:#【双模】白色",
            ),
        )
        self.assertEqual(
            generation.unresolved_attribute_issues(sample)[0]["kind"], "sku_dependent"
        )
        with self.assertRaises(ValueError):
            generation.apply_attribute_review(
                sample,
                {
                    "1": {
                        "fields": {
                            "接口": {
                                "status": "confirmed",
                                "value": "USB",
                                "source": "测试夹具",
                            }
                        }
                    }
                },
            )

    def test_explicit_verified_sku_restores_only_its_attributes(self) -> None:
        """明确SKU和证据才能恢复接口，原冲突不能再次清除已确认事实。"""
        sample = generation.normalize_product(
            "1",
            record(
                "键盘",
                "接口类型#:#USB#;#接口类型#:#蓝牙#;#颜色分类#:#【USB】黑色#;#颜色分类#:#【双模】白色",
            ),
        )
        reviews = {
            "1": {
                "selected_sku": {
                    "sku_id": "测试USB款",
                    "source": "测试夹具SKU表",
                    "attributes": {"接口": "USB"},
                }
            }
        }
        after = generation.apply_attribute_review(sample, reviews)
        self.assertEqual(after["facts"]["接口"], "USB")
        self.assertTrue(after["conflicts"])
        self.assertFalse(generation.unresolved_attribute_issues(after))
        payload = generation.make_payload(after, args())
        sent = generation.generation_payload(
            payload, generation.unresolved_attribute_issues(after)
        )
        self.assertEqual(sent["facts"]["接口"], "USB")
        self.assertEqual(sent["sku_context"]["sku_id"], "测试USB款")
        self.assertNotIn("source_conflicts", payload)
        self.assertNotIn("测试夹具SKU表", json.dumps(sent, ensure_ascii=False))

    def test_sku_cannot_invent_values_or_combinations(self) -> None:
        """SKU候选值不能任意补造，来源为空也不可用。"""
        sample = generation.normalize_product(
            "1", record("T恤", "颜色分类#:#黑色#;#颜色分类#:#白色")
        )
        for sku in [
            {"sku_id": "测试款", "source": "测试夹具", "attributes": {"颜色": "红色"}},
            {"sku_id": "测试款", "source": "", "attributes": {"颜色": "黑色"}},
        ]:
            with self.assertRaises(ValueError):
                generation.apply_attribute_review(sample, {"1": {"selected_sku": sku}})

    def test_mixed_color_labels_can_be_manually_reclassified(self) -> None:
        """保留原始款式标签，不推断颜色或任意组合，人工确认后解除字段歧义。"""
        sample = generation.normalize_product(
            "1", record("夹克", "颜色#:#休闲男#;#颜色#:#极地女")
        )
        entry = {
            "fields": {
                "颜色": {
                    "status": "reclassified",
                    "target_field": "款式",
                    "value": ["休闲男", "极地女"],
                    "source": "测试夹具字段说明",
                }
            }
        }
        after = generation.apply_attribute_review(sample, {"1": entry})
        self.assertNotIn("颜色", after["facts"])
        self.assertEqual(after["options"]["款式"], ["休闲男", "极地女"])
        self.assertFalse(generation.unresolved_attribute_issues(after))
        self.assertEqual(
            generation.make_payload(after, args())["option_types"]["款式"],
            ["休闲男", "极地女"],
        )
        self.assertTrue(after["conflicts"])

    def test_reclassification_cannot_drop_or_add_labels(self) -> None:
        """字段重分类必须保留完整候选标签，不改变原始SKU组合。"""
        sample = generation.normalize_product(
            "1", record("夹克", "颜色#:#休闲男#;#颜色#:#极地女")
        )
        for values in [["休闲男"], ["休闲男", "极地女", "极地男"]]:
            with self.assertRaises(ValueError):
                generation.apply_attribute_review(
                    sample,
                    {
                        "1": {
                            "fields": {
                                "颜色": {
                                    "status": "reclassified",
                                    "target_field": "款式",
                                    "value": values,
                                    "source": "测试夹具",
                                }
                            }
                        }
                    },
                )

    def test_inconsistent_material_review_is_rejected(self) -> None:
        """核验值本身仍然矛盾时不能解除状态。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        with self.assertRaises(ValueError):
            generation.apply_attribute_review(
                sample,
                {
                    "1": {
                        "fields": {
                            "材质": {
                                "status": "confirmed",
                                "value": "棉混纺布",
                                "source": "测试夹具",
                            },
                            "面料成分": {
                                "status": "confirmed",
                                "value": "棉100%",
                                "source": "测试夹具",
                            },
                        }
                    }
                },
            )

    def test_historical_conflicts_no_longer_block_confirmed_payload(self) -> None:
        """确认后历史冲突留档，但当前文案检查不再永久待核验。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        review = {
            "1": {
                "fields": {
                    "材质": {
                        "status": "confirmed",
                        "value": "纯棉",
                        "source": "测试夹具",
                    },
                    "面料成分": {
                        "status": "confirmed",
                        "value": "棉100%",
                        "source": "测试夹具",
                    },
                }
            }
        }
        confirmed = generation.apply_attribute_review(sample, review)
        output = {
            "title": "纯棉衬衫",
            "description": "这款衬衫采用100%棉面料，适合日常穿着。",
            "selling_points": ["纯棉面料，适合日常穿搭。"],
        }
        checks = generation.validate(
            json.dumps(output, ensure_ascii=False),
            generation.make_payload(confirmed, args()),
        )[1]
        self.assertFalse(checks["input_review_required"])
        self.assertTrue(checks["quality_gate_pass"])
        self.assertTrue(confirmed["conflicts"])

    def test_legacy_conflicts_cannot_be_cleared_by_empty_issue_list(self) -> None:
        """旧格式仍保守处理，空列表不能跳过历史未解决字段。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        sample["attribute_issues"] = []
        self.assertTrue(generation.unresolved_attribute_issues(sample))

    def test_review_json_rejects_duplicate_keys(self) -> None:
        """重复商品ID或属性键不能被JSON解析静默覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text('{"1":{},"1":{}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                generation.load_attribute_reviews(path)

    def test_review_template_is_pending_and_loadable(self) -> None:
        """导出的是待填模板，不包含假确认值。"""
        sample = generation.normalize_product(
            "1", record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.json"
            generation.write_json(path, generation.attribute_review_template([sample]))
            reviews = generation.load_attribute_reviews(path)
            self.assertTrue(
                all(
                    item["status"] == "pending" and item["value"] == ""
                    for item in reviews["1"]["fields"].values()
                )
            )
            self.assertTrue(
                generation.unresolved_attribute_issues(
                    generation.apply_attribute_review(sample, reviews)
                )
            )

    def test_attribute_review_dry_run_records_source_and_current_status(self) -> None:
        """新运行记录核验文件指纹、事实、审计及当前状态，源文件不变。"""
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "raw.json"
            review_path = folder / "attributes.json"
            generation.write_json(
                source, {"1": record("衬衫", "面料#:#棉混纺布#;#材质成分#:#棉100%")}
            )
            original = source.read_bytes()
            generation.write_json(
                review_path,
                {
                    "1": {
                        "fields": {
                            "材质": {
                                "status": "confirmed",
                                "value": "纯棉",
                                "source": "测试夹具",
                            },
                            "面料成分": {
                                "status": "confirmed",
                                "value": "棉100%",
                                "source": "测试夹具",
                            },
                        }
                    }
                },
            )
            a = args()
            a.samples = source
            a.limit = 0
            a.attribute_review = review_path
            a.run_dir = folder / "run"
            a.dry_run = True
            a.model_dir = folder / "no_model"
            a.repair_attempts = 1
            generation.run(a)
            manifest = generation.read_json(a.run_dir / "manifest.json")
            self.assertEqual(
                manifest["attribute_review"]["sha256"],
                hashlib.sha256(review_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["samples"][0]["facts"]["材质"], "纯棉")
            self.assertNotIn("source_conflicts", manifest["payloads"][0])
            audit = generation.read_json(a.run_dir / "attribute_audit.json")
            self.assertEqual(audit["unresolved_products"], 0)
            self.assertEqual(audit["reviewed_products"], 1)
            self.assertEqual(
                generation.read_json(a.run_dir / "summary.json")["overall"][
                    "input_attribute_review"
                ]["historical_conflict_products"],
                1,
            )
            self.assertEqual(source.read_bytes(), original)

    def test_review_and_resume_are_incompatible(self) -> None:
        """已保存的运行不能在断点续跑时悄悄换事实。"""
        a = args()
        a.resume_dir = Path("not_read")
        a.attribute_review = Path("not_read")
        with self.assertRaisesRegex(ValueError, "resume"):
            generation.run(a)

    def test_atomic_batch_review_does_not_mutate_earlier_samples(self) -> None:
        """后面一条核验错误时，前面样本也不应被就地修改。"""
        samples = [
            generation.normalize_product("1", record("T恤", "材质#:#棉")),
            generation.normalize_product("2", record("T恤", "材质#:#棉")),
        ]
        before = json.dumps(samples, ensure_ascii=False, sort_keys=True)
        reviews = {
            "1": {
                "fields": {
                    "材质": {
                        "status": "confirmed",
                        "value": "纯棉",
                        "source": "测试夹具",
                    }
                }
            },
            "2": {
                "fields": {
                    "错误属性": {
                        "status": "confirmed",
                        "value": "棉",
                        "source": "测试夹具",
                    }
                }
            },
        }
        with self.assertRaises(ValueError):
            generation.apply_attribute_reviews(samples, reviews)
        self.assertEqual(
            json.dumps(samples, ensure_ascii=False, sort_keys=True), before
        )

    def test_reapplying_same_review_does_not_duplicate_history(self) -> None:
        """prepare后再次run应用相同核验文件不应重复嵌套证据和日志。"""
        sample = generation.normalize_product("1", record("T恤", "材质#:#棉"))
        reviews = {
            "1": {
                "fields": {
                    "材质": {
                        "status": "confirmed",
                        "value": "纯棉",
                        "source": "测试夹具",
                    }
                }
            }
        }
        once = generation.apply_attribute_review(sample, reviews)
        twice = generation.apply_attribute_review(once, reviews)
        self.assertEqual(once, twice)

    def test_sku_and_reclassification_cannot_conflict_on_same_field(self) -> None:
        """同一字段不能既改成款式选项，又继续宣称为颜色。"""
        sample = generation.normalize_product(
            "1", record("夹克", "颜色#:#休闲男#;#颜色#:#极地女")
        )
        reviews = {
            "1": {
                "fields": {
                    "颜色": {
                        "status": "reclassified",
                        "value": ["休闲男", "极地女"],
                        "target_field": "款式",
                        "source": "测试夹具",
                    }
                },
                "selected_sku": {
                    "sku_id": "测试款",
                    "source": "测试夹具",
                    "attributes": {"颜色": "休闲男"},
                },
            }
        }
        with self.assertRaises(ValueError):
            generation.apply_attribute_review(sample, reviews)

    def test_normalized_sku_confirmation_requires_sku_scope(self) -> None:
        """持久化样本中的已确认SKU字段也必须有匹配范围，不能只改状态跳过。"""
        sample = generation.normalize_product(
            "1",
            record(
                "键盘",
                "接口类型#:#USB#;#接口类型#:#蓝牙#;#颜色分类#:#【USB】黑色#;#颜色分类#:#【双模】白色",
            ),
        )
        sample["facts"]["接口"] = "USB"
        sample["attribute_issues"][0]["status"] = "confirmed"
        sample["attribute_issues"][0]["resolution"] = {
            "value": "USB",
            "source": "测试夹具",
            "scope": "product",
        }
        with self.assertRaises(ValueError):
            generation.unresolved_attribute_issues(sample)

    def test_volume_review_must_correct_or_explain_metric_scope(self) -> None:
        """复写相同矛盾数值不能解除；折叠尺寸与套装口径必须明确写出。"""
        sample = generation.normalize_product(
            "1", record("收纳箱", "尺寸#:#32cmx22cmx16cm#;#容量#:#50L")
        )
        fields = {
            "尺寸": {
                "status": "confirmed",
                "value": "32cmx22cmx16cm",
                "source": "测试夹具",
            },
            "容量": {"status": "confirmed", "value": "50L", "source": "测试夹具"},
        }
        with self.assertRaisesRegex(ValueError, "尺寸容量"):
            generation.apply_attribute_review(sample, {"1": {"fields": fields}})
        fields["容量"]["value"] = "10L"
        corrected = generation.apply_attribute_review(sample, {"1": {"fields": fields}})
        self.assertFalse(generation.unresolved_attribute_issues(corrected))
        fields["容量"]["value"] = "50L"
        fields["尺寸"]["value"] = "折叠后32cmx22cmx16cm"
        scoped = generation.apply_attribute_review(sample, {"1": {"fields": fields}})
        self.assertFalse(generation.unresolved_attribute_issues(scoped))
        self.assertIn("折叠后", scoped["facts"]["尺寸"])

    def test_stacked_title_is_repaired_when_attributes_exist_in_body(self) -> None:
        """属性堆砌标题压缩后保留品牌和身份，其他属性在详情中保留。"""
        sample = generation.normalize_product(
            "1",
            record(
                "衬衫",
                "品牌#:#Princeyan#;#材质#:#棉#;#领型#:#翻领尖领#;#袖长#:#长袖#;#图案#:#格子#;#风格#:#学院",
            ),
        )
        options = args()
        options.title_max = 80
        payload = generation.make_payload(sample, options)
        output = {
            "title": "Princeyan 衬衫 棉 翻领尖领 长袖 格子 学院风",
            "description": "Princeyan棉衬衫采用翻领尖领长袖设计，搭配格子图案，呈现学院风格。",
            "selling_points": ["格子图案搭配长袖设计，适合日常穿搭。"],
        }
        self.assertTrue(
            generation.title_expression_issues(output["title"], sample["facts"])
        )
        answer, trace = generation.repair_title(
            json.dumps(output, ensure_ascii=False), payload
        )
        after = json.loads(answer)
        self.assertTrue(trace)
        self.assertIn("Princeyan", after["title"])
        self.assertIn("衬衫", after["title"])
        self.assertLess(len(after["title"]), len(output["title"]))
        self.assertEqual(after["description"], output["description"])
        self.assertFalse(
            generation.title_expression_issues(after["title"], sample["facts"])
        )

    def test_long_identity_and_two_features_are_not_title_stacking(self) -> None:
        """品牌型号本身的空格和两项必要特征不当作属性词串。"""
        self.assertFalse(
            generation.title_expression_issues(
                "Thom Browne 深蓝色直筒休闲裤",
                {
                    "品牌": "Thom Browne",
                    "商品类型": "休闲裤",
                    "颜色": "深蓝色",
                    "版型": "直筒",
                },
            )
        )
        self.assertFalse(
            generation.title_expression_issues(
                "Brand Model X Pro 2026 显示器 32英寸",
                {
                    "品牌": "Brand",
                    "型号": "Model X Pro 2026",
                    "商品类型": "显示器",
                    "尺寸": "32英寸",
                },
            )
        )

    def test_title_compression_cannot_drop_unique_attributes(self) -> None:
        """其他字段没有的标题事实不能在本地压缩时丢掉。"""
        sample = generation.normalize_product(
            "1",
            record(
                "衬衫",
                "品牌#:#Princeyan#;#材质#:#棉#;#领型#:#翻领尖领#;#袖长#:#长袖#;#图案#:#格子#;#风格#:#学院",
            ),
        )
        options = args()
        options.title_max = 80
        payload = generation.make_payload(sample, options)
        output = {
            "title": "Princeyan 衬衫 棉 翻领尖领 长袖 格子 学院风",
            "description": "这款衬衫适合日常穿着。",
            "selling_points": ["衬衫适合日常搭配。"],
        }
        original = json.dumps(output, ensure_ascii=False)
        self.assertEqual(generation.repair_title(original, payload), (original, []))
        checks = generation.validate(original, payload)[1]
        self.assertIn("title", checks["field_issues"])
        request = generation.build_repair_request(payload, original, checks)
        self.assertIn("description", request["repair"]["target_fields"])
        self.assertIn("领型", request["repair"]["title_facts_to_move_to_description"])
        self.assertEqual(
            request["repair"]["title_facts_to_move_to_description"]["领型"], "翻领尖领"
        )

    def test_title_expression_repair_respects_partial_scope(self) -> None:
        """局部只修卖点时不得额外重组标题。"""
        payload = {
            "facts": {
                "品牌": "P",
                "商品类型": "衬衫",
                "材质": "棉",
                "领型": "翻领",
                "袖长": "长袖",
                "图案": "格子",
            },
            "length_limits": {"title": 80},
        }
        output = {
            "title": "P 衬衫 棉 翻领 长袖 格子",
            "description": "棉质格子长袖翻领衬衫。",
            "selling_points": ["长袖设计适合日常穿搭。"],
        }
        original = json.dumps(output, ensure_ascii=False)
        self.assertEqual(
            generation.repair_title(original, payload, ["selling_points"]),
            (original, []),
        )

    def test_whole_description_repetition_is_expression_issue(self) -> None:
        """移除品牌套词的同一句仍是重复，诊断应指向卖点。"""
        sample = generation.normalize_product(
            "1", record("车载充电器", "品牌#:#YX#;#USB接口数量#:#2个")
        )
        payload = generation.make_payload(sample, args())
        output = {
            "title": "YX车载充电器，2个接口",
            "description": "YX品牌车载充电器提供2个接口。",
            "selling_points": ["YX车载充电器提供2个接口"],
        }
        checks = generation.validate(json.dumps(output, ensure_ascii=False), payload)[1]
        self.assertTrue(
            any("直接复述" in issue for issue in checks["copywriting_issues"])
        )
        self.assertTrue(
            any(
                "直接复述" in issue
                for issue in checks["field_issues"]["selling_points"]
            )
        )
        self.assertFalse(checks["quality_gate_pass"])

    def test_necessary_attribute_overlap_is_not_whole_copy(self) -> None:
        """详情和卖点可以复用关键参数，只要各自承担介绍和提炼作用。"""
        output = {
            "title": "YX双接口车载充电器",
            "description": "YX车载充电器配备2个接口，用于车内设备充电。",
            "selling_points": ["双接口配置，便于按需安排充电连接。"],
        }
        self.assertFalse(generation.cross_field_repetition(output))

    def test_point_equal_to_title_is_not_a_selling_point(self) -> None:
        """标题不能直接复制成唯一核心卖点。"""
        output = {
            "title": "YX车载充电器，2个接口",
            "description": "这款YX车载充电器提供2个接口，用于车内设备充电。",
            "selling_points": ["YX车载充电器 2个接口"],
        }
        self.assertTrue(generation.cross_field_repetition(output))

    def test_intended_user_does_not_prove_special_design(self) -> None:
        """适用对象和音乐播放功能不等于专门设计的依据。"""
        self.assertTrue(
            generation.claim_issues(
                "这款水杯专为成人设计", {"商品类型": "水杯", "用途": "成人"}
            )
        )
        self.assertTrue(
            generation.claim_issues(
                "该耳机专为音乐聆听设计", {"商品类型": "耳机", "功能": "支持音乐"}
            )
        )
        self.assertFalse(
            generation.claim_issues(
                "这款水杯适用于成人日常饮水", {"商品类型": "水杯", "用途": "成人"}
            )
        )
        self.assertFalse(
            generation.claim_issues("这款水杯专为成人设计", {"设计对象": "成人"})
        )

    def test_applicability_rewrite_is_grounded_and_traceable(self) -> None:
        """成人适用属性只能改为适用于成人，保留字段和修改记录。"""
        payload = {"facts": {"商品类型": "水杯", "用途": "成人", "材质": "不锈钢"}}
        output = {
            "title": "不锈钢水杯",
            "description": "这款不锈钢水杯专为成人设计。",
            "selling_points": ["不锈钢水杯适合日常饮水。"],
        }
        answer, trace = generation.repair_applicability_wording(
            json.dumps(output, ensure_ascii=False), payload, []
        )
        after = json.loads(answer)
        self.assertEqual(after["description"], "这款不锈钢水杯适用于成人。")
        self.assertEqual(after["selling_points"], output["selling_points"])
        self.assertEqual(trace[0]["before"], output["description"])
        self.assertEqual(trace[0]["after"], after["description"])
        self.assertFalse(
            generation.claim_issues(after["description"], payload["facts"])
        )

    def test_applicability_rewrite_does_not_infer_target_or_escape_scope(self) -> None:
        """没有对应适用属性不猜目标，局部修复也不改其他字段。"""
        output = {
            "title": "水杯",
            "description": "水杯专为成人设计。",
            "selling_points": ["水杯适合日常饮水。"],
        }
        original = json.dumps(output, ensure_ascii=False)
        self.assertEqual(
            generation.repair_applicability_wording(
                original, {"facts": {"商品类型": "水杯"}}, []
            ),
            (original, []),
        )
        self.assertEqual(
            generation.repair_applicability_wording(
                original,
                {"facts": {"商品类型": "水杯", "用途": "成人"}},
                ["selling_points"],
            ),
            (original, []),
        )

    def test_adjectives_cannot_hide_parameter_fragments(self) -> None:
        """加舒适实用等词不能让无结构的属性清单自动通过。"""
        facts = {"商品类型": "衬衫", "材质": "棉"}
        for phrase in ["舒适实用百搭", "材质为棉，舒适实用", "棉 面料 舒适 实用"]:
            self.assertTrue(generation.point_is_bare(phrase, facts))
        self.assertFalse(
            generation.point_is_bare("棉质面料，亲肤舒适，适合日常穿搭。", facts)
        )

    def test_after_claim_deletion_bare_point_still_fails(self) -> None:
        """删掉耐用承诺后，若只剩参数，也必须继续判表达不合格。"""
        sample = generation.normalize_product("1", record("收纳箱", "材质#:#塑料"))
        payload = generation.make_payload(sample, args())
        output = {
            "title": "塑料收纳箱",
            "description": "塑料收纳箱用于整理物品。",
            "selling_points": ["材质为塑料，稳固耐用。"],
        }
        answer, trace = generation.repair_grounded_copy(
            json.dumps(output, ensure_ascii=False), payload
        )
        checks = generation.validate(answer, payload)[1]
        self.assertTrue(trace)
        self.assertNotIn("耐用", answer)
        self.assertFalse(checks["copywriting_rule_pass"])
        self.assertFalse(checks["quality_gate_pass"])

    def test_negated_design_evidence_does_not_support_positive_claim(self) -> None:
        """否定设计说明不能因包含相同短语而证明肯定宣传。"""
        self.assertTrue(
            generation.design_claim_issues(
                "专为成人设计", {"设计说明": "并非专为成人设计"}
            )
        )
        self.assertFalse(
            generation.design_claim_issues("专为成人设计", {"设计说明": "专为成人设计"})
        )

    def test_generic_use_only_point_needs_a_known_feature(self) -> None:
        """用途可作为连接表达，但不应独自充当无商品特征的卖点。"""
        payload = {"facts": {"商品类型": "衬衫", "袖长": "长袖", "图案": "格子"}}
        result = {
            "title": "格子长袖衬衫",
            "description": "格子图案搭配长袖设计，适合日常穿搭。",
            "selling_points": ["适用于日常通勤或休闲场合。"],
        }
        self.assertTrue(
            any(
                "泛化用途" in issue
                for issue in generation.copywriting_issues(result, payload)
            )
        )
        result["selling_points"] = ["格子图案搭配长袖设计，适合日常穿搭。"]
        self.assertFalse(
            any(
                "泛化用途" in issue
                for issue in generation.copywriting_issues(result, payload)
            )
        )

    def test_numbered_parallel_power_claims_require_capacity_evidence(self) -> None:
        """两台、数字台数和多台供电共用证据要求，接口数不能直接放行。"""
        for text in [
            "可同时为两台设备供电",
            "连接两台设备充电",
            "同时为2台手机充电",
            "为多台设备供电",
        ]:
            self.assertTrue(
                generation.parallel_power_issues(
                    text, {"商品类型": "车载充电器", "接口数量": "2个"}
                )
            )
        self.assertFalse(
            generation.parallel_power_issues(
                "同时为两台设备供电", {"充电能力": "同时为2台设备充电"}
            )
        )
        self.assertTrue(
            generation.parallel_power_issues(
                "同时为3台设备供电", {"充电能力": "同时为2台设备充电"}
            )
        )
        self.assertFalse(
            generation.parallel_power_issues(
                "配置2个接口，用于车内充电", {"接口数量": "2个"}
            )
        )

    def test_internal_usage_instruction_is_not_public_copy(self) -> None:
        """模型不能照抄通用用途等提示词，须写出具体的已知品类用途。"""
        payload = {"facts": {"商品类型": "水杯", "材质": "钛合金"}}
        result = {
            "title": "钛合金水杯",
            "description": "这款水杯采用钛合金材质。",
            "selling_points": ["采用钛合金材质，结合水杯通用用途。"],
        }
        self.assertTrue(
            any(
                "内部写作指令" in item
                for item in generation.copywriting_issues(result, payload)
            )
        )
        result["selling_points"] = ["钛合金材质水杯，适合日常饮水。"]
        self.assertFalse(
            any(
                "内部写作指令" in item
                for item in generation.copywriting_issues(result, payload)
            )
        )
        self.assertEqual(generation.usage_reference(payload), "日常饮水")

    def test_generic_parallel_power_and_hub_evidence_are_separate(self) -> None:
        """集线器不证明充电，明确多设备充电功能也不能被连接规则误报。"""
        self.assertTrue(
            generation.claim_issues("多设备同时充电", {"商品类型": "USB集线器"})
        )
        self.assertFalse(
            generation.claim_issues("多设备同时充电", {"功能": "同时为两台设备充电"})
        )
        self.assertFalse(
            generation.claim_issues("多设备同时充电", {"功能": "多设备同时充电"})
        )

    def test_parallel_power_without_explicit_device_count_is_still_a_claim(
        self,
    ) -> None:
        """去掉两台等量词仍不能绕过同时供电的证据要求。"""
        self.assertTrue(
            generation.parallel_power_issues(
                "可同时为车内设备充电", {"接口数量": "2个"}
            )
        )
        self.assertFalse(
            generation.parallel_power_issues(
                "可同时为车内设备充电", {"充电能力": "支持同时为两台设备充电"}
            )
        )

    def test_implicit_parallel_claim_cannot_prove_specific_device_count(self) -> None:
        """未量化的同时供电不能证明两台，单台供电也不能扩大成多台。"""
        self.assertTrue(
            generation.parallel_power_issues(
                "同时为两台设备充电", {"充电能力": "可同时为车内设备充电"}
            )
        )
        self.assertTrue(
            generation.parallel_power_issues(
                "同时为两台设备充电", {"充电能力": "同时为1台设备充电并提供照明"}
            )
        )
        self.assertFalse(
            generation.parallel_power_issues(
                "可同时为车内设备充电", {"充电能力": "可同时为车内设备充电"}
            )
        )


if __name__ == "__main__":
    unittest.main()
