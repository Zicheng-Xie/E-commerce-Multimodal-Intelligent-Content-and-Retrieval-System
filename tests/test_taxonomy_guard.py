"""Regress category substring collisions and unsupported parameter claims."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from multimodal_ecommerce import cli, safety
from multimodal_ecommerce.content_contract import generate_content
from multimodal_ecommerce.content_guard import (
    check_claim_support,
    check_source_consistency,
)
from multimodal_ecommerce.dataset_release import read_rows, write_rows
from multimodal_ecommerce.safety import sha256_file, write_json
from multimodal_ecommerce.taxonomy import (
    load_policy,
    release_taxonomy,
    resolve_category,
)

POLICY = Path(__file__).resolve().parents[1] / "configs/taxonomy_v2.json"


@pytest.mark.parametrize(
    "label,title,expected,status",
    [
        ("平板卫生纸", "洁云擦手纸", "家居日用", "mapped"),
        ("办公电脑桌", "液晶屏隐藏电脑桌", "家居日用", "mapped"),
        ("盖帽配饰", "摩托车气门嘴帽", None, "out_of_scope"),
        ("手机包", "韩系女包水桶包", "服饰穿搭", "mapped"),
        ("手机包", "5.5寸真皮手机包", "3C数码", "mapped"),
        ("手机包", "5.5寸手机包女包", None, "needs_review"),
        ("新型电脑桌配件", "手机电脑床", None, "needs_review"),
        ("充电器", "宽电压磷酸铁锂充电器", None, "needs_review"),
    ],
)
def test_taxonomy_is_exact_and_abstains(
    label: str, title: str, expected: object, status: str
) -> None:
    """Verify that taxonomy is exact and abstains.

    Args:
        label: Label used by this operation.
        title: Public product title used as input.
        expected: Expected used by this operation.
        status: Status used by this operation.
    """
    decision = resolve_category(
        {"id": "fixture", "label": label, "text": title}, load_policy(POLICY)
    )
    assert decision["category"] == expected
    assert decision["status"] == status
    assert decision["human_reviewed"] is False


def test_release_preserves_ids_roles_originals_and_quarantines_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that release preserves ids roles originals and quarantines unknown.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through (source / 'image.jpg').write_bytes,
        source.mkdir, write_rows.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "old"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"fixture-only-image")
    row: dict = {
        "id": "known",
        "label": "平板卫生纸",
        "text": "纸巾",
        "category": "3C数码",
        "category_rule": "substring",
        "project_role": "heldout",
        "image_path": "image.jpg",
        "image_sha256": sha256_file(source / "image.jpg"),
    }
    write_rows(
        source / "products.jsonl", [row, {**row, "id": "unknown", "label": "陌生品类"}]
    )
    before = sha256_file(source / "products.jsonl")
    output = tmp_path / "new"
    result = release_taxonomy(source / "products.jsonl", POLICY, output)
    assert result["accepted_count"] == result["quarantined_count"] == 1
    new = read_rows(output / "products.jsonl")[0]
    assert (new["id"], new["project_role"], new["category"]) == (
        "known",
        "heldout",
        "家居日用",
    )
    assert new["lineage"]["previous_category"] == "3C数码"
    assert before == sha256_file(source / "products.jsonl")
    with pytest.raises(FileExistsError):
        release_taxonomy(source / "products.jsonl", POLICY, output)
    (source / "image.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed source image"):
        release_taxonomy(source / "products.jsonl", POLICY, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_overlapping_normalized_policy_labels_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that overlapping normalized policy labels rejected.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through write_json.
    """
    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    policy = load_policy(POLICY)
    policy["exact_labels"][" 手机包 "] = "服饰穿搭"
    write_json(tmp_path / "policy.json", policy)
    with pytest.raises(ValueError, match="Overlapping"):
        load_policy(tmp_path / "policy.json")


class NeverGenerate:
    """Fail if a rejected input reaches model generation."""

    def generate_copy(self, product: dict) -> dict:
        """Apply the generate copy workflow step.

        Args:
            product: Product used by this operation.

        Returns:
            Result mapping produced by the workflow.

        Raises:
            AssertionError: Conflicting source must not invoke the model.
        """
        raise AssertionError("Conflicting source must not invoke the model")


def test_actual_cup_conflict_is_blocked_before_inference() -> None:
    """Verify that actual cup conflict is blocked before inference."""
    product: dict = {
        "name": "纸杯",
        "category": "家居日用",
        "attributes": {
            "容量": ["301mL(含)-400mL(含)"],
            "只数": ["200只"],
            "颜色分类": ["蓝500ml+平盖1000套", "蓝700ml杯500只"],
        },
    }
    result = generate_content(NeverGenerate(), product, "friendly")
    assert result["status"] == "blocked_source_conflict"
    assert result["factual_copy"] is result["marketing_line"] is None
    assert {i["field"] for i in result["source_consistency"]["issues"]} == {
        "容量",
        "只数",
    }


def test_units_and_sku_memory_are_not_guessed() -> None:
    """Verify that units and sku memory are not guessed."""
    source: dict = {
        "attributes": {
            "容量": ["1L"],
            "只数": ["200只"],
            "内存容量": ["16GB"],
            "颜色分类": ["1000ml+1000套", "512GB SSD/8GB显存"],
        }
    }
    assert not check_source_consistency(source)["issues"]
    source["attributes"]["颜色分类"].append("8G内存Z8-TA5NS")
    assert {i["field"] for i in check_source_consistency(source)["issues"]} == {
        "内存容量"
    }


def test_explicit_gpu_option_conflict_does_not_infer_bare_model_numbers() -> None:
    """Verify that explicit gpu option conflict does not infer bare model numbers."""
    source: dict = {
        "attributes": {"显卡类型": ["NVIDIA RTX3070"], "颜色分类": ["3060Z8-CA7NP"]}
    }
    assert not check_source_consistency(source)["issues"]
    source["attributes"]["颜色分类"].append("RTX 3060版")
    assert (
        check_source_consistency(source)["issues"][0]["code"]
        == "fixed_gpu_vs_sku_option"
    )


def test_title_cannot_authorize_pure_cotton_and_rated_power_is_not_adjustable() -> None:
    """Verify that title cannot authorize pure cotton and rated power is not
    adjustable.
    """
    product: dict = {
        "name": "纯棉四季可用",
        "attributes": {"材质": ["棉"], "成分含量": ["95%以上"], "功率": ["6W-10W"]},
    }
    copy: dict = {
        "title": "纯棉四季可用",
        "description": "支持6W至10W功率调节",
        "selling_points": ["100%全棉裆部"],
    }
    codes = {i["code"] for i in check_claim_support(product, copy)["issues"]}
    assert codes == {"cotton_purity", "all_seasons", "adjustable_power"}


def test_positive_material_support_and_negation_are_distinct() -> None:
    """Verify that positive material support and negation are distinct."""
    copy: dict = {"title": "纯棉上衣", "description": "", "selling_points": []}
    assert not check_claim_support({"attributes": {"材质": ["100%棉"]}}, copy)["issues"]
    assert check_claim_support({"attributes": {"材质": ["非纯棉"]}}, copy)["issues"]
    assert check_claim_support(
        {"attributes": {"材质": ["实木"]}}, {"title": "全实木柜"}
    )["issues"]


def test_comparison_to_cotton_is_not_a_cotton_composition_claim() -> None:
    """Verify that comparison to cotton is not a cotton composition claim."""
    result = check_claim_support(
        {"attributes": {"材质": ["其他"]}}, {"title": "比纯棉更柔软"}
    )
    assert {i["code"] for i in result["issues"]} == {"comparative_softness"}
    assert result["issues"][0]["quote"] == "比纯棉更柔软"


def test_claim_guard_blocks_marketing_even_when_old_rule_allows_output() -> None:
    """Verify that claim guard blocks marketing even when old rule allows output."""

    class UnsupportedClient:
        """Return an unsupported claim to exercise post-generation fact checks."""

        def generate_copy(self, product: dict) -> dict:
            """Return generate copy for this workflow.

            Args:
                product: Product used by this operation.

            Returns:
                Mapping with fields: title, description, selling_points, _meta.
            """
            return {
                "title": "纯棉上衣",
                "description": "",
                "selling_points": [],
                "_meta": {"release_status": "draft_requires_human_review"},
            }

    result = generate_content(
        UnsupportedClient(),
        {"name": "上衣", "category": "服饰穿搭", "attributes": {"材质": ["棉"]}},
        "friendly",
    )
    assert result["status"] == "blocked_by_contract_rules"
    assert result["marketing_line"] is None
    assert result["factual_copy"]["title"] == "纯棉上衣"


def test_default_cli_generate_uses_contract_and_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that default cli generate uses contract and preserves existing output.

    Args:
        tmp_path: Isolated temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.

    Notes:
        Writes derived artifacts through write_json.
    """

    class Client(NeverGenerate):
        """Provide controlled model responses for the surrounding isolated test
        scenario.
        """

        def __init__(self, *args: object) -> None:
            """Initialize the component and validate its configured dependencies.
            *args: Additional positional values forwarded by this interface.
            """
            pass

        def close(self) -> None:
            """Release the resources held by this component."""
            pass

    monkeypatch.setattr(safety, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", Client)
    monkeypatch.setattr(
        cli,
        "configuration",
        lambda: {
            "source_root": "protected",
            "samples_root": "samples",
            "ollama_base_url": "unused",
            "generation_model": "unused",
        },
    )
    source = tmp_path / "product.json"
    write_json(
        source,
        {
            "name": "纸杯",
            "category": "家居日用",
            "attributes": {"容量": ["300ml"], "颜色分类": ["500ml"]},
        },
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "generate", "--product", str(source)])
    cli.main()
    output = tmp_path / "artifacts/generated_content.json"
    assert (
        json.loads(output.read_text(encoding="utf-8"))["status"]
        == "blocked_source_conflict"
    )
    before = sha256_file(output)
    with pytest.raises(FileExistsError):
        cli.main()
    assert before == sha256_file(output)
