"""Public-only swap safety, compound nouns, and conditional metric guarantees."""

from __future__ import annotations

import itertools

import pytest

from module2_mainline_conflicts import _capacity, _swap_evidence
from multimodal_ecommerce.public_conflict_guard import reorder_public_conflicts
from multimodal_ecommerce.retrieval_strategies import extended_metrics


def _chair() -> tuple[list[str], dict[str, str], dict]:
    order = [str(i) for i in range(24)]
    titles = {c: "标题未提供主体词" for c in order}
    titles.update({"1": "女士短靴", "12": "实木餐椅"})
    return order, titles, dict(product="餐椅", appearance="", unclear=False)


def test_matching_replacement_changes_only_two_positions_and_keeps_full_gallery() -> (
    None
):
    """The tail must not be shifted when a known-negative position is replaced."""
    order, titles, caption = _chair()
    revised, observations = reorder_public_conflicts(
        order, titles, caption=caption, options=dict(protect=0)
    )
    assert revised[1] == "12" and revised[12] == "1"
    assert [i for i in range(len(order)) if order[i] != revised[i]] == [1, 12]
    assert set(revised) == set(order) and order[1] == "1"
    assert observations["swaps"][0]["position"] == 2
    protected, _ = reorder_public_conflicts(
        order, titles, caption=caption, options=dict(protect=2)
    )
    assert protected == order


def test_unknown_tail_titles_cannot_fill_conflicts_and_unclear_query_abstains() -> None:
    """Missing positive title evidence cannot be replaced by a random fallback."""
    order, titles, caption = _chair()
    titles["12"] = "未知商品"
    assert (
        reorder_public_conflicts(
            order, titles, caption=caption, options=dict(protect=0)
        )[0]
        == order
    )
    caption["unclear"] = True
    assert (
        reorder_public_conflicts(
            order, titles, caption=caption, options=dict(protect=0)
        )[0]
        == order
    )


def test_bag_compound_is_not_treated_as_a_bed_pillow() -> None:
    """The same compound noun correction applies to candidate and query titles."""
    order, titles, _ = _chair()
    titles.update({"1": "波士顿枕头包手提包", "12": "电脑内胆包"})
    caption = dict(product="枕头包", appearance="皮革手提包", unclear=False)
    revised, observations = reorder_public_conflicts(
        order, titles, caption=caption, options=dict(protect=0)
    )
    assert revised == order
    assert observations["evidence"]["1"]["title_type"] == "bag"


def test_battery_case_replacement_needs_a_phone_case_title() -> None:
    """Plain phone products cannot become replacements for a protective case."""
    order, titles, _ = _chair()
    titles.update({"1": "背夹式电池移动电源", "11": "老人手机", "12": "苹果手机保护壳"})
    revised, _ = reorder_public_conflicts(
        order, titles, text="手机保护壳", options=dict(protect=0)
    )
    assert revised[1] == "12" and revised[11] == "11"


def test_negative_downswap_nondecreases_every_metric_for_every_unknown_completion() -> (
    None
):
    """Prove the rule's claimed consequence by enumerating unresolved relevance."""
    order, titles, caption = _chair()
    revised, _ = reorder_public_conflicts(
        order, titles, caption=caption, options=dict(protect=0)
    )
    for bits in itertools.product((False, True), repeat=4):
        positive = {"0", "3", "20"} | {
            c for c, on in zip(("4", "10", "12", "18"), bits) if on
        }
        before = extended_metrics(order, positive)
        after = extended_metrics(revised, positive)
        assert before.keys() == after.keys()
        assert all(after[k] + 1e-12 >= before[k] for k in before)
    q: dict = {"q": dict(sources=[], judgments={"1": dict(human=True, grade=0)})}
    proof = _swap_evidence({"q": order}, {"q": revised}, q)
    assert proof["all_six_metrics_nondecreasing_for_all_human_label_completions"]
    q["q"]["judgments"]["1"] = dict(human=False, grade=0)
    assert not _swap_evidence({"q": order}, {"q": revised}, q)[
        "all_six_metrics_nondecreasing_for_all_human_label_completions"
    ]


def test_capacity_depth_is_diagnostic_and_does_not_mutate_top10_contract() -> None:
    """A larger hypothetical depth must not rewrite the reported Top10 capacity."""
    rows = {"q": dict(mode="text", human_positive=16, human_unresolved=100)}
    assert _capacity(rows, "text", 10) == pytest.approx(0.625)
    assert _capacity(rows, "text", 20) == 1
    assert _capacity(rows, "image", 10) is None


def test_foreign_candidates_and_invalid_protection_are_rejected() -> None:
    """Full-gallery identity must be preserved before public evidence is used."""
    order, titles, caption = _chair()
    with pytest.raises(ValueError):
        reorder_public_conflicts(order + [order[0]], titles, caption=caption)
    with pytest.raises(ValueError):
        reorder_public_conflicts(order, {**titles, "foreign": "餐椅"}, caption=caption)
    with pytest.raises(ValueError):
        reorder_public_conflicts(
            order, titles, caption=caption, options=dict(protect=1)
        )
