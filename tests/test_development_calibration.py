"""Check public-only ranking inputs, grouped human learning and prop-only rewriting."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from module2_calibrated_development import crossfit, run, swap_proof
from module2_subject_description_development import execute
from multimodal_ecommerce.calibrated_ranking import (
    FEATURES,
    CalibrationContext,
    calibrated_order,
    fit_pairwise,
    preference_scores,
    public_features,
)
from multimodal_ecommerce.subject_description import clean_caption, subject_swap


@pytest.mark.parametrize("entry", [run, execute])
def test_development_entry_rejects_reserved_output_before_loading_data(
    entry: Callable[[Path], dict],
) -> None:
    """Neither experiment may write results over the sealed final acceptance data.

    Args:
        entry: Automatic development entry to validate before any model load.
    """
    with pytest.raises(PermissionError):
        entry(Path("artifacts/module2_final_acceptance_20261110/forbidden_output"))


def _context() -> CalibrationContext:
    order = [f"p{i:02}" for i in range(24)]
    return CalibrationContext(
        order=order,
        original=order,
        scores={c: float(20 - i) for i, c in enumerate(order[:20])},
        titles={c: "黑色无线耳机" for c in order},
        text="无线耳机",
    )


def test_public_feature_identity_and_inference_do_not_use_ids_or_labels() -> None:
    """Renaming IDs preserves public features; unknown annotation fields are rejected."""
    context = _context()
    before, group = public_features(context)
    names = {c: "renamed_" + c for c in context.order}
    renamed = CalibrationContext(
        order=[names[c] for c in context.order],
        original=[names[c] for c in context.original],
        scores={names[c]: s for c, s in context.scores.items()},
        titles={names[c]: t for c, t in context.titles.items()},
        text=context.text,
    )
    after, other = public_features(renamed)
    assert np.array_equal(before, after) and other == group
    untrusted: dict = dict(
        order=context.order,
        original=context.original,
        scores=context.scores,
        titles=context.titles,
        human_grade=1,
    )
    # The invalid extra field is intentional in this boundary rejection test.
    # pylint: disable=unexpected-keyword-arg
    with pytest.raises(TypeError):
        CalibrationContext(**untrusted)
    # pylint: enable=unexpected-keyword-arg


def test_pairwise_training_uses_only_binary_pairs_and_reduces_fixed_loss() -> None:
    """Unresolved examples do not become negative training targets."""
    matrix = np.zeros((6, len(FEATURES)))
    matrix[:, 0] = [3, 2, -3, -2, 100, -100]
    model = fit_pairwise([(matrix, [1, 1, 0, 0, None, None])])
    assert model["pairs"] == 4
    assert model["final_loss"] < model["initial_loss"]
    scores = preference_scores(matrix, model)
    assert min(scores[:2]) > max(scores[2:4])
    assert not any(k in model for k in ("query_ids", "candidate_ids", "grades"))
    with pytest.raises(ValueError, match="explicit human"):
        fit_pairwise([(matrix, [True, 1, 0, 0, None, None])])


def test_group_exclusion_is_invariant_to_scored_family_answer_changes() -> None:
    """A family's own human labels cannot influence the model scoring that family."""
    visible, queries, baseline = {}, {}, {}
    for i, group in enumerate(("a", "b", "c")):
        order = [f"p{j:02}" for j in range(24)]
        matrix = np.random.default_rng(i).normal(size=(20, len(FEATURES)))
        visible[group] = dict(matrix=matrix, group=group)
        baseline[group] = order
        queries[group] = dict(
            judgments={
                c: dict(human=True, grade=int(j < 10)) for j, c in enumerate(order[:20])
            }
        )
    before, models, audit = crossfit(visible, queries, baseline)
    changed = copy.deepcopy(queries)
    for value in changed["a"]["judgments"].values():
        value["grade"] = 1 - value["grade"]
    after, other, _ = crossfit(visible, changed, baseline)
    assert models["a"] == other["a"]
    assert before["pairwise_one_swap"]["a"] == after["pairwise_one_swap"]["a"]
    assert all(
        group not in row["train_families"]
        and not set(row["train_queries"]) & set(row["test_queries"])
        for group, row in audit.items()
    )


@pytest.mark.parametrize("defect", ["nan", "zero_scale", "wrong_features"])
def test_invalid_preference_parameters_fail(defect: str) -> None:
    """Corrupted learned parameters never silently generate a ranking.

    Args:
        defect: Numeric or feature-identity corruption.
    """
    matrix = np.zeros((4, len(FEATURES)))
    matrix[:, 0] = [3, 2, -2, -3]
    model = fit_pairwise([(matrix, [1, 1, 0, 0])])
    if defect == "nan":
        model["weights"][0] = float("nan")
    elif defect == "zero_scale":
        model["scale"][0] = 0
    else:
        model["features"] = ["query_id"]
    with pytest.raises(ValueError):
        preference_scores(matrix, model)


def test_conservative_swap_keeps_prefix_and_does_not_hide_unresolved_changes() -> None:
    """Protect anchors, retain the whole gallery and fail adoption on unjudged changes."""
    order = _context().order
    scores = np.ones(20)
    scores[7], scores[12] = -2, 3
    after = calibrated_order(order, scores)
    assert after[:2] == order[:2] and set(after) == set(order)
    assert sum(a != b for a, b in zip(order, after, strict=True)) == 2
    queries: dict = dict(
        q=dict(
            judgments={
                order[7]: dict(human=True, grade=1),
                order[12]: dict(human=True, grade=0),
            }
        )
    )
    proof = swap_proof(dict(q=order), dict(q=after), queries)
    assert (
        proof["exact_net_hits"] == -1 and not proof["eligible_for_further_development"]
    )
    queries["q"]["judgments"][order[12]]["grade"] = None
    assert swap_proof(dict(q=order), dict(q=after), queries)["unresolved_pairs"] == 1


def test_subject_description_removes_foreign_footwear_but_preserves_shoe_queries() -> (
    None
):
    """A suit expansion drops footwear props; a shoe expansion keeps its own evidence."""
    caption = dict(
        product="西装套装",
        appearance="深蓝色双排扣西装，脚穿拼色尖头短靴",
        unclear=False,
    )
    result = clean_caption(caption)
    assert result["changed"] and "短靴" not in result["cleaned"]["appearance"]
    assert result["original"] == caption and "other_shoes" in result["removed_types"]
    shoes = dict(product="短靴", appearance="脚穿黑色尖头短靴", unclear=False)
    assert not clean_caption(shoes)["changed"]
    uncertain = dict(caption, unclear=True)
    assert not clean_caption(uncertain)["changed"]


def test_compound_caption_abstains_instead_of_removing_its_own_bag_evidence() -> None:
    """The previous bag-noun safeguard remains effective in the new description path."""
    caption = dict(product="枕头包", appearance="手持棕色手提包", unclear=False)
    assert not clean_caption(caption)["changed"]


def test_subject_swap_requires_prop_conflict_and_stronger_real_score() -> None:
    """Only the identified foreign prop can leave the preserved Top10."""
    order = [f"p{i:02}" for i in range(24)]
    titles = {c: "蓝色西装外套" for c in order}
    titles[order[9]] = "拼色尖头短靴"
    observation = clean_caption(
        dict(product="西装套装", appearance="蓝色西装，脚穿尖头短靴", unclear=False)
    )
    scores = {c: 0.0 for c in order[:20]}
    scores[order[10]], scores[order[9]] = 2.0, -2.0
    inputs: dict = dict(
        order=order,
        pool=order[:20],
        titles=titles,
        scores=scores,
        observation=observation,
    )
    result, trace = subject_swap(inputs)
    assert result[9] == order[10] and result[10] == order[9] and trace["changed"]
    assert result[:2] == order[:2] and set(result) == set(order)
    scores[order[9]] = 3.0
    result, trace = subject_swap(inputs)
    assert result == order and not trace["changed"]
    inputs["type_veto"] = True
    result, trace = subject_swap(inputs)
    assert result[9] == order[10] and trace["type_veto"]


def test_hard_subject_constraint_keeps_explicit_query_attributes() -> None:
    """A subject veto cannot replace a long-trouser outfit with an explicit short one."""
    order = [f"p{i:02}" for i in range(24)]
    titles = {c: "蓝色西装长裤套装" for c in order}
    titles[order[9]], titles[order[10]] = "尖头短靴", "西装短裤套装"
    observation = clean_caption(
        dict(product="西装套装", appearance="西装外套配长裤，脚穿短靴", unclear=False)
    )
    scores = {c: 0.0 for c in order[:20]}
    scores[order[10]], scores[order[11]] = 4.0, 2.0
    result, trace = subject_swap(
        dict(
            order=order,
            pool=order[:20],
            titles=titles,
            scores=scores,
            observation=observation,
            type_veto=True,
        )
    )
    assert result[9] == order[11] and trace["incoming"] == order[11]
