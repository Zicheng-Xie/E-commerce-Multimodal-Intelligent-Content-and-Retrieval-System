"""Guard candidate-only feature inputs, native completion and explicit abstention."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import balanced_supplement_features as pipeline


def raw_profile() -> dict:
    """Return raw profile for this workflow.

    Returns:
        Mapping with fields: model, done, done_reason, prompt_eval_count,
        eval_count, message.
    """
    return dict(
        model=pipeline.acquisition.MODEL,
        done=True,
        done_reason="stop",
        prompt_eval_count=100,
        eval_count=100,
        message=dict(
            role="assistant",
            content=json.dumps(
                dict(
                    product="耳机保护套",
                    appearance="编织外壳",
                    visible_text="",
                    primary_unclear=False,
                )
            ),
        ),
    )


def test_candidate_only_request_rejects_evaluator_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that candidate only request rejects evaluator fields.

    Args:
        monkeypatch: Pytest fixture for restoring patched attributes after the scenario.
    """
    monkeypatch.setattr(
        pipeline.acquisition, "encoded", lambda p: ("nativePNG", "pixels")
    )
    monkeypatch.setattr(pipeline, "load", lambda p: dict(digest="full-model"))
    doc: dict = dict(id="candidate", title="保护套", image_path="original.png")
    request, identity = pipeline.profile_payload(doc)
    assert request["messages"][0]["images"] == ["nativePNG"]
    assert identity["queries_labels_and_rankings_supplied"] is False
    assert request["options"]["num_ctx"] == 16384
    assert request["keep_alive"] == "10m"
    for key in ("grade", "query", "rank", "bucket", "candidate_product_type"):
        with pytest.raises(AssertionError):
            pipeline.profile_payload(dict(doc, **{key: "forbidden"}))


def test_native_profile_preserves_real_unknown_and_rejects_truncation() -> None:
    """Verify that native profile preserves real unknown and rejects truncation."""
    raw = raw_profile()
    assert pipeline.profile_value(raw)["primary_unclear"] is False
    value = json.loads(raw["message"]["content"])
    value["primary_unclear"] = True
    raw["message"]["content"] = json.dumps(value)
    assert pipeline.profile_value(raw)["primary_unclear"] is True
    for key, bad in [
        ("done", False),
        ("done_reason", "length"),
        ("prompt_eval_count", 15360),
        ("eval_count", 1024),
        ("prompt_eval_count", True),
    ]:
        changed = copy.deepcopy(raw)
        changed[key] = bad
        with pytest.raises(AssertionError):
            pipeline.profile_value(changed)


def test_profile_schema_rejects_duplicate_or_relation_fields() -> None:
    """Verify that profile schema rejects duplicate or relation fields."""
    raw = raw_profile()
    raw["message"]["content"] = '{"product":"x","product":"y"}'
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.profile_value(raw)
    raw = raw_profile()
    value = json.loads(raw["message"]["content"])
    value["grades"] = [1]
    raw["message"]["content"] = json.dumps(value)
    with pytest.raises(AssertionError):
        pipeline.profile_value(raw)


def test_full_feature_shapes_and_explicit_zero_semantic_abstention() -> None:
    """Verify that full feature shapes and explicit zero semantic abstention."""
    bank: dict = dict(
        views=np.ones((6, 2048), dtype=np.float32),
        title=np.ones(2048, dtype=np.float32),
        profile=np.ones(2048, dtype=np.float32),
    )
    padded, title, profile = pipeline.checked_vectors(bank, abstain=False)
    assert padded.shape == (7, 2048) and np.all(padded[-1] == 0)
    assert np.array_equal(padded[:6], bank["views"])
    assert np.array_equal(title, bank["title"]) and np.array_equal(
        profile, bank["profile"]
    )
    with pytest.raises(AssertionError):
        pipeline.checked_vectors(bank, abstain=True)
    bank["profile"][:] = 0
    pipeline.checked_vectors(bank, abstain=True)
    with pytest.raises(AssertionError):
        pipeline.checked_vectors(bank, abstain=False)
    for value in (np.zeros((6, 2048)), np.ones((6, 1024)), np.full((6, 2048), np.nan)):
        with pytest.raises(AssertionError):
            pipeline.checked_vectors(dict(bank, views=value), abstain=True)
