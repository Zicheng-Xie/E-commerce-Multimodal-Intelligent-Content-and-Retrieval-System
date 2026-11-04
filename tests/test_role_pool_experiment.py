"""Attribution controls: retain candidates, isolate uncertain profiles, exact fusion."""

from __future__ import annotations

import numpy as np
import pytest

from grounded_subject_runtime import GroundedSubjectRuntimeFuseInputs, fuse
from role_pool_experiment import (
    ARMS,
    COMPARISONS,
    PRIMARY,
    extend_pool,
    role_text,
)


def test_extension_retains_every_original_candidate_and_uses_fixed_top20() -> None:
    """Verify that extension retains every original candidate and uses fixed top20."""
    ids = [f"d{i:02d}" for i in range(40)]
    parent = ids[25:]
    out = extend_pool(ids, parent, ids)
    assert out[: len(parent)] == parent
    assert out[len(parent) :] == ids[:20]
    assert set(parent) <= set(out)
    assert len(out) == len(set(out))
    with pytest.raises(ValueError):
        extend_pool(ids, parent, ids[:-1])


def test_uncertain_role_abstains_and_keeps_type_separate_from_host() -> None:
    """Verify that uncertain role abstains and keeps type separate from host."""
    p: dict = dict(
        sale_item="包内胆",
        role="accessory",
        host_object="手提包",
        display_props=["手提包"],
        function="收纳",
        appearance="独立多格袋体",
        primary_unclear=False,
    )
    text = role_text(p)
    assert text.startswith("主售商品：包内胆")
    assert "配套或使用对象：手提包" in text
    assert role_text(dict(p, primary_unclear=True)) == ""
    assert role_text(dict(p, role="unclear")) == ""


def test_dual_fusion_demands_real_scores_and_preserves_same_gallery_tail() -> None:
    """Verify that dual fusion demands real scores and preserves same gallery tail."""
    ids = [f"d{i}" for i in range(15)]
    pool = ids[:4]
    scores: dict = dict(zip(pool, [2.0, 1.0, 4.0, 3.0]))
    base = np.arange(15, dtype=float)
    order = fuse(
        GroundedSubjectRuntimeFuseInputs(ids, pool, base, scores, scores, mode="dual")
    )
    assert set(order[:4]) == set(pool)
    assert set(order) == set(ids) and len(order) == 15
    with pytest.raises(ValueError):
        fuse(
            GroundedSubjectRuntimeFuseInputs(
                ids, pool, base, {pool[0]: 1.0}, scores, mode="dual"
            )
        )
    assert (ARMS[1], ARMS[0]) in COMPARISONS
    assert (ARMS[2], ARMS[0]) in COMPARISONS
    assert (PRIMARY, ARMS[1]) in COMPARISONS
    assert (PRIMARY, ARMS[2]) in COMPARISONS
