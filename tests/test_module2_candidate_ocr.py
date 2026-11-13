"""Guard against identifying a compatible host as the sold product."""

from __future__ import annotations

from evaluate_module2_candidate_ocr import merge_candidate, ocr_signature
from primary_subject_ranker import signature


def test_explicit_offer_needs_title_support_before_correcting_host() -> None:
    """A bag liner offer beats the pictured handbag only with matching title evidence."""
    host = signature("枕头包")
    liner = signature("内胆包")
    text = ocr_signature(
        [
            dict(text="枕头包", confidence=0.99),
            dict(text="专属内胆包", confidence=0.99),
        ]
    )
    assert text is not None
    assert merge_candidate(host, liner, text, override=False) == host
    assert merge_candidate(host, liner, text, override=True) == liner
    assert merge_candidate(host, host, text, override=True) == host


def test_short_product_phrase_rejects_compatibility_and_ambiguous_lines() -> None:
    """A readable description can identify a type, but host or conflicting text cannot."""
    pillow = [dict(text="魔性枕头", confidence=0.99)]
    assert ocr_signature(pillow) is None
    assert ocr_signature(pillow, qualified=True) == {
        **signature("枕头"),
        "offer": False,
    }
    assert (
        ocr_signature([dict(text="适用蓝牙耳机", confidence=0.99)], qualified=True)
        is None
    )
    assert (
        ocr_signature([*pillow, dict(text="蓝牙耳机", confidence=0.99)], qualified=True)
        is None
    )
