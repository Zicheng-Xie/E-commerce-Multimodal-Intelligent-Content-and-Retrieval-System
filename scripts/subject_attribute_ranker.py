"""Public sale-item attributes and full-evidence hard-negative rank fusion."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from learned_fusion_model import FEATURE_NAMES as ORIGINAL_FEATURES
from learned_fusion_model import POLICY
from primary_subject_ranker import penalty, signature

ATTRIBUTE_NAMES = [
    "primary_types_known",
    "primary_type_match",
    "primary_type_conflict",
    "subtypes_known",
    "subtype_match",
    "subtype_conflict",
    "sale_roles_known",
    "sale_role_match",
    "sale_role_conflict",
    "phone_models_known",
    "phone_model_match",
    "phone_model_conflict",
    "visible_structure_conflict",
    "whole_minus_crop_CE_rank",
    "whole_crop_CE_disagreement",
    "whole_minus_ROI_FG_rank",
    "whole_ROI_FG_disagreement",
    "regional_minus_global_cosine",
]
FEATURE_NAMES = [*ORIGINAL_FEATURES, *ATTRIBUTE_NAMES]
PUBLIC_KEYS = {"product", "appearance", "visible_text", "primary_unclear"}


def phone_models(text: str) -> list[str]:
    """Extract explicit supported model strings; missing text means abstention.

    Args:
        text: Text input used by the operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\s_-]+", "", text)
    result = set()
    patterns = [
        (
            "iphone",
            r"(?:iphone|苹果)(x(?:s)?|se[23]?|\d{1,2})(promax|pro|max|plus|mini)?(?!\d)",
        ),
        ("huawei", r"(mate\d{1,2}|nova\d{1,2})(pro|plus|i)?(?!\d)"),
        ("honor", r"(?:honor|荣耀)(x?\d{1,2})(pro|plus|s)?(?!\d)"),
        ("vivo", r"vivo(x\d{1,2}|y\d{1,3}|s\d{1,2})(pro|plus|s)?(?!\d)"),
        ("iqoo", r"iqoo(neo\d{0,3}|\d{1,2})(pro|plus)?(?!\d)"),
    ]
    if any(word in text for word in ("华为", "huawei")):
        patterns.append(("huawei", r"(p\d{1,2})(pro|plus)?(?!\d)"))
    if any(word in text for word in ("三星", "samsung", "galaxy")):
        patterns.append(
            ("samsung", r"(s\d{1,2}|a\d{1,2}|note\d{1,2})(ultra|plus|e|s|\+)?(?!\d)")
        )
    for brand, pattern in patterns:
        for found in re.finditer(pattern, text):
            result.add(brand + ":" + "".join(part or "" for part in found.groups()))
    return sorted(result)


def public_descriptor(value: dict) -> dict:
    """Return public descriptor for this workflow.

    Args:
        value: Input value to validate or transform.

    Returns:
        Mapping with fields: family, subtype, structures, role, phone_models.

    Raises:
        ValueError: Exactly query/candidate-only public fields required; Public
            descriptions must be strings.
    """
    if set(value) != PUBLIC_KEYS or type(value["primary_unclear"]) is not bool:
        raise ValueError("Exactly query/candidate-only public fields required")
    if not all(isinstance(value[k], str) for k in PUBLIC_KEYS - {"primary_unclear"}):
        raise ValueError("Public descriptions must be strings")
    head, appearance, title = (
        value["product"],
        value["appearance"],
        value["visible_text"],
    )
    if value["primary_unclear"]:
        return dict(
            family=None, subtype=None, structures=[], role=None, phone_models=[]
        )
    parsed = signature(head, appearance)
    family, subtype = parsed["family"], parsed["subtype"]
    # Restrict title refinements to an already identified sale-item family.
    # Context objects in a handbag/phone title cannot redefine an unrelated item.
    if family == "bag":
        if any(
            k in head + title
            for k in ("笔记本电脑", "电脑内胆包", "笔记本包", "macbook")
        ):
            subtype = "laptop"
        elif any(k in head for k in ("相机包", "摄影包")):
            subtype = "camera"
    role = "main" if family else None
    if family in (
        "phone_case",
        "phone_accessory",
        "audio_accessory",
        "shoe_accessory",
        "pillow_accessory",
        "cup_accessory",
    ):
        role = "accessory"
    if family == "bag" and subtype in ("organizer", "laptop", "camera"):
        role = "accessory"
    if family == "phone_case" and any(
        k in head + title for k in ("滴胶材料", "手机壳材料", "自制材料", "手机壳配件")
    ):
        family, subtype, role = "craft_material", "phone_case_material", "material"
    models = (
        phone_models(head + " " + appearance + " " + title)
        if family == "phone_case"
        else []
    )
    return dict(
        family=family,
        subtype=subtype,
        structures=parsed["structures"],
        role=role,
        phone_models=models,
    )


def attribute_vector(
    original: np.ndarray, query: dict, candidate: dict
) -> tuple[np.ndarray, dict]:
    """Return attribute vector for this workflow.

    Args:
        original: Original used by this operation.
        query: Query record or representation used by this operation.
        candidate: Candidate used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.

    Raises:
        ValueError: Complete original real-model evidence required.
    """
    original = np.asarray(original, dtype=float)
    if original.shape != (len(ORIGINAL_FEATURES),) or not np.isfinite(original).all():
        raise ValueError("Complete original real-model evidence required")
    q, d = public_descriptor(query), public_descriptor(candidate)

    def pair(left: dict, right: dict) -> list[float]:
        """Return pair for this workflow.

        Args:
            left: Left used by this operation.
            right: Right used by this operation.

        Returns:
            Values produced by this operation, in the stated return order.
        """
        known = left is not None and right is not None
        return [
            float(known),
            float(known and left == right),
            float(known and left != right),
        ]

    models_known = bool(q["phone_models"] and d["phone_models"])
    model_match = bool(set(q["phone_models"]) & set(d["phone_models"]))
    structural = (
        penalty(
            {k: q[k] for k in ("family", "subtype", "structures")},
            {k: d[k] for k in ("family", "subtype", "structures")},
        )["reason"]
        == "explicit incompatible visible structures"
    )
    old, new, crop = [
        original[ORIGINAL_FEATURES.index(name)]
        for name in ("old_whole_CE_rank", "new_whole_CE_rank", "actual_crop_CE_rank")
    ]
    whole, roi = [
        original[ORIGINAL_FEATURES.index(name)]
        for name in ("whole_FG_fusion_rank", "ROI_FG_fusion_rank")
    ]
    delta = 0.5 * (old + new) - crop
    values = [
        *pair(q["family"], d["family"]),
        *pair(q["subtype"], d["subtype"]),
        *pair(q["role"], d["role"]),
        float(models_known),
        float(models_known and model_match),
        float(models_known and not model_match),
        float(structural),
        delta,
        abs(delta),
        whole - roi,
        abs(whole - roi),
        original[1] - original[0],
    ]
    assert len(values) == len(ATTRIBUTE_NAMES)
    return np.asarray(values, dtype=float), dict(query=q, candidate=d)


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    base: np.ndarray,
    *,
    names: list[str],
) -> dict:
    """Same 400-step RankNet recipe, with explicit complete feature dimensions.

    Args:
        features: Feature array aligned with the candidate or query records.
        labels: Judgments used for the stated development or evaluation protocol.
        allowed: Allowed used by this operation.
        base: Base used by this operation.
        names: Names to inspect or resolve.

    Returns:
        Mapping with fields: feature_names, mean, scale, weights, score_location,
        score_scale, training_cells, positive, negative, optimization_rows, policy,
        human_labels, formal_metrics.

    Raises:
        ValueError: Feature configuration mismatch; Both training classes required
            inside the family mask; Missing real training evidence; No source-diverse
            confirmed training pairs.
    """
    if features.shape[-1] != len(names):
        raise ValueError("Feature configuration mismatch")
    known = allowed & np.isfinite(labels)
    if not known.any() or set(np.unique(labels[known])) != {0.0, 1.0}:
        raise ValueError("Both training classes required inside the family mask")
    if not np.isfinite(features[known]).all():
        raise ValueError("Missing real training evidence")
    mean, scale = features[known].mean(0), features[known].std(0)
    scale = np.where(scale < 1e-6, 1, scale)
    standardized = (features - mean) / scale
    xs, weights = [], []
    for q in range(len(labels)):
        positives = np.flatnonzero(known[q] & (labels[q] == 1))
        negatives = np.flatnonzero(known[q] & (labels[q] == 0))
        if not len(positives) or not len(negatives):
            continue
        negatives = negatives[
            np.argsort(-base[q, negatives], kind="stable")[: POLICY["hard_negatives"]]
        ]
        for p in positives:
            xs.extend(standardized[q, p] - standardized[q, negatives])
            weights.extend([1 / (len(positives) * len(negatives))] * len(negatives))
    if not xs:
        raise ValueError("No source-diverse confirmed training pairs")
    x, sample = np.asarray(xs), np.asarray(weights)
    sample /= sample.sum()
    w = np.zeros(len(names))
    for _ in range(POLICY["steps"]):
        probability = 1 / (1 + np.exp(-np.clip(x @ w, -40, 40)))
        w -= POLICY["learning_rate"] * (
            x.T @ ((probability - 1) * sample) + POLICY["l2"] * w
        )
    raw = standardized[known] @ w
    location, spread = float(raw.mean()), float(raw.std())
    return dict(
        feature_names=list(names),
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=w.tolist(),
        score_location=location,
        score_scale=spread if spread > 1e-6 else 1.0,
        training_cells=int(known.sum()),
        positive=int((labels[known] == 1).sum()),
        negative=int((labels[known] == 0).sum()),
        optimization_rows=len(x),
        policy=POLICY,
        human_labels=0,
        formal_metrics=None,
    )


def predict(features: np.ndarray, model: dict) -> np.ndarray:
    """Return predict for this workflow.

    Args:
        features: Feature array aligned with the candidate or query records.
        model: Model used by this operation.

    Returns:
        Computed array in the order of the supplied records.

    Raises:
        ValueError: Feature version mismatch.
    """
    if features.shape[-1] != len(model["feature_names"]):
        raise ValueError("Feature version mismatch")
    return (
        ((features - model["mean"]) / model["scale"]) @ model["weights"]
        - model["score_location"]
    ) / model["score_scale"]
