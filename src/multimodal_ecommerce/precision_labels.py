"""Strict business-label imports for the existing development Top30 packet."""

from __future__ import annotations

from collections import Counter

EDITABLE = {"grade", "note", "reviewer", "review_origin"}


def validate_precision_labels(templates: list[dict], reviewed: list[dict]) -> dict:
    """Permit pending labels; reject tampering, duplicate pairs and fake provenance.

    Args:
        templates: Templates used by this operation.
        reviewed: Reviewed used by this operation.

    Returns:
        Mapping with fields: total, completed, pending, positive, negative,
        completed_rows, formal_acceptance, full_gallery_recall_computable,
        provenance_limit.

    Raises:
        ValueError: Packet row count mismatch; Unknown or duplicate judgment; Immutable
            packet fields changed; Only this development packet is supported; Grade must
            be null or integer 0/1; Reviewer and note must be text; Only declared human
            judgments may enter this label stream; Judgments and review notes require a
            human declaration.
    """
    expected = {r["id"]: r for r in templates}
    if len(expected) != len(templates) or len(reviewed) != len(templates):
        raise ValueError("Packet row count mismatch")
    seen = set()
    completed = []
    for row in reviewed:
        identifier = row.get("id")
        if identifier not in expected or identifier in seen:
            raise ValueError("Unknown or duplicate judgment")
        seen.add(identifier)
        template = expected[identifier]
        if set(row) != set(template) or any(
            row[k] != template[k] for k in template if k not in EDITABLE
        ):
            raise ValueError("Immutable packet fields changed")
        if template["role"] != "development_training_only":
            raise ValueError("Only this development packet is supported")
        grade = row["grade"]
        if grade is not None and (type(grade) is not int or grade not in (0, 1)):
            raise ValueError("Grade must be null or integer 0/1")
        if not isinstance(row["note"], str) or not isinstance(row["reviewer"], str):
            raise ValueError("Reviewer and note must be text")
        if row["review_origin"] not in (None, "human"):
            raise ValueError(
                "Only declared human judgments may enter this label stream"
            )
        if grade is not None or row["note"].strip():
            if row["review_origin"] != "human" or not row["reviewer"].strip():
                raise ValueError(
                    "Judgments and review notes require a human declaration"
                )
        if grade is not None:
            completed.append(row)
    counts = Counter(r["grade"] for r in completed)
    return {
        "total": len(templates),
        "completed": len(completed),
        "pending": len(templates) - len(completed),
        "positive": counts[1],
        "negative": counts[0],
        "completed_rows": completed,
        "formal_acceptance": False,
        "full_gallery_recall_computable": False,
        "provenance_limit": "Reviewer declaration is validated, not independently authenticated",
    }


def human_training_entries(entries: list[dict], reviewed: list[dict]) -> list[dict]:
    """Use only explicit human 0/1 labels; never convert pending relations to
    negatives.

    Args:
        entries: Entries used by this operation.
        reviewed: Reviewed used by this operation.

    Returns:
        Values produced by this operation, in the stated return order.
    """
    judgments = {
        (r["query_id"], r["candidate_id"]): r["grade"]
        for r in reviewed
        if r["grade"] is not None
    }
    output = []
    for entry in entries:
        selected = [
            i
            for i, c in enumerate(entry["candidate_ids"])
            if (entry["query_id"], c) in judgments
        ]
        labels = [
            bool(judgments[(entry["query_id"], entry["candidate_ids"][i])])
            for i in selected
        ]
        if not any(labels) or all(labels):
            continue
        output.append(
            {
                **entry,
                "candidate_ids": [entry["candidate_ids"][i] for i in selected],
                "candidate_groups": [entry["candidate_groups"][i] for i in selected],
                "features": [entry["features"][i] for i in selected],
                "teacher": [entry["teacher"][i] for i in selected],
                "labels": labels,
                "label_origin": "declared_human_business_relevance",
            }
        )
    return output
