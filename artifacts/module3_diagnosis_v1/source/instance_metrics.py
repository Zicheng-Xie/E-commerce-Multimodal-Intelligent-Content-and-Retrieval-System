"""Class-aware greedy instance mask AP; no COCO crowd/area conventions."""

import numpy as np


def instance_ap(images, thresholds=None):
    """Evaluate image-local masks, class labels and scores at 101 recall points.

    images: list of {gt_labels, gt_masks, labels, masks, scores}.
    Classes without ground truth have AP=None. False positives, duplicates and
    missed targets remain in the denominator. No oracle matching or class filter.
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 10)
    classes = sorted(
        {int(c) for row in images for c in list(row["gt_labels"]) + list(row["labels"])}
    )
    results = {}
    for c in classes:
        count = sum(sum(int(v) == c for v in row["gt_labels"]) for row in images)
        if not count:
            results[str(c)] = None
            continue
        predictions = sorted(
            [
                (float(score), i, j)
                for i, row in enumerate(images)
                for j, (label, score) in enumerate(zip(row["labels"], row["scores"]))
                if int(label) == c
            ],
            key=lambda v: -v[0],
        )
        aps = []
        for threshold in thresholds:
            used = set()
            tp = []
            for _, i, j in predictions:
                row = images[i]
                pred = np.asarray(row["masks"][j], dtype=bool)
                best = None
                best_iou = -1
                for k, (label, mask) in enumerate(
                    zip(row["gt_labels"], row["gt_masks"])
                ):
                    if int(label) != c or (i, k) in used:
                        continue
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape != pred.shape:
                        raise ValueError("Mask shape mismatch")
                    union = np.logical_or(pred, mask).sum()
                    iou = (
                        float(np.logical_and(pred, mask).sum() / union) if union else 0
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best = k
                matched = best is not None and best_iou >= threshold
                tp.append(int(matched))
                if matched:
                    used.add((i, best))
            cumulative = np.cumsum(tp)
            recall = cumulative / count
            precision = cumulative / np.arange(1, len(tp) + 1)
            aps.append(
                float(
                    np.mean(
                        [
                            max(precision[recall >= r], default=0)
                            for r in np.linspace(0, 1, 101)
                        ]
                    )
                )
            )
        results[str(c)] = dict(ap=float(np.mean(aps)), ap50=aps[0], ground_truth=count)
    valid = [v for v in results.values() if v is not None]
    return dict(
        mask_map=float(np.mean([v["ap"] for v in valid])) if valid else None,
        mask_ap50=float(np.mean([v["ap50"] for v in valid])) if valid else None,
        per_class=results,
        protocol="greedy_class_aware_101_point_no_crowd_no_area_ranges",
    )
