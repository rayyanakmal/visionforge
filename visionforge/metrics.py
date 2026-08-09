"""Detection metrics: AP (COCO 101-pt protocol), precision/recall/F1, IoU stats, confusion.

Implements the COCO evaluation protocol (Lin et al. 2014, pycocotools BSD)
in pure numpy — no Cython, no torch. Cross-checked against pycocotools once
in tests (see test_coco_crosscheck if present).

Protocol details:
- AP uses 101-point recall interpolation (recall thresholds 0:0.01:1)
- Precision at each recall threshold = max precision over all points with recall >= threshold
- mAP@0.5:0.95 averages AP over 10 IoU thresholds [0.5:0.95] and all classes
- Classes with zero GT are excluded from mAP (COCO sets precision to -1 there)
"""

import numpy as np

from visionforge.matching import compute_iou, match_predictions
from visionforge.schema import GroundTruth, Prediction

COCO_IOU_SWEEP = [0.5 + 0.05 * i for i in range(10)]  # [0.5, 0.55, ..., 0.95]


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Average precision via 101-point recall interpolation (COCO protocol)."""
    if len(precisions) == 0:
        return 0.0
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interp = np.zeros(101)
    for i, r in enumerate(recall_thresholds):
        mask = recalls >= r
        if mask.any():
            interp[i] = precisions[mask].max()
    return float(interp.mean())


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts. Zero counts → zero (not NaN)."""
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_iou_stats(iou_values: list[float]) -> dict:
    """IoU distribution stats across matched detections."""
    if not iou_values:
        return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "count": 0}
    arr = np.array(iou_values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "count": len(arr),
    }


def _group_gt_by_image(gt: GroundTruth) -> dict[int, list]:
    """GT annotations grouped by image_id."""
    groups: dict[int, list] = {}
    for ann in gt.annotations:
        groups.setdefault(ann.image_id, []).append(ann)
    return groups


def _group_preds_by_image(preds: list[Prediction]) -> dict[int, list[Prediction]]:
    groups: dict[int, list[Prediction]] = {}
    for p in preds:
        groups.setdefault(p.image_id, []).append(p)
    return groups


def _class_pr_curve(
    class_preds: list[Prediction],
    class_gt_boxes: dict[int, list[list[float]]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precision/recall curve for one class at one IoU threshold.

    Matches per (image, category) — a prediction can only match a GT box
    in the SAME image (COCO protocol). A match requires IoU >= iou_threshold.
    Sorts predictions by score desc across all images, accumulates TP/FP,
    returns (precisions, recalls).
    """
    if not class_preds:
        return np.array([]), np.array([])

    # Group preds by image for per-image matching
    preds_by_image: dict[int, list[Prediction]] = {}
    for p in class_preds:
        preds_by_image.setdefault(p.image_id, []).append(p)

    # Per-image greedy match: pred -> GT index (in that image's GT list)
    matched_pred_ids: set[int] = set()  # id() of matched predictions

    for image_id, image_preds in preds_by_image.items():
        gt_boxes = class_gt_boxes.get(image_id, [])
        gt_used: set[int] = set()
        order = sorted(range(len(image_preds)), key=lambda i: image_preds[i].score, reverse=True)
        for pi in order:
            best_gt = -1
            best_iou = iou_threshold
            for gi, gbox in enumerate(gt_boxes):
                if gi in gt_used:
                    continue
                iou = compute_iou(image_preds[pi].bbox, gbox)
                if iou >= iou_threshold and iou >= best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0:
                gt_used.add(best_gt)
                matched_pred_ids.add(id(image_preds[pi]))

    total_gt = sum(len(boxes) for boxes in class_gt_boxes.values())

    # Sort ALL class predictions by score desc for the PR curve
    order = sorted(range(len(class_preds)), key=lambda i: class_preds[i].score, reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []

    for pi in order:
        if id(class_preds[pi]) in matched_pred_ids:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum) if tp_cum + fp_cum > 0 else 0.0)
        recalls.append(tp_cum / total_gt if total_gt > 0 else 0.0)

    return np.array(precisions), np.array(recalls)


def _compute_ap_at_threshold(
    class_preds: list[Prediction],
    class_gt_boxes: dict[int, list[list[float]]],
    iou_threshold: float,
) -> float:
    """AP for one class at one IoU threshold (COCO protocol)."""
    if not class_gt_boxes or sum(len(v) for v in class_gt_boxes.values()) == 0:
        return -1.0  # no GT → excluded from mAP (COCO convention)
    precisions, recalls = _class_pr_curve(class_preds, class_gt_boxes, iou_threshold)
    return compute_ap(precisions, recalls)


def compute_confusion(preds: list[Prediction], gt: GroundTruth, iou_threshold: float = 0.5) -> np.ndarray:
    """Confusion matrix with background row/col ((nc+1, nc+1)).

    Row = GT class, col = predicted class. Diagonal = correct matches.
    Last row = background (phantom detections / false positives).
    Last col = missed GT (false negatives).
    """
    nc = len(gt.categories)
    matrix = np.zeros((nc + 1, nc + 1), dtype=int)
    cat_to_idx = {c.id: i for i, c in enumerate(gt.categories)}

    gt_by_image = _group_gt_by_image(gt)
    preds_by_image = _group_preds_by_image(preds)

    for image_id, image_preds in preds_by_image.items():
        image_gt = gt_by_image.get(image_id, [])
        if not image_gt:
            for p in image_preds:
                matrix[nc][cat_to_idx.get(p.category_id, nc)] += 1
            continue

        gt_used: set[int] = set()
        for p in image_preds:
            best_gt = -1
            best_iou = iou_threshold
            for gi, ann in enumerate(image_gt):
                if gi in gt_used:
                    continue
                iou = compute_iou(p.bbox, ann.bbox)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0:
                gt_used.add(best_gt)
                gt_row = cat_to_idx.get(image_gt[best_gt].category_id, nc)
                pred_col = cat_to_idx.get(p.category_id, nc)
                matrix[gt_row][pred_col] += 1
            else:
                pred_col = cat_to_idx.get(p.category_id, nc)
                matrix[nc][pred_col] += 1  # phantom detection

        for gi, ann in enumerate(image_gt):
            if gi not in gt_used:
                gt_row = cat_to_idx.get(ann.category_id, nc)
                matrix[gt_row][nc] += 1  # missed GT

    return matrix


def evaluate(
    preds: list[Prediction],
    gt: GroundTruth,
    iou_thresholds: list[float] | None = None,
) -> dict:
    """Full evaluation: aggregate + per-class + per-image + IoU stats + confusion.

    Returns:
        {
          "aggregate": {map50, map, precision, recall, f1, tp, fp, fn},
          "per_class": {cat_id: {ap50, precision, recall, f1, tp, fp, fn}},
          "per_image": {image_id: {tp, fp, fn}},
          "iou_stats": {mean, median, p50, p90, count},
          "confusion": np.ndarray (nc+1, nc+1),
        }
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    gt_by_image = _group_gt_by_image(gt)
    preds_by_image = _group_preds_by_image(preds)
    cat_ids = [c.id for c in gt.categories]

    # Per-class at IoU 0.5 (P/R/F1 + counts) and AP at requested thresholds
    per_class: dict[int, dict] = {}
    class_aps_50: list[float] = []
    class_aps_sweep: list[float] = []

    for cat_id in cat_ids:
        class_preds = [p for p in preds if p.category_id == cat_id]
        class_gt_anns = [ann for ann in gt.annotations if ann.category_id == cat_id]
        # GT boxes grouped by image (per-image matching, COCO protocol)
        class_gt_by_image: dict[int, list[list[float]]] = {}
        for ann in class_gt_anns:
            class_gt_by_image.setdefault(ann.image_id, []).append(ann.bbox)

        # P/R/F1 at IoU 0.5 using all predictions (counts from match).
        # Filter GT to this category so other classes' boxes don't count as FN.
        class_gt_filtered = GroundTruth(
            images=gt.images,
            annotations=class_gt_anns,
            categories=gt.categories,
        )
        match_result = match_predictions(class_preds, class_gt_filtered, iou_threshold=0.5)
        tp = match_result.true_positives
        fp = match_result.false_positives
        fn = match_result.false_negatives
        precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)

        ap50 = _compute_ap_at_threshold(class_preds, class_gt_by_image, 0.5)
        ap_values = {
            "ap50": ap50,
        }
        for t in iou_thresholds:
            if t != 0.5:
                ap_values[f"ap{t:g}"] = _compute_ap_at_threshold(class_preds, class_gt_by_image, t)

        per_class[cat_id] = {
            "ap50": ap50,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        if ap50 >= 0:
            class_aps_50.append(ap50)
        # Sweep APs for mAP@0.5:0.95
        sweep_aps = [_compute_ap_at_threshold(class_preds, class_gt_by_image, t) for t in COCO_IOU_SWEEP]
        valid_sweep = [a for a in sweep_aps if a >= 0]
        if valid_sweep:
            class_aps_sweep.append(float(np.mean(valid_sweep)))

    # Aggregate P/R/F1 (macro-average of per-class at IoU 0.5, matching match_predictions counts)
    total_tp = sum(pc["tp"] for pc in per_class.values())
    total_fp = sum(pc["fp"] for pc in per_class.values())
    total_fn = sum(pc["fn"] for pc in per_class.values())
    agg_precision, agg_recall, agg_f1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)

    aggregate = {
        "map50": float(np.mean(class_aps_50)) if class_aps_50 else 0.0,
        "map": float(np.mean(class_aps_sweep)) if class_aps_sweep else 0.0,
        "precision": agg_precision,
        "recall": agg_recall,
        "f1": agg_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    # Per-image TP/FP/FN at IoU 0.5
    per_image: dict[int, dict] = {}
    for image_id in set(gt_by_image) | set(preds_by_image):
        image_preds = preds_by_image.get(image_id, [])
        image_gt = gt_by_image.get(image_id, [])
        # Build a GT filtered to this image so other images' boxes don't
        # count as false negatives here.
        image_gt_filtered = GroundTruth(
            images=[i for i in gt.images if i.id == image_id],
            annotations=image_gt,
            categories=gt.categories,
        )
        match_result = match_predictions(image_preds, image_gt_filtered, iou_threshold=0.5)
        per_image[image_id] = {
            "tp": match_result.true_positives,
            "fp": match_result.false_positives,
            "fn": match_result.false_negatives,
        }

    # IoU stats at 0.5 (matched pair IoUs)
    iou_values: list[float] = []
    for image_id, image_preds in preds_by_image.items():
        image_gt = gt_by_image.get(image_id, [])
        gt_used: set[int] = set()
        for p in image_preds:
            best_gt = -1
            best_iou = 0.5
            for gi, ann in enumerate(image_gt):
                if gi in gt_used:
                    continue
                iou = compute_iou(p.bbox, ann.bbox)
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0:
                gt_used.add(best_gt)
                iou_values.append(best_iou)
    iou_stats = compute_iou_stats(iou_values)

    confusion = compute_confusion(preds, gt, iou_threshold=0.5)

    return {
        "aggregate": aggregate,
        "per_class": per_class,
        "per_image": per_image,
        "iou_stats": iou_stats,
        "confusion": confusion,
    }
