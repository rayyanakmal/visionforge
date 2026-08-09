"""Greedy IoU matching between predictions and ground truth (COCO convention).

COCO matching rules (pycocotools, BSD):
- Detections sorted by score descending (mergesort for consistency)
- Per (image, category): each detection matched to the best available GT
  (highest IoU) above the threshold; each GT used at most once
- IoU >= threshold counts as a match (threshold boundary inclusive)
"""

from dataclasses import dataclass, field

from visionforge.schema import GroundTruth, Prediction


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between two [x, y, w, h] boxes (COCO format)."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    """Outcome of greedy matching at one IoU threshold."""

    matches: list[tuple[int, int]] = field(default_factory=list)  # (pred_idx, gt_idx)
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def true_positives(self) -> int:
        return len(self.matches)


def _match_one_group(
    preds: list[Prediction],
    gt_boxes: list[list[float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int]]:
    """Greedy match one (image, category) group.

    Returns (matches, matched_pred_indices). GT indices are used at most once.
    """
    # Sort preds by score descending (stable)
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    gt_used: set[int] = set()
    matches: list[tuple[int, int]] = []
    matched_preds: set[int] = set()

    for pi in order:
        best_gt = -1
        best_iou = iou_threshold
        for gi, gbox in enumerate(gt_boxes):
            if gi in gt_used:
                continue
            iou = compute_iou(preds[pi].bbox, gbox)
            # COCO caps match IoU at min(t, 1-1e-10) — any IoU >= t counts
            if iou >= iou_threshold and iou >= best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0:
            gt_used.add(best_gt)
            matches.append((pi, best_gt))
            matched_preds.add(pi)

    return matches, matched_preds


def match_predictions(
    preds: list[Prediction],
    gt: GroundTruth,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy per-category matching across all images.

    Groups predictions and GT by (image_id, category_id), matches each group
    independently (COCO convention), aggregates matches + FP/FN counts.
    """
    # Group GT by (image_id, category_id)
    gt_groups: dict[tuple[int, int], list[list[float]]] = {}
    for ann in gt.annotations:
        key = (ann.image_id, ann.category_id)
        gt_groups.setdefault(key, []).append(ann.bbox)

    # Group preds by (image_id, category_id)
    pred_groups: dict[tuple[int, int], list[Prediction]] = {}
    for p in preds:
        key = (p.image_id, p.category_id)
        pred_groups.setdefault(key, []).append(p)

    result = MatchResult()
    all_keys = set(gt_groups) | set(pred_groups)

    for key in all_keys:
        group_preds = pred_groups.get(key, [])
        group_gt = gt_groups.get(key, [])
        matches, matched_preds = _match_one_group(group_preds, group_gt, iou_threshold)

        # Translate group-relative indices back to global pred indices
        # (preds in the group are the same objects, so indices are stable)
        for pi, gi in matches:
            result.matches.append((pi, gi))

        result.false_positives += len(group_preds) - len(matched_preds)
        result.false_negatives += len(group_gt) - len(matches)

    return result
