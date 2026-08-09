"""One-time cross-check: our numpy COCO protocol vs pycocotools reference.

This validates that the numpy implementation matches the canonical COCO
implementation on the shared fixture. Skipped automatically when pycocotools
is not installed (it is an optional validation dependency, not required).
"""

import json

import numpy as np
import pytest

pycocotools = pytest.importorskip("pycocotools")

from visionforge.metrics import evaluate
from visionforge.schema import GroundTruth, PredictionsFile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_coco_format(gt: GroundTruth, preds: PredictionsFile):
    # pycocotools requires iscrowd + area on GT annotations; our schema
    # doesn't, so synthesize them (area = w*h, the COCO default).
    annotations = []
    for a in gt.annotations:
        d = a.model_dump()
        d["iscrowd"] = 0
        d["area"] = d["bbox"][2] * d["bbox"][3]
        annotations.append(d)
    gt_dict = {
        "images": [i.model_dump() for i in gt.images],
        "annotations": annotations,
        "categories": [c.model_dump() for c in gt.categories],
    }
    pred_list = [p.model_dump() for p in preds.predictions]
    return gt_dict, pred_list


def _coco_ap50(gt_dict, pred_list):
    """Reference mAP@0.5 via pycocotools."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(pred_list)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0]


class TestPycocotoolsCrossCheck:
    def test_map50_matches_reference(self, gt, preds_full):
        gt_dict, pred_list = _to_coco_format(gt, preds_full)
        reference_ap50 = _coco_ap50(gt_dict, pred_list)

        metrics = evaluate(preds_full.predictions, gt)
        our_ap50 = metrics["aggregate"]["map50"]

        assert our_ap50 == pytest.approx(reference_ap50, abs=0.01), (
            f"our AP@0.5 ({our_ap50:.4f}) diverges from pycocotools ({reference_ap50:.4f})"
        )

    def test_map50_matches_reference_missing(self, gt, preds_missing):
        gt_dict, pred_list = _to_coco_format(gt, preds_missing)
        reference_ap50 = _coco_ap50(gt_dict, pred_list)

        metrics = evaluate(preds_missing.predictions, gt)
        our_ap50 = metrics["aggregate"]["map50"]

        assert our_ap50 == pytest.approx(reference_ap50, abs=0.01)
