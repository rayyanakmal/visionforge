"""Metrics tests — golden values hand-computed from the shared fixture.

Fixture recap (see conftest): 2 images, class 1 (person), GT boxes on each.
Preds: p1 (img1, IoU 1.0, score .9), p2 (img1, IoU 0.391, score .8), p3 (img2, IoU 1.0, score .7).

Hand-computed at IoU 0.5 (full preds):
- TP=2, FP=1, FN=0 → P=2/3, R=1.0, F1=0.8
- AP@0.5 = 253/303 ≈ 0.8350 (101-pt interpolation, see compute_ap)
- mAP@0.5:0.95 = 253/303 (all sweep thresholds give identical matching for this fixture)
- per-image: img1 {TP:1, FP:1, FN:0}; img2 {TP:1, FP:0, FN:0}
- iou_stats: mean=1.0, median=1.0 (both matches are perfect)
- confusion (nc+1=2): [[2, 0], [1, 0]]

Hand-computed with preds_missing (p1 + p2 only):
- TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
- AP@0.5 = 51/101 ≈ 0.5050
"""

import numpy as np
import pytest

from visionforge.metrics import (
    compute_ap,
    compute_confusion,
    compute_iou_stats,
    compute_precision_recall_f1,
    evaluate,
)


class TestComputeAP:
    def test_perfect_detection_ap_one(self):
        # One perfect match: PR point (R=1.0, P=1.0) → AP 1.0
        assert compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)

    def test_fixture_ap(self):
        # PR points: (R=0.5, P=1.0), (R=0.5, P=0.5), (R=1.0, P=2/3)
        precisions = np.array([1.0, 0.5, 2 / 3])
        recalls = np.array([0.5, 0.5, 1.0])
        assert compute_ap(precisions, recalls) == pytest.approx(253 / 303)

    def test_empty_ap_zero(self):
        assert compute_ap(np.array([]), np.array([])) == pytest.approx(0.0)


class TestPRF1:
    def test_fixture_counts(self):
        p, r, f1 = compute_precision_recall_f1(tp=2, fp=1, fn=0)
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(0.8)

    def test_zero_counts(self):
        p, r, f1 = compute_precision_recall_f1(tp=0, fp=0, fn=0)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_no_false_positives(self):
        p, r, f1 = compute_precision_recall_f1(tp=2, fp=0, fn=1)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(2 / 3)
        assert f1 == pytest.approx(0.8)


class TestIoUStats:
    def test_perfect_matches(self):
        stats = compute_iou_stats([1.0, 1.0])
        assert stats["mean"] == pytest.approx(1.0)
        assert stats["median"] == pytest.approx(1.0)

    def test_empty(self):
        stats = compute_iou_stats([])
        assert stats["mean"] == pytest.approx(0.0)


class TestConfusion:
    def test_fixture_confusion(self, gt, preds_full):
        matrix = compute_confusion(preds_full.predictions, gt)
        assert matrix.shape == (2, 2)  # nc+1
        assert matrix[0][0] == 2  # person correctly detected twice
        assert matrix[1][0] == 1  # phantom person (background row)


class TestEvaluate:
    def test_full_fixture(self, gt, preds_full):
        result = evaluate(preds_full.predictions, gt)
        agg = result["aggregate"]
        assert agg["map50"] == pytest.approx(253 / 303)
        assert agg["map"] == pytest.approx(253 / 303)
        assert agg["precision"] == pytest.approx(2 / 3)
        assert agg["recall"] == pytest.approx(1.0)
        assert agg["f1"] == pytest.approx(0.8)

        per_class = result["per_class"]
        assert 1 in per_class
        assert per_class[1]["ap50"] == pytest.approx(253 / 303)
        assert per_class[1]["precision"] == pytest.approx(2 / 3)
        assert per_class[1]["recall"] == pytest.approx(1.0)

        per_image = result["per_image"]
        assert per_image[1] == {"tp": 1, "fp": 1, "fn": 0}
        assert per_image[2] == {"tp": 1, "fp": 0, "fn": 0}

        assert result["iou_stats"]["mean"] == pytest.approx(1.0)
        assert result["iou_values"] == [1.0, 1.0]
        assert result["confusion"].shape == (2, 2)

    def test_missing_fixture(self, gt, preds_missing):
        result = evaluate(preds_missing.predictions, gt)
        agg = result["aggregate"]
        assert agg["map50"] == pytest.approx(51 / 101)
        assert agg["precision"] == pytest.approx(0.5)
        assert agg["recall"] == pytest.approx(0.5)
        assert agg["f1"] == pytest.approx(0.5)

    def test_empty_preds(self, gt, preds_empty):
        result = evaluate(preds_empty.predictions, gt)
        agg = result["aggregate"]
        assert agg["map50"] == pytest.approx(0.0)
        assert agg["precision"] == pytest.approx(0.0)
        assert agg["recall"] == pytest.approx(0.0)
        # all GT missed
        assert result["aggregate"]["f1"] == pytest.approx(0.0)

    def test_empty_gt(self, preds_full):
        from visionforge.schema import GroundTruth

        gt = GroundTruth(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[],
            categories=[{"id": 1, "name": "person"}],
        )
        result = evaluate(preds_full.predictions, gt)
        assert result["aggregate"]["map50"] == pytest.approx(0.0)

    def test_threshold_sweep_differs(self):
        """AP@0.5 must differ from AP@0.5:0.95 when boxes have mid-range IoU.

        Regression guard for the bug where the IoU threshold was ignored in
        the PR curve (mAP@0.5 == mAP@0.5:0.95 on real data). A box with
        IoU 0.75 matches at 0.5 but NOT at 0.8 → APs must differ.
        """
        from visionforge.schema import GroundTruth, PredictionsFile

        gt = GroundTruth(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            categories=[{"id": 1, "name": "person"}],
        )
        # pred box [0, 0, 10, 7.5] → IoU with GT = 0.75 (area 75, overlap 75, union 100)
        preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 7.5], "score": 0.9},
        ])
        result = evaluate(preds.predictions, gt)
        # At IoU 0.5 and 0.75 it matches (AP=1.0); at 0.8+ it doesn't (AP=0)
        # mAP@0.5 = 1.0, mAP@0.5:0.95 = (6*1.0 + 4*0.0)/10 = 0.6
        assert result["aggregate"]["map50"] == pytest.approx(1.0)
        assert result["aggregate"]["map"] == pytest.approx(0.6)
        assert result["aggregate"]["map50"] > result["aggregate"]["map"]
