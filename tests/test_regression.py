"""Regression tests — run A vs run B comparison (hand-computed).

Using the shared fixture: full preds (run A) has TP=2, FP=1, FN=0 at IoU 0.5.
preds_missing (run B) has only p1+p2 → TP=1, FP=1, FN=1.

Hand-computed:
- Run A: mAP@0.5 = 253/303 ≈ 0.8350, F1 = 0.8
- Run B: mAP@0.5 = 51/101 ≈ 0.5050, F1 = 0.5
- delta mAP@0.5 = 0.5050 - 0.8350 = -0.3300 < -0.05 → REGRESSED
"""

import pytest

from visionforge.metrics import evaluate
from visionforge.regression import compare_runs


def _metrics(preds, gt):
    return evaluate(preds, gt)


class TestCompareRuns:
    def test_identical_runs_pass(self, gt, preds_full):
        run_a = _metrics(preds_full.predictions, gt)
        result = compare_runs(run_a, run_a)
        assert result["verdict"] == "PASS"
        assert all(abs(d["delta_map50"]) < 1e-9 for d in result["deltas"].values())

    def test_regression_detected(self, gt, preds_full, preds_missing):
        run_a = _metrics(preds_full.predictions, gt)
        run_b = _metrics(preds_missing.predictions, gt)
        result = compare_runs(run_a, run_b)
        assert result["verdict"] == "REGRESSED"
        assert 1 in result["deltas"]
        assert result["deltas"][1]["delta_map50"] == pytest.approx(51 / 101 - 253 / 303)
        assert result["deltas"][1]["regressed"] is True

    def test_improvement_passes(self, gt, preds_full, preds_missing):
        run_a = _metrics(preds_missing.predictions, gt)  # worse baseline
        run_b = _metrics(preds_full.predictions, gt)     # better candidate
        result = compare_runs(run_a, run_b)
        assert result["verdict"] == "PASS"
        assert result["deltas"][1]["delta_map50"] == pytest.approx(253 / 303 - 51 / 101)

    def test_threshold_boundary(self, gt, preds_full, preds_missing):
        run_a = _metrics(preds_full.predictions, gt)
        run_b = _metrics(preds_missing.predictions, gt)
        # delta is -0.33; with a huge threshold (1.0) it should NOT flag
        result = compare_runs(run_a, run_b, regression_threshold=1.0)
        assert result["verdict"] == "PASS"
        # with a tiny threshold (0.01) it flags
        result = compare_runs(run_a, run_b, regression_threshold=0.01)
        assert result["verdict"] == "REGRESSED"

    def test_class_disappears_in_run_b(self, gt, preds_full, preds_missing):
        """Class 2 has GT; run A detects it, run B has zero predictions for it.

        A disappeared class with GT is a REAL regression (AP 1.0 → 0.0),
        not an undefined delta — the delta must be -1.0 and flagged.
        """
        from visionforge.schema import GroundTruth, PredictionsFile

        # Multi-class GT: class 1 (person) + class 2 (car) with GT on img2
        gt2 = GroundTruth(
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20]},
            ],
            categories=[{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        )
        # Run A predicts both classes, run B predicts only class 1
        run_a_preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.7},
        ])
        run_b_preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        ])
        run_a = _metrics(run_a_preds.predictions, gt2)
        run_b = _metrics(run_b_preds.predictions, gt2)
        result = compare_runs(run_a, run_b)
        # Class 2: run A AP=1.0 (perfect match), run B AP=0.0 (no preds)
        assert result["verdict"] == "REGRESSED"
        assert 2 in result["deltas"]
        assert result["deltas"][2]["delta_map50"] == pytest.approx(-1.0)
        assert result["deltas"][2]["regressed"] is True
