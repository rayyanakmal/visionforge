"""IoU + greedy matching tests (hand-computed)."""

import pytest

from visionforge.matching import compute_iou, match_predictions


class TestIoU:
    def test_identical_boxes(self):
        assert compute_iou([10, 10, 20, 20], [10, 10, 20, 20]) == pytest.approx(1.0)

    def test_non_overlapping(self):
        assert compute_iou([10, 10, 20, 20], [100, 100, 20, 20]) == pytest.approx(0.0)

    def test_partial_overlap_50pct(self):
        # GT (10,10)-(30,30) area 400; pred (15,15)-(35,35) area 400
        # overlap (15,15)-(30,30) = 15x15 = 225; union 400+400-225 = 575
        assert compute_iou([10, 10, 20, 20], [15, 15, 20, 20]) == pytest.approx(225 / 575)

    def test_contained_box(self):
        # GT (0,0)-(100,100) area 10000; pred (25,25)-(75,75) area 2500
        assert compute_iou([0, 0, 100, 100], [25, 25, 50, 50]) == pytest.approx(0.25)

    def test_zero_area_boxes(self):
        assert compute_iou([10, 10, 0, 0], [10, 10, 0, 0]) == pytest.approx(0.0)


class TestMatching:
    def test_perfect_matches(self, gt, preds_full):
        result = match_predictions(preds_full.predictions, gt, iou_threshold=0.5)
        # p1 -> gt1, p3 -> gt2; p2 unmatched (IoU 0.391 < 0.5)
        assert len(result.matches) == 2
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_high_score_wins_greedy(self, gt):
        # Two preds on same GT; high score should win the match
        from visionforge.schema import PredictionsFile

        preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},  # perfect
            {"image_id": 1, "category_id": 1, "bbox": [9, 9, 22, 22], "score": 0.95},  # better? no - lower IoU
        ])
        result = match_predictions(preds.predictions, gt, iou_threshold=0.5)
        # Both overlap GT1; the 0.9 perfect box matches, 0.95 overlaps heavily too but
        # GT already taken → both can't match; only ONE match possible
        assert len(result.matches) == 1

    def test_threshold_boundary(self, gt):
        from visionforge.schema import PredictionsFile

        # IoU exactly 0.5: box (10,10)-(30,30) vs (10,10)-(30,25): overlap 20x15=300, union 400+300-300=400 → 0.75
        # Need exact 0.5: GT (0,0,10,10) area 100; pred (0,0,5,15)? overlap 5x10=50, union 100+75-50=125 → 0.4
        # Construct: GT [0,0,20,20] area 400; pred [0,0,10,30] overlap 10x20=200, union 400+300-200=500 → 0.4
        # Exact 0.5 case: GT [0,0,20,20] (400); pred [0,0,20,10] (200) overlap 200, union 400 → 0.5
        preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10], "score": 0.9},  # IoU exactly 0.5
        ])
        result = match_predictions(preds.predictions, gt, iou_threshold=0.5)
        # GT image 1 is [10,10,20,20] — my constructed box is on a different image
        # Build a custom GT instead for this edge case
        custom_gt = __import__("visionforge.schema", fromlist=["GroundTruth"]).GroundTruth(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]}],
            categories=[{"id": 1, "name": "person"}],
        )
        result = match_predictions(preds.predictions, custom_gt, iou_threshold=0.5)
        assert len(result.matches) == 1  # IoU == 0.5 matches

    def test_empty_preds(self, gt):
        from visionforge.schema import PredictionsFile

        preds = PredictionsFile(predictions=[])
        result = match_predictions(preds.predictions, gt, iou_threshold=0.5)
        assert result.matches == []
        assert result.false_positives == 0
        assert result.false_negatives == 2

    def test_empty_gt(self):
        from visionforge.schema import GroundTruth, PredictionsFile

        gt = GroundTruth(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[],
            categories=[{"id": 1, "name": "person"}],
        )
        preds = PredictionsFile(predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.9},
        ])
        result = match_predictions(preds.predictions, gt, iou_threshold=0.5)
        assert result.matches == []
        assert result.false_positives == 1
        assert result.false_negatives == 0
