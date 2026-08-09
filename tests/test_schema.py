"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from visionforge.schema import GroundTruth, Prediction, PredictionsFile, validate_consistency


class TestGTValidation:
    def test_valid_gt_parses(self, gt):
        assert len(gt.images) == 2
        assert len(gt.annotations) == 2
        assert gt.categories[0].name == "person"

    def test_empty_annotations_allowed(self):
        gt = GroundTruth(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[],
            categories=[{"id": 1, "name": "person"}],
        )
        assert gt.annotations == []

    def test_invalid_bbox_length_rejected(self):
        with pytest.raises(ValidationError):
            GroundTruth(
                images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3]}],
                categories=[{"id": 1, "name": "person"}],
            )

    def test_negative_bbox_rejected(self):
        with pytest.raises(ValidationError):
            GroundTruth(
                images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [-1, 2, 3, 4]}],
                categories=[{"id": 1, "name": "person"}],
            )


class TestPredictionValidation:
    def test_valid_preds_parse(self, preds_full):
        assert len(preds_full.predictions) == 3

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Prediction(image_id=1, category_id=1, bbox=[1, 2, 3, 4], score=1.5)

    def test_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            Prediction(image_id=1, category_id=1, bbox=[1, 2, 3, 4], score=-0.1)

    def test_score_boundaries_accepted(self):
        p = Prediction(image_id=1, category_id=1, bbox=[1, 2, 3, 4], score=0.0)
        assert p.score == 0.0
        p = Prediction(image_id=1, category_id=1, bbox=[1, 2, 3, 4], score=1.0)
        assert p.score == 1.0


class TestConsistency:
    def test_consistent_no_warnings(self, gt, preds_full):
        warnings = validate_consistency(gt, preds_full)
        assert warnings == []

    def test_unknown_category_warns(self, gt, preds_full):
        # Prediction category 99 does not exist in GT categories
        preds_full.predictions[0].category_id = 99
        warnings = validate_consistency(gt, preds_full)
        assert any("99" in w for w in warnings)

    def test_unknown_image_warns(self, gt, preds_full):
        preds_full.predictions[0].image_id = 999
        warnings = validate_consistency(gt, preds_full)
        assert any("999" in w for w in warnings)

    def test_empty_preds_no_warnings(self, gt, preds_empty):
        assert validate_consistency(gt, preds_empty) == []
