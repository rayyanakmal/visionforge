"""Shared test fixtures: tiny hand-computed GT + predictions.

Fixture geometry (all boxes [x, y, w, h]):
- Image 1 GT:  [10, 10, 20, 20]  → box spans (10,10)-(30,30)
- Image 2 GT:  [50, 50, 20, 20]  → box spans (50,50)-(70,70)

Predictions:
- p1: img1 [10,10,20,20] score 0.9  → IoU 1.0 vs img1 GT (perfect match)
- p2: img1 [15,15,20,20] score 0.8  → IoU 0.391 vs img1 GT (< 0.5, false positive)
- p3: img2 [50,50,20,20] score 0.7  → IoU 1.0 vs img2 GT (perfect match)

Hand-computed expectations at IoU 0.5:
- TP=2 (p1, p3), FP=1 (p2), FN=0
- P = 2/3 ≈ 0.6667, R = 2/2 = 1.0, F1 = 2*P*R/(P+R) = 0.8
- AP@0.5 = 253/303 ≈ 0.8350 (101-pt interpolation, see test_metrics)
- Per-image: img1 TP=1 FP=1 FN=0; img2 TP=1 FP=0 FN=0
- Confusion (nc+1=2): [[2, 0], [1, 0]]
"""

import pytest
from visionforge.schema import GroundTruth, PredictionsFile


@pytest.fixture
def gt() -> GroundTruth:
    return GroundTruth(
        images=[
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 20, 20]},
        ],
        categories=[{"id": 1, "name": "person"}],
    )


@pytest.fixture
def preds_full() -> PredictionsFile:
    """All three predictions: 2 TP + 1 FP at IoU 0.5."""
    return PredictionsFile(
        predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [15, 15, 20, 20], "score": 0.8},
            {"image_id": 2, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.7},
        ]
    )


@pytest.fixture
def preds_missing() -> PredictionsFile:
    """Only p1 + p2: 1 TP + 1 FP, 1 GT missed (FN)."""
    return PredictionsFile(
        predictions=[
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [15, 15, 20, 20], "score": 0.8},
        ]
    )


@pytest.fixture
def preds_empty() -> PredictionsFile:
    return PredictionsFile(predictions=[])
