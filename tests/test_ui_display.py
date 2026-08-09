"""Tests for the UI display layer (pure functions, no Streamlit needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from visionforge.metrics import evaluate
from visionforge.regression import compare_runs
from visionforge.schema import GroundTruth
from visionforge.ui.display import (
    aggregate_row,
    category_names,
    compare_files,
    confusion_frame,
    delta_frame,
    evaluate_files,
    format_score,
    iou_histogram,
    per_class_frame,
    per_image_frame,
)

GT_PATH = "examples/gt_sample.json"
PREDS_A_PATH = "examples/preds_run_a.json"
PREDS_B_PATH = "examples/preds_run_b.json"


@pytest.fixture(scope="module")
def gt():
    return GroundTruth.model_validate(json.load(open(GT_PATH)))


def make_gt() -> GroundTruth:
    """Tiny hand-built GT: 1 image, 2 cats (person, car), 3 boxes."""
    return GroundTruth(
        images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10]},
        ],
        categories=[
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
        ],
    )


class TestFormatScore:
    def test_normal(self):
        assert format_score(0.314159) == "0.314"

    def test_none(self):
        assert format_score(None) == "n/a"

    def test_no_gt(self):
        assert format_score(-1.0) == "n/a"


class TestPerClassFrame:
    def test_columns_and_names(self):
        gt = make_gt()
        preds = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10], "score": 0.8},
        ]
        from visionforge.schema import PredictionsFile

        metrics = evaluate(PredictionsFile(predictions=preds).predictions, gt)
        df = per_class_frame(metrics["per_class"], category_names(gt))
        assert list(df.columns) == [
            "class_id", "class", "ap50", "precision", "recall", "f1", "tp", "fp", "fn",
        ]
        assert df["class"].tolist() == ["car", "person"]  # sorted by AP desc
        # person has 2 GT, 1 TP → recall 0.5; car has 1 GT, 1 TP → recall 1.0
        assert df.loc[df["class"] == "car", "recall"].iloc[0] == "1.000"
        assert df.loc[df["class"] == "person", "recall"].iloc[0] == "0.500"

    def test_unknown_id_falls_back(self):
        df = per_class_frame({99: {"ap50": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1}}, {})
        assert df.iloc[0]["class"] == "id 99"

    def test_sorted_by_ap_desc(self):
        df = per_class_frame(
            {
                1: {"ap50": 0.2, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1},
                2: {"ap50": 0.9, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1},
            },
            {1: "low", 2: "high"},
        )
        assert df["class"].tolist() == ["high", "low"]

    def test_no_gt_class_sorted_last(self):
        """-1 (no GT) must sort AFTER real values, not first (string-sort trap)."""
        df = per_class_frame(
            {
                1: {"ap50": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
                2: {"ap50": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1},
            },
            {1: "no_gt", 2: "has_gt"},
        )
        assert df["class"].tolist() == ["has_gt", "no_gt"]
        assert df.iloc[1]["ap50"] == "n/a"


class TestPerImageFrame:
    def test_sorts_by_fn_desc(self):
        df = per_image_frame({1: {"tp": 1, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 3}})
        assert df["image_id"].tolist() == [2, 1]

    def test_file_names(self):
        df = per_image_frame({1: {"tp": 1, "fp": 0, "fn": 0}}, images=[{"id": 1, "file_name": "x.jpg"}])
        assert df.iloc[0]["file_name"] == "x.jpg"


class TestConfusionFrame:
    def test_labels_with_background(self):
        gt = make_gt()
        names = category_names(gt)
        # person row has a TP, car col has a TP → both active; background present
        cf = confusion_frame(
            np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=int), names, gt.categories
        )
        assert list(cf.index) == ["person", "car", "background"]
        assert list(cf.columns) == ["person", "car", "background"]
        assert cf.loc["person", "person"] == 1

    def test_inactive_class_dropped(self):
        gt = make_gt()
        names = category_names(gt)
        # car has no GT and no predictions → dropped from the frame
        cf = confusion_frame(
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int), names, gt.categories
        )
        assert list(cf.index) == ["person", "background"]

    def test_phantom_only_class_kept(self):
        gt = make_gt()
        names = category_names(gt)
        # car predicted only as a phantom (background row, car col) → kept
        cf = confusion_frame(
            np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=int), names, gt.categories
        )
        assert list(cf.index) == ["person", "car", "background"]
        assert cf.loc["background", "car"] == 1

    def test_all_fn_class_kept(self):
        """A class whose GT was entirely missed must NOT vanish from the matrix.

        Its counts live in the background (FN) column — the row-sum filter
        must include that column, or the matrix hides exactly the classes
        that broke in a regression.
        """
        gt = make_gt()
        names = category_names(gt)
        # car has GT but every one missed → its row is [0, 0, 1] (FN col)
        cf = confusion_frame(
            np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=int), names, gt.categories
        )
        assert list(cf.index) == ["person", "car", "background"]
        assert cf.loc["car", "background"] == 1


class TestIouHistogram:
    def test_counts_sum_to_input(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df = iou_histogram(values, bins=10)
        assert df["count"].sum() == len(values)

    def test_empty(self):
        df = iou_histogram([], bins=10)
        assert df.empty

    def test_bin_count(self):
        df = iou_histogram([0.5] * 5, bins=20)
        assert len(df) == 20


class TestDeltaFrame:
    def test_status_badges(self):
        compare = {
            "deltas": {
                1: {"delta_map50": -0.1, "delta_f1": -0.2, "regressed": True},
                2: {"delta_map50": 0.2, "delta_f1": 0.1, "regressed": False},
                3: {"delta_map50": 0.01, "delta_f1": 0.0, "regressed": False},
                4: {"delta_map50": None, "delta_f1": None, "regressed": False},
            }
        }
        df = delta_frame(compare, {1: "a", 2: "b", 3: "c", 4: "d"})
        statuses = dict(zip(df["class"], df["status"]))
        assert statuses == {"a": "regressed", "b": "improved", "c": "ok", "d": "n/a"}

    def test_regressed_sorted_first(self):
        compare = {
            "deltas": {
                1: {"delta_map50": 0.1, "delta_f1": 0.1, "regressed": False},
                2: {"delta_map50": -0.1, "delta_f1": -0.1, "regressed": True},
            }
        }
        df = delta_frame(compare, {1: "ok_class", 2: "bad_class"})
        assert df["class"].tolist() == ["bad_class", "ok_class"]

    def test_negative_delta_not_hidden(self):
        """Negatives are real deltas, not the 'no GT' sentinel (-1)."""
        compare = {
            "deltas": {
                1: {"delta_map50": -0.577, "delta_f1": -0.25, "regressed": True},
            }
        }
        df = delta_frame(compare, {1: "person"})
        row = df.iloc[0]
        assert row["delta_map50"] == "-0.577"
        assert row["delta_f1"] == "-0.250"
        assert row["status"] == "regressed"


class TestAggregateRow:
    def test_rows(self):
        agg = {"map50": 0.5, "map": 0.4, "precision": 0.6, "recall": 0.7, "f1": 0.65, "tp": 3, "fp": 2, "fn": 1}
        df = aggregate_row(agg)
        assert len(df) == 6
        assert df["metric"].tolist() == ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "TP / FP / FN"]
        assert df.loc[df["metric"] == "mAP@0.5", "value"].iloc[0] == "0.500"


class TestEvaluateFiles:
    def test_sample_files_load(self):
        metrics = evaluate_files(GT_PATH, PREDS_A_PATH)
        assert "aggregate" in metrics
        assert metrics["aggregate"]["map50"] > 0

    def test_compare_files_sample(self):
        result = compare_files(GT_PATH, PREDS_A_PATH, PREDS_B_PATH)
        assert result["compare"]["verdict"] == "REGRESSED"
        assert "person" in result["names"].values()

    def test_bad_gt_path(self):
        with pytest.raises(ValueError):
            evaluate_files("does/not/exist.json", PREDS_A_PATH)
