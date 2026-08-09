"""Pure display helpers: metrics dicts → pandas DataFrames for the UI.

All functions here are Streamlit-free so they can be unit-tested without
launching a browser. Widgets in app.py stay thin and call these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visionforge.metrics import evaluate as evaluate_metrics
from visionforge.regression import compare_runs
from visionforge.schema import GroundTruth, PredictionsFile, load_ground_truth, load_predictions, validate_consistency


def category_names(gt: GroundTruth) -> dict[int, str]:
    """Map category_id → name for display."""
    return {c.id: c.name for c in gt.categories}


def format_score(value: float | None) -> str:
    """3-decimal format; -1 (no GT) or None renders as n/a."""
    if value is None:
        return "n/a"
    if value < 0:
        return "n/a"
    return f"{value:.3f}"


def format_delta(value: float | None) -> str:
    """3-decimal format for deltas — negatives are real values, not n/a."""
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def per_class_frame(per_class: dict[int, dict], names: dict[int, str]) -> pd.DataFrame:
    """Per-class metrics as a display DataFrame (sorted by AP desc, no-GT last)."""
    rows = []
    for cat_id, pc in per_class.items():
        rows.append(
            {
                "class_id": cat_id,
                "class": names.get(cat_id, f"id {cat_id}"),
                "_ap": pc["ap50"],  # numeric for sorting; -1 (no GT) goes last
                "ap50": format_score(pc["ap50"]),
                "precision": format_score(pc["precision"]),
                "recall": format_score(pc["recall"]),
                "f1": format_score(pc["f1"]),
                "tp": pc["tp"],
                "fp": pc["fp"],
                "fn": pc["fn"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_ap", ascending=False, na_position="last").drop(columns="_ap").reset_index(drop=True)
    return df


def per_image_frame(
    per_image: dict[int, dict], images: list[dict] | None = None
) -> pd.DataFrame:
    """Per-image TP/FP/FN as a DataFrame, with file names when known."""
    file_by_id = {}
    if images:
        for img in images:
            file_by_id[img.get("id")] = img.get("file_name", "")
    rows = []
    for img_id, counts in per_image.items():
        rows.append(
            {
                "image_id": img_id,
                "file_name": file_by_id.get(img_id, ""),
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("fn", ascending=False).reset_index(drop=True)
    return df


def confusion_frame(
    confusion: np.ndarray, names: dict[int, str], gt_categories: list
) -> pd.DataFrame:
    """Confusion matrix as a DataFrame with class names on both axes.

    Rows = GT class, cols = predicted class. Last row/col = background.
    Only categories with any GT or any prediction are shown (keeps the
    matrix readable when the dataset has 80+ categories but a few active).
    """
    nc = len(gt_categories)
    counts = np.asarray(confusion)
    # Full row sum > 0 → has GT (INCLUDING the background/FN column — a class
    # whose GT was entirely missed would otherwise vanish, hiding the classes
    # that broke). Full col sum > 0 → has predictions (a class predicted only
    # as a phantom still shows up).
    active = [
        i
        for i in range(nc)
        if counts[i, :].sum() > 0 or counts[:, i].sum() > 0
    ]
    labels = [names.get(gt_categories[i].id, f"id {gt_categories[i].id}") for i in active] + ["background"]
    idx = active + [nc]
    return pd.DataFrame(counts[np.ix_(idx, idx)], index=labels, columns=labels)


def iou_histogram(iou_values: list[float], bins: int = 10) -> pd.DataFrame:
    """IoU values → histogram counts for st.bar_chart.

    Index = numeric bin centers (so the chart x-axis is a clean 0.0→1.0
    scale, not 20 string labels that render as a text blob).
    """
    if not iou_values:
        return pd.DataFrame({"count": []})
    counts, edges = np.histogram(iou_values, bins=bins, range=(0.0, 1.0))
    centers = [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(edges) - 1)]
    return pd.DataFrame({"count": counts}, index=centers)


def _delta_status(delta: float | None, regressed: bool, threshold: float) -> str:
    """Badge text for one class delta."""
    if delta is None:
        return "n/a"
    if regressed:
        return "regressed"
    if delta > threshold:
        return "improved"
    return "ok"


def delta_frame(
    compare: dict, names: dict[int, str], threshold: float = 0.05
) -> pd.DataFrame:
    """Per-class deltas from compare_runs() as a display DataFrame."""
    rows = []
    for cat_id in sorted(compare["deltas"]):
        d = compare["deltas"][cat_id]
        rows.append(
            {
                "class_id": cat_id,
                "class": names.get(cat_id, f"id {cat_id}"),
                "delta_map50": format_delta(d["delta_map50"]),
                "delta_f1": format_delta(d["delta_f1"]),
                "status": _delta_status(d["delta_map50"], d["regressed"], threshold),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"regressed": 0, "improved": 1, "ok": 2, "n/a": 3}
        df["_order"] = df["status"].map(lambda s: order.get(s, 3))
        df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def aggregate_row(aggregate: dict) -> pd.DataFrame:
    """Aggregate metrics as a single-row DataFrame (for comparison tables)."""
    return pd.DataFrame(
        [
            {
                "metric": "mAP@0.5",
                "value": format_score(aggregate["map50"]),
            },
            {
                "metric": "mAP@0.5:0.95",
                "value": format_score(aggregate["map"]),
            },
            {
                "metric": "Precision",
                "value": format_score(aggregate["precision"]),
            },
            {
                "metric": "Recall",
                "value": format_score(aggregate["recall"]),
            },
            {
                "metric": "F1",
                "value": format_score(aggregate["f1"]),
            },
            {
                "metric": "TP / FP / FN",
                "value": f"{aggregate['tp']} / {aggregate['fp']} / {aggregate['fn']}",
            },
        ]
    )


# ---------------------------------------------------------------------------
# Higher-level: full evaluation pipeline for the UI (load → validate → eval)
# ---------------------------------------------------------------------------


def evaluate_files(gt_path: str, preds_path: str) -> dict:
    """Load GT + predictions from paths and return the full metrics dict.

    Raises ValueError with a friendly message when the inputs are invalid or
    inconsistent (unknown categories/images would silently produce zeros).
    """
    try:
        gt = load_ground_truth(gt_path)
    except Exception as e:
        raise ValueError(f"ground truth file is not valid COCO JSON: {e}") from e
    try:
        preds = load_predictions(preds_path)
    except Exception as e:
        raise ValueError(f"predictions file is not valid COCO JSON: {e}") from e

    warnings = validate_consistency(gt, preds)
    if warnings:
        raise ValueError("; ".join(warnings))

    return evaluate_metrics(preds.predictions, gt)


def predictions_from_data(data: dict | list) -> PredictionsFile:
    """Build a PredictionsFile from in-memory JSON data (upload flow)."""
    return PredictionsFile.load(data)


def evaluate_from_models(preds: PredictionsFile, gt: GroundTruth) -> dict:
    """Evaluate already-loaded models (upload flow — no paths needed)."""
    warnings = validate_consistency(gt, preds)
    if warnings:
        raise ValueError("; ".join(warnings))
    return evaluate_metrics(preds.predictions, gt)


def compare_files(gt_path: str, base_path: str, candidate_path: str) -> dict:
    """Load GT + two prediction runs and return the full comparison result."""
    run_a = evaluate_files(gt_path, base_path)
    run_b = evaluate_files(gt_path, candidate_path)
    return {
        "run_a": run_a,
        "run_b": run_b,
        "compare": compare_runs(run_a, run_b),
        "names": category_names(load_ground_truth(gt_path)),
    }
