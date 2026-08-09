"""VisionForge CLI — evaluate detection models against ground truth."""

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from visionforge.metrics import evaluate as evaluate_metrics
from visionforge.regression import compare_runs
from visionforge.reporting import write_json_report, write_markdown_report, write_regression_report
from visionforge.schema import load_ground_truth, load_predictions, validate_consistency

app = typer.Typer(help="Vision model report card: grade detections against labeled data.")


def _load_inputs(gt_path: str, preds_path: str):
    """Load + validate GT and predictions. Exits with code 1 on error."""
    try:
        gt = load_ground_truth(gt_path)
    except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
        typer.echo(f"error: could not load ground truth: {e}", err=True)
        raise typer.Exit(1)
    try:
        preds = load_predictions(preds_path)
    except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
        typer.echo(f"error: could not load predictions: {e}", err=True)
        raise typer.Exit(1)

    for warning in validate_consistency(gt, preds):
        typer.echo(f"warning: {warning}", err=True)

    return gt, preds


@app.command()
def evaluate(
    gt: str = typer.Option(..., help="Path to COCO-style ground truth JSON"),
    preds: str = typer.Option(..., help="Path to predictions JSON"),
    thresholds: str = typer.Option("0.5,0.75", help="IoU thresholds (comma-separated)"),
    out: str = typer.Option("report", help="Output basename (writes .json and .md)"),
) -> None:
    """Compute a metric report card for one prediction run."""
    gt_model, preds_model = _load_inputs(gt, preds)

    try:
        thr = [float(t) for t in thresholds.split(",")]
    except ValueError:
        typer.echo(f"error: invalid thresholds: {thresholds}", err=True)
        raise typer.Exit(1)

    metrics = evaluate_metrics(preds_model.predictions, gt_model, iou_thresholds=thr)

    write_json_report(metrics, f"{out}.json")
    write_markdown_report(metrics, f"{out}.md")

    agg = metrics["aggregate"]
    typer.echo(f"mAP@0.5: {agg['map50']:.3f}")
    typer.echo(f"mAP@0.5:0.95: {agg['map']:.3f}")
    typer.echo(f"precision/recall/F1: {agg['precision']:.3f}/{agg['recall']:.3f}/{agg['f1']:.3f}")
    typer.echo(f"reports written: {out}.json, {out}.md")


@app.command()
def compare(
    gt: str = typer.Option(..., help="Path to COCO-style ground truth JSON"),
    base: str = typer.Option(..., help="Path to baseline predictions JSON (run A)"),
    candidate: str = typer.Option(..., help="Path to candidate predictions JSON (run B)"),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit non-zero when verdict is REGRESSED"
    ),
    out: str = typer.Option("compare", help="Output basename (writes .json and .md)"),
) -> None:
    """Compare two prediction runs and report per-class deltas + verdict."""
    gt_model, base_model = _load_inputs(gt, base)
    try:
        cand_model = load_predictions(candidate)
    except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
        typer.echo(f"error: could not load candidate predictions: {e}", err=True)
        raise typer.Exit(1)

    run_a = evaluate_metrics(base_model.predictions, gt_model)
    run_b = evaluate_metrics(cand_model.predictions, gt_model)
    result = compare_runs(run_a, run_b)

    write_regression_report(run_a, run_b, result, out)

    typer.echo(f"verdict: {result['verdict']}")
    for cat_id in result["summary"]["regressed_classes"]:
        d = result["deltas"][cat_id]
        typer.echo(f"  class {cat_id} REGRESSED: delta mAP@0.5 = {d['delta_map50']:.3f}")

    if fail_on_regression and result["verdict"] == "REGRESSED":
        typer.echo("regression detected — failing", err=True)
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Print input schema and a minimal example."""
    typer.echo("VisionForge input formats")
    typer.echo("")
    typer.echo("Ground truth (COCO-style JSON):")
    typer.echo('  {"images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],')
    typer.echo('   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40]}],')
    typer.echo('   "categories": [{"id": 1, "name": "person"}]}')
    typer.echo("")
    typer.echo("Predictions (COCO results JSON — a list, or {\"predictions\": [...]}):")
    typer.echo('  [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40], "score": 0.87}]')
    typer.echo("")
    typer.echo("Example:")
    typer.echo("  visionforge evaluate --gt gt.json --preds preds.json")
    typer.echo("  visionforge compare --gt gt.json --base preds_a.json --candidate preds_b.json --fail-on-regression")


if __name__ == "__main__":
    app()
