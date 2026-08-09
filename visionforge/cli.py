"""VisionForge CLI — evaluate detection models against ground truth."""

import typer

app = typer.Typer(help="Vision model report card: grade detections against labeled data.")


@app.command()
def evaluate(
    gt: str = typer.Option(..., help="Path to COCO-style ground truth JSON"),
    preds: str = typer.Option(..., help="Path to predictions JSON"),
    thresholds: str = typer.Option("0.5,0.75", help="IoU thresholds (comma-separated)"),
    out: str = typer.Option("report", help="Output basename (writes .json and .md)"),
) -> None:
    """Compute a metric report card for one prediction run."""
    typer.echo("evaluate: not yet implemented")
    raise typer.Exit(code=1)


@app.command()
def compare(
    gt: str = typer.Option(..., help="Path to COCO-style ground truth JSON"),
    base: str = typer.Option(..., help="Path to baseline predictions JSON (run A)"),
    candidate: str = typer.Option(..., help="Path to candidate predictions JSON (run B)"),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit non-zero on REGRESSED"),
    out: str = typer.Option("compare", help="Output basename (writes .json and .md)"),
) -> None:
    """Compare two prediction runs and report per-class deltas + verdict."""
    typer.echo("compare: not yet implemented")
    raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Print input schema and a minimal example."""
    typer.echo("info: not yet implemented")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
