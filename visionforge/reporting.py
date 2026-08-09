"""Report writers: JSON (machine-readable) + markdown (human-readable).

Style rules: tables, no emoji, numbers rounded to 3 decimals.
"""

import json
from pathlib import Path


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def _ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_json_report(metrics: dict, path: str) -> None:
    """Write the full metrics dict as JSON (numpy arrays converted to lists)."""
    _ensure_parent(path)
    payload = {
        "aggregate": metrics["aggregate"],
        "per_class": metrics["per_class"],
        "per_image": metrics["per_image"],
        "iou_stats": metrics["iou_stats"],
        "confusion": metrics["confusion"].tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_markdown_report(metrics: dict, path: str) -> None:
    """Write a human-readable report card (tables, no emoji)."""
    _ensure_parent(path)
    agg = metrics["aggregate"]
    lines: list[str] = []

    lines.append("# VisionForge Evaluation Report")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| mAP@0.5 | {_fmt(agg['map50'])} |")
    lines.append(f"| mAP@0.5:0.95 | {_fmt(agg['map'])} |")
    lines.append(f"| Precision | {_fmt(agg['precision'])} |")
    lines.append(f"| Recall | {_fmt(agg['recall'])} |")
    lines.append(f"| F1 | {_fmt(agg['f1'])} |")
    lines.append(f"| TP / FP / FN | {agg['tp']} / {agg['fp']} / {agg['fn']} |")
    lines.append("")

    per_class = metrics["per_class"]
    if per_class:
        lines.append("## Per Class")
        lines.append("")
        lines.append("| Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|--------|-----------|--------|-----|----|----|----|")
        for cat_id in sorted(per_class):
            pc = per_class[cat_id]
            lines.append(
                f"| {cat_id} | {_fmt(pc['ap50'])} | {_fmt(pc['precision'])} | "
                f"{_fmt(pc['recall'])} | {_fmt(pc['f1'])} | "
                f"{pc['tp']} | {pc['fp']} | {pc['fn']} |"
            )
        lines.append("")

    per_image = metrics["per_image"]
    if per_image:
        lines.append("## Per Image")
        lines.append("")
        lines.append("| Image | TP | FP | FN |")
        lines.append("|-------|----|----|----|")
        for img_id in sorted(per_image):
            pi = per_image[img_id]
            lines.append(f"| {img_id} | {pi['tp']} | {pi['fp']} | {pi['fn']} |")
        lines.append("")

    iou = metrics["iou_stats"]
    lines.append("## IoU Statistics (matched detections)")
    lines.append("")
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Mean | {_fmt(iou['mean'])} |")
    lines.append(f"| Median | {_fmt(iou['median'])} |")
    lines.append(f"| p90 | {_fmt(iou['p90'])} |")
    lines.append(f"| Matched count | {iou['count']} |")
    lines.append("")

    confusion = metrics["confusion"]
    lines.append("## Confusion Matrix (rows=GT, cols=pred, last=background)")
    lines.append("")
    lines.append("```")
    lines.append(str(confusion))
    lines.append("```")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_regression_report(
    base_metrics: dict,
    candidate_metrics: dict,
    compare_result: dict,
    basename: str,
) -> None:
    """Write regression comparison outputs (JSON + markdown)."""
    _ensure_parent(f"{basename}.json")
    _ensure_parent(f"{basename}.md")
    # JSON: full machine-readable comparison
    payload = {
        "verdict": compare_result["verdict"],
        "deltas": compare_result["deltas"],
        "summary": compare_result["summary"],
        "base_aggregate": base_metrics["aggregate"],
        "candidate_aggregate": candidate_metrics["aggregate"],
    }
    with open(f"{basename}.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown: human-readable diff
    lines: list[str] = []
    lines.append("# VisionForge Regression Report")
    lines.append("")
    lines.append(f"## Verdict: {compare_result['verdict']}")
    lines.append("")
    lines.append("## Per-Class Deltas (candidate - baseline)")
    lines.append("")
    lines.append("| Class | Δ mAP@0.5 | Δ F1 | Status |")
    lines.append("|-------|-----------|------|--------|")
    for cat_id in sorted(compare_result["deltas"]):
        d = compare_result["deltas"][cat_id]
        status = "REGRESSED" if d["regressed"] else "ok"
        lines.append(
            f"| {cat_id} | {_fmt(d['delta_map50'])} | {_fmt(d['delta_f1'])} | {status} |"
        )
    lines.append("")
    lines.append("## Baseline (run A)")
    lines.append("")
    _append_aggregate(lines, base_metrics)
    lines.append("")
    lines.append("## Candidate (run B)")
    lines.append("")
    _append_aggregate(lines, candidate_metrics)
    lines.append("")

    with open(f"{basename}.md", "w") as f:
        f.write("\n".join(lines))


def _append_aggregate(lines: list[str], metrics: dict) -> None:
    agg = metrics["aggregate"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| mAP@0.5 | {_fmt(agg['map50'])} |")
    lines.append(f"| mAP@0.5:0.95 | {_fmt(agg['map'])} |")
    lines.append(f"| Precision | {_fmt(agg['precision'])} |")
    lines.append(f"| Recall | {_fmt(agg['recall'])} |")
    lines.append(f"| F1 | {_fmt(agg['f1'])} |")
