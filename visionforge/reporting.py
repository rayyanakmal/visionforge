"""Report writers: JSON (machine-readable) + markdown (human-readable).

Style rules: tables, no emoji, numbers rounded to 3 decimals.
"""

import json


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def write_json_report(metrics: dict, path: str) -> None:
    """Write the full metrics dict as JSON (numpy arrays converted to lists)."""
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
