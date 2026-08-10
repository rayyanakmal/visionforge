"""Generate simulated "v2" candidate runs for the dashboard upload demo.

The Pi never runs a model (design D5) — real inference happens once on the
Mac (see generate_demo_predictions.py). These candidates are honest
perturbations of the REAL YOLOv8n run (examples/preds_run_a.json), not
from-scratch fabrications:

  examples/preds_v2_mixed.json
      person recall dropped (remove the ~40% lowest-score person
      detections) + dining-table detections added (GT-perfect boxes).
      -> person REGRESSES, dining table IMPROVES. Aggregate mAP stays
      roughly flat while a class broke — the mixed-delta story the
      built-in sample (conf 0.25 vs 0.90) doesn't show.

  examples/preds_v2_clean.json
      only additions (dining-table + chair GT-perfect boxes).
      -> every class same-or-better: PASS verdict, "clean release" banner.

Upload kit: examples/gt_sample.json + examples/preds_run_a.json (baseline)
+ one of the v2 files (candidate). Same GT, same format, both verdicts.

Usage:
    python data/generate_upload_demo.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
GT_PATH = EXAMPLES / "gt_sample.json"
PREDS_A_PATH = EXAMPLES / "preds_run_a.json"

# Classes in GT with plenty of ground truth but few detections in run A —
# perfect "this model got better here" candidates.
IMPROVE_CLASSES = {
    "dining table": 10,  # add this many GT-perfect boxes
    "chair": 6,
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def category_id_by_name(gt: dict) -> dict:
    return {c["name"]: c["id"] for c in gt["categories"]}


def gt_boxes_for_category(gt: dict, cat_id: int) -> list[dict]:
    return [a for a in gt["annotations"] if a["category_id"] == cat_id]


def add_perfect_boxes(preds: list[dict], gt: dict, cat_id: int, count: int) -> list[dict]:
    """Add GT-perfect detections (score 0.92) for a category — simulates a
    model that now catches boxes it used to miss."""
    boxes = gt_boxes_for_category(gt, cat_id)
    for ann in boxes[:count]:
        preds.append(
            {
                "image_id": ann["image_id"],
                "category_id": cat_id,
                "bbox": list(ann["bbox"]),
                "score": 0.92,
            }
        )
    return preds


def drop_lowest_score(preds: list[dict], cat_id: int, count: int) -> list[dict]:
    """Remove the `count` lowest-score detections of a category — simulates
    recall loss on that class."""
    keep, drop_pool = [], []
    for p in preds:
        (drop_pool if p["category_id"] == cat_id else keep).append(p)
    drop_pool.sort(key=lambda p: p["score"])
    keep.extend(drop_pool[count:])
    return keep


def main() -> None:
    gt = load_json(GT_PATH)
    run_a = load_json(PREDS_A_PATH)
    cat_ids = category_id_by_name(gt)

    # --- v2 mixed: person regressed, dining table improved ---
    mixed = list(run_a)
    person_preds = [p for p in mixed if p["category_id"] == cat_ids["person"]]
    drop_n = max(1, int(len(person_preds) * 0.4))
    mixed = drop_lowest_score(mixed, cat_ids["person"], drop_n)
    mixed = add_perfect_boxes(mixed, gt, cat_ids["dining table"], IMPROVE_CLASSES["dining table"])

    # --- v2 clean: only improvements ---
    clean = list(run_a)
    clean = add_perfect_boxes(clean, gt, cat_ids["dining table"], IMPROVE_CLASSES["dining table"])
    clean = add_perfect_boxes(clean, gt, cat_ids["chair"], IMPROVE_CLASSES["chair"])

    out_mixed = EXAMPLES / "preds_v2_mixed.json"
    out_clean = EXAMPLES / "preds_v2_clean.json"
    out_mixed.write_text(json.dumps(mixed, indent=2), encoding="utf-8")
    out_clean.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    print(f"run_a preds: {len(run_a)}")
    print(f"v2_mixed: {len(mixed)} preds (dropped {drop_n} person, "
          f"added {IMPROVE_CLASSES['dining table']} dining table) -> {out_mixed.name}")
    print(f"v2_clean: {len(clean)} preds (added {IMPROVE_CLASSES['dining table']} dining "
          f"table + {IMPROVE_CLASSES['chair']} chair) -> {out_clean.name}")


if __name__ == "__main__":
    main()
