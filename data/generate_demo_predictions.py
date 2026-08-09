"""Generate demo prediction files with YOLO — run on Mac (needs torch + ultralytics).

The Pi never runs a model (design decision D5). Run this on the Mac once:

  cd ~/Desktop/Portfolio\ Project/visionforge   # or wherever the repo is on Mac
  python3 -m venv /tmp/vf_yolo_venv
  /tmp/vf_yolo_venv/bin/pip install -q ultralytics pillow
  /tmp/vf_yolo_venv/bin/python data/generate_demo_predictions.py

It loads the committed sample images + GT, runs YOLOv8n twice:
  - Run A: conf 0.25 (normal)   -> examples/preds_run_a.json
  - Run B: conf 0.90 (degraded) -> examples/preds_run_b.json  (fewer detections -> recall drop -> REGRESSED)

Output: COCO-results-format JSON (bare list of detections), the same format
visionforge's loader accepts.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
IMAGES_DIR = EXAMPLES / "images"
GT_PATH = EXAMPLES / "gt_sample.json"


def load_gt() -> dict:
    with open(GT_PATH) as f:
        return json.load(f)


def category_id_by_name(gt: dict) -> dict:
    return {c["name"]: c["id"] for c in gt["categories"]}


def results_to_coco(results, image_id: int, name_to_cat_id: dict, conf_threshold: float) -> list[dict]:
    """Convert one ultralytics Results object to COCO-results detections."""
    out = []
    if results.boxes is None:
        return out
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
        if conf < conf_threshold:
            continue
        name = names[int(cls)]
        cat_id = name_to_cat_id.get(name)
        if cat_id is None:
            continue  # only keep classes present in our GT
        out.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(conf),
        })
    return out


def run_model(model, gt: dict, conf_threshold: float) -> list[dict]:
    name_to_cat_id = category_id_by_name(gt)
    img_id_by_name = {img["file_name"]: img["id"] for img in gt["images"]}
    all_dets = []

    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        image_id = img_id_by_name.get(img_path.name)
        if image_id is None:
            print(f"  skip {img_path.name} (not in GT)")
            continue
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        dets = results_to_coco(results[0], image_id, name_to_cat_id, conf_threshold)
        all_dets.extend(dets)
        print(f"  {img_path.name}: {len(dets)} detections @ conf {conf_threshold}")

    return all_dets


def main() -> None:
    if not GT_PATH.exists():
        sys.exit("examples/gt_sample.json not found — run data/download_sample.py first")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — see header comment for the one-time setup")

    gt = load_gt()
    model = YOLO("yolov8n.pt")  # downloads weights on first run

    print("=== Run A: conf 0.25 (normal) ===")
    run_a = run_model(model, gt, conf_threshold=0.25)
    (EXAMPLES / "preds_run_a.json").write_text(json.dumps(run_a, indent=2))
    print(f"run A: {len(run_a)} detections -> examples/preds_run_a.json")

    print("=== Run B: conf 0.90 (degraded) ===")
    run_b = run_model(model, gt, conf_threshold=0.90)
    (EXAMPLES / "preds_run_b.json").write_text(json.dumps(run_b, indent=2))
    print(f"run B: {len(run_b)} detections -> examples/preds_run_b.json")

    print("\nDone. Copy preds_run_a.json + preds_run_b.json back to the Pi repo (or commit from Mac).")


if __name__ == "__main__":
    main()
