<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-82%20passing-brightgreen" alt="82 tests passing">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/version-v0.1.0-blue" alt="v0.1.0">
</p>

<h1 align="center">👁️ VisionForge</h1>
<p align="center"><em>The report card for your vision models — grades the boxes <strong>and</strong> catches the regression.</em></p>

<p align="center">
  <a href="https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/"><strong>🚀 Try the Live Demo</strong></a> ·
  <a href="#what-is-this"><strong>What is this</strong></a> ·
  <a href="#how-it-works"><strong>How it works</strong></a> ·
  <a href="#quick-start"><strong>Quickstart</strong></a> ·
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#the-regression-view"><strong>The regression view</strong></a> ·
  <a href="SPEC.md"><strong>Spec</strong></a> ·
  <a href="ARCHITECTURE.md"><strong>Architecture</strong></a>
</p>

---

## What is this?

**VisionForge grades detection models against labeled images — and shows when a new model version makes them worse.**

Object detection models output bounding boxes with confidence scores. The question is always: *how good are those boxes?* VisionForge answers it the same way the COCO benchmark does: compare every prediction against ground truth, count true positives, false positives (drew a box where nothing is), and false negatives (missed a real object), and turn the counts into a report card.

The special feature is the **regression view**: grade the same images with the old model and the new model, and VisionForge tells you which object types got *worse* — before your customers see it.

<div align="center">
  <a href="https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/">
    <img src="assets/compare.png" alt="VisionForge compare view — run A vs run B with the verdict banner and per-class delta table" width="700">
  </a>
  <p><em>Live app: verdict banner, aggregate side-by-side, and per-class deltas with regressed/improved badges.</em></p>
</div>

---

## How it works

```
  1. LABEL          2. RUN              3. GRADE
  ─────────────     ──────────────      ──────────────
  Ground truth =    Your detector      VisionForge compares
  a human-drawn     produces boxes     every box against GT,
  answer key        with confidence    counts TP/FP/FN, and
  (COCO JSON)       scores (COCO JSON) produces the report card
```

- **You provide the answer key** — COCO-style ground truth JSON (images, annotations, categories)
- **Your model provides its guesses** — COCO-style results JSON (image_id, category_id, bbox, score)
- **VisionForge does the grading** — greedy IoU matching per class (validated against pycocotools), COCO protocol metrics, and a per-image failure table
- **Compare two runs** — same GT, old vs new predictions → per-class deltas and a PASS / REGRESSED verdict

No model training, no API keys, no external calls. The demo runs entirely in your browser.

---

## Live Demo

Try the deployed app: **https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/**

The demo ships pre-loaded with a real sample: 12 COCO val2017 images, ground truth, and two YOLOv8n prediction runs (confidence 0.25 vs 0.90). The Report card tab shows run A's full card; load run B and both cards appear as nested tabs; the Compare tab shows the verdict and per-class deltas. You can also upload your own COCO-format JSON — same formats, same views.

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → connect the `visionforge` repo
3. Main file path: `visionforge/ui/app.py`
4. Deploy — no secrets, no API keys, no external services. The app is fully self-contained.

---

## Quick Start

```bash
# Install (core CLI + UI)
pip install -e ".[ui]"

# Evaluate one run — writes report.json + report.md
visionforge evaluate --gt examples/gt_sample.json --preds examples/preds_run_a.json

# Compare two runs — the regression view
visionforge compare --gt examples/gt_sample.json \
    --base examples/preds_run_a.json \
    --candidate examples/preds_run_b.json --fail-on-regression
# → verdict: REGRESSED (exit code 1)

# Launch the dashboard
streamlit run visionforge/ui/app.py
```

The repo ships a real demo: 12 COCO val2017 images with ground truth, plus two YOLOv8n prediction runs (confidence 0.25 vs 0.90). Run the commands above and you get real numbers — mAP@0.5 0.315 for run A, 0.047 for run B, verdict REGRESSED.

For the dashboard's upload flow, the repo also ships a simulated "v2" kit derived from the real run-A output (`data/generate_upload_demo.py`):
- `examples/preds_v2_mixed.json` — person regresses (−0.146) while dining table improves (+0.743): aggregate mAP actually rises 0.315 → 0.338 as a class breaks (the "mAP hides it" story)
- `examples/preds_v2_clean.json` — only improvements → verdict PASS, "clean release" banner

---

## Features

### 📊 The report card
- **COCO protocol metrics** — mAP@0.5, mAP@0.5:0.95, precision/recall/F1, all per-class
- **IoU distribution** — p50/p90 + histogram of matched detections
- **Confusion matrix** with background row/column — catches phantom-object failures that mAP hides
- **Per-image TP/FP/FN** — see exactly which images are hardest

### 🚨 Catch regressions before they ship
- **Before/after comparison** — same ground truth, old vs new predictions, per-class deltas
- **Verdict PASS / REGRESSED** — a class regresses when its mAP@0.5 delta drops below `-0.05` (strict `<`; classes in only one run are reported `n/a`, never silently zero)
- **CI gate** — `--fail-on-regression` exits 1 on regression, so a bad release fails the build

### 🔬 Deterministic and reproducible
- **Pure Python + numpy core** — no torch, no GPU, no Cython; runs anywhere
- **Byte-identical reports** — same inputs, same output, every time
- **Schema validation** with consistency warnings (unknown categories/images)

### 🧩 Detector-agnostic
- Works with predictions from YOLO, DETR, R-CNN, or anything that exports COCO-format JSON
- Ground truth and predictions are plain JSON files — no vendor lock-in

### 🖥️ Web dashboard
- **Report card tab** — hero metrics, per-class table, IoU histogram, confusion matrix, per-image detail; both runs' cards as nested tabs when two are loaded
- **Compare tab** — verdict banner, aggregate side-by-side, per-class delta table
- **About tab** — input formats and CLI examples
- **Upload support** — drop in your own COCO JSON, same views
- **Zero config** — no API keys, no secrets, no external calls

---

## The regression view

Run A vs run B on the same ground truth:

| Class | Delta mAP@0.5 | Status |
|-------|---------------|--------|
| person | -0.409 | regressed |
| car | -1.000 | regressed |
| dog | -0.505 | regressed |
| dining table | -0.089 | regressed |
| ... | | |
| chair | +0.000 | ok |

A class regresses when its mAP@0.5 delta drops below `-0.05` (strict `<`, so exactly -0.05 does not flag). Classes present in only one run are reported as `n/a` — never silently zero.

---

## Metrics

All metrics follow the COCO evaluation protocol (Lin et al., 2014), implemented independently in numpy and cross-checked against pycocotools:

- **mAP@0.5** — mean average precision at IoU 0.5 (the "does it find objects" score)
- **mAP@0.5:0.95** — averaged over 10 IoU thresholds (the "how precise are the boxes" score)
- **Precision / Recall / F1** at IoU 0.5
- **Per-class** versions of all of the above
- **IoU distribution** of matched detections (p50/p90, histogram)
- **Confusion matrix** with background row/column — catches phantom-object failures that mAP hides
- **Per-image TP/FP/FN** — which images are hardest

---

## Input format (COCO-style)

Ground truth:
```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40]}],
  "categories": [{"id": 1, "name": "person"}]
}
```

Predictions (bare COCO results list, or `{"predictions": [...]}`):
```json
[{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40], "score": 0.87}]
```

Bounding boxes are `[x, y, width, height]`. `visionforge info` prints this schema.

---

## Example Output

`visionforge evaluate` prints a one-line summary and writes `report.json` + `report.md`:

```
mAP@0.5: 0.315
mAP@0.5:0.95: 0.249
precision/recall/F1: 0.660/0.330/0.440
reports written: report.json, report.md
```

The JSON report (trimmed) looks like:

```json
{
  "aggregate": {
    "map50": 0.315,
    "map": 0.249,
    "precision": 0.660,
    "recall": 0.330,
    "f1": 0.440,
    "tp": 35,
    "fp": 18,
    "fn": 71
  },
  "per_class": {
    "person": {"ap50": 0.577, "precision": 0.789, "recall": 0.625, "tp": 15, "fp": 4, "fn": 9},
    "car":    {"ap50": 1.0,   "precision": 1.0,   "recall": 1.0,   "tp": 2,  "fp": 0, "fn": 0}
  },
  "iou_stats": {"mean": 0.72, "median": 0.75, "p50": 0.75, "p90": 0.91},
  "confusion": { "...": "nc+1 matrix with background row/column" }
}
```

`visionforge compare` prints the per-class regression table:

```
verdict: REGRESSED
  person       REGRESSED: delta mAP@0.5 = -0.409
  car          REGRESSED: delta mAP@0.5 = -1.000
  dog          REGRESSED: delta mAP@0.5 = -0.505
  dining table REGRESSED: delta mAP@0.5 = -0.089
```

---

## Web dashboard

`streamlit run visionforge/ui/app.py` (or the [live app](https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/)) — three tabs:

- **Report card** — full per-run card: hero metrics, per-class table, IoU histogram, confusion matrix, per-image detail. When two runs are loaded (sample or upload) both cards appear as nested tabs.
- **Compare** — verdict banner, aggregate side-by-side, and the per-class delta table with regressed/improved badges. Full report cards live in the Report card tab.
- **About** — input formats and CLI examples.

The built-in "Sample demo" is the real YOLOv8n pair above (Run A = conf 0.25, Run B = conf 0.90). The upload flow accepts the same formats and renders the same views.

---

## Architecture

```
CLI Layer (typer)
    └─▶ evaluate | compare | info
            │
Core Engine Layer
    ├─ schema (pydantic) — GT + predictions validation
    ├─ matching — greedy per-image per-class IoU matching
    ├─ metrics — AP (101-pt), precision/recall/F1, IoU stats, confusion
    └─ regression — per-class deltas, PASS/REGRESSED verdict
            │
UI Layer (Streamlit)
    ├─ upload → report card (per-run tabs when 2 runs) → compare
    └─ (verdict + deltas) → about
```

**Key design principle (V4):** matching produces a per-detection IoU table ONCE. All thresholds (0.5, 0.75, 0.5:0.95) are derived at report time from that table — never re-run inference or re-match per threshold.

---

## Project Status

**v0.1.0** — Core evaluation engine (pure numpy, COCO protocol), regression comparison with CI gate, CLI, and a three-tab Streamlit dashboard. See [Versions](#versions).

### Roadmap

- [x] Core engine: greedy IoU matching, COCO protocol metrics
- [x] Regression comparison with PASS/REGRESSED verdict + CI gate
- [x] CLI (`evaluate`, `compare`, `info`)
- [x] Streamlit dashboard (report card / compare / about)
- [x] Sample demo (12 COCO val images, 2 YOLOv8n runs)
- [x] Upload demo kit (simulated v2 stories)
- [ ] Tracking / MOT metrics (v2)
- [ ] Video processing (v2)
- [ ] Custom IoU thresholds + confusion-matrix export

---

## Versions

| Version | What it is |
|---------|-----------|
| **v0.1.0** | First public release — COCO protocol evaluation, regression comparison, CLI, dashboard |

**Live demo:** https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/ — load the built-in sample pair (or upload your own COCO JSON) to see the report card and compare views in action.

---

## References

- [SPEC.md](SPEC.md) — Full behavior spec with acceptance criteria
- [ARCHITECTURE.md](ARCHITECTURE.md) — Design, interfaces, extension points
- [CHANGELOG.md](CHANGELOG.md) — Version history
- [assets/compare.png](assets/compare.png) — Compare view screenshot (regenerable via scripts/capture_shots.py)
- [assets/report-card.png](assets/report-card.png) — Report card screenshot (regenerable via scripts/capture_shots.py)
- [examples/gt_sample.json](examples/gt_sample.json) — Demo ground truth (12 COCO val2017 images, 106 annotations, 25 categories)
- [examples/preds_run_a.json](examples/preds_run_a.json) — YOLOv8n conf 0.25 run (mAP@0.5 0.315)
- [examples/preds_run_b.json](examples/preds_run_b.json) — YOLOv8n conf 0.90 run (mAP@0.5 0.047)
- [examples/preds_v2_mixed.json](examples/preds_v2_mixed.json) / [preds_v2_clean.json](examples/preds_v2_clean.json) — Simulated v2 upload demo (generated by data/generate_upload_demo.py)
- [data/generate_upload_demo.py](data/generate_upload_demo.py) — Simulated v2 generator (derived from real run-A output)

---

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/rayyanakmal">@rayyanakmal</a>
</p>
