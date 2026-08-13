# VisionForge — Vision Model Report Card

## Overview

VisionForge evaluates object detection models against labeled images and tells you when a new model version makes things worse. You give it ground truth boxes (COCO JSON) and prediction boxes from any detector (YOLO, DETR, R-CNN, SAM — anything that exports COCO format), and it produces a report card: mAP, precision/recall/F1 per class, IoU statistics, confusion counts, and a run A vs run B regression diff with a PASS/REGRESSED verdict.

The name combines **Vision** + **Forge** (verdictlab's sibling — the place where raw model outputs are shaped into measurable results).

---

## Prerequisites (Human Tasks)

| Task | Detail |
|------|--------|
| Python 3.11+ | Managed by uv (`.venv` created) |
| uv | Installed at `~/.local/bin/uv` |
| Git repo | Created at `~/projects/visionforge/` |
| Demo prediction files | Generated once on Mac via YOLO (Phase 4), committed as JSON |

---

## US-1: Core Evaluation — Metric Report Card

**AC-1.1:** Given a COCO-style GT JSON + COCO-style predictions JSON, when evaluated, then the report contains aggregate metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, F1.

**AC-1.2:** Given a predictions file with per-class predictions, when evaluated, then the report contains per-class precision/recall/F1/AP.

**AC-1.3:** Given predictions with confidence scores, when evaluated, then the mAP computation follows the COCO protocol exactly: 101-point recall interpolation, IoU thresholds [0.5:0.95] step 0.05, area ranges (all/small/medium/large), maxDets [1,10,100].

**AC-1.4:** Given a predictions file, when evaluated, then the report contains per-image TP/FP/FN counts identifying the hardest images.

**AC-1.5:** Given a predictions file, when evaluated, then the report contains a confusion matrix with a background row/col capturing phantom detections.

Edge cases:
- Empty predictions → metrics all zero, no crash
- Empty GT annotations → warning, metrics undefined where no GT exists
- Prediction category_ids not present in GT categories → schema validation warning (silent zero-match trap)
- Determinism: same inputs → byte-identical report

---

## US-2: Regression Comparison — Run A vs Run B

**AC-2.1:** Given two prediction files (run A baseline, run B candidate) + GT, when compared, then the report contains per-class deltas (B − A) for mAP@0.5 and F1.

**AC-2.2:** Given a regression threshold (default 0.05), when a class delta < −threshold, then that class is flagged REGRESSED; if any class is flagged, the overall verdict is REGRESSED, else PASS.

**AC-2.3:** Given `--fail-on-regression` flag, when the verdict is REGRESSED, then the CLI exits non-zero (CI-able).

Edge cases:
- Identical runs → PASS, all deltas 0
- Delta exactly −0.05 → not flagged (strict <)
- A class present in run B but not run A → delta treated as undefined/NaN, reported, not silently zero

---

## US-3: IoU Statistics

**AC-3.1:** Given a predictions file + GT, when evaluated, then the report contains IoU distribution stats (mean, median, p50, p90) across matched detections.

---

## US-4: CLI + UI Parity

**AC-4.1:** Given the `visionforge evaluate` CLI with GT + preds paths, when run, then it writes a JSON report and prints a human-readable markdown summary.

**AC-4.2:** Given the `visionforge compare` CLI with GT + base + candidate paths, when run, then it writes a regression report and exits 0 (PASS) or 1 (REGRESSED with `--fail-on-regression`).

**AC-4.3:** Given the Streamlit UI, when a user uploads GT + preds (or loads sample data), then they see the same report card + compare views as the CLI. Zero API keys required.

**AC-4.4:** Given two runs loaded in the UI (sample or upload), when viewing the Report card tab, then both runs' full report cards are shown as nested tabs; the Compare tab shows the verdict banner, aggregate side-by-side, and per-class delta table — no repeated full report cards.

---

## Out of Scope (explicitly rejected)

- Model training/fine-tuning — we evaluate, we don't train
- Tracking/MOT metrics (ID-switch, track fragmentation) — v2
- Annotation tools / GT creation UI — GT is an input file
- Video processing (frame extraction, streaming eval) — v2
- Custom metric reimplementation where standard math suffices — COCO protocol implemented in numpy per spec (no pycocotools/Cython dependency), cross-checked in one test
- API keys / network calls in the core — deterministic, offline
- MarveCount / qwen-fish data (classified — never in this repo)
