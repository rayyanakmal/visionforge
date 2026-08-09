# Changelog

## v0.1.0 — 2026-08-09

First public release.

### Features

- Core evaluation engine (pure numpy, COCO protocol):
  - Greedy per-image per-class IoU matching, validated against pycocotools
  - mAP@0.5, mAP@0.5:0.95 (101-pt recall interpolation, 10-IoU sweep)
  - Per-class precision / recall / F1
  - IoU distribution stats (mean, median, p50, p90) + raw values
  - Confusion matrix with background row/column (nc+1)
  - Per-image TP/FP/FN table
- Regression comparison (run A vs run B):
  - Per-class deltas with regressed/improved/ok badges
  - Verdict PASS/REGRESSED with configurable threshold (default 0.05)
  - `--fail-on-regression` CI flag (exit code 1 on regression)
- CLI (`visionforge evaluate|compare|info`), typer-based
- Streamlit UI:
  - Report card: hero metrics, per-class table, IoU histogram, confusion matrix
  - Compare: verdict banner, side-by-side aggregates, delta table
  - About: input format docs + CLI examples
  - Sample demo pre-loaded (12 COCO val images, 2 YOLOv8n runs)
  - Zero API keys, no external calls
- Schema validation with consistency warnings (unknown categories/images)

### Fixed

- IoU threshold now enforced in PR-curve matching (mAP@0.5:0.95 was identical
  to mAP@0.5 when the parameter was ignored) — pinned by regression test
- Report writers create parent directories instead of crashing
- UI delta table: negative deltas were rendered "n/a" (the -1 no-GT sentinel
  was colliding with real decreases) — fixed with a dedicated formatter

### Demo data

- `examples/gt_sample.json`: 12 COCO val2017 images, 106 annotations, 25 categories
- `examples/preds_run_a.json`: YOLOv8n conf 0.25 (mAP@0.5 0.315)
- `examples/preds_run_b.json`: YOLOv8n conf 0.90 (mAP@0.5 0.047) — REGRESSED
