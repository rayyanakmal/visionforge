# VisionForge — Architecture Design

> **Version:** 0.1.0
> **Design Date:** 2026-08-09
> **Status:** Draft
> **Sources:** COCO evaluation protocol (Lin et al. 2014, pycocotools BSD); landscape research 2026-08-09 (see plan `~/.hermes/plans/20260809-183450-visionforge.md`)

---

## 1. Component Tree & Data Flow

### Component Tree (Layered)

```
┌──────────────────────────────────────────────────────────────┐
│                       CLI Layer (typer)                      │
│   ┌────────────┐  ┌─────────────┐  ┌───────────────┐        │
│   │  evaluate  │  │  compare    │  │    info       │        │
│   └─────┬──────┘  └──────┬──────┘  └───────┬───────┘        │
│         │                │                 │                │
└─────────┼────────────────┼─────────────────┼────────────────┘
          │                │                 │
┌─────────▼────────────────▼─────────────────▼────────────────┐
│                     Core Engine Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │  schema  │  │ matching │  │  metrics   │  │ regression│  │
│  │ (pydantic)│ │ (IoU)    │  │ (AP/P/R/F1)│  │ (deltas)  │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └─────┬─────┘  │
│       │             │              │               │        │
│       └─────────────┴──────────────┴───────────────┘        │
│                         │                                   │
│                    ┌────▼────┐                              │
│                    │reporting│                              │
│                    └────┬────┘                              │
└─────────────────────────┼────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                        UI Layer (Streamlit)                  │
│  upload → report card (per-run tabs when 2 runs) → compare   │
│           (verdict + deltas) → about                         │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
GT JSON (COCO) ──┐
                ├──► schema.validate ──► matching (greedy IoU per class)
Preds JSON ─────┘          │                    │
                           │                    ▼
                           │         per-detection IoU table (stored once)
                           │                    │
                           │                    ▼
                           │         metrics (AP/P/R/F1/IoU stats/confusion)
                           │                    │
                           │                    ▼
                           └────────► reporting (JSON + markdown)
                                            │
                              regression (run A vs run B) ──► verdict
```

**Key design principle (V4):** matching produces a per-detection IoU table ONCE. All thresholds (0.5, 0.75, 0.5:0.95) are derived at report time from that table — never re-run inference or re-match per threshold.

---

## 2. File/Module Structure

```
visionforge/
├── SPEC.md
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── visionforge/
│   ├── __init__.py
│   ├── schema.py          # pydantic models: GTImage, GTAnnotation, GTCategory,
│   │                      #   GroundTruth, Prediction, PredictionsFile
│   ├── matching.py        # compute_iou, match_predictions (greedy, score-desc)
│   ├── metrics.py         # compute_ap (101-pt), precision/recall/f1,
│   │                      #   compute_iou_stats, evaluate()
│   ├── regression.py      # compare_runs → per-class deltas + verdict
│   ├── reporting.py       # write_json_report, write_markdown_report
│   ├── cli.py             # typer: evaluate / compare / info
│   └── ui/
│       └── app.py         # Streamlit app
├── tests/
│   ├── conftest.py        # fixtures: tiny GT + preds (hand-computed)
│   ├── test_schema.py
│   ├── test_matching.py
│   ├── test_metrics.py
│   ├── test_regression.py
│   └── test_cli.py
├── data/
│   ├── README.md          # sample dataset notes (NOT committed)
│   ├── download_sample.py # downloads COCO val subset (~100 images)
│   └── generate_upload_demo.py  # simulated v2 candidate runs (derived from real run A)
└── examples/
    ├── gt_sample.json     # committed tiny sample (5-10 images) for smoke tests
    ├── preds_run_a.json, preds_run_b.json  # real YOLOv8n, Phase 4, committed
    └── preds_v2_mixed.json, preds_v2_clean.json  # upload demo kit (simulated)
```

---

## 3. Interface Specifications

### schema.py

```python
class GTImage(BaseModel):
    id: int; file_name: str; width: int; height: int

class GTAnnotation(BaseModel):
    id: int; image_id: int; category_id: int; bbox: list[float]  # [x,y,w,h]

class GTCategory(BaseModel):
    id: int; name: str

class GroundTruth(BaseModel):
    images: list[GTImage]; annotations: list[GTAnnotation]; categories: list[GTCategory]

class Prediction(BaseModel):
    image_id: int; category_id: int; bbox: list[float]; score: float = Field(ge=0, le=1)

class PredictionsFile(BaseModel):
    predictions: list[Prediction]
```

### matching.py

```python
def compute_iou(box_a: list[float], box_b: list[float]) -> float
def match_predictions(preds, gt_anns, iou_threshold=0.5) -> MatchResult
    # MatchResult: matches [(pred_idx, gt_idx)], unmatched_preds, unmatched_gt
    # Greedy: sort preds by score desc; per category, assign best available GT
    #   with IoU >= threshold; iou capped at min(t, 1-1e-10) per COCO
```

### metrics.py

```python
def compute_ap(precisions, recalls) -> float          # 101-pt interpolation
def compute_precision_recall_f1(tp, fp, fn) -> tuple[float, float, float]
def compute_iou_stats(iou_values) -> dict             # mean/median/p50/p90
def evaluate(preds, gt, iou_thresholds=[0.5, 0.75]) -> dict
    # → {aggregate: {...}, per_class: {...}, per_image: {...},
    #    iou_stats: {...}, confusion: {...}}
```

### regression.py

```python
def compare_runs(run_a_metrics, run_b_metrics, regression_threshold=0.05) -> dict
    # → {verdict: 'PASS'|'REGRESSED', deltas: {class: {delta_ap50, delta_f1, ...}},
    #    summary: {...}}
```

### reporting.py

```python
def write_json_report(metrics, path) -> None
def write_markdown_report(metrics, path) -> None   # tables, no emoji
```

### cli.py

```
visionforge evaluate --gt <gt.json> --preds <preds.json> [--thresholds 0.5,0.75] [--out report]
visionforge compare --gt <gt.json> --base <preds_a.json> --candidate <preds_b.json> [--fail-on-regression] [--out report]
visionforge info   # print input schema + example
```

---

## 4. Extension Point Map

| Extension | Where | Notes |
|-----------|-------|-------|
| New input format (point annotations, YOLO txt) | schema.py loader + adapter | Stretch goal D10 |
| Segmentation masks (RLE) | metrics.py (IoU via mask) | v2 |
| Tracking/MOT metrics | metrics.py new module | v2, rejected for v1 |
| Video eval | new runner over frames | v2, rejected for v1 |
| New report formats (HTML, CSV) | reporting.py | trivial add |

---

## 5. Tech Stack & Rationale

| Choice | Why |
|--------|-----|
| Python 3.11 + uv | Matches evalforge; uv is fast and already proven |
| numpy only for core | Pure-Python metrics, no torch, no Cython → runs on Pi, deterministic |
| pydantic | Input validation catches label/bbox mismatches before silent zero-matches (known pitfall) |
| typer | CLI parity with evalforge, zero-boilerplate |
| Streamlit + pandas | UI parity with evalforge; report tables render natively |
| pytest | Golden hand-computed tests (V2), TDD loop |

**Deliberately NOT used:** pycocotools (Cython dep — implement the documented protocol in numpy, cross-check in one test), torchmetrics (pulls torch), Ultralytics (AGPL-3.0 — structure referenced, code written independently), FiftyOne (MongoDB platform — overkill).

---

## 6. Concurrency Model

None needed. Single-threaded, deterministic, pure functions. CLI and UI both call the same core synchronously. Sample data is tiny (<100 images, ~10MB). No async, no parallelism — YAGNI.

---

## 7. Scenario-to-Component Mapping

| Scenario | Path |
|----------|------|
| "Does my new YOLO version regress?" | cli compare → regression → verdict |
| "Which classes is my model worst at?" | metrics per_class → report table |
| "Which images fail hardest?" | metrics per_image (TP/FP/FN) → UI table |
| "Is the model hallucinating objects?" | confusion matrix background row |
| "How tight are my boxes?" | iou_stats distribution |

---

## 8. Research Verdict Integration (2026-08-09)

Applied from landscape research:
1. COCO protocol in numpy (~60 lines) — no Cython dep, numbers comparable to any vendor
2. Per-image TP/FP/FN table — Ultralytics `image_metrics` pattern (structure only, not code)
3. Confusion matrix (nc+1, nc+1) background row/col — same
4. `--fail-on-regression` CI exit code — HoneyHive `compare_runs()` pattern
5. evalforge tracker pattern (accumulate → summarize, zeros-not-errors) — schema/metrics design

Rejected: FiftyOne platform model, torchmetrics, Ultralytics code (AGPL), VLMEvalKit benchmark runner, W&B experiment tracking.
