# VisionForge Evaluation Report

## Aggregate

| Metric | Value |
|--------|-------|
| mAP@0.5 | 1.000 |
| mAP@0.5:0.95 | 1.000 |
| Precision | 0.500 |
| Recall | 0.500 |
| F1 | 0.500 |
| TP / FP / FN | 1 / 1 / 1 |

## Per Class

| Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|-------|--------|-----------|--------|-----|----|----|----|
| 1 | 1.000 | 0.500 | 0.500 | 0.500 | 1 | 1 | 1 |

## Per Image

| Image | TP | FP | FN |
|-------|----|----|----|
| 1 | 0 | 2 | 1 |
| 2 | 1 | 0 | 0 |

## IoU Statistics (matched detections)

| Stat | Value |
|------|-------|
| Mean | 1.000 |
| Median | 1.000 |
| p90 | 1.000 |
| Matched count | 2 |

## Confusion Matrix (rows=GT, cols=pred, last=background)

```
[[1 1]
 [1 0]]
```
