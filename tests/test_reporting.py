"""Reporting tests — JSON + markdown writers."""

import json

import pytest

from visionforge.metrics import evaluate
from visionforge.reporting import write_json_report, write_markdown_report


class TestJSONReport:
    def test_roundtrip(self, gt, preds_full, tmp_path):
        metrics = evaluate(preds_full.predictions, gt)
        out = tmp_path / "report.json"
        write_json_report(metrics, str(out))

        data = json.loads(out.read_text())
        assert data["aggregate"]["map50"] == pytest.approx(253 / 303)
        assert data["aggregate"]["precision"] == pytest.approx(2 / 3)
        assert "per_class" in data
        assert "per_image" in data
        assert "iou_stats" in data

    def test_confusion_serializable(self, gt, preds_full, tmp_path):
        """Confusion matrix is numpy — must be list-ified for JSON."""
        metrics = evaluate(preds_full.predictions, gt)
        out = tmp_path / "report.json"
        write_json_report(metrics, str(out))
        data = json.loads(out.read_text())
        assert isinstance(data["confusion"], list)
        assert len(data["confusion"]) == 2


class TestMarkdownReport:
    def test_contains_expected_sections(self, gt, preds_full, tmp_path):
        metrics = evaluate(preds_full.predictions, gt)
        out = tmp_path / "report.md"
        write_markdown_report(metrics, str(out))
        text = out.read_text()

        assert "# VisionForge" in text or "VisionForge" in text
        assert "mAP@0.5" in text
        assert "mAP@0.5:0.95" in text
        assert "precision" in text.lower()
        assert "recall" in text.lower()
        assert "0.835" in text  # hand-computed mAP@0.5

    def test_no_emoji(self, gt, preds_full, tmp_path):
        """Report style rule: no emoji."""
        metrics = evaluate(preds_full.predictions, gt)
        out = tmp_path / "report.md"
        write_markdown_report(metrics, str(out))
        text = out.read_text()
        assert "🚀" not in text
        assert "✅" not in text
        assert "❌" not in text

    def test_empty_preds_report(self, gt, preds_empty, tmp_path):
        metrics = evaluate(preds_empty.predictions, gt)
        out = tmp_path / "report.md"
        write_markdown_report(metrics, str(out))
        text = out.read_text()
        assert "0.000" in text  # all-zero metrics render without crash
