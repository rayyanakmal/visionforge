"""CLI end-to-end tests (typer CliRunner)."""

import json

import pytest
from typer.testing import CliRunner

from visionforge.cli import app

runner = CliRunner()


def _write_fixture_files(tmp_path, gt, preds):
    gt_path = tmp_path / "gt.json"
    preds_path = tmp_path / "preds.json"
    gt_path.write_text(gt.model_dump_json())
    preds_path.write_text(preds.model_dump_json())
    return gt_path, preds_path


class TestEvaluateCLI:
    def test_evaluate_success(self, gt, preds_full, tmp_path):
        gt_path, preds_path = _write_fixture_files(tmp_path, gt, preds_full)
        out = tmp_path / "report"
        result = runner.invoke(
            app,
            ["evaluate", "--gt", str(gt_path), "--preds", str(preds_path), "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["aggregate"]["map50"] == pytest.approx(253 / 303)
        assert "0.835" in (tmp_path / "report.md").read_text()

    def test_evaluate_invalid_score_fails(self, gt, tmp_path):
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(gt.model_dump_json())
        bad_preds = tmp_path / "bad_preds.json"
        bad_preds.write_text(json.dumps([
            {"image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 1.5}
        ]))
        result = runner.invoke(
            app,
            ["evaluate", "--gt", str(gt_path), "--preds", str(bad_preds)],
        )
        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_evaluate_missing_file_fails(self, gt, tmp_path):
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(gt.model_dump_json())
        result = runner.invoke(
            app,
            ["evaluate", "--gt", str(gt_path), "--preds", str(tmp_path / "nope.json")],
        )
        assert result.exit_code != 0

    def test_evaluate_warns_on_unknown_category(self, gt, preds_full, tmp_path):
        gt_path, preds_path = _write_fixture_files(tmp_path, gt, preds_full)
        # Corrupt the preds file to use an unknown category
        raw = json.loads(preds_path.read_text())
        raw["predictions"][0]["category_id"] = 99
        preds_path.write_text(json.dumps(raw))
        result = runner.invoke(
            app,
            ["evaluate", "--gt", str(gt_path), "--preds", str(preds_path)],
        )
        assert result.exit_code == 0  # warning, not failure
        assert "warning" in result.output.lower()


class TestCompareCLI:
    def test_compare_pass(self, gt, preds_full, tmp_path):
        gt_path, preds_path = _write_fixture_files(tmp_path, gt, preds_full)
        result = runner.invoke(
            app,
            ["compare", "--gt", str(gt_path), "--base", str(preds_path),
             "--candidate", str(preds_path)],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_compare_regression_reported_but_exits_zero(self, gt, preds_full, preds_missing, tmp_path):
        gt_path, base_path = _write_fixture_files(tmp_path, gt, preds_full)
        cand_path = tmp_path / "cand.json"
        cand_path.write_text(preds_missing.model_dump_json())
        result = runner.invoke(
            app,
            ["compare", "--gt", str(gt_path), "--base", str(base_path),
             "--candidate", str(cand_path)],
        )
        assert result.exit_code == 0  # report-only mode
        assert "REGRESSED" in result.output

    def test_compare_fail_on_regression(self, gt, preds_full, preds_missing, tmp_path):
        gt_path, base_path = _write_fixture_files(tmp_path, gt, preds_full)
        cand_path = tmp_path / "cand.json"
        cand_path.write_text(preds_missing.model_dump_json())
        result = runner.invoke(
            app,
            ["compare", "--gt", str(gt_path), "--base", str(base_path),
             "--candidate", str(cand_path), "--fail-on-regression"],
        )
        assert result.exit_code == 1
        assert "REGRESSED" in result.output

    def test_compare_writes_reports(self, gt, preds_full, tmp_path):
        gt_path, base_path = _write_fixture_files(tmp_path, gt, preds_full)
        out = tmp_path / "cmp"
        result = runner.invoke(
            app,
            ["compare", "--gt", str(gt_path), "--base", str(base_path),
             "--candidate", str(base_path), "--out", str(out)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "cmp.json").exists()
        assert (tmp_path / "cmp.md").exists()


class TestInfoCLI:
    def test_info(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Ground truth" in result.output
        assert "Predictions" in result.output
