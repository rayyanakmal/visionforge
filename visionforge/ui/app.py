"""VisionForge — Streamlit UI.

Screens:
- Report card: hero metrics, per-class table, IoU histogram, confusion matrix.
- Compare: run A vs run B deltas with regressed/improved badges + verdict banner.
- About: what it does, input formats, repo link.

Zero API keys, no external calls, sample data pre-loaded (evalforge precedent).
"""

import html
import json
import sys
from pathlib import Path

# streamlit run puts the script's dir on sys.path, not the repo root —
# make the visionforge package importable when the app is launched as
# `streamlit run visionforge/ui/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from visionforge.schema import GroundTruth  # noqa: E402
from visionforge.ui import display  # noqa: E402

st.set_page_config(page_title="VisionForge — Vision Model Report Card", page_icon="▦", layout="wide")

CSS = Path(__file__).resolve().parent / "style.css"
if CSS.exists():
    st.markdown(
        """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
GT_SAMPLE = str(EXAMPLES / "gt_sample.json")
PREDS_A = str(EXAMPLES / "preds_run_a.json")
PREDS_B = str(EXAMPLES / "preds_run_b.json")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def parse_upload(uploaded, kind: str):
    """Parse an uploaded JSON file into a dict (or None on error)."""
    try:
        return json.loads(uploaded.getvalue().decode("utf-8"))
    except Exception as e:
        st.error(f"Could not parse {uploaded.name}: {e}")
        return None


def load_sample_data() -> dict | None:
    """Load the committed sample (GT + run A + run B) or explain why not."""
    missing = [p for p in (GT_SAMPLE, PREDS_A, PREDS_B) if not Path(p).exists()]
    if missing:
        st.error("Sample data not found. Run `python data/generate_demo_predictions.py` first.")
        return None
    try:
        with open(GT_SAMPLE) as f:
            gt_meta = GroundTruth.model_validate(json.load(f))
        run_a = display.evaluate_files(GT_SAMPLE, PREDS_A)
        run_b = display.evaluate_files(GT_SAMPLE, PREDS_B)
        return {
            "run_a": run_a,
            "run_b": run_b,
            "gt_meta": gt_meta,
        }
    except ValueError as e:
        st.error(f"Sample data failed to load: {e}")
        return None


def build_state() -> dict:
    """Load data from the sidebar selection into a dict with gt / run_a / run_b."""
    state: dict = {}

    with st.sidebar:
        st.title("VisionForge")
        source = st.radio("Data source", ["Sample demo", "Upload your own"], index=0)

        if source == "Sample demo":
            st.caption("Run A = YOLOv8n conf 0.25 · Run B = conf 0.90 (degraded)")
            data = load_sample_data()
            if data is None:
                st.stop()
            return data

        # Upload flow
        gt_file = st.file_uploader("Ground truth (COCO JSON)", type=["json"])
        preds_files = st.file_uploader(
            "Predictions JSON (one or two runs)",
            type=["json"],
            accept_multiple_files=True,
        )
        if not gt_file or not preds_files:
            st.info("Upload a ground truth file and one or two prediction files.")
            st.stop()

        gt_data = parse_upload(gt_file, "ground truth")
        if gt_data is None:
            st.stop()
        try:
            gt = GroundTruth.model_validate(gt_data)
            state["gt_meta"] = gt
        except Exception as e:
            st.error(f"Ground truth file is not valid COCO JSON: {e}")
            st.stop()

        runs = []
        for f in preds_files:
            preds_data = parse_upload(f, "predictions")
            if preds_data is None:
                continue
            try:
                preds = display.predictions_from_data(preds_data)
                runs.append((f.name, preds))
            except Exception as e:
                st.error(f"Predictions file {f.name} is not valid COCO JSON: {e}")

        if not runs:
            st.stop()

        # Rename runs A/B by upload order for the compare view
        run_a = runs[0][1] if len(runs) >= 1 else None
        run_b = runs[1][1] if len(runs) >= 2 else None

        # Evaluate against GT (we hold the loaded GT model, not a path)
        gt_model = state["gt_meta"]
        try:
            if run_a is not None:
                state["run_a"] = display.evaluate_from_models(run_a, gt_model)
            if run_b is not None:
                state["run_b"] = display.evaluate_from_models(run_b, gt_model)
        except ValueError as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

        return state


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def verdict_banner(compare: dict, run_a: dict, run_b: dict, label_a: str, label_b: str):
    """Verdict callout for the compare view."""
    verdict = compare["verdict"]
    agg_a = run_a["aggregate"]
    agg_b = run_b["aggregate"]
    n_regressed = len(compare["summary"]["regressed_classes"])

    # Aggregate mAP can rise while individual classes regressed (others
    # improved more) — report the mean drop across regressed classes instead.
    regressed_drops = [
        abs(compare["deltas"][cid]["delta_map50"])
        for cid in compare["summary"]["regressed_classes"]
        if compare["deltas"][cid]["delta_map50"] is not None
    ]
    avg_drop = sum(regressed_drops) / len(regressed_drops) if regressed_drops else 0.0

    if verdict == "REGRESSED":
        st.markdown(
            f"""<div class="callout danger">
  <span class="callout-label">{label_a} → {label_b}</span>
  <span>mAP@0.5 <strong>{agg_a['map50']:.3f} → {agg_b['map50']:.3f}</strong> —
  <strong>{n_regressed} class{'es' if n_regressed != 1 else ''}</strong>
  got worse, averaging {avg_drop:.3f} per class. See the table below for which object types moved.</span>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="callout success">
  <span class="callout-label">{label_a} → {label_b}</span>
  <span>mAP@0.5 <strong>{agg_a['map50']:.3f} → {agg_b['map50']:.3f}</strong> —
  no regressions detected. Clean release.</span>
</div>""",
            unsafe_allow_html=True,
        )


def status_badge(status: str) -> str:
    """Badge pill HTML for a delta status."""
    tone = {"regressed": "red", "improved": "green", "ok": "neutral", "n/a": "neutral"}
    return f'<span class="badge {tone.get(status, "neutral")}">{status}</span>'


def render_report_card(metrics: dict, gt_meta, title: str):
    """Single-run report card: hero metrics, per-class, IoU, confusion."""
    st.subheader(title)
    agg = metrics["aggregate"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP@0.5", f"{agg['map50']:.3f}")
    c2.metric("mAP@0.5:0.95", f"{agg['map']:.3f}")
    c3.metric("F1", f"{agg['f1']:.3f}")
    c4.metric("TP / FP / FN", f"{agg['tp']} / {agg['fp']} / {agg['fn']}")

    names = display.category_names(gt_meta)
    per_class = display.per_class_frame(metrics["per_class"], names)

    st.markdown("**Per-class results** — best first, 3 decimals.")
    st.dataframe(per_class, width="stretch", hide_index=True)

    # IoU histogram + confusion matrix side by side
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**IoU of matched detections**")
        hist = display.iou_histogram(metrics["iou_values"])
        if hist.empty:
            st.caption("No matched detections.")
        else:
            st.bar_chart(hist, height=280)
    with col_r:
        st.markdown("**Confusion matrix** — rows = GT, cols = predicted, last = background")
        cf = display.confusion_frame(metrics["confusion"], names, gt_meta.categories)
        st.dataframe(cf, width="stretch", height=280)

    # Per-image detail
    with st.expander("Per-image detail — hardest images first"):
        img_df = display.per_image_frame(metrics["per_image"])
        st.dataframe(img_df, width="stretch", hide_index=True)


def render_compare(compare: dict, run_a: dict, run_b: dict, gt_meta, label_a: str, label_b: str):
    """Two-run comparison: verdict banner, aggregate side-by-side, delta table."""
    st.subheader(f"Compare: {label_a} vs {label_b}")
    verdict_banner(compare, run_a, run_b, label_a, label_b)

    # Aggregate side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{label_a} — aggregate**")
        st.dataframe(display.aggregate_row(run_a["aggregate"]), width="stretch", hide_index=True)
    with col_b:
        st.markdown(f"**{label_b} — aggregate**")
        st.dataframe(display.aggregate_row(run_b["aggregate"]), width="stretch", hide_index=True)

    st.markdown("**Per-class deltas (candidate − baseline)** — regressed first.")
    names = display.category_names(gt_meta)
    deltas = display.delta_frame(compare, names)
    if deltas.empty:
        st.caption("No classes to compare.")
    else:
        # Badges as HTML in the status column (internal enum, safe).
        # Class names come from the user's GT file — escape them so a crafted
        # category name cannot inject HTML/JS into the page (XSS).
        deltas["status"] = deltas["status"].map(status_badge)
        for col in ("class", "delta_map50", "delta_f1"):
            deltas[col] = deltas[col].map(html.escape)
        st.markdown(deltas.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Report card for each run in tabs
    tabs = st.tabs([f"{label_a} — full report", f"{label_b} — full report"])
    with tabs[0]:
        render_report_card(run_a, gt_meta, f"{label_a} report card")
    with tabs[1]:
        render_report_card(run_b, gt_meta, f"{label_b} report card")


def render_about():
    """About screen: what it is, formats, repo."""
    st.subheader("About VisionForge")
    st.markdown(
        "VisionForge grades detection models against labeled images and shows when a "
        "new model version makes them worse. Feed it COCO-style ground truth plus "
        "predictions from any detector (YOLO, DETR, R-CNN) and get a report card: "
        "mAP, per-class precision/recall/F1, IoU distribution, and a confusion matrix. "
        "Compare two runs to catch regressions before customers do."
    )
    st.markdown(
        "**Why it exists:** model metrics like mAP hide the detail that matters — "
        "*which* object types broke and *which images* are hardest. This tool surfaces "
        "both, with a `--fail-on-regression` CI flag so a bad release fails the build."
    )
    st.markdown("**Input format (COCO):**")
    st.code(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40]}],
                "categories": [{"id": 1, "name": "person"}],
            },
            indent=2,
        ),
        language="json",
    )
    st.markdown(
        "Predictions are the same but a bare list with a `score` per detection:\n"
        '`[{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40], "score": 0.87}]`'
    )
    st.markdown(
        "**Try it from the CLI:**\n"
        "```\n"
        "visionforge evaluate --gt examples/gt_sample.json --preds examples/preds_run_a.json\n"
        "visionforge compare --gt examples/gt_sample.json --base examples/preds_run_a.json \\\n"
        "    --candidate examples/preds_run_b.json --fail-on-regression\n"
        "```"
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="hero">
  <span class="eyebrow">Vision model evaluation</span>
  <h1 class="hero-title">The report card for your vision models</h1>
  <p class="hero-sub">Grade detection models against labeled images — mAP, per-class
  precision/recall, IoU, confusion. Compare two runs to catch a release that made
  things worse, before your customers do.</p>
</div>""",
    unsafe_allow_html=True,
)

state = build_state()

tab_report, tab_compare, tab_about = st.tabs(["Report card", "Compare", "About"])

with tab_report:
    if "run_a" in state:
        render_report_card(state["run_a"], state["gt_meta"], "Run A — report card")
    else:
        st.info("Upload at least one prediction file to see a report card.")

with tab_compare:
    if "run_a" in state and "run_b" in state:
        render_compare(
            display.compare_runs(state["run_a"], state["run_b"]),
            state["run_a"],
            state["run_b"],
            state["gt_meta"],
            "Run A",
            "Run B",
        )
    elif "run_a" in state:
        st.info("Upload a second prediction file to compare runs.")
    else:
        st.info("Upload two prediction files to compare runs.")

with tab_about:
    render_about()

st.markdown(
    """<div class="footer">
  <span>VisionForge v0.1.0 · MIT</span>
  <span><a href="https://github.com/rayyanakmal/visionforge">GitHub</a> · Streamlit</span>
</div>""",
    unsafe_allow_html=True,
)
