"""Capture README screenshots of the deployed VisionForge dashboard.

- report-card.png: the Report card tab (hero metrics, per-class table, IoU
  histogram, confusion matrix) — the default tab on load. When the default
  sample pair is loaded, both run cards show as nested tabs (Run A / Run B).
- compare.png: the Compare tab (verdict banner, aggregate side-by-side,
  per-class delta table with regressed/improved badges).

The Streamlit Cloud app is wrapped in an iframe at /~/+/; the outer page's
full_page height tracks the viewport, so a tall viewport captures the app.
"""
import time

from playwright.sync_api import sync_playwright

URL = "https://visionforge-sxbb3ltfcuftm66zfzvgkx.streamlit.app/"
OUT = "/home/rayyan/projects/visionforge/assets"


def find_app_frame(page):
    deadline = time.time() + 150
    while time.time() < deadline:
        for f in page.frames:
            if "/~/+/" in f.url:
                try:
                    if "VisionForge" in f.inner_text("body"):
                        return f
                except Exception:
                    continue
        page.wait_for_timeout(3_000)
    return None


def wait_for_text(frame, text, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if text in frame.inner_text("body"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 3000})
        page.goto(URL, wait_until="domcontentloaded", timeout=180_000)
        page.wait_for_timeout(5_000)

        frame = find_app_frame(page)
        if frame is None:
            print("ERROR: app frame not found after 150s")
            browser.close()
            return

        # Default tab is Report card (with nested Run A / Run B cards).
        ok = wait_for_text(frame, "Report card")
        print("Report card tab marker found:", ok)
        page.wait_for_timeout(12_000)  # let tables/plots settle
        page.screenshot(path=f"{OUT}/report-card.png", full_page=True)
        print("saved report-card.png")

        # Click the Compare tab (main tab row: div[role=tab][data-testid=stTab]).
        try:
            frame.locator(
                'div[data-testid="stTab"][role="tab"]', has_text="Compare"
            ).first.click(timeout=15_000)
            print("clicked Compare tab")
        except Exception as e:
            print("Compare tab click failed:", e)

        ok2 = wait_for_text(frame, "classes got worse")
        print("Compare verdict marker found:", ok2)
        page.wait_for_timeout(12_000)  # let the delta table render
        page.screenshot(path=f"{OUT}/compare.png", full_page=True)
        print("saved compare.png")

        browser.close()


if __name__ == "__main__":
    main()
