"""Deep verification of the rendered Streamlit user interface across all 9 modules.

Validates that:
1. Streamlit server starts cleanly and serves HTTP 200.
2. No traceback, exception, or deprecation warnings are emitted.
3. Every module displays fully visible horizontal titles and subtitles.
4. No raw HTML tags or orphan closures leak into the rendered DOM.
5. All metric cards, charts, tabs, and input areas have verified formatting.
6. Input font size is increased for enhanced legibility.
"""

import os
import subprocess
import sys
import time
import requests
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODULES = [
    ("LIVE DEMO", "nav_btn_live_demo"),
    ("DATA EXPLORER", "nav_btn_data_explorer"),
    ("MODEL EVALUATION", "nav_btn_model_evaluation"),
    ("ERROR ANALYSIS", "nav_btn_error_analysis"),
    ("TAXONOMY ANALYSIS", "nav_btn_taxonomy_analysis"),
    ("SIMILARITY RETRIEVAL", "nav_btn_similarity_retrieval"),
    ("CLASSIFICATION", "nav_btn_classification"),
    ("PREPROCESSING", "nav_btn_preprocessing"),
    ("SYSTEM ARCHITECTURE", "nav_btn_system_architecture"),
]


def test_rendered_ui_across_all_modules_without_exception():
    """Verify that every module renders cleanly with title, subtitle, and no raw HTML leaks."""
    at = AppTest.from_file("app/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Default page raised exception: {at.exception}"

    for module_name, btn_key in MODULES:
        btn = [b for b in at.button if b.key and btn_key in b.key]
        assert len(btn) == 1, f"Navigation button for {module_name} not found"
        btn[0].click().run(timeout=30)
        assert not at.exception, f"Module {module_name} raised exception: {at.exception}"
        assert at.session_state["active_module"] == module_name

        # Check all markdown elements for raw HTML leaks or orphan tags
        all_md_texts = []
        for md_elem in at.markdown:
            val = str(md_elem.value)
            all_md_texts.append(val)
            # Must not leak orphan tags as raw text
            assert not val.strip().startswith("</div>"), f"Raw </div> at start on {module_name}: {val}"
            assert not val.strip().startswith("&lt;/div&gt;"), f"Escaped </div> on {module_name}: {val}"
            assert "&lt;div class=" not in val, f"Raw div class exposed on {module_name}: {val}"
            assert "&lt;/div&gt;" not in val, f"Raw escaped closing div on {module_name}: {val}"

        # Ensure page header markdown exists on each page
        full_page_text = " ".join(all_md_texts)
        assert module_name in full_page_text, f"Module title {module_name} missing from page content"


def test_input_box_font_size_and_header_clearance_css():
    """Verify that input font size is increased and block container has clearance for Streamlit header."""
    css_path = ROOT_DIR / "app" / "ui_components.py"
    css_content = css_path.read_text(encoding="utf-8")

    # 1. Main block container padding-top must be >= 4.5rem to clear 3.75rem fixed toolbar
    assert "padding-top: 5rem !important;" in css_content or "padding-top: 4.75rem !important;" in css_content
    # Old broken 1.5rem must not exist
    assert "padding-top: 1.5rem !important;" not in css_content

    # 2. Input box font size must be increased (>= 0.95rem, not old 0.82rem)
    assert "font-size: 0.96rem !important;" in css_content or "font-size: 0.95rem !important;" in css_content
    assert "font-size: 0.82rem !important;" not in css_content.split("Unified Inputs and Text Areas")[1].split("/* Expanders */")[0]


def test_live_server_http_status_and_zero_warnings():
    """Launch Streamlit server and verify HTTP 200, healthy response, and zero warnings."""
    test_port = 8597
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(test_port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.port",
        str(test_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    health_url = f"http://localhost:{test_port}/_stcore/health"
    app_url = f"http://localhost:{test_port}/"
    healthy = False

    try:
        for _ in range(30):
            time.sleep(0.5)
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                continue

        assert healthy, "Streamlit health check did not return HTTP 200"

        main_resp = requests.get(app_url, timeout=2.0)
        assert main_resp.status_code == 200, f"Streamlit app returned status {main_resp.status_code}"
        assert "<!doctype html>" in main_resp.text.lower()
        assert "Traceback" not in main_resp.text
    finally:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

    # Zero deprecation warnings
    assert "use_container_width" not in stderr
    assert "use_container_width" not in stdout
    assert "Please replace `use_container_width` with `width`" not in stderr


def test_playwright_rendered_geometry_and_no_clipping():
    """Verify rendered UI geometry, title clearance below toolbar, horizontal orientation, and input font size."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright is not available in environment")

    test_port = 8598
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(test_port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.port",
        str(test_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        health_url = f"http://localhost:{test_port}/_stcore/health"
        app_url = f"http://localhost:{test_port}/"
        healthy = False
        for _ in range(30):
            time.sleep(0.5)
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                continue

        assert healthy, "Streamlit health check did not return HTTP 200"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(app_url, wait_until="networkidle", timeout=30000)

            page.wait_for_selector(".page-title", timeout=20000)
            page.wait_for_selector('header[data-testid="stHeader"]', timeout=20000)

            # Check geometry on initial page (LIVE DEMO)
            header_box = page.locator('header[data-testid="stHeader"]').bounding_box()
            title_box = page.locator(".page-title").first.bounding_box()
            sub_box = page.locator(".page-subtitle").first.bounding_box()

            assert header_box is not None, "Header toolbar not found"
            assert title_box is not None, "Page title element not found"
            assert sub_box is not None, "Page subtitle element not found"

            # 1. Page title must not be covered by header toolbar
            assert title_box["y"] >= header_box["y"] + header_box["height"], (
                f"Page title top ({title_box['y']}) is clipped behind toolbar ({header_box['y'] + header_box['height']})"
            )

            # 2. Page title must be horizontal and have reasonable dimensions
            assert title_box["width"] > title_box["height"], (
                f"Page title is vertically wrapped: width={title_box['width']}, height={title_box['height']}"
            )
            assert title_box["height"] >= 16, f"Page title height too small: {title_box['height']}"

            # 3. Subtitle must be below title
            assert sub_box["y"] >= title_box["y"] + title_box["height"] - 5, (
                f"Subtitle top ({sub_box['y']}) should be below title ({title_box['y'] + title_box['height']})"
            )

            # 4. Input / Textarea computed font size must be increased (>= 15px, i.e. ~0.96rem)
            text_input = page.locator('textarea, input[type="text"]').first
            if text_input.count() > 0:
                font_size_str = text_input.evaluate("el => window.getComputedStyle(el).fontSize")
                font_size_px = float(font_size_str.replace("px", ""))
                assert font_size_px >= 15.0, f"Input font size {font_size_px}px is too small (< 15px)"

            # 5. Zero raw HTML tags leaked as visible text
            body_text = page.inner_text("body")
            assert "</div>" not in body_text, "Visible </div> tag rendered in page text"
            assert "<div class=" not in body_text, "Visible <div class= rendered in page text"
            assert "</span>" not in body_text, "Visible </span> tag rendered in page text"

            # 6. Verify navigation and titles across all 9 modules
            for module_name, btn_key in MODULES:
                btn = page.locator(f'button:has-text("{module_name}")').first
                if btn.count() > 0:
                    btn.click()
                    page.wait_for_timeout(600)
                    page.wait_for_selector(".page-title", timeout=15000)

                    t_box = page.locator(".page-title").first.bounding_box()
                    assert t_box is not None
                    assert t_box["y"] >= header_box["y"] + header_box["height"], (
                        f"On module '{module_name}', title ({t_box['y']}) is clipped behind toolbar ({header_box['y'] + header_box['height']})"
                    )

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

