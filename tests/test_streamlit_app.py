"""Integration test for Streamlit application startup and HTTP 200 response."""

import os
import subprocess
import sys
import time
import requests
import pytest

PORT = 8599


def test_streamlit_app_startup_http_200():
    """Start Streamlit app in headless mode and verify it responds with HTTP 200."""
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(PORT)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    url = f"http://localhost:{PORT}/_stcore/health"
    app_url = f"http://localhost:{PORT}/"
    healthy = False

    try:
        # Poll for server readiness up to 15 seconds
        for _ in range(30):
            time.sleep(0.5)
            try:
                resp = requests.get(url, timeout=1.0)
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                continue

        assert healthy, "Streamlit health endpoint did not respond with HTTP 200"

        # Verify main application page returns HTTP 200
        main_resp = requests.get(app_url, timeout=2.0)
        assert main_resp.status_code == 200, f"Main app returned status {main_resp.status_code}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_apptest_live_demo_defaults_and_navigation():
    """Verify AppTest initializes on LIVE DEMO and switches modules cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py")
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["active_module"] == "LIVE DEMO"
    assert at.text_area[0].value == ""

    # Switch to DATA EXPLORER
    nav_btns = [b for b in at.button if b.key and "nav_btn_data_explorer" in b.key]
    assert len(nav_btns) == 1
    nav_btns[0].click().run(timeout=30)
    assert not at.exception
    assert at.session_state["active_module"] == "DATA EXPLORER"

    # Switch back to LIVE DEMO
    demo_btns = [b for b in at.button if b.key and "nav_btn_live_demo" in b.key]
    assert len(demo_btns) == 1
    demo_btns[0].click().run(timeout=30)
    assert not at.exception
    assert at.session_state["active_module"] == "LIVE DEMO"


def test_apptest_live_demo_lifecycle_and_actions():
    """Verify example population, model inference, and clear actions via AppTest."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py")
    at.run(timeout=30)
    assert not at.exception

    # 1. Click example button (Unauthorized Credit Card Payment)
    ex_btn = [b for b in at.button if b.key and "ex_btn_0" in b.key][0]
    ex_btn.click().run(timeout=30)
    assert not at.exception
    assert "unauthorized transactions" in at.text_area[0].value.lower()
    # Ensure clicking example did NOT automatically run inference
    assert "live_analyzed_data" not in at.session_state or at.session_state["live_analyzed_data"] is None

    # 2. Click ANALYZE COMPLAINT
    analyze_btn = [b for b in at.button if b.label == "ANALYZE COMPLAINT"][0]
    analyze_btn.click().run(timeout=30)
    assert not at.exception
    assert "live_analyzed_data" in at.session_state
    analyzed = at.session_state["live_analyzed_data"]
    assert analyzed is not None
    assert "Credit card" in analyzed["pred_category"]
    assert 0.0 <= analyzed["confidence"] <= 1.0
    assert analyzed["feature_dim"] == 237148
    assert analyzed["active_nnz"] > 0

    # 3. Click CLEAR button
    clear_btn = [b for b in at.button if b.label == "CLEAR"][0]
    clear_btn.click().run(timeout=30)
    assert not at.exception
    assert at.text_area[0].value == ""
    assert at.session_state["live_analyzed_data"] is None

    # 4. Click ANALYZE COMPLAINT on empty input -> warning displayed, no crash
    analyze_btn = [b for b in at.button if b.label == "ANALYZE COMPLAINT"][0]
    analyze_btn.click().run(timeout=30)
    assert not at.exception
    assert len(at.warning) >= 1
    assert any("Please enter a customer complaint" in w.value for w in at.warning)
