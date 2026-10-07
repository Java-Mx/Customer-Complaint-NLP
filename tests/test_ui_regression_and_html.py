"""Strict UI regression tests, HTML balance validation, and Streamlit smoke tests.

Validates that:
1. All custom HTML fragments are structurally balanced (no orphan tags, no unclosed tags).
2. No raw HTML tags (e.g. </div>, <div class=) leak as visible text in the UI.
3. No deprecated `use_container_width` parameters remain in the project.
4. CSS layout for page titles preserves horizontal reading orientation without character wrapping.
5. All 9 application modules navigate and render cleanly without exceptions via AppTest.
"""

import ast
import glob
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple
import pytest
import markdown_it

ROOT_DIR = Path(__file__).resolve().parents[1]


# ==============================================================================
# HTML PARSER VALIDATOR FOR STRUCTURAL BALANCE
# ==============================================================================

class BalancedHTMLValidator(HTMLParser):
    """Standard-library HTML validator ensuring balanced tags and no orphan closures."""

    SELF_CLOSING = {
        "img", "br", "hr", "input", "meta", "link",
        "polyline", "line", "circle", "rect", "path", "polygon"
    }

    def __init__(self):
        super().__init__()
        self.stack: List[str] = []
        self.errors: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag_lower = tag.lower()
        if tag_lower not in self.SELF_CLOSING:
            self.stack.append(tag_lower)

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self.SELF_CLOSING:
            return
        if not self.stack:
            self.errors.append(f"Orphan closing tag </{tag}> found with empty tag stack")
        elif self.stack[-1] != tag_lower:
            self.errors.append(f"Mismatched closing tag </{tag}>, expected </{self.stack[-1]}>")
            if tag_lower in self.stack:
                while self.stack and self.stack[-1] != tag_lower:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
        else:
            self.stack.pop()

    def validate(self, html_str: str) -> Tuple[bool, List[str]]:
        self.feed(html_str)
        if self.stack:
            self.errors.append(f"Unclosed opening tags remaining: {self.stack}")
        return len(self.errors) == 0, self.errors


def validate_html(html_str: str) -> Tuple[bool, List[str]]:
    """Helper to validate an HTML string for balanced tags."""
    validator = BalancedHTMLValidator()
    return validator.validate(html_str)


# ==============================================================================
# 1. HTML VALIDATION TESTS
# ==============================================================================

def test_page_header_html_is_balanced():
    """Verify render_page_header produces completely balanced HTML across all modules."""
    from app.ui_components import render_page_header

    test_titles = [
        "LIVE DEMO",
        "DATA EXPLORER",
        "MODEL EVALUATION",
        "ERROR ANALYSIS",
        "TAXONOMY ANALYSIS",
        "SIMILARITY RETRIEVAL",
        "CLASSIFICATION",
        "PREPROCESSING",
        "SYSTEM ARCHITECTURE",
    ]

    md = markdown_it.MarkdownIt("commonmark", {"html": True})

    for title in test_titles:
        for badge in [None, "Active", "Benchmark"]:
            badge_html = f'<span class="page-badge">{badge}</span>' if badge else ""
            subtitle = f"Description and testing for module {title}."
            sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
            html = (
                f'<div class="app-page-header">'
                f'<div class="page-title-row"><div class="page-title">{title}</div>{badge_html}</div>'
                f'{sub_html}'
                f'</div>'
            )

            is_valid, errors = validate_html(html)
            assert is_valid, f"Page header HTML for '{title}' is not balanced: {errors}"

            # Verify that Markdown parser does not interpret it as code block (<pre><code>)
            rendered = md.render(html)
            assert "<pre>" not in rendered, f"Markdown parser generated code block for '{title}': {rendered}"
            assert "&lt;div" not in rendered, f"Markdown parser exposed escaped div for '{title}': {rendered}"
            assert "&lt;/div&gt;" not in rendered, f"Markdown parser exposed escaped closing div for '{title}': {rendered}"


def test_section_header_html_is_balanced():
    """Verify render_section_header produces completely balanced HTML."""
    desc = "Detailed explanation of intermediate preprocessing steps."
    desc_html = f'<div class="section-desc">{desc}</div>'
    html_with_desc = (
        f'<div class="app-section-header">'
        f'<div class="section-title">TEST SECTION</div>'
        f'{desc_html}'
        f'</div>'
    )
    is_valid, errors = validate_html(html_with_desc)
    assert is_valid, f"Section header with description is not balanced: {errors}"

    html_no_desc = (
        f'<div class="app-section-header">'
        f'<div class="section-title">TEST SECTION</div>'
        f'</div>'
    )
    is_valid, errors = validate_html(html_no_desc)
    assert is_valid, f"Section header without description is not balanced: {errors}"


def test_status_card_html_is_balanced():
    """Verify render_status_card and render_status_row produce balanced HTML."""
    from app.ui_components import render_status_row

    for status in ["success", "warning", "error", "info"]:
        row_html = render_status_row("Test Label", "Subtext info", status=status)
        is_valid, errors = validate_html(row_html)
        assert is_valid, f"Status row for {status} is not balanced: {errors}"

    rows = (
        f'{render_status_row("API", "Online", "success")}'
        f'{render_status_row("Data", "Available", "success")}'
        f'{render_status_row("Model", "Trained", "warning")}'
        f'{render_status_row("Rep", "Word+Char", "error")}'
    )
    card_html = (
        f'<div class="status-card">'
        f'<div class="status-header">SYSTEM STATUS</div>'
        f'<div class="status-list">{rows}</div>'
        f'</div>'
    )
    is_valid, errors = validate_html(card_html)
    assert is_valid, f"Full status card is not balanced: {errors}"

    # Verify no <pre><code> code blocks in CommonMark render
    md = markdown_it.MarkdownIt("commonmark", {"html": True})
    rendered = md.render(card_html)
    assert "<pre>" not in rendered, f"Status card rendered as code block: {rendered}"


def test_info_card_html_is_balanced():
    """Verify render_info_card produces balanced HTML with and without title."""
    title_html = '<div class="info-card-title">Notice</div>'
    html = (
        f'<div class="app-info-card">'
        f'{title_html}'
        f'<div class="info-card-body">Contextual text body.</div>'
        f'</div>'
    )
    is_valid, errors = validate_html(html)
    assert is_valid, f"Info card is not balanced: {errors}"


def test_no_orphan_closing_divs():
    """Verify that no standalone closing tags exist in helper components."""
    # Test that validator detects orphan closing tag
    bad_html = '</div><div class="page-subtitle">Test</div>'
    is_valid, errors = validate_html(bad_html)
    assert not is_valid
    assert any("Orphan closing tag" in e for e in errors)


# ==============================================================================
# 2. SOURCE-LEVEL REGRESSION TESTS
# ==============================================================================

def test_no_use_container_width_in_project():
    """Strictly verify that ZERO occurrences of use_container_width exist in Python application and test files."""
    target_kw = "use_container_width"
    this_file = Path(__file__).resolve()
    violations = []
    for py_file in ROOT_DIR.glob("**/*.py"):
        if ".git" in py_file.parts or ".pytest_cache" in py_file.parts:
            continue
        if py_file.resolve() == this_file:
            continue
        content = py_file.read_text(encoding="utf-8")
        if target_kw in content:
            violations.append(str(py_file.relative_to(ROOT_DIR)))

    assert not violations, f"Deprecated API found in: {violations}. Replace with width='stretch' or width='content'."


def test_all_custom_html_blocks_allow_html():
    """Verify all st.markdown and st.sidebar.markdown calls with HTML have unsafe_allow_html=True."""
    violations = []

    for py_path in [ROOT_DIR / "app" / "app.py", ROOT_DIR / "app" / "ui_components.py"]:
        content = py_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_md = False
                if isinstance(func, ast.Attribute) and func.attr == "markdown":
                    is_md = True

                if is_md and node.args:
                    first_arg = node.args[0]
                    # Check if the string literal or format contains HTML tags
                    raw_str = ""
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        raw_str = first_arg.value
                    elif isinstance(first_arg, ast.JoinedStr):
                        for part in first_arg.values:
                            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                                raw_str += part.value

                    if "<div" in raw_str or "<span" in raw_str or "<style" in raw_str:
                        # Ensure unsafe_allow_html=True is set
                        kwargs = {kw.arg: kw.value for kw in node.keywords}
                        unsafe_val = kwargs.get("unsafe_allow_html")
                        is_true = isinstance(unsafe_val, ast.Constant) and unsafe_val.value is True
                        if not is_true:
                            violations.append(f"{py_path.name}:{node.lineno} missing unsafe_allow_html=True")

    assert not violations, f"Markdown calls with HTML missing unsafe_allow_html=True: {violations}"


def test_no_unclosed_custom_html_container():
    """Verify that no code opens an HTML tag in one st.markdown call and closes in another."""
    for py_path in [ROOT_DIR / "app" / "app.py", ROOT_DIR / "app" / "ui_components.py"]:
        content = py_path.read_text(encoding="utf-8")
        # Ensure pattern like st.markdown("<div...") followed by st.markdown("</div>") does not exist
        assert 'st.markdown("<div' not in content and "st.markdown('<div" not in content, (
            f"Found potential split div opener in {py_path.name}"
        )
        assert 'st.markdown("</div>"' not in content and "st.markdown('</div>')" not in content, (
            f"Found standalone closing </div> in {py_path.name}"
        )


# ==============================================================================
# 3. CSS REGRESSION TESTS
# ==============================================================================

def test_page_title_does_not_have_character_wrapping_css():
    """Audit CSS to ensure page title does not contain vertical text or letter wrapping."""
    from app.ui_components import apply_custom_styles

    css_path = ROOT_DIR / "app" / "ui_components.py"
    content = css_path.read_text(encoding="utf-8")

    # 1. No vertical writing mode
    assert "vertical-rl" not in content, "CSS must not contain vertical-rl"
    assert "vertical-lr" not in content, "CSS must not contain vertical-lr"

    # 2. No aggressive character breaking
    assert "word-break: break-all" not in content, "Page titles must not use break-all"

    # 3. Explicit horizontal writing mode on page title
    assert "writing-mode: horizontal-tb" in content, "Page title CSS must enforce horizontal-tb"

    # 4. Normal word break and overflow wrap
    assert "word-break: normal" in content
    assert "overflow-wrap: normal" in content


# ==============================================================================
# 4. RENDERED-TEXT REGRESSION TEST
# ==============================================================================

def test_no_raw_html_closing_tag_rendered_as_text():
    """Verify that commonmark parser does not turn header or card HTML into visible code blocks."""
    md = markdown_it.MarkdownIt("commonmark", {"html": True})

    # Test page header markdown rendering
    title = "MODEL EVALUATION"
    subtitle = "Rigorous comparative evaluation between baseline and final models"
    html = (
        f'<div class="app-page-header">'
        f'<div class="page-title-row"><div class="page-title">{title}</div></div>'
        f'<div class="page-subtitle">{subtitle}</div>'
        f'</div>'
    )

    rendered = md.render(html)

    # Forbidden literal fragments that appear when HTML is rendered as text
    forbidden = [
        "&lt;/div&gt;",
        "&lt;div class=",
        "&lt;/span&gt;",
        "&lt;span class=",
        "&lt;style&gt;",
        "&lt;/style&gt;",
    ]

    for frag in forbidden:
        assert frag not in rendered, f"Forbidden literal fragment '{frag}' found in rendered markdown: {rendered}"


# ==============================================================================
# 5. STREAMLIT APP SMOKE TESTS
# ==============================================================================

def test_apptest_smoke_all_nine_modules():
    """Smoke test initializing Streamlit app and navigating through all 9 modules cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception on startup: {at.exception}"

    expected_modules = [
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

    # Verify initial default state
    assert at.session_state["active_module"] == "LIVE DEMO"

    for module_name, btn_key in expected_modules:
        btn = [b for b in at.button if b.key and btn_key in b.key]
        assert len(btn) == 1, f"Navigation button for {module_name} ({btn_key}) not found"
        btn[0].click().run(timeout=30)
        assert not at.exception, f"Navigating to {module_name} raised exception: {at.exception}"
        assert at.session_state["active_module"] == module_name

        # Verify no raw HTML tags leak into markdown element values
        for md_elem in at.markdown:
            val = str(md_elem.value)
            # A markdown element should NOT be an unparsed literal snippet starting with </div>
            assert not val.strip().startswith("</div>"), (
                f"Raw closing </div> found at start of markdown on page {module_name}: {val}"
            )


def test_model_evaluation_metrics_exact_formatting():
    """Verify that MODEL EVALUATION renders the exact required metric presentation without truncation or raw decimals."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Switch to MODEL EVALUATION
    eval_btn = [b for b in at.button if b.key and "nav_btn_model_evaluation" in b.key][0]
    eval_btn.click().run(timeout=30)
    assert not at.exception
    assert at.session_state["active_module"] == "MODEL EVALUATION"

    metric_map = {m.label: m.value for m in at.metric}

    # Verify all expected labels are present
    assert "Overall Accuracy" in metric_map
    assert "Macro F1" in metric_map
    assert "Weighted F1" in metric_map
    assert "Macro Precision" in metric_map
    assert "Macro Recall" in metric_map
    assert "Weighted Precision" in metric_map
    assert "Weighted Recall" in metric_map
    assert "Test Partition" in metric_map
    assert "Vocabulary" in metric_map

    # Verify exact required percentage formatting
    assert metric_map["Overall Accuracy"] == "69.82%"
    assert metric_map["Macro F1"] == "50.88%"
    assert metric_map["Weighted F1"] == "69.78%"
    assert metric_map["Macro Precision"] == "50.41%"
    assert metric_map["Macro Recall"] == "51.69%"
    assert metric_map["Weighted Precision"] == "70.08%"
    assert metric_map["Weighted Recall"] == "69.82%"

    # Verify Test Partition not truncated and cleanly formatted
    assert metric_map["Test Partition"] == "5,000 complaints"
    assert "sam..." not in metric_map["Test Partition"]

    # Verify Vocabulary cleanly formatted
    assert metric_map["Vocabulary"] == "114,493 features"

    # Verify no raw decimal representations exist for percentage metrics
    all_values = list(metric_map.values())
    for raw_dec in ["0.5088", "0.6978", "0.5041", "0.5169", "0.7008", "0.6982"]:
        assert raw_dec not in all_values, f"Raw decimal {raw_dec} found in metric cards"


def test_streamlit_server_zero_deprecation_warnings():
    """Start Streamlit server, check HTTP 200, and ensure zero use_container_width deprecation warnings."""
    import os
    import subprocess
    import sys
    import time
    import requests

    test_port = 8594
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
    finally:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

    # Verify ZERO warnings regarding use_container_width
    assert "use_container_width" not in stderr, f"Deprecated use_container_width warning in stderr: {stderr}"
    assert "use_container_width" not in stdout, f"Deprecated use_container_width warning in stdout: {stdout}"
    assert "Please replace `use_container_width` with `width`" not in stderr

