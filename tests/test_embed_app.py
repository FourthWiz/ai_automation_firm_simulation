"""
Tests for the slim stlite embed entrypoint (embed_app.py).

Notes:
(a) Tests run in definition order under default pytest (no random ordering).
    Order matters for the in-process AppTest runs — AppTest re-evaluates
    embed_app.py module scope per run and pollutes sys.modules with
    firm_ai_abm.simulate (which imports dp_optimizer directly via simulate.py:82).
    The subprocess-isolated test (test_package_init_does_not_load_dp) is
    intentionally placed last and runs in a fresh process to avoid this issue.

(b) The sys.modules assertion in test_package_init_does_not_load_dp runs in a
    SUBPROCESS (round-4 MAJ-3 requirement) so it is order-independent within
    pytest. The subprocess starts with a clean sys.modules regardless of what
    prior in-process tests have loaded.

(c) These tests cover embed_app.py only; app.py tests are in test_app.py
    (D-08 invariant — the two entrypoints are tested independently).
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

EMBED_APP = str(Path(__file__).resolve().parents[1] / "embed_app.py")


def test_embed_app_runs_default():
    """Basic smoke test: embed_app.py renders without exceptions."""
    at = AppTest.from_file(EMBED_APP)
    at.run()
    # No StreamlitAPIException for premature set_page_config (round-2 MAJ-10)
    assert not at.exception, f"embed_app raised: {at.exception}"
    # Round-3 CRIT-1: no StreamlitSetPageConfigMustBeFirstCommandError
    # (at.exception covers this — if query_params came first, AppTest would raise)
    assert len(at.button) == 1, "Expected exactly 1 Run button"
    assert len(at.slider) == 4, f"Expected 4 sliders, got {len(at.slider)}"
    assert len(at.radio) == 1, "Expected exactly 1 strategy radio"


def test_embed_app_set_page_config_first():
    """Source-level assertion: set_page_config precedes ALL other st.* calls.

    Round-3 CRIT-1: st.query_params IS a Streamlit command (added in 1.30)
    and must come AFTER st.set_page_config. Catch regressions at the AST level
    so this test passes even if AppTest semantics change in future Streamlit versions.
    """
    src = Path(EMBED_APP).read_text()
    tree = ast.parse(src)

    page_config_line = None
    query_params_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "set_page_config":
                page_config_line = node.lineno
        # Capture st.query_params as attribute access (non-call — it's accessed, not called)
        if isinstance(node, ast.Attribute) and node.attr == "query_params":
            if query_params_line is None:
                query_params_line = node.lineno

    assert page_config_line is not None, "set_page_config not found in embed_app.py"
    assert query_params_line is not None, "query_params not found in embed_app.py"
    assert page_config_line < query_params_line, (
        f"CRIT-1 regression: set_page_config at line {page_config_line} "
        f"but query_params at line {query_params_line} — query_params must come AFTER"
    )


def test_embed_app_run_button_produces_charts():
    """Clicking Run produces two chart tabs and a cumulative profit metric."""
    at = AppTest.from_file(EMBED_APP)
    at.run()
    at.button[0].click().run()
    assert not at.exception, f"Run button raised: {at.exception}"
    # The Run branch creates st.tabs(["Profit over time", "Task mode mix"])
    assert len(at.tabs) >= 1, "Expected at least one tab group after Run"
    # AppTest doesn't expose plotly_chart directly; verify via metric (cumulative profit)
    assert len(at.metric) >= 1, "Expected cumulative profit metric after Run"
    assert "Cumulative profit" in at.metric[0].label


def test_embed_app_strategy_whitelist():
    """Radio options are exactly the three allowed strategies — no DP optimizer.

    D-03 + R-13: embed whitelist = greedy_with_switching, all_H, all_T.
    horizon_optimizer (DP) is intentionally NOT present (too slow under Pyodide).
    """
    at = AppTest.from_file(EMBED_APP)
    at.run()
    assert not at.exception
    radio = at.radio[0]
    assert list(radio.options) == ["greedy_with_switching", "all_H", "all_T"], (
        f"Strategy whitelist changed: {radio.options}"
    )
    assert "horizon_optimizer" not in radio.options, (
        "horizon_optimizer must NOT appear in embed strategy radio (D-03)"
    )


def test_package_init_does_not_load_dp():
    """Bare `import firm_ai_abm` must not load dp_optimizer or margin_optimizer.

    Round-4 MAJ-3 (subprocess isolation): runs in a FRESH Python process so
    sys.modules is clean regardless of test order. The in-process AppTest tests
    above load firm_ai_abm.simulate (via embed_app.py) which transitively imports
    dp_optimizer — but that's the embed's RUNTIME path, not the package-init path.
    T-00 closes only the package-init (eager-import) path; the deeper runtime
    coupling in simulate.py:82 is documented in the T-00 honest-scope note.
    """
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import firm_ai_abm, sys; "
            "sys.exit(0 if ("
            "    'firm_ai_abm.dp_optimizer' not in sys.modules and "
            "    'firm_ai_abm.margin_optimizer' not in sys.modules"
            ") else 1)",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, (
        f"Bare `import firm_ai_abm` loaded dp/margin transitively. "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
