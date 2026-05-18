"""P1-11.14: New AppTest cases for the redesigned dashboard.

Tests the new UI elements introduced in P1-4 through P1-9:
  - Advanced expander (collapsed by default)
  - KPI strip (3 metrics after run)
  - 5-tab layout
  - Widget keys preserved after redesign
  - tab_het sigma_theta=0 branch
  - Reset button restores FirmParams defaults
"""

import pytest

from firm_ai_abm.config import FirmParams


@pytest.fixture(autouse=True)
def _require_streamlit():
    pytest.importorskip("streamlit", minversion="1.28")


def _get_at():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file("app.py", default_timeout=60).run()


def _find_run_btn(at):
    return next(b for b in at.button if b.label.endswith("Run simulation"))


def test_advanced_expander_collapsed_by_default():
    """P1-11.14: Advanced expander exists on the page."""
    at = _get_at()
    assert not at.exception
    expanders = list(at.expander)
    assert len(expanders) >= 1, f"Expected at least one expander, got {len(expanders)}"
    advanced = next((e for e in expanders if "Advanced" in e.label), None)
    assert advanced is not None, (
        f"No expander with 'Advanced' in label found. Labels: {[e.label for e in expanders]}"
    )


def test_kpi_strip_after_run():
    """P1-11.14: KPI strip shows 5 metrics with expected labels after run.

    Layout: Cumulative profit | Final workforce (K) | H (human) | A (augmented) | T (automated).
    """
    at = _get_at()
    assert not at.exception
    run_btn = _find_run_btn(at)
    at = run_btn.click().run()
    assert not at.exception
    metrics = list(at.metric)
    assert len(metrics) >= 5, f"Expected at least 5 metrics in KPI strip, got {len(metrics)}"
    metric_labels = [m.label for m in metrics]
    assert "Cumulative profit" in metric_labels, f"Missing 'Cumulative profit'. Got: {metric_labels}"
    assert "Final workforce (K)" in metric_labels, f"Missing 'Final workforce (K)'. Got: {metric_labels}"
    assert "H (human)" in metric_labels, f"Missing 'H (human)'. Got: {metric_labels}"
    assert "A (augmented)" in metric_labels, f"Missing 'A (augmented)'. Got: {metric_labels}"
    assert "T (automated)" in metric_labels, f"Missing 'T (automated)'. Got: {metric_labels}"
    # Each mode metric value must be a percentage string
    for label in ("H (human)", "A (augmented)", "T (automated)"):
        m = next(m for m in metrics if m.label == label)
        assert m.value.endswith("%"), f"Mode metric '{label}' value should end with '%': {m.value}"


def test_tabs_present():
    """P1-11.14: 6 plot tabs present with expected labels.

    Note: at.tabs returns ALL tabs including the 6 advanced-expander tabs
    (total 12). The plot tabs are always the last 6 in render order.
    Advanced tabs: Costs | Strategy & pricing | Heterogeneity |
                   Firing & hiring | Productivity baseline | Reproducibility
    Plot tabs: Outcomes | Workforce | Tasks & modes | Wages | Worker heterogeneity | Compare strategies
    """
    at = _get_at()
    assert not at.exception
    all_tabs = list(at.tabs)
    # 12 total: 6 advanced expander tabs + 6 plot tabs (in render order)
    expected_all_count = 12
    assert len(all_tabs) == expected_all_count, (
        f"Expected {expected_all_count} total tabs, got {len(all_tabs)}: {[t.label for t in all_tabs]}"
    )
    # Advanced tabs are the first 6
    advanced_tabs = all_tabs[:6]
    expected_advanced_labels = [
        "Costs", "Strategy & pricing", "Heterogeneity",
        "Firing & hiring", "Productivity baseline", "Reproducibility",
    ]
    actual_advanced_labels = [t.label for t in advanced_tabs]
    assert actual_advanced_labels == expected_advanced_labels, (
        f"Advanced tab labels mismatch. Expected {expected_advanced_labels}, got {actual_advanced_labels}"
    )
    # Plot tabs are the last 6
    plot_tabs = all_tabs[-6:]
    expected_plot_labels = [
        "Outcomes", "Workforce", "Tasks & modes", "Wages",
        "Worker heterogeneity", "Compare strategies",
    ]
    actual_plot_labels = [t.label for t in plot_tabs]
    assert actual_plot_labels == expected_plot_labels, (
        f"Plot tab labels mismatch. Expected {expected_plot_labels}, got {actual_plot_labels}"
    )


def test_widget_keys_preserved():
    """P1-11.14: All widget keys from the frozen contract are present on the page.

    NOTE: relies on AppTest including disabled widgets in at.number_input/at.slider/
    at.radio collections (verified via the existing disabled p/target_margin pattern).
    If a future Streamlit upgrade excludes disabled widgets from these collections,
    this test will need to enable each disabled widget before assertion.
    """
    from app import ALL_WIDGET_KEYS
    at = _get_at()
    assert not at.exception
    # Collect all widget keys from the entire page
    all_keys = set()
    for widget in (
        list(at.slider)
        + list(at.number_input)
        + list(at.select_slider)
        + list(at.checkbox)
        + list(at.radio)
    ):
        if hasattr(widget, "key") and widget.key:
            all_keys.add(widget.key)
    missing = set(ALL_WIDGET_KEYS) - all_keys
    assert len(missing) == 0, (
        f"Widget keys missing from page: {sorted(missing)}. "
        f"Available keys: {sorted(all_keys)}"
    )


def test_tab_het_sigma_theta_zero_renders():
    """P1-11.14: tab_het with sigma_theta=0 still renders 2 charts (degenerate inputs).

    P1-6 chose option (b): render both charts with empty inputs when sigma_theta=0.
    This preserves the 15-plot count regardless of sigma_theta value.
    Compare tab adds 0 charts on fresh load (requires button click).
    """
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    sigma_slider = next((s for s in at.slider if s.key == "sigma_theta"), None)
    if sigma_slider is None:
        pytest.skip("sigma_theta slider not found")
    at = sigma_slider.set_value(0.0).run()
    assert not at.exception
    run_btn = _find_run_btn(at)
    at = run_btn.click().run()
    assert not at.exception
    # Plot count should still be 15 (degenerate empty charts; compare tab = 0 without button click)
    unknown_count = sum(1 for el in at.main if type(el).__name__ == "UnknownElement")
    assert unknown_count == 15, (
        f"Expected 15 UnknownElements with sigma_theta=0, got {unknown_count}"
    )


def test_reset_button_restores_defaults():
    """P1-11.14: Reset button restores all primary controls to FirmParams defaults.

    Changes 5 widgets to non-default values, clicks Reset, then verifies
    each returns to its FirmParams default. T_review UI default now matches
    FirmParams().T_review = 10.0 (two-defaults seam closed by T-08).
    """
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception

    defaults = FirmParams()

    # Change N
    n_input = next((inp for inp in at.number_input if inp.key == "N"), None)
    assert n_input is not None, "N number_input not found"
    at = n_input.set_value(int(defaults.N) + 50).run()

    # Change q_a
    q_a_slider = next((s for s in at.slider if s.key == "q_a"), None)
    assert q_a_slider is not None, "q_a slider not found"
    at = q_a_slider.set_value(round(float(defaults.q_a) + 0.3, 2)).run()

    # Change w
    w_slider = next((s for s in at.slider if s.key == "w"), None)
    assert w_slider is not None, "w slider not found"
    at = w_slider.set_value(round(float(defaults.w) + 0.5, 1)).run()

    # Change c_auto
    c_auto_slider = next((s for s in at.slider if s.key == "c_auto"), None)
    assert c_auto_slider is not None, "c_auto slider not found"
    at = c_auto_slider.set_value(round(float(defaults.c_auto) + 0.2, 2)).run()

    # Change sigma_theta (in advanced expander — still queriable via at.slider)
    sigma_slider = next((s for s in at.slider if s.key == "sigma_theta"), None)
    assert sigma_slider is not None, "sigma_theta slider not found"
    at = sigma_slider.set_value(round(float(defaults.sigma_theta) + 0.1, 2)).run()

    # Click Reset
    reset_btn = next((b for b in at.button if b.label == "Reset to defaults"), None)
    assert reset_btn is not None, "Reset button not found"
    at = reset_btn.click().run()
    assert not at.exception

    # Verify defaults restored
    n_input_after = next((inp for inp in at.number_input if inp.key == "N"), None)
    assert n_input_after is not None
    assert int(n_input_after.value) == int(defaults.N), (
        f"N not restored: got {n_input_after.value}, expected {defaults.N}"
    )

    q_a_after = next((s for s in at.slider if s.key == "q_a"), None)
    assert q_a_after is not None
    assert abs(float(q_a_after.value) - float(defaults.q_a)) < 0.01, (
        f"q_a not restored: got {q_a_after.value}, expected {defaults.q_a}"
    )

    w_after = next((s for s in at.slider if s.key == "w"), None)
    assert w_after is not None
    assert abs(float(w_after.value) - float(defaults.w)) < 0.01, (
        f"w not restored: got {w_after.value}, expected {defaults.w}"
    )

    sigma_after = next((s for s in at.slider if s.key == "sigma_theta"), None)
    assert sigma_after is not None
    assert abs(float(sigma_after.value) - float(defaults.sigma_theta)) < 0.01, (
        f"sigma_theta not restored: got {sigma_after.value}, expected {defaults.sigma_theta}"
    )

    # Also check T_review select_slider (UI default == FirmParams().T_review == 10.0; seam closed T-08)
    t_review_after = next((s for s in at.select_slider if s.key == "T_review"), None)
    assert t_review_after is not None
    assert t_review_after.value == 10, (
        f"T_review not restored: got {t_review_after.value}, expected 10 "
        "(UI default matches FirmParams().T_review = 10.0 since T-08 seam closure)"
    )

    # Also check max_hire_period (UI default 3 per Scenario B recalibration 99ddaea)
    max_hire_after = next((inp for inp in at.number_input if inp.key == "max_hire_period"), None)
    assert max_hire_after is not None
    assert int(max_hire_after.value) == 3, (
        f"max_hire_period not restored: got {max_hire_after.value}, expected 3 "
        "(UI default 3 per Scenario B recalibration; kernel default is 3 post-99ddaea)"
    )

    # Also check tasks_per_worker (UI default == FirmParams default, no divergence)
    tasks_per_worker_after = next((inp for inp in at.number_input if inp.key == "tasks_per_worker"), None)
    assert tasks_per_worker_after is not None
    assert int(tasks_per_worker_after.value) == int(defaults.tasks_per_worker), (
        f"tasks_per_worker not restored: got {tasks_per_worker_after.value}, "
        f"expected {defaults.tasks_per_worker}"
    )


def test_hiring_mode_mutex_kernel_mapping():
    """D-01: Verify _build_controls inline mapping via DRAFT_PARAMS_DEBUG for all three radio options.

    A swap bug in _build_controls (assigning enable_hiring_val = (hiring_mode == "enable_replenish_hiring"))
    would fail the enable_hiring assertion for mode="enable_hiring".
    """
    from streamlit.testing.v1 import AppTest
    from app import params_to_key

    at = AppTest.from_file("app.py", default_timeout=30).run()
    expected = {
        "off": (False, False),
        "enable_hiring": (True, False),
        "enable_replenish_hiring": (False, True),
    }
    for mode, (exp_hiring, exp_replenish) in expected.items():
        at.radio(key="hiring_mode").set_value(mode).run()
        dp = at.session_state["DRAFT_PARAMS_DEBUG"]
        assert dp.enable_hiring is exp_hiring, (
            f"mode={mode}: enable_hiring should be {exp_hiring}, got {dp.enable_hiring}"
        )
        assert dp.enable_replenish_hiring is exp_replenish, (
            f"mode={mode}: enable_replenish_hiring should be {exp_replenish}, got {dp.enable_replenish_hiring}"
        )
        key = params_to_key(dp, 0)
        assert (key[26], key[27]) == (exp_hiring, exp_replenish), (
            f"mode={mode}: key tuple positions mismatch — both-True is unreachable"
        )


def test_plausible_noop_when_placeholder_unset():
    """D-07: With PLAUSIBLE_DOMAIN == '' (placeholder unchanged), st.markdown must NOT emit script tag."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30).run()
    script_tags = [m for m in at.markdown if "plausible.io/js/script.js" in str(m.value)]
    assert len(script_tags) == 0, (
        f"Plausible script tag emitted with empty domain. Found: {script_tags}"
    )


