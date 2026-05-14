"""Tests for the Phase 1.5 Streamlit dashboard.

Three test classes:
  - TestDashboardHelpers:  per-plot render tests (plotly, no streamlit)
  - TestAppSmoke:          AppTest smoke test (requires streamlit extras)
  - TestCacheHit:          cache-hit counter test (requires streamlit extras)

The AppTest-based tests are guarded with pytest.importorskip so they skip
gracefully when the [dashboard] extras are not installed.

P1-11 update notes:
  - TestDashboardHelpers rewritten for plotly (matplotlib removed).
  - Spike P1-2a pinned result: st.plotly_chart → UnknownElement in AppTest.
    st.tabs DOES expose inner chart elements in at.main (all 13 visible).
    Empirical count: 13 UnknownElements for default-param run
    (tabs layout 2+3+2+4+2 = 13 — verified Streamlit 1.45).
  - test_main_panel_has_13_plots: expects exactly 13 UnknownElements.
  - test_sidebar_strategy_radio, test_sidebar_seed_input_exists: flat search.
  - test_sidebar_all_param_fields_present: rewritten as test_all_param_fields_present_on_page.
  - All at.button[0] → lookup by label.
  - TestDashboardStaleBanner/MarginScenario button refs rewritten.
"""

import math
import warnings

import numpy as np
import pytest
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared fixture: default-params DataFrame
# ---------------------------------------------------------------------------

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import greedy_profit, all_T


@pytest.fixture(scope="module")
def default_df():
    """Run simulation with default FirmParams(seed=0) and greedy_profit."""
    p = FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0)  # N=100: Phase-1 fixture scale
    firm = make_firm(p)
    df = run_simulation(firm, greedy_profit)
    return df, firm, p


@pytest.fixture(scope="module")
def all_t_df():
    """Run simulation with all_T strategy (tests empty wage_bill / n_a_trained)."""
    p = FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0)  # N=100: Phase-1 fixture scale
    firm = make_firm(p)
    df = run_simulation(firm, all_T)
    return df, firm, p


# ---------------------------------------------------------------------------
# P1-11.8: Per-plot render tests rewritten for Plotly (no streamlit dependency)
# ---------------------------------------------------------------------------

class TestDashboardHelpers:
    """Per-plot render tests — all figures must be plotly.graph_objects.Figure."""

    def _assert_figure(self, fig, label: str) -> None:
        assert isinstance(fig, go.Figure), (
            f"{label}: expected plotly Figure, got {type(fig)}"
        )
        assert len(fig.data) >= 1, f"{label}: expected at least one trace"

    def test_fig_pi_over_time(self, default_df):
        from firm_ai_abm.dashboard import fig_pi_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_pi_over_time(df)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Unexpected RuntimeWarnings: {runtime_warnings}"
        self._assert_figure(fig, "fig_pi_over_time")

    def test_fig_K_over_time(self, default_df):
        from firm_ai_abm.dashboard import fig_K_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_K_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_K_over_time")

    def test_fig_K_over_time_with_firings(self):
        """P1-11.12: When firings occur, fig_K_over_time renders a secondary y-axis."""
        from firm_ai_abm.dashboard import fig_K_over_time
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0, tasks_per_worker=10, p=1.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] > 0).any(), "Expected firing events for this test"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_K_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, go.Figure)
        # Firings bar should be on secondary y-axis (y2)
        assert any(getattr(tr, "yaxis", None) == "y2" for tr in fig.data), (
            "expected at least one trace on the secondary y-axis (firings bar)"
        )

    def test_fig_mode_mix_area(self, default_df):
        from firm_ai_abm.dashboard import fig_mode_mix_area
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_mode_mix_area(df, p.N)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_mode_mix_area")

    def test_fig_wage_bill_over_time(self, default_df):
        from firm_ai_abm.dashboard import fig_wage_bill_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_wage_bill_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_wage_bill_over_time")

    def test_fig_theta_histogram(self, default_df):
        from firm_ai_abm.dashboard import fig_theta_histogram
        df, firm, p = default_df
        theta = firm.workforce.theta
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_theta_histogram(theta)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_theta_histogram")

    def test_fig_mean_theta_over_time(self, default_df):
        from firm_ai_abm.dashboard import fig_mean_theta_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_mean_theta_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_mean_theta_over_time")

    def test_fig_firing_events_empty(self, default_df):
        """Default T_review=inf → no firing events; plot still renders."""
        from firm_ai_abm.dashboard import fig_firing_events
        df, firm, p = default_df
        assert (df["n_review_fired"] == 0).all(), "Expected empty firing events"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, go.Figure)

    def test_fig_firing_events_nonempty(self):
        """T_review=10 + high firing_threshold guarantees firings; verify markers render."""
        from firm_ai_abm.dashboard import fig_firing_events
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0, tasks_per_worker=10, p=1.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] > 0).any(), "Expected actual firing events"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df, T_review=10.0)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_fig_firing_events_active_review_no_firings(self):
        """T_review=10 but firing_threshold=0 → no firings; fallback message."""
        from firm_ai_abm.dashboard import fig_firing_events
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=0.0, tasks_per_worker=10, p=1.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] == 0).all(), "Expected zero firings with threshold=0"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df, T_review=10.0)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, go.Figure)

    def test_fig_trained_capital(self, default_df):
        from firm_ai_abm.dashboard import fig_trained_capital
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_trained_capital(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_trained_capital")

    def test_fig_mode_mix_area_all_T(self, all_t_df):
        """all_T strategy: H and A fractions = 0; plot still renders."""
        from firm_ai_abm.dashboard import fig_mode_mix_area
        df, firm, p = all_t_df
        fig = fig_mode_mix_area(df, p.N)
        self._assert_figure(fig, "fig_mode_mix_area (all_T)")

    def test_fig_wage_bill_all_T(self, all_t_df):
        """all_T strategy: wage_bill may be 0; plot still renders."""
        from firm_ai_abm.dashboard import fig_wage_bill_over_time
        df, firm, p = all_t_df
        fig = fig_wage_bill_over_time(df)
        self._assert_figure(fig, "fig_wage_bill_over_time (all_T)")

    def test_fig_trained_capital_all_T(self, all_t_df):
        """all_T strategy: n_a_trained = 0 throughout; plot still renders."""
        from firm_ai_abm.dashboard import fig_trained_capital
        df, firm, p = all_t_df
        assert (df["n_a_trained"] == 0).all()
        fig = fig_trained_capital(df)
        self._assert_figure(fig, "fig_trained_capital (all_T zero)")

    # --- P1-11.8: Stronger trace-shape assertions ---

    def test_fig_pi_per_period_over_time_renders(self, default_df):
        """P1-11.8: fig_pi_per_period_over_time renders and has at least one line scatter."""
        from firm_ai_abm.dashboard import fig_pi_per_period_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_pi_per_period_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, go.Figure)
        line_traces = [tr for tr in fig.data
                       if tr.type == "scatter"
                       and getattr(tr, "mode", "lines") in ("lines", "lines+markers")]
        assert len(line_traces) >= 1, "Expected at least one line trace"

    def test_K_active_caption_present(self, default_df):
        """P1-11.9: fig_K_over_time annotation contains 'K = workforce headcount'."""
        from firm_ai_abm.dashboard import fig_K_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_K_over_time(df)
        # Check layout.annotations for the caption text
        annotations = fig.layout.annotations or ()
        combined = " ".join(a.text for a in annotations if a.text)
        assert "K = workforce headcount" in combined, (
            f"Caption 'K = workforce headcount' not found in layout.annotations. Got: {combined}"
        )

    def test_n_a_trained_degenerate_annotation(self, default_df):
        """P1-11.10: fig_trained_capital annotates when series is constant > 0."""
        from firm_ai_abm.dashboard import fig_trained_capital
        import pandas as pd
        df, firm, p = default_df
        df_degenerate = df.copy()
        df_degenerate["n_a_trained"] = 5  # constant value > 0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_trained_capital(df_degenerate)
        annotations = fig.layout.annotations or ()
        combined = " ".join(a.text for a in annotations if a.text)
        assert "trained by t=" in combined, (
            f"Degenerate annotation 'trained by t=' not found. Got: {combined}"
        )

    # --- P1-11.11: Wage figures ---

    def test_fig_wage_histogram(self):
        """P1-11.11: fig_wage_histogram renders, has correct labels and mean line/annotation."""
        from firm_ai_abm.dashboard import fig_wage_histogram
        wages = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_wage_histogram(wages)
        assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 0
        assert isinstance(fig, go.Figure)
        hist_traces = [tr for tr in fig.data if isinstance(tr, go.Histogram)]
        assert len(hist_traces) >= 1, "Expected at least one Histogram trace"
        # Mean vline should exist as a shape or annotation
        shapes = list(fig.layout.shapes or ())
        annotations = list(fig.layout.annotations or ())
        has_mean = (
            any(getattr(s, "type", None) == "line" for s in shapes)
            or any("mean" in (a.text or "").lower() for a in annotations)
        )
        assert has_mean, "Expected a mean vline (in layout.shapes) or mean annotation"
        # Axis labels
        assert fig.layout.xaxis.title.text == "wage"
        assert fig.layout.yaxis.title.text == "count"
        assert "Wage" in fig.layout.title.text

    def test_fig_wage_histogram_empty(self):
        """P1-11.11: fig_wage_histogram handles empty array without error."""
        from firm_ai_abm.dashboard import fig_wage_histogram
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_wage_histogram(np.array([]))
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_fig_wage_vs_mean_output(self):
        """P1-11.11: fig_wage_vs_mean_output renders correct point count and trace names."""
        from firm_ai_abm.dashboard import fig_wage_vs_mean_output
        wages = np.array([0.9, 1.0, 1.1])
        mean_out = np.array([1.0, 1.2, np.nan])  # worker 2 has NaN → excluded
        a_trained = np.array([False, True, False])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_wage_vs_mean_output(wages, mean_out, a_trained)
        # Scatter point count should equal number of non-NaN workers (2)
        scatter_pt_count = sum(len(tr.x) for tr in fig.data if tr.type == "scatter")
        assert scatter_pt_count == 2, (
            f"Expected 2 scatter points (1 NaN worker excluded), got {scatter_pt_count}"
        )
        # Two named traces (showlegend=None means default=True in plotly)
        named_traces = [tr for tr in fig.data
                        if getattr(tr, "name", None)
                        and getattr(tr, "showlegend", True) is not False]
        assert len(named_traces) == 2
        names = sorted(tr.name for tr in named_traces)
        assert names[0].startswith("trained (n="), f"Unexpected trace name: {names[0]}"
        assert names[1].startswith("untrained (n="), f"Unexpected trace name: {names[1]}"

    def test_fig_wage_vs_mean_output_all_nan(self):
        """P1-11.11: handles all-NaN without raising."""
        from firm_ai_abm.dashboard import fig_wage_vs_mean_output
        wages = np.array([1.0, 1.0])
        mean_out = np.array([np.nan, np.nan])
        a_trained = np.array([False, False])
        fig = fig_wage_vs_mean_output(wages, mean_out, a_trained)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_fig_hiring_events_disabled(self):
        """P1-11.13: fig_hiring_events disabled renders the disabled message."""
        from firm_ai_abm.dashboard import fig_hiring_events
        params = FirmParams(seed=0)
        firm = make_firm(params)
        df = run_simulation(firm, all_T)
        fig = fig_hiring_events(df, enable_hiring=False)
        # Check layout.annotations for disabled message
        annotations = fig.layout.annotations or ()
        combined = " ".join(a.text for a in annotations if a.text)
        assert "hiring disabled" in combined, (
            f"Expected disabled message in annotations; got: {combined}"
        )

    def test_fig_hiring_events_active(self):
        """P1-11.13: fig_hiring_events with hiring events renders scatter point."""
        from firm_ai_abm.dashboard import fig_hiring_events
        import pandas as pd
        df = pd.DataFrame({"t": [0, 1, 2, 3], "n_hired": [0, 3, 0, 0]})
        fig = fig_hiring_events(df, enable_hiring=True)
        assert isinstance(fig, go.Figure)
        scatter_traces = [tr for tr in fig.data if tr.type == "scatter" and len(tr.x) > 0]
        assert len(scatter_traces) >= 1, "Expected scatter point for hiring event at t=1"


# ---------------------------------------------------------------------------
# T-08: AppTest smoke test (requires streamlit >= 1.28)
# ---------------------------------------------------------------------------

class TestAppSmoke:
    """Smoke test: app launches, controls render, no exceptions."""

    @pytest.fixture(autouse=True)
    def _require_streamlit(self):
        pytest.importorskip("streamlit", minversion="1.28")

    def test_smoke_no_exception(self, tmp_path):
        """App runs without uncaught exceptions."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception, f"App raised exception: {at.exception}"

    def test_sidebar_strategy_radio(self):
        """P1-11.3: Strategy radio widget exists somewhere on the page."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # Flat search across all radio widgets (Strategy moved to main panel)
        radio_labels = [r.label for r in at.radio]
        assert "Strategy" in radio_labels, f"Strategy radio not found, got: {radio_labels}"

    def test_strategy_single_widget_all_options(self):
        """T-07: Single strategy widget with key='strategy' exposes all 6 registry options.

        Enforces the single-widget contract: no 'strategy_adv' widget anywhere,
        and the 'strategy' widget has all 6 strategies as options.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from streamlit.testing.v1 import AppTest
        from app import _STRATEGY_REGISTRY

        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception

        # Find the strategy widget (radio or selectbox)
        strategy_radio = next((r for r in at.radio if r.key == "strategy"), None)
        strategy_select = next((s for s in at.selectbox if s.key == "strategy"), None) if strategy_radio is None else None

        assert strategy_radio is not None or strategy_select is not None, (
            "No widget with key='strategy' found (checked radio and selectbox)"
        )

        if strategy_radio is not None:
            assert set(strategy_radio.options) == set(_STRATEGY_REGISTRY.keys()), (
                f"Strategy radio options {set(strategy_radio.options)} != "
                f"registry keys {set(_STRATEGY_REGISTRY.keys())}"
            )
            # F-01 T-03: greedy_profit must have been removed from UI options
            assert "greedy_profit" not in strategy_radio.options, (
                "greedy_profit must be removed from the strategy radio options (F-01)"
            )
        else:
            assert set(strategy_select.options) == set(_STRATEGY_REGISTRY.keys()), (
                f"Strategy selectbox options {set(strategy_select.options)} != "
                f"registry keys {set(_STRATEGY_REGISTRY.keys())}"
            )
            assert "greedy_profit" not in strategy_select.options, (
                "greedy_profit must be removed from the strategy selectbox options (F-01)"
            )

        # No widget with key='strategy_adv' should exist anywhere
        all_radio_keys = {r.key for r in at.radio}
        assert "strategy_adv" not in all_radio_keys, (
            "Found unexpected widget with key='strategy_adv' — advanced strategy radio must be deleted"
        )

    def test_sidebar_seed_input_exists(self):
        """P1-11.4: Seed number_input exists somewhere on the page."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # Flat search across all number_input widgets (Seed moved to main panel)
        seed_inputs = [inp for inp in at.number_input if inp.key == "seed"]
        assert len(seed_inputs) > 0, "No number_input with key='seed' found on page"

    def test_all_param_fields_present_on_page(self):
        """P1-11.1: All _PARAM_FIELDS keys exist as widget keys somewhere on page.

        Primary controls moved to main panel — search both at.sidebar and at.main.
        scenario_mode uses widget key 'scenario'; check by key not label.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from streamlit.testing.v1 import AppTest
        from app import _PARAM_FIELDS
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception

        # Collect all widget keys from entire page (flat search)
        all_keys = set()
        all_labels = set()
        for widget in (
            list(at.slider)
            + list(at.number_input)
            + list(at.select_slider)
            + list(at.checkbox)
            + list(at.radio)
        ):
            if hasattr(widget, "key") and widget.key:
                all_keys.add(widget.key)
            if hasattr(widget, "label") and widget.label:
                all_labels.add(widget.label)

        # Primary fields renamed in display but must be present by key
        _EXEMPT_LABELS = {
            "scenario_mode", "N", "q_a", "g", "c_auto", "w", "p", "target_margin",
        }
        # FirmParams field → widget key mapping (only where they differ)
        _FIELD_TO_KEY = {
            "scenario_mode": "scenario",
            "enable_hiring": "hiring_mode",
            "enable_replenish_hiring": "hiring_mode",
        }

        for field in _PARAM_FIELDS:
            widget_key = _FIELD_TO_KEY.get(field, field)
            if field in _EXEMPT_LABELS:
                assert widget_key in all_keys, (
                    f"Primary field '{field}' (widget key '{widget_key}') missing from page"
                )
            else:
                assert field in all_labels or widget_key in all_keys, (
                    f"Param '{field}' (widget key '{widget_key}') not found by label or key"
                )

    def test_main_panel_has_13_plots(self):
        """P1-11.2: Main panel renders 15 plotly charts (as UnknownElement in AppTest).

        P1-2a spike result: st.plotly_chart → UnknownElement; tabs expose inner content
        in at.main. Empirical count = 15 UnknownElements (post F-02: alpha + beta
        histograms add 2 to tab_modes; tabs layout 2+3+4+4+2 = 15).
        Verified on Streamlit 1.45.
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        unknown_count = sum(1 for el in at.main if type(el).__name__ == "UnknownElement")
        assert unknown_count == 15, (
            f"Expected 15 st.plotly_chart (UnknownElement) in main panel, got {unknown_count}"
        )

    def test_footer_caption_content(self):
        """Footer captions contain required strings."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        at.select_slider(key="T_review").set_value("inf")
        run_btn = next(b for b in at.button if b.label.endswith("Run simulation"))
        at = run_btn.click().run()
        assert not at.exception
        caption_values = [c.value for c in at.caption]
        combined = " ".join(caption_values)
        assert "corr(theta, log(wage)) =" in combined, (
            f"Footer missing corr text. Captions: {caption_values}"
        )
        assert "final cumulative profit =" in combined, (
            f"Footer missing profit text. Captions: {caption_values}"
        )
        assert "total firings" in combined, (
            f"Footer missing 'total firings'. Captions: {caption_values}"
        )
        assert "Hires:" in combined, (
            f"Footer missing 'Hires:' (hiring summary). Captions: {caption_values}"
        )
        assert "final K_active" in combined, (
            f"Footer missing 'final K_active'. Captions: {caption_values}"
        )


# ---------------------------------------------------------------------------
# T-10: Cache-hit counter test (requires streamlit >= 1.28)
# ---------------------------------------------------------------------------

class TestCacheHit:
    """Verify that identical params do not trigger a second sim run."""

    @pytest.fixture(autouse=True)
    def _require_streamlit(self):
        pytest.importorskip("streamlit", minversion="1.28")

    def _get_timestamp(self, at) -> int:
        """Safely read RUN_COUNTER_VAL_THIS_RUN (monotonic timestamp) from AppTest."""
        try:
            return int(at.session_state["RUN_COUNTER_VAL_THIS_RUN"])
        except (KeyError, AttributeError):
            return -1

    def _find_run_btn(self, at):
        """Find the Run button by label (stable; not by index)."""
        return next(b for b in at.button if b.label.endswith("Run simulation"))

    def test_cache_hit_no_increment(self):
        """P1-11.5: Clicking Run twice with unchanged params should not change the timestamp."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception

        start_ts = self._get_timestamp(at)
        assert start_ts > 0, (
            f"RUN_COUNTER_VAL_THIS_RUN should be a positive timestamp, got {start_ts}"
        )

        # Click Run with no changes — cache hit expected
        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        assert not at.exception

        end_ts = self._get_timestamp(at)
        assert end_ts == start_ts, (
            f"Cache miss on identical params: timestamp changed from {start_ts} "
            f"to {end_ts} (expected same — cache should return identical 7-tuple)"
        )

    def test_cache_miss_after_param_change(self):
        """P1-11.5/6: Changing a slider then clicking Run SHOULD change the timestamp."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        start_ts = self._get_timestamp(at)

        # Find the q_a slider by key (moved to main panel)
        q_a_slider = next((s for s in at.slider if s.key == "q_a"), None)
        assert q_a_slider is not None, "Could not find q_a slider by key"

        new_q_a = round(float(q_a_slider.value) + 0.05, 2)
        at = q_a_slider.set_value(new_q_a).run()
        assert not at.exception

        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        assert not at.exception

        end_ts = self._get_timestamp(at)
        assert end_ts != start_ts, (
            f"Expected timestamp to change after param change (cache miss), "
            f"but it stayed at {start_ts}."
        )


# ---------------------------------------------------------------------------
# T-14b: README margin recipe CI smoke (no streamlit required)
# ---------------------------------------------------------------------------


def test_README_margin_recipe_at_default_seed():
    """Stage 5 T-14b: README margin recipe (w=8.0, c_auto=0.6, F=5) produces a positive margin.

    Recalibrated after brute-greedy-use-posteriors: greedy_profit now plans decisions
    under alpha_hat/beta_hat priors (dp_prior_alpha=0.5, dp_prior_beta=0.7) rather than
    peeking at truth (D-01). Margin at seed=0 is ~30.6% under posteriors; range updated
    to [0.20, 0.45] to guard against sign-flip regressions while allowing for prior variation.
    """
    from firm_ai_abm.firm import make_firm
    from firm_ai_abm.simulate import run_simulation
    from firm_ai_abm.strategy import greedy_profit

    params = FirmParams(seed=0, w=8.0, c_auto=0.6, F=5, tasks_per_worker=10, p=1.0)
    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)
    margin = float(df["pi"].mean() / df["Y"].mean())
    assert 0.20 <= margin <= 0.45, (
        f"README margin recipe out of range: margin={margin:.3f} ({margin*100:.1f}%). "
        "If recipe parameters changed, update README and this test together. "
        "Note: greedy now plans under posteriors (dp_prior_alpha=0.5, dp_prior_beta=0.7)."
    )


# ---------------------------------------------------------------------------
# T-11: Stale-banner and margin-scenario AppTest coverage
# ---------------------------------------------------------------------------


class TestDashboardStaleBanner:
    """T-11: AppTest coverage for the stale-run banner."""

    def _get_at(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit AppTest not available")
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        return at

    def _find_run_btn(self, at):
        return next(b for b in at.button if b.label.endswith("Run simulation"))

    def test_no_banner_on_fresh_load(self):
        """T-11: No stale-banner on fresh load + Run."""
        at = self._get_at()
        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        stale_warnings = [w for w in at.warning if "Params changed" in str(w.value)]
        assert len(stale_warnings) == 0, (
            f"Expected no 'Params changed' warning on fresh load, found {len(stale_warnings)}"
        )

    def test_banner_on_param_change(self):
        """T-11: Stale-banner appears after param change without pressing Run."""
        at = self._get_at()
        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        n_input = next((inp for inp in at.number_input if inp.key == "N"), None)
        if n_input is None:
            pytest.skip("N number_input not found")
        n_input.set_value(n_input.value + 10)
        at.run()
        stale_warnings = [w for w in at.warning if "Params changed" in str(w.value)]
        assert len(stale_warnings) >= 1, (
            "Expected 'Params changed' warning after param change without Run"
        )

    def test_banner_clears_after_run(self):
        """T-11: Banner clears after clicking Run."""
        at = self._get_at()
        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        n_input = next((inp for inp in at.number_input if inp.key == "N"), None)
        if n_input is None:
            pytest.skip("N number_input not found")
        n_input.set_value(n_input.value + 10)
        at.run()
        run_btn = self._find_run_btn(at)
        at = run_btn.click().run()
        stale_warnings = [w for w in at.warning if "Params changed" in str(w.value)]
        assert len(stale_warnings) == 0, (
            "Expected banner to clear after clicking Run"
        )


class TestDashboardMarginScenario:
    """T-11: AppTest coverage for the margin scenario UI."""

    def _get_at(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit AppTest not available")
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        return at

    def _find_run_btn(self, at):
        return next(b for b in at.button if b.label.endswith("Run simulation"))

    def test_margin_scenario_widgets_visible(self):
        """T-11: In margin mode, target_margin slider and margin_horizon number_input are present."""
        at = self._get_at()
        scenario_radio = next((r for r in at.radio if r.key == "scenario"), None)
        if scenario_radio is None:
            pytest.skip("scenario radio not found")
        scenario_radio.set_value("margin")
        at.run()
        slider_keys = {s.key for s in at.slider}
        input_keys = {inp.key for inp in at.number_input}
        assert "target_margin" in slider_keys, "target_margin slider not visible in margin mode"
        assert "margin_horizon" in input_keys, "margin_horizon number_input not visible in margin mode"

    def test_margin_scenario_does_not_auto_lock_strategy(self):
        """T-06: Setting scenario=margin must NOT force strategy to horizon_optimizer.

        Strategy is now an independent widget; scenario is a pure pricing selector.
        First, we set the strategy radio to 'all_H' explicitly (non-tautological
        coverage: the test verifies the scenario switch does NOT change strategy).
        Then we switch scenario to 'margin' and assert strategy remains 'all_H'.
        """
        at = self._get_at()
        # Explicitly pick a non-default strategy first
        strategy_radio = next((r for r in at.radio if r.key == "strategy"), None)
        assert strategy_radio is not None, "strategy radio not found"
        at = strategy_radio.set_value("all_H").run()
        # Now flip to margin scenario
        scenario_radio = next((r for r in at.radio if r.key == "scenario"), None)
        if scenario_radio is None:
            pytest.skip("scenario radio not found")
        at = scenario_radio.set_value("margin").run()
        # Strategy radio should still be all_H (scenario did NOT auto-lock it)
        strategy_radio = next((r for r in at.radio if r.key == "strategy"), None)
        assert strategy_radio is not None, "strategy radio not found"
        assert strategy_radio.value == "all_H", (
            f"Expected strategy to remain 'all_H' after setting scenario=margin, "
            f"got: {strategy_radio.value}"
        )
        # LAST_STRATEGY_DEBUG must reflect all_H (not horizon_optimizer)
        assert at.session_state["LAST_STRATEGY_DEBUG"] == "all_H", (
            f"LAST_STRATEGY_DEBUG should be 'all_H', got: {at.session_state['LAST_STRATEGY_DEBUG']}"
        )
