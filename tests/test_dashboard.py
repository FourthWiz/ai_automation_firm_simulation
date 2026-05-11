"""Tests for the Stage-4 Streamlit dashboard.

Three test classes:
  - TestDashboardHelpers:  per-plot render tests (matplotlib only, no streamlit)
  - TestAppSmoke:          AppTest smoke test (requires streamlit extras)
  - TestCacheHit:          cache-hit counter test (requires streamlit extras)

The AppTest-based tests are guarded with pytest.importorskip so they skip
gracefully when the [dashboard] extras are not installed.
"""

import math
import warnings

import matplotlib
import matplotlib.figure
import numpy as np
import pytest

# Guard: AppTest-based tests skip cleanly if streamlit < 1.28 is absent
# (importorskip raises Skipped, not an error)
# This guard is at the TOP of the file so the whole file is importable
# even without streamlit; only the AppTest classes are individually guarded.

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
    p = FirmParams(seed=0)
    firm = make_firm(p)
    df = run_simulation(firm, greedy_profit)
    return df, firm, p


@pytest.fixture(scope="module")
def all_t_df():
    """Run simulation with all_T strategy (tests empty wage_bill / n_a_trained)."""
    p = FirmParams(seed=0)
    firm = make_firm(p)
    df = run_simulation(firm, all_T)
    return df, firm, p


# ---------------------------------------------------------------------------
# T-09: Per-plot render tests (no streamlit dependency)
# ---------------------------------------------------------------------------

class TestDashboardHelpers:
    """8 helpers × (Figure type + axes count + no unexpected warnings)."""

    def _assert_figure(self, fig, label: str) -> None:
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"{label}: expected Figure, got {type(fig)}"
        )
        assert len(fig.axes) >= 1, f"{label}: expected at least 1 Axes"
        matplotlib.pyplot.close(fig)

    def test_fig_pi_over_time(self, default_df):
        from firm_ai_abm.dashboard import fig_pi_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_pi_over_time(df)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        # UserWarnings from matplotlib are tolerated; RuntimeWarning is not
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
        """When firings occur, fig_K_over_time renders a twin-axis with firing bars."""
        from firm_ai_abm.dashboard import fig_K_over_time
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] > 0).any(), "Expected firing events for this test"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_K_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2, "Expected twin axes when firings present"
        matplotlib.pyplot.close(fig)

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
        # Default params have T_review=math.inf → n_review_fired always 0
        assert (df["n_review_fired"] == 0).all(), "Expected empty firing events"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_firing_events (empty)")

    def test_fig_firing_events_nonempty(self):
        """T_review=10 + high firing_threshold guarantees firings; verify markers render."""
        from firm_ai_abm.dashboard import fig_firing_events
        # firing_threshold=100.0 (kernel) >> any realistic surplus (~8-9), so all
        # workers are fired at every review period — guarantees non-empty markers.
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] > 0).any(), "Expected actual firing events"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df, T_review=10.0)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
        matplotlib.pyplot.close(fig)

    def test_fig_firing_events_active_review_no_firings(self):
        """T_review=10 but firing_threshold=0 → no firings; fallback message distinguishes from inf."""
        from firm_ai_abm.dashboard import fig_firing_events
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=0.0)
        firm = make_firm(p)
        df = run_simulation(firm, greedy_profit)
        assert (df["n_review_fired"] == 0).all(), "Expected zero firings with threshold=0"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_firing_events(df, T_review=10.0)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

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


# ---------------------------------------------------------------------------
# T-08: AppTest smoke test (requires streamlit >= 1.28)
# ---------------------------------------------------------------------------

class TestAppSmoke:
    """Smoke test: app launches, sidebar renders, no exceptions."""

    @pytest.fixture(autouse=True)
    def _require_streamlit(self):
        pytest.importorskip("streamlit", minversion="1.28")

    def test_smoke_no_exception(self, tmp_path):
        """App runs without uncaught exceptions."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception, f"App raised exception: {at.exception}"

    def test_sidebar_strategy_radio(self):
        """Strategy radio widget exists with correct label."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        radio_labels = [r.label for r in at.sidebar.radio]
        assert "Strategy" in radio_labels, f"Strategy radio not found, got: {radio_labels}"

    def test_sidebar_seed_input_exists(self):
        """Seed number_input exists in sidebar."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # Seed input should exist — any number_input with a non-None value
        assert len(at.sidebar.number_input) > 0, "No number_input widgets in sidebar"

    def test_sidebar_all_param_fields_present(self):
        """All 22 param field names appear as widget labels in sidebar."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from streamlit.testing.v1 import AppTest
        from app import _PARAM_FIELDS
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # Collect all widget labels from sidebar (WidgetList is not directly summable)
        all_labels = set()
        for widget in list(at.sidebar.slider) + list(at.sidebar.number_input) + list(at.sidebar.select_slider):
            all_labels.add(widget.label)
        # T_review is rendered as select_slider with label "T_review"
        # The sidebar uses exact field names as labels (per plan T-05)
        for field in _PARAM_FIELDS:
            assert field in all_labels, (
                f"Param field '{field}' not found as a sidebar widget label. "
                f"Available labels: {sorted(all_labels)}"
            )

    def test_main_panel_has_8_plots(self):
        """Main panel renders 8 pyplot figures (as UnknownElement in AppTest 1.57)."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # st.pyplot renders as UnknownElement in Streamlit 1.57 AppTest
        unknown_count = sum(1 for el in at.main if type(el).__name__ == "UnknownElement")
        assert unknown_count == 8, (
            f"Expected 8 st.pyplot (UnknownElement) in main panel, got {unknown_count}"
        )

    def test_footer_caption_content(self):
        """Footer captions contain required strings."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        caption_values = [c.value for c in at.caption]
        combined = " ".join(caption_values)
        assert "corr(theta, log(wage)) =" in combined, (
            f"Footer missing corr text. Captions: {caption_values}"
        )
        assert "final cumulative profit =" in combined, (
            f"Footer missing profit text. Captions: {caption_values}"
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

    def test_cache_hit_no_increment(self):
        """Clicking Run twice with unchanged params should not change the timestamp.

        run_cached returns a monotonic timestamp (computed_at = time.monotonic_ns())
        captured at actual computation time. @st.cache_data returns the SAME cached
        5-tuple on a hit, so computed_at does not change. The app writes computed_at
        to session_state["RUN_COUNTER_VAL_THIS_RUN"] on every script run, so the
        value is always present. A cache hit → same timestamp; cache miss → new one.
        """
        from streamlit.testing.v1 import AppTest

        # First run: app loads. computed_at is set (either fresh or cached).
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception

        start_ts = self._get_timestamp(at)
        assert start_ts > 0, (
            f"RUN_COUNTER_VAL_THIS_RUN should be a positive timestamp, got {start_ts}"
        )

        # Click Run with no slider changes — identical params_key → cache hit.
        # The cached 5-tuple is returned → same computed_at timestamp.
        at = at.button[0].click().run()
        assert not at.exception

        end_ts = self._get_timestamp(at)
        assert end_ts == start_ts, (
            f"Cache miss on identical params: timestamp changed from {start_ts} "
            f"to {end_ts} (expected same — cache should return identical 5-tuple)"
        )

    def test_cache_miss_after_param_change(self):
        """Changing a slider then clicking Run SHOULD change the timestamp.

        A new params_key → cache miss → run_cached executes → new computed_at
        timestamp. This proves the cache-hit test is not a false positive.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        start_ts = self._get_timestamp(at)

        # Find the q_a slider by label and change its value
        q_a_slider = None
        for s in list(at.sidebar.slider):
            if s.label == "q_a":
                q_a_slider = s
                break
        assert q_a_slider is not None, "Could not find q_a slider in sidebar"

        # Set a different value, re-run, then click Run
        new_q_a = round(float(q_a_slider.value) + 0.05, 2)
        at = q_a_slider.set_value(new_q_a).run()
        assert not at.exception
        at = at.button[0].click().run()
        assert not at.exception

        end_ts = self._get_timestamp(at)
        assert end_ts != start_ts, (
            f"Expected timestamp to change after param change (cache miss), "
            f"but it stayed at {start_ts}. "
            f"This may indicate the new params_key is incorrectly cached."
        )
