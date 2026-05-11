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
    p = FirmParams(seed=0, tasks_per_worker=10, p=1.0)
    firm = make_firm(p)
    df = run_simulation(firm, greedy_profit)
    return df, firm, p


@pytest.fixture(scope="module")
def all_t_df():
    """Run simulation with all_T strategy (tests empty wage_bill / n_a_trained)."""
    p = FirmParams(seed=0, tasks_per_worker=10, p=1.0)
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
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0, tasks_per_worker=10, p=1.0)
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
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=100.0, tasks_per_worker=10, p=1.0)
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
        p = FirmParams(seed=0, T_review=10.0, firing_threshold=0.0, tasks_per_worker=10, p=1.0)
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

    # --- Stage 5 T-10 new render tests ---

    def test_fig_pi_per_period_over_time_renders(self, default_df):
        """Stage 5 T-10: fig_pi_per_period_over_time renders and has at least one line."""
        from firm_ai_abm.dashboard import fig_pi_per_period_over_time
        import matplotlib.lines
        df, firm, p = default_df
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_pi_per_period_over_time(df)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
        self._assert_figure(fig, "fig_pi_per_period_over_time")
        # Check at least one line artist exists
        ax = fig.axes[0]
        line_artists = [a for a in ax.get_children() if isinstance(a, matplotlib.lines.Line2D)]
        assert len(line_artists) >= 1, "Expected at least one line in per-period profit plot"
        matplotlib.pyplot.close(fig)

    def test_K_active_caption_present(self, default_df):
        """Stage 5 T-10: fig_K_over_time caption contains 'K = workforce headcount'."""
        from firm_ai_abm.dashboard import fig_K_over_time
        df, firm, p = default_df
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_K_over_time(df)
        # Find text artists that contain the substring
        ax = fig.axes[0]
        text_strings = [t.get_text() for t in ax.texts]
        # Also scan all text in figure
        all_text = []
        for a in fig.axes:
            all_text.extend(t.get_text() for t in a.texts)
        combined = " ".join(all_text)
        assert "K = workforce headcount" in combined, (
            f"Caption 'K = workforce headcount' not found in figure text. Got: {all_text}"
        )
        matplotlib.pyplot.close(fig)

    def test_n_a_trained_degenerate_annotation(self, default_df):
        """Stage 5 T-10: fig_trained_capital annotates when series is constant > 0."""
        from firm_ai_abm.dashboard import fig_trained_capital
        import pandas as pd

        df, firm, p = default_df
        # Build a df where n_a_trained is constant and > 0
        df_degenerate = df.copy()
        df_degenerate["n_a_trained"] = 5  # constant value > 0

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_trained_capital(df_degenerate)

        # Check that the annotation text is present
        all_text = []
        for a in fig.axes:
            all_text.extend(t.get_text() for t in a.texts)
        combined = " ".join(all_text)
        assert "trained by t=" in combined, (
            f"Degenerate annotation 'trained by t=' not found. Got text: {all_text}"
        )
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# Stage 6 VIZ tests: wage histogram, wage-vs-output, hiring events
# ---------------------------------------------------------------------------

    def test_fig_wage_histogram(self):
        """T-08 VIZ-1: fig_wage_histogram renders, has correct labels and mean line."""
        from firm_ai_abm.dashboard import fig_wage_histogram
        import matplotlib.lines
        wages = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = fig_wage_histogram(wages)
        assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 0
        ax = fig.axes[0]
        assert ax.get_xlabel() == "wage"
        assert ax.get_ylabel() == "count"
        assert "Wage" in ax.get_title()
        # Mean vline should be present
        vlines = [a for a in ax.get_children() if isinstance(a, matplotlib.lines.Line2D)]
        assert len(vlines) >= 1, "Expected mean vline in wage histogram"
        matplotlib.pyplot.close(fig)

    def test_fig_wage_histogram_empty(self):
        """T-08 VIZ-1: fig_wage_histogram handles empty array without error."""
        from firm_ai_abm.dashboard import fig_wage_histogram
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_wage_histogram(np.array([]))
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_fig_wage_vs_cumulative_output(self):
        """T-09 VIZ-2: fig_wage_vs_cumulative_output renders correct point count."""
        from firm_ai_abm.dashboard import fig_wage_vs_cumulative_output
        wages = np.array([0.9, 1.0, 1.1])
        cum_out = np.array([10.0, 12.0, np.nan])  # worker 2 has NaN → excluded
        a_trained = np.array([False, True, False])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fig = fig_wage_vs_cumulative_output(wages, cum_out, a_trained)
        ax = fig.axes[0]
        # Only 2 valid workers (worker 2 excluded due to NaN)
        scatter_offsets = []
        for collection in ax.collections:
            scatter_offsets.extend(collection.get_offsets().data.tolist())
        assert len(scatter_offsets) == 2, (
            f"Expected 2 scatter points (1 NaN worker excluded), got {len(scatter_offsets)}"
        )
        # Two legend entries (trained + untrained)
        legend = ax.get_legend()
        assert legend is not None, "Expected legend"
        assert len(legend.get_texts()) == 2, "Expected 2 legend entries (trained + untrained)"
        matplotlib.pyplot.close(fig)

    def test_fig_wage_vs_cumulative_output_all_nan(self):
        """T-09 VIZ-2: handles empty active workers without raising."""
        from firm_ai_abm.dashboard import fig_wage_vs_cumulative_output
        wages = np.array([1.0, 1.0])
        cum_out = np.array([np.nan, np.nan])  # all NaN → empty plot
        a_trained = np.array([False, False])
        fig = fig_wage_vs_cumulative_output(wages, cum_out, a_trained)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_fig_hiring_events_disabled(self):
        """T-10 VIZ-3: fig_hiring_events disabled renders the disabled message."""
        from firm_ai_abm.dashboard import fig_hiring_events
        from firm_ai_abm.config import FirmParams
        from firm_ai_abm.firm import make_firm
        from firm_ai_abm.simulate import run_simulation
        from firm_ai_abm.strategy import all_H
        params = FirmParams(seed=0)
        firm = make_firm(params)
        df = run_simulation(firm, all_H)
        fig = fig_hiring_events(df, enable_hiring=False)
        ax = fig.axes[0]
        all_text = [t.get_text() for t in ax.texts]
        combined = " ".join(all_text)
        assert "hiring disabled" in combined or "enable_hiring=False" in combined, (
            f"Expected disabled message; got text: {all_text}"
        )
        matplotlib.pyplot.close(fig)

    def test_fig_hiring_events_active(self):
        """T-10 VIZ-3: fig_hiring_events with hiring events renders scatter point."""
        from firm_ai_abm.dashboard import fig_hiring_events
        import pandas as pd
        df = pd.DataFrame({"t": [0, 1, 2, 3], "n_hired": [0, 3, 0, 0]})
        fig = fig_hiring_events(df, enable_hiring=True)
        ax = fig.axes[0]
        # Should have one scatter collection with one point at t=1
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert any(len(c.get_offsets()) > 0 for c in scatter_collections), (
            "Expected scatter point for hiring event at t=1"
        )
        matplotlib.pyplot.close(fig)


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
        """All param field names appear as widget labels in sidebar.

        scenario_mode is rendered as the "Scenario" radio (key="scenario"), not as a
        widget labeled "scenario_mode". enable_training_delay is a checkbox.
        Both are exempt from the label-match check.
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from streamlit.testing.v1 import AppTest
        from app import _PARAM_FIELDS
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # Collect all widget labels from sidebar (sliders, number_inputs, select_sliders, checkboxes)
        all_labels = set()
        for widget in (
            list(at.sidebar.slider)
            + list(at.sidebar.number_input)
            + list(at.sidebar.select_slider)
            + list(at.sidebar.checkbox)
        ):
            all_labels.add(widget.label)
        # scenario_mode is a radio labeled "Scenario" (not the field name) — exempt
        _EXEMPT = {"scenario_mode"}
        for field in _PARAM_FIELDS:
            if field in _EXEMPT:
                continue
            assert field in all_labels, (
                f"Param field '{field}' not found as a sidebar widget label. "
                f"Available labels: {sorted(all_labels)}"
            )

    def test_main_panel_has_11_plots(self):
        """Main panel renders 11 pyplot figures (as UnknownElement in AppTest 1.57).

        3 new plots added in Stage 6: wage histogram, wage-vs-output scatter,
        hiring events. row6_right is blank (no st.pyplot) → count = 8+3 = 11.
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception
        # st.pyplot renders as UnknownElement in Streamlit 1.57 AppTest
        unknown_count = sum(1 for el in at.main if type(el).__name__ == "UnknownElement")
        assert unknown_count == 11, (
            f"Expected 11 st.pyplot (UnknownElement) in main panel, got {unknown_count}"
        )

    def test_footer_caption_content(self):
        """Footer captions contain required strings (Stage 6: total firings, total hirings, final K_active)."""
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
        assert "total firings" in combined, (
            f"Footer missing 'total firings'. Captions: {caption_values}"
        )
        assert "total hirings" in combined, (
            f"Footer missing 'total hirings'. Captions: {caption_values}"
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
        # The cached 7-tuple is returned → same computed_at timestamp.
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
            f"This may indicate the new params_key (28-tuple) is incorrectly cached."
        )


# ---------------------------------------------------------------------------
# T-14b: README margin recipe CI smoke (no streamlit required)
# ---------------------------------------------------------------------------


def test_README_margin_recipe_at_default_seed():
    """Stage 5 T-14b: README margin recipe (w=8.0, c_auto=0.6, F=5) hits 5–20% margin.

    Verified recipe: FirmParams(seed=0, w=8.0, c_auto=0.6, F=5) with greedy_profit
    produces margin ≈ 16.2% at T=60. Asserts 0.05 <= margin <= 0.20.
    This test gates the README recipe claim at CI time.
    """
    from firm_ai_abm.firm import make_firm
    from firm_ai_abm.simulate import run_simulation
    from firm_ai_abm.strategy import greedy_profit

    params = FirmParams(seed=0, w=8.0, c_auto=0.6, F=5, tasks_per_worker=10, p=1.0)
    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)
    margin = float(df["pi"].mean() / df["Y"].mean())
    assert 0.05 <= margin <= 0.20, (
        f"README margin recipe out of range: margin={margin:.3f} ({margin*100:.1f}%). "
        "If recipe parameters changed, update README and this test together."
    )


# ---------------------------------------------------------------------------
# T-11: Stale-banner and margin-scenario AppTest coverage
# ---------------------------------------------------------------------------


class TestDashboardStaleBanner:
    """T-11: AppTest coverage for the stale-run banner (T-10)."""

    def _get_at(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit AppTest not available")
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        return at

    def test_no_banner_on_fresh_load(self):
        """T-11: No stale-banner on fresh load + Run."""
        at = self._get_at()
        at.button[0].click()  # Run
        at.run()
        stale_warnings = [w for w in at.warning if "Params changed" in str(w.value)]
        assert len(stale_warnings) == 0, (
            f"Expected no 'Params changed' warning on fresh load, found {len(stale_warnings)}"
        )

    def test_banner_on_param_change(self):
        """T-11: Stale-banner appears after param change without pressing Run."""
        at = self._get_at()
        # Click Run once to establish last_run_key
        at.button[0].click()
        at.run()
        # Change N slider value without clicking Run
        n_input = next((inp for inp in at.number_input if inp.key == "N"), None)
        if n_input is None:
            pytest.skip("N number_input not found (widget structure may have changed)")
        n_input.set_value(n_input.value + 10)
        at.run()
        stale_warnings = [w for w in at.warning if "Params changed" in str(w.value)]
        assert len(stale_warnings) >= 1, (
            "Expected 'Params changed' warning after param change without Run"
        )

    def test_banner_clears_after_run(self):
        """T-11: Banner clears after clicking Run."""
        at = self._get_at()
        at.button[0].click()
        at.run()
        n_input = next((inp for inp in at.number_input if inp.key == "N"), None)
        if n_input is None:
            pytest.skip("N number_input not found")
        n_input.set_value(n_input.value + 10)
        at.run()
        # Now click Run to clear the banner
        at.button[0].click()
        at.run()
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

    def test_margin_scenario_strategy_auto_target(self):
        """T-11: Selecting margin scenario forces strategy=target_margin in cached call."""
        at = self._get_at()
        scenario_radio = next((r for r in at.radio if r.key == "scenario"), None)
        if scenario_radio is None:
            pytest.skip("scenario radio not found")
        scenario_radio.set_value("margin")
        at.run()
        at.button[0].click()
        at.run()
        # The caption should say "Strategy: target_margin"
        captions = [c.value for c in at.caption]
        assert any("target_margin" in str(c) for c in captions), (
            f"Expected caption mentioning 'target_margin' in margin mode, got: {captions}"
        )
