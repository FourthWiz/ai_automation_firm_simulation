"""Tests for F-02 alpha/beta histogram helpers and their integration into run_cached.

Five sub-tests:
1. test_fig_alpha_histogram_returns_figure  — function returns a go.Figure with a Histogram trace
2. test_fig_alpha_histogram_empty_array     — empty input returns a figure with no vline annotation
3. test_fig_beta_histogram_returns_figure   — symmetric for beta
4. test_histograms_render_in_tab_modes      — AppTest smoke: charts appear in the rendered app
5. test_alpha_beta_arrays_flow_through_cache — run_cached 9-tuple contains alpha/beta arrays
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
import pytest

# Ensure repo root is importable when running from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firm_ai_abm.dashboard import fig_alpha_histogram, fig_beta_histogram


class TestFigAlphaHistogram:
    """Tests for fig_alpha_histogram helper."""

    def test_fig_alpha_histogram_returns_figure(self):
        """Sub-test 1: returns a go.Figure with at least one Histogram trace."""
        rng = np.random.default_rng(0)
        alpha = rng.uniform(0, 1, 500)
        fig = fig_alpha_histogram(alpha)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) >= 1, "Expected at least one Histogram trace"

    def test_fig_alpha_histogram_empty_array(self):
        """Sub-test 2: empty input returns a Figure without raising; no vline annotation."""
        fig = fig_alpha_histogram(np.array([]))
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        # No vline should be added for empty array (no annotation)
        # layout.shapes are None/empty when no vline was added
        shapes = fig.layout.shapes if fig.layout.shapes else []
        assert len(shapes) == 0, (
            f"Expected no vline shapes for empty input, got {shapes}"
        )


class TestFigBetaHistogram:
    """Tests for fig_beta_histogram helper."""

    def test_fig_beta_histogram_returns_figure(self):
        """Sub-test 3: returns a go.Figure with at least one Histogram trace."""
        rng = np.random.default_rng(42)
        beta = rng.uniform(0, 1, 500)
        fig = fig_beta_histogram(beta)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) >= 1, "Expected at least one Histogram trace"

    def test_fig_beta_histogram_empty_array(self):
        """Sub-test 3b: empty input returns a Figure without raising; no vline."""
        fig = fig_beta_histogram(np.array([]))
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        shapes = fig.layout.shapes if fig.layout.shapes else []
        assert len(shapes) == 0, (
            f"Expected no vline shapes for empty input, got {shapes}"
        )

    def test_fig_alpha_histogram_mean_annotation(self):
        """fig_alpha_histogram with 3-element input has mean=0.500 annotation."""
        fig = fig_alpha_histogram(np.array([0.0, 0.5, 1.0]))
        annotation_texts = [a.text for a in fig.layout.annotations] if fig.layout.annotations else []
        assert any("mean=0.500" in t for t in annotation_texts), (
            f"Expected 'mean=0.500' in annotations, got: {annotation_texts}"
        )


class TestHistogramsInApp:
    """AppTest smoke: histograms render inside the running app (sub-test 4)."""

    @pytest.fixture(autouse=True)
    def _require_streamlit(self):
        pytest.importorskip("streamlit", minversion="1.28")

    def test_histograms_render_in_tab_modes(self):
        """Sub-test 4: AppTest smoke — chart count went from 13 → 15; named keys present."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=60).run()
        assert not at.exception

        # Verify session state was populated (run happened)
        assert "RUN_COUNTER_VAL_THIS_RUN" in at.session_state, (
            "RUN_COUNTER_VAL_THIS_RUN not set — initial run did not happen"
        )

        # Chart count should be 16 (13 baseline + 2 new histograms + 1 compare chart)
        unknown_count = sum(1 for el in at.main if type(el).__name__ == "UnknownElement")
        assert unknown_count == 16, (
            f"Expected 16 charts (13 baseline + 2 new histograms + 1 compare), got {unknown_count}"
        )


class TestAlphaBetaFlowThroughCache:
    """run_cached 9-tuple contains alpha/beta arrays (sub-test 5)."""

    @pytest.fixture(autouse=True)
    def _require_streamlit(self):
        pytest.importorskip("streamlit", minversion="1.28")

    def test_alpha_beta_arrays_flow_through_cache(self):
        """Sub-test 5: run_cached returns 9-tuple; [7] is alpha list, [8] is beta list."""
        from app import run_cached, params_to_key
        from firm_ai_abm.config import FirmParams

        params = FirmParams(seed=0)
        key = params_to_key(params, 0)
        result = run_cached(key, "all_H")
        assert len(result) == 9, f"Expected 9-tuple from run_cached, got {len(result)}"

        alpha_list = result[7]
        beta_list = result[8]

        assert len(alpha_list) == params.N, (
            f"Expected alpha_list length {params.N}, got {len(alpha_list)}"
        )
        assert len(beta_list) == params.N, (
            f"Expected beta_list length {params.N}, got {len(beta_list)}"
        )

        # All values should be in [0, 1] (Beta distribution)
        alpha_arr = np.array(alpha_list)
        beta_arr = np.array(beta_list)
        assert float(alpha_arr.min()) >= 0.0, f"alpha min below 0: {alpha_arr.min()}"
        assert float(alpha_arr.max()) <= 1.0, f"alpha max above 1: {alpha_arr.max()}"
        assert float(beta_arr.min()) >= 0.0, f"beta min below 0: {beta_arr.min()}"
        assert float(beta_arr.max()) <= 1.0, f"beta max above 1: {beta_arr.max()}"
