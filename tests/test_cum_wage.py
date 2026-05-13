"""T-N3..T-N6: Cumulative wage tracking tests (margin-floor-and-worker-wage-accumulation task).

T-N3: numeric correctness — no firings, sigma=0, mean_accum_wage == T * w at final period
T-N3b: K=0 edge case — all rows NaN for mean_accum_wage, 0 for ever_worked_count
T-N4: firing path — closed_worker_wages grows by exactly n_fired_total
T-N5: reset clears closed_worker_wages and zeroes cum_wage (CRIT-1 / D-05 fix)
T-N5b: cross-run isolation — second run equals first (stale accumulation blocked by reset)
T-N6: byte-parity guard — dormant path (T_review=inf) bit-identical for existing columns
"""
import math

import numpy as np
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H


def test_TN3_cum_wage_no_firings():
    """T-N3: No-firing run — mean_accum_wage at final period == T * w within 1e-9."""
    params = FirmParams(
        seed=0,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=math.inf,
        enable_hiring=False,
        enable_replenish_hiring=False,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    assert firm.closed_worker_wages == [], "no firings expected"
    expected_per_worker = params.T * params.w
    assert np.allclose(firm.workforce.cum_wage, expected_per_worker, atol=1e-9), (
        f"Expected each worker's cum_wage == {expected_per_worker}, "
        f"got min={firm.workforce.cum_wage.min():.6f} max={firm.workforce.cum_wage.max():.6f}"
    )
    assert df["mean_accum_wage"].iloc[-1] == pytest.approx(expected_per_worker, abs=1e-9)
    assert df["ever_worked_count"].iloc[-1] == firm.workforce.K


def test_TN3b_K0_edge_case():
    """T-N3b: N=5, tasks_per_worker=10 → K0=0 workers; all mean_accum_wage rows NaN, ever_worked_count==0."""
    params = FirmParams(
        seed=0,
        N=5,
        tasks_per_worker=10,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=10,
        T_review=math.inf,
        enable_hiring=False,
        enable_replenish_hiring=False,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    assert df["mean_accum_wage"].isna().all(), "K=0 → all mean_accum_wage must be NaN"
    assert (df["ever_worked_count"] == 0).all(), "K=0 → ever_worked_count must be 0"


def test_TN4_cum_wage_with_firings():
    """T-N4: Firing path — closed_worker_wages grows by exactly n_fired_total workers."""
    params = FirmParams(
        seed=0,
        T=40,
        T_review=10.0,
        firing_threshold=10.0,   # high threshold → many firings
        p=0.22,
        tasks_per_worker=5,
        enable_replenish_hiring=False,
        enable_hiring=False,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    n_fired_total = int(df["n_review_fired"].sum())
    assert n_fired_total > 0, "high firing_threshold must produce at least one firing"
    assert len(firm.closed_worker_wages) == n_fired_total, (
        f"Expected {n_fired_total} closed_worker_wages entries, "
        f"got {len(firm.closed_worker_wages)}"
    )
    # All recorded wages must be non-negative (wages are positive by construction)
    assert all(v >= 0.0 for v in firm.closed_worker_wages), (
        "closed_worker_wages contains negative values"
    )


def test_TN5_reset_clears_closed_wages():
    """T-N5: firm.reset() clears closed_worker_wages and zeroes workforce.cum_wage (D-05 / CRIT-1 fix)."""
    params = FirmParams(
        seed=0,
        T=20,
        T_review=10.0,
        firing_threshold=10.0,
        p=0.22,
    )
    firm = make_firm(params)
    run_simulation(firm, all_H)

    # Ensure closed_worker_wages is non-empty; if not, inject a sentinel
    if len(firm.closed_worker_wages) == 0:
        firm.closed_worker_wages = [42.0]

    firm.reset()

    assert firm.closed_worker_wages == [], "reset() must clear closed_worker_wages"
    assert firm.workforce.cum_wage.sum() == 0.0, (
        "reset() must zero workforce.cum_wage (D-05 CRIT-1 fix)"
    )


def test_TN5b_cross_run_cum_wage_isolation():
    """T-N5b: Second run on same firm (after reset) equals first run — no stale accumulation.

    This test directly fails on code missing the D-05 fix (cum_wage not zeroed in reset).
    """
    params = FirmParams(
        seed=0,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=math.inf,
        enable_hiring=False,
        enable_replenish_hiring=False,
    )
    firm = make_firm(params)

    # Run 1
    run_simulation(firm, all_H)
    pre_reset_sum = firm.workforce.cum_wage.sum()
    assert pre_reset_sum > 0.0, "sanity: accumulated wages must be positive after run"

    # Reset — D-05 fix must zero cum_wage
    firm.reset()
    assert firm.workforce.cum_wage.sum() == 0.0, (
        "reset() must zero cum_wage (FAILS on un-fixed D-05 code)"
    )

    # Run 2 on same firm (same theta/wage preserved across reset)
    run_simulation(firm, all_H)
    post_second_sum = firm.workforce.cum_wage.sum()

    # Second run must equal first (not 2x due to stale accumulation)
    assert post_second_sum == pytest.approx(pre_reset_sum, rel=1e-9), (
        f"Run 2 cum_wage sum {post_second_sum:.6f} != Run 1 sum {pre_reset_sum:.6f}. "
        "Stale accumulation from Run 1 was not cleared by reset()."
    )


def test_TN6_byte_parity_dormant_path():
    """T-N6: Dormant-path byte-parity — existing pi/Y/C/K columns unchanged after adding mean_accum_wage."""
    from firm_ai_abm import validate

    firm_factory = lambda: make_firm(
        FirmParams(
            seed=0,
            sigma_theta=0.0,
            sigma_w=0.0,
            tasks_per_worker=10,
            p=1.0,
        )
    )
    passed, details = validate.check7_phase1_parity(firm_factory)
    assert passed, f"Byte-parity failed: {details}"
