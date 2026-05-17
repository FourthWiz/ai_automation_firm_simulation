"""Stage 3 / Stage 5 tests: periodic firing review (T-08 .. T-17).

Stage 5 changes (D-03): apply_firings_and_replace removed; replaced by
  apply_firings (K shrinks) + replace_to_target (opt-in hire-back).

Test plan reference: stage-3/current-plan.md and stage-5/current-plan.md.
Tests T-08..T-17 cover:
  T-08: firing_review fires negative-surplus workers, leaves positive-surplus alone
  T-09: surplus calculation correctness (mean_output - wage matches definition)
  T-10a: apply_firings drops K and re-indexes survivors by descending tenure
  T-10b: replace_to_target restores K; no-op identity fast-path
  T-10 (run_simulation): K drops at review period; K_clamp_events > 0 after K=0
  T-11: Tier-A check8 Stage 3 neutrality (T_review=inf == Stage 2 fixtures)
  T-12: replacement workers have tenure=0, hire_t=t, a_trained=False
  T-13: c_train_lost is metric-only, NOT charged into pi
  T-14: T_review=inf neutrality fast-path (inf == T_review=999 byte-identical)
  T-15: output_per_worker NaN handling (sub1: all-NaN no RuntimeWarning;
        sub2: pre-hire NaN boundary with replace_to_target; sub3: dropped cols are NaN)
  T-16: numeraire invariance with firing review ACTIVE
  T-17: greedy gaming smoke — SKIPPED (Stage 5 D-03: no-replace semantics)
"""
import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.review import apply_firings, replace_to_target, firing_review
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import (
    all_H,
    all_A,
    all_T,
    greedy_profit,
    greedy_with_switching,
)
from firm_ai_abm.workers import Workforce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_workforce(theta, wage, a_trained=None, tenure=None, hire_t=None):
    """Build a Workforce with explicit arrays for unit tests."""
    K = len(theta)
    return Workforce(
        theta=np.array(theta, dtype=np.float64),
        wage=np.array(wage, dtype=np.float64),
        a_trained=np.zeros(K, dtype=bool) if a_trained is None else np.array(a_trained, dtype=bool),
        tenure=np.zeros(K, dtype=int) if tenure is None else np.array(tenure, dtype=int),
        hire_t=np.zeros(K, dtype=int) if hire_t is None else np.array(hire_t, dtype=int),
    )


def _default_params(**kwargs):
    """Return FirmParams with sigma_theta=0, sigma_w=0 and any overrides.

    Pins tasks_per_worker=10, p=1.0 by default to preserve Stage 1–2 fixtures.
    """
    defaults = {"tasks_per_worker": 10, "p": 1.0}
    defaults.update(kwargs)
    return FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, **defaults)


def _make_zero_acpw(opw: np.ndarray) -> np.ndarray:
    """Build an aug_cost_per_worker matrix matching opw shape: 0.0 where opw is not NaN."""
    acpw = np.full_like(opw, np.nan, dtype=np.float64)
    acpw[~np.isnan(opw)] = 0.0
    return acpw


# ---------------------------------------------------------------------------
# T-08: fire negative-surplus, keep positive-surplus; c_train_lost accounting
# ---------------------------------------------------------------------------


def test_T08_firing_review_fires_negative_surplus():
    """firing_review fires only negative-surplus workers (surplus < 0).

    F=0.0 so F-share does not affect the fire mask (isolates wage-vs-revenue logic).
    Surplus formula: p * mean_output - wage - mean_aug_cost - F/K_review.
    With F=0 and all H-mode (aug_cost=0): reduces to p * mean_output - wage.
    """
    params = _default_params(T_review=10.0, firing_threshold=0.0, F=0.0)

    wf = _make_fake_workforce(
        theta=[1.5, 1.0, 0.5],
        wage=[1.0, 1.0, 1.0],
        a_trained=[False, False, False],
    )
    # Manually fill output_per_worker: worker 0 → mean=1.5, worker 1 → mean=1.0, worker 2 → mean=0.4
    opw = np.full((20, 3), np.nan)
    opw[0:10, 0] = 1.5   # mean 1.5 > wage 1.0 → surplus > 0
    opw[0:10, 1] = 1.0   # mean 1.0 == wage 1.0 → surplus = 0.0 (not < 0)
    opw[0:10, 2] = 0.4   # mean 0.4 < wage 1.0 → surplus = -0.6 < 0 → fired
    acpw = _make_zero_acpw(opw)

    fire_indices, c_train_lost = firing_review(
        wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
    )

    assert np.array_equal(fire_indices, np.array([2])), (
        f"Expected [2], got {fire_indices}"
    )
    assert c_train_lost == 0.0, f"Expected 0.0 (untrained), got {c_train_lost}"


def test_T08_firing_review_c_train_lost_when_trained():
    """c_train_lost reflects trained-worker count when a_trained=True for fired worker.

    F=0.0 so only worker 2 fires (surplus = 0.4-1.0-0-0 = -0.6). With default F=5.0
    and K=3, F_share=1.667 would tip workers 0 and 1 into firing too, changing the
    expected fire mask.
    """
    params = _default_params(T_review=10.0, firing_threshold=0.0, c_train=5.0, F=0.0)

    wf = _make_fake_workforce(
        theta=[1.5, 1.0, 0.5],
        wage=[1.0, 1.0, 1.0],
        a_trained=[False, False, True],  # worker 2 is trained
    )
    opw = np.full((20, 3), np.nan)
    opw[0:10, 0] = 1.5
    opw[0:10, 1] = 1.0
    opw[0:10, 2] = 0.4   # fired
    acpw = _make_zero_acpw(opw)

    fire_indices, c_train_lost = firing_review(
        wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
    )

    assert np.array_equal(fire_indices, np.array([2]))
    # 1 trained worker fired × c_train=5.0
    assert np.isclose(c_train_lost, 5.0), f"Expected 5.0, got {c_train_lost}"


def test_T08_firing_review_T_review_inf_returns_empty():
    """T_review=math.inf → no review, returns empty regardless of surplus.

    math.isinf short-circuit fires before reading aug_cost_per_worker (D-11).
    """
    params = _default_params(T_review=math.inf)
    wf = _make_fake_workforce(theta=[0.1], wage=[10.0])
    opw = np.full((20, 1), -999.0)  # absurdly negative
    acpw = _make_zero_acpw(opw)

    fire_indices, c_train_lost = firing_review(
        wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
    )
    assert fire_indices.size == 0
    assert c_train_lost == 0.0


# ---------------------------------------------------------------------------
# T-09: surplus calculation correctness
# ---------------------------------------------------------------------------


def test_T09_surplus_calculation_correctness():
    """Surplus = p * mean_output - wage - mean_aug_cost - F/K_review.

    F=0.0 and aug_cost=0 so formula reduces to p*mean_output - wage, letting us
    verify the price-scaled revenue term in isolation.
    surplus[0] = 1.0*1.8 - 2.0 - 0 - 0 = -0.2; surplus[1] = 1.0*0.6 - 0.5 - 0 - 0 = 0.1.
    """
    params_base = _default_params(T_review=10.0, F=0.0)
    wf = _make_fake_workforce(theta=[1.0, 1.0], wage=[2.0, 0.5])

    opw = np.full((20, 2), np.nan)
    opw[0:10, 0] = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0])  # mean=1.8
    opw[0:10, 1] = 0.6   # mean=0.6
    acpw = _make_zero_acpw(opw)

    # threshold=0.0: fire worker 0 (surplus=-0.2 < 0)
    params_0 = replace(params_base, firing_threshold=0.0)
    fire_0, _ = firing_review(wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params_0)
    assert np.array_equal(fire_0, np.array([0])), f"Expected [0], got {fire_0}"

    # threshold=-0.5: both surplus above -0.5 → no firing
    params_neg = replace(params_base, firing_threshold=-0.5)
    fire_neg, _ = firing_review(wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params_neg)
    assert fire_neg.size == 0, f"Expected empty, got {fire_neg}"

    # threshold=0.05: worker 0 still fired (surplus=-0.2 < 0.05); worker 1 not (surplus=0.1 >= 0.05)
    params_pos = replace(params_base, firing_threshold=0.05)
    fire_pos, _ = firing_review(wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params_pos)
    assert np.array_equal(fire_pos, np.array([0])), f"Expected [0], got {fire_pos}"


# ---------------------------------------------------------------------------
# T-10a/T-10b: apply_firings drops K; replace_to_target restores K
# Stage 5 D-03: K may drop after firings
# ---------------------------------------------------------------------------


def test_T10a_apply_firings_drops_K_and_reindexes_tenure():
    """Stage 5 D-03: apply_firings drops K (no auto-replace) and re-indexes survivors."""
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    firm.workforce = Workforce(
        theta=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        wage=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        a_trained=np.zeros(5, dtype=bool),
        tenure=np.array([10, 8, 6, 4, 2], dtype=int),
        hire_t=np.zeros(5, dtype=int),
    )

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    acpw = _make_zero_acpw(opw)
    fire_indices = np.array([1, 3], dtype=int)

    new_wf, new_opw, new_acpw = apply_firings(
        firm, fire_indices, t=10, output_per_worker=opw, aug_cost_per_worker=acpw
    )

    # Stage 5 D-03: K DROPS (was 5, fired 2 → K=3)
    assert new_wf.K == 5 - len(fire_indices), (
        f"Expected K={5 - len(fire_indices)}, got {new_wf.K}"
    )

    # Survivors sorted by descending tenure (D-05)
    assert np.all(np.diff(new_wf.tenure) <= 0), (
        f"Expected non-increasing tenure, got {new_wf.tenure}"
    )

    # Trailing inactive columns [new_wf.K : K_max] must be all-NaN in both arrays
    assert np.all(np.isnan(new_opw[:, new_wf.K:])), (
        "Trailing columns after K_new must be all-NaN in output_per_worker"
    )
    assert new_acpw.shape == new_opw.shape, (
        f"aug_cost shape {new_acpw.shape} must match output shape {new_opw.shape}"
    )


def test_T10b_replace_to_target_restores_K():
    """Stage 5 D-03: replace_to_target restores K after apply_firings; no-op identity fast-path."""
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    firm.workforce = Workforce(
        theta=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        wage=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        a_trained=np.zeros(5, dtype=bool),
        tenure=np.array([10, 8, 6, 4, 2], dtype=int),
        hire_t=np.zeros(5, dtype=int),
    )

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    acpw = _make_zero_acpw(opw)
    fire_indices = np.array([1, 3], dtype=int)

    new_wf, new_opw, new_acpw = apply_firings(
        firm, fire_indices, t=10, output_per_worker=opw, aug_cost_per_worker=acpw
    )
    assert new_wf.K == 3

    # replace_to_target restores K to 5
    firm.workforce = new_wf
    restored_wf, restored_opw, restored_acpw = replace_to_target(
        firm, K_target=5, t=10, output_per_worker=new_opw, aug_cost_per_worker=new_acpw
    )
    assert restored_wf.K == 5, f"Expected K=5 after replace_to_target, got {restored_wf.K}"

    # Descending tenure preserved after combine+reorder
    assert np.all(np.diff(restored_wf.tenure) <= 0), (
        f"Expected non-increasing tenure, got {restored_wf.tenure}"
    )

    # New replacement workers (hire_t=10) columns are all-NaN for pre-hire rows
    repl_slots = np.where(restored_wf.hire_t == 10)[0]
    for k in repl_slots:
        assert np.all(np.isnan(restored_opw[:10, k])), (
            f"Slot {k} (hire_t=10) should have NaN for rows 0..9"
        )

    # No-op fast-path: empty fire_indices → apply_firings returns SAME objects (identity)
    empty_fire = np.array([], dtype=int)
    firm2 = make_firm(params)
    ret_wf, ret_opw, ret_acpw = apply_firings(
        firm2, empty_fire, t=5, output_per_worker=opw, aug_cost_per_worker=acpw
    )
    assert ret_wf is firm2.workforce, "Empty fire_indices must return same workforce (identity)"
    assert ret_opw is opw, "Empty fire_indices must return same opw (identity)"
    assert ret_acpw is acpw, "Empty fire_indices must return same acpw (identity)"


def test_T10_run_simulation_K_drops_at_review_period():
    """Stage 5 D-03: run_simulation K drops at review period and stays down (no auto-restore)."""
    # firing_threshold=10.0 fires all workers (surplus = 10.0 - 1.0 = 9.0 < 10.0)
    # After all fired, workforce.K=0. At t=11, step 2.5 clamps all_H proposal to all-T.
    params = _default_params(T=15, T_review=10.0, firing_threshold=10.0)
    K_target = params.N // params.tasks_per_worker  # 10

    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    # K_active at t=10 should have dropped (all workers fired)
    assert int(df.iloc[10]["K_active"]) < K_target, (
        f"K_active at t=10 should have dropped below {K_target}, got {df.iloc[10]['K_active']}"
    )

    # K stays at the dropped value (no auto-restore)
    assert int(df.iloc[11]["K_active"]) <= int(df.iloc[10]["K_active"]), (
        f"K at t=11 should not exceed K at t=10 (no auto-replace); "
        f"t=10: {df.iloc[10]['K_active']}, t=11: {df.iloc[11]['K_active']}"
    )

    # K_clamp_events > 0 at t=11 (all_H proposed K_max tasks but workforce.K=0 → clamped to all-T)
    assert int(df.iloc[11]["K_clamp_events"]) > 0, (
        f"Expected K_clamp_events > 0 at t=11 (all_H clamped to all-T with K=0); "
        f"got {df.iloc[11]['K_clamp_events']}"
    )

    assert int(df.iloc[10]["n_review_fired"]) >= 1, (
        f"Expected >=1 firing at t=10, got {df.iloc[10]['n_review_fired']}"
    )


# ---------------------------------------------------------------------------
# T-11: Tier-A check8 Stage 3 neutrality
# ---------------------------------------------------------------------------


def test_T11_check8_stage3_neutrality_passes():
    """Tier-A check8 must pass: T_review=inf output byte-identical to Stage 2 fixtures."""
    from firm_ai_abm.validate import run_tier_a

    result = run_tier_a()
    assert result["check8"]["passed"], (
        f"check8_stage3_neutrality FAILED: {result['check8']['details']}"
    )


# ---------------------------------------------------------------------------
# T-12: replacement workers have tenure=0, hire_t=t, a_trained=False
# ---------------------------------------------------------------------------


def test_T12_replacement_worker_initial_state():
    """Stage 5 D-03: after apply_firings + replace_to_target, replacements have
    tenure=0, hire_t=t, a_trained=False.

    Sequence (proc:T-13): apply_firings → firm.workforce = new_wf → replace_to_target.
    F_share = F/K_max = 5.0/10 = 0.5. surplus = 0.0 - 1.0 - 0 - 0.5 = -1.5 < 10 threshold → all fired.
    """
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    # Fill first 10 rows with sub-threshold output so all workers get fired
    opw[0:10, :K_max] = 0.0  # output=0, wage=1.0, F_share=0.5, surplus=-1.5 < 10 → all fired
    acpw = _make_zero_acpw(opw)

    fire_indices, _ = firing_review(
        firm.workforce, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
    )
    assert fire_indices.size == K_max, f"Expected all {K_max} workers fired, got {fire_indices.size}"

    # Stage 5: apply_firings (K drops to 0) then replace_to_target (restore K)
    new_wf, new_opw, new_acpw = apply_firings(
        firm, fire_indices, t=10, output_per_worker=opw, aug_cost_per_worker=acpw
    )
    assert new_wf.K == 0, f"After firing all, K should be 0; got {new_wf.K}"
    firm.workforce = new_wf  # update in-place for replace_to_target
    restored_wf, _, _ = replace_to_target(
        firm, K_target=K_max, t=10, output_per_worker=new_opw, aug_cost_per_worker=new_acpw
    )

    # All replacements should have tenure=0 immediately (before the step-11.5 increment)
    assert np.all(restored_wf.tenure == 0), (
        f"Expected all tenure=0 immediately post-replace; got {restored_wf.tenure}"
    )
    assert np.all(restored_wf.hire_t == 10), (
        f"Expected all hire_t=10; got {restored_wf.hire_t}"
    )
    assert np.all(~restored_wf.a_trained), (
        f"Expected all a_trained=False; got {restored_wf.a_trained}"
    )
    assert restored_wf.K == K_max, (
        f"K-target preserved: expected {K_max}, got {restored_wf.K}"
    )
    assert np.all(np.isfinite(restored_wf.wage) & (restored_wf.wage > 0)), (
        f"All replacement wages should be positive finite; got {restored_wf.wage}"
    )


def test_T12_replacement_worker_tenure_after_run():
    """Supplemental: replacements hired at t=10 have tenure=1 at end of t=10 (step-11.5)."""
    params = _default_params(T=15, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)
    run_simulation(firm, all_H)

    # After run_simulation, all workers were replaced at t=10 and tenure was incremented once
    # by step-11.5 on that period. They are hired at t=10 and end t=10 with tenure=1.
    # Periods 11..14 add 4 more increments → end tenure = 5 after T=15 periods.
    # But the KEY assertion is that hire_t == 10 and tenure > 0.
    hired_at_10 = firm.workforce.hire_t == 10
    if hired_at_10.any():
        # They've been tenure-incremented once per period from t=10 to t=14 inclusive = 5 times
        expected_tenure = 5  # periods 10, 11, 12, 13, 14
        assert np.all(firm.workforce.tenure[hired_at_10] == expected_tenure), (
            f"Workers hired at t=10 should have tenure={expected_tenure} after T=15 run; "
            f"got {firm.workforce.tenure[hired_at_10]}"
        )


# ---------------------------------------------------------------------------
# T-13: c_train_lost is metric-only, NOT charged into pi
# ---------------------------------------------------------------------------


def test_T13_c_train_lost_not_in_pi():
    """c_train_lost records capital destroyed but does NOT enter C or pi."""
    params = _default_params(
        T=15,
        T_review=10.0,
        firing_threshold=10.0,  # fires all workers
        c_train=0.1,
        c_fire=2.0,
        F=5.0,
        p=1.0,
        q_h=1.0,
        w=1.0,
        g=0.5,
        q_a=1.2,
        c_aug=0.05,
        c_auto=0.4,
        c_hire=0.5,
    )
    firm = make_firm(params)
    df = run_simulation(firm, greedy_with_switching)

    row = df.iloc[10]

    # c_train_lost > 0 only if some workers were trained (a_trained=True) before firing
    # With c_train=0.1 and greedy_with_switching, some workers may have been trained.
    # We check that c_train_lost recorded in history is consistent.
    c_train_lost_recorded = float(row["c_train_lost"])
    assert c_train_lost_recorded >= 0.0

    # Manually reconstruct pi at t=10 WITHOUT c_train_lost:
    # pi = p*Y - (task_costs + wage_bill + F + adj_cost + c_review_fire)
    Y_row = float(row["Y"])
    C_row = float(row["C"])
    pi_row = float(row["pi"])
    expected_pi = params.p * Y_row - C_row

    assert np.isclose(pi_row, expected_pi, rtol=1e-12, atol=1e-9), (
        f"pi mismatch at t=10: got {pi_row}, expected {expected_pi}"
    )

    # Verify c_train_lost is NOT in C: if it were, pi would differ by c_train_lost
    if c_train_lost_recorded > 0:
        # pi should NOT equal p*Y - (C + c_train_lost)
        # (if it did, c_train_lost was double-charged)
        pi_if_double_charged = params.p * Y_row - (C_row + c_train_lost_recorded)
        assert not np.isclose(pi_row, pi_if_double_charged, rtol=1e-12, atol=1e-9), (
            "c_train_lost appears to be double-charged into pi!"
        )


def test_T13_c_train_lost_zero_when_no_trained_workers():
    """Sanity: c_train=0 → no worker ever trained → c_train_lost always zero."""
    params = _default_params(T=15, T_review=10.0, firing_threshold=10.0, c_train=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)
    assert float(df["c_train_lost"].sum()) == 0.0


# ---------------------------------------------------------------------------
# T-14: T_review=inf neutrality (fast-path vs T_review=999)
# ---------------------------------------------------------------------------


def test_T14_T_review_inf_equals_T_review_999():
    """T_review=inf and T_review=999 (never fires in T=60 window) produce identical history.

    NOTE (firing-timing-and-horizon-fix T-02): adj_cost and C are intentionally excluded
    from parity_cols. Under finite T_review (999), adj_cost zeros the c_fire term for H→T
    / A→T transitions (Fix 2a). C includes adj_cost, so it differs too at t=0 for any
    strategy that switches modes (e.g., all_T: H→T at t=0 charges c_fire×K under inf but
    0 under 999). Y and K remain byte-identical because no workers are fired within T=60
    periods (T_review=999 > 60). pi is excluded because it depends on C. See plan D-02
    and lesson 2026-05-09.
    """
    parity_cols = ["t", "Y", "K"]
    strategies = [
        ("all_H", all_H),
        ("all_A", all_A),
        ("all_T", all_T),
        ("greedy_profit", greedy_profit),
        ("greedy_with_switching", greedy_with_switching),
    ]

    params_inf = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=math.inf, tasks_per_worker=10, p=1.0)
    params_999 = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=999.0, tasks_per_worker=10, p=1.0)

    for strat_name, strat in strategies:
        firm_inf = make_firm(params_inf)
        df_inf = run_simulation(firm_inf, strat)

        firm_999 = make_firm(params_999)
        df_999 = run_simulation(firm_999, strat)

        for col in parity_cols:
            assert np.array_equal(df_inf[col].values, df_999[col].values), (
                f"Column {col} differs for {strat_name}: "
                f"inf vs 999. Max dev = {np.max(np.abs(df_inf[col].values.astype(float) - df_999[col].values.astype(float)))}"
            )


# ---------------------------------------------------------------------------
# T-15: output_per_worker NaN handling (all-NaN column; pre-hire NaN boundary)
# ---------------------------------------------------------------------------


def test_T15_subtest1_all_nan_column_no_RuntimeWarning():
    """Sub-test 1: all-NaN worker column does not produce RuntimeWarning and is not fired.

    F=0.0 so workers 1 and 2 retain positive surplus (surplus = output - wage - 0 - 0 > 0).
    With default F=5.0 and K=3, F_share=1.667 would push workers 1 and 2 negative.
    """
    params = _default_params(T_review=10.0, firing_threshold=0.0, F=0.0)

    K = 3
    wf = _make_fake_workforce(
        theta=[1.0, 1.0, 1.0],
        wage=[0.5, 0.5, 0.5],
        a_trained=[False, False, False],
    )

    opw = np.full((20, K), np.nan, dtype=np.float64)
    # Worker 0: entirely NaN (simulates all-T worker)
    # Workers 1 and 2: positive output
    opw[0:10, 1] = 1.0   # mean=1.0, surplus=0.5 > 0 → safe
    opw[0:10, 2] = 0.8   # mean=0.8, surplus=0.3 > 0 → safe
    acpw = _make_zero_acpw(opw)

    # Assert no RuntimeWarning from the per-column mask approach (CRIT-1 pin)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fire_indices, _ = firing_review(
            wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
        )
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, (
            f"RuntimeWarning(s) emitted: {[str(x.message) for x in runtime_warnings]}"
        )

    # Worker 0 (all-NaN) must NOT be fired (insufficient evidence, D-01)
    assert 0 not in fire_indices, (
        f"Worker 0 (all-NaN column) must not be a firing candidate; got fire_indices={fire_indices}"
    )
    # Workers 1 and 2 have positive surplus → not fired
    assert fire_indices.size == 0, f"No worker should be fired; got {fire_indices}"


def test_T15_subtest2_pre_hire_NaN_with_replace_to_target():
    """Sub-test 2 (Stage 5): replacement workers via replace_to_target have NaN pre-hire rows.

    Stage 5 update: uses explicit replace_to_target call within run_simulation (via
    apply_firings; run_simulation no longer auto-replaces). The invariant holds because
    replacement workers get fresh all-NaN columns with the same pre-hire guarantee.

    firing_threshold=9.5 ensures all workers fire at t=10 and t=20 (surplus=9.0 < 9.5).
    After run(T=25), workers have hire_t=20.
    """
    params = _default_params(T=25, T_review=10.0, firing_threshold=9.5)

    firm = make_firm(params)
    df = run_simulation(firm, all_H)  # noqa: F841

    opw = firm.output_per_worker  # exposed seam (T-05)
    wf = firm.workforce

    # Stage 5: after firing-only (no auto-replace), workforce.K may be small or 0 at end
    # We test the invariant only for slots that were filled at some point
    assert int(df["n_review_fired"].sum()) > 0, (
        "Expected some workers to be fired; n_review_fired sum = 0. "
        f"firing_threshold={params.firing_threshold}, T_review={params.T_review}"
    )

    # For each slot k still present in the final workforce:
    # rows before hire_t[k] should be ALL NaN (pre-hire boundary)
    for k in range(wf.K):
        hire_t_k = int(wf.hire_t[k])
        if hire_t_k > 0:
            pre_hire_slice = opw[0:hire_t_k, k]
            assert np.all(np.isnan(pre_hire_slice)), (
                f"Slot {k} (hire_t={hire_t_k}): expected all-NaN for rows 0..{hire_t_k-1}; "
                f"got non-NaN entries"
            )


def test_T15_subtest3_dropped_columns_become_NaN():
    """Sub-test 3 (Stage 5 NEW): after bare apply_firings (no replace), trailing cols are NaN.

    Both output_per_worker and aug_cost_per_worker must have all-NaN in dropped columns.
    """
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), 1.0, dtype=np.float64)   # non-NaN to show the wipe
    acpw = np.full((params.T, K_max), 1.0, dtype=np.float64)  # non-NaN to show the wipe
    fire_indices = np.array([0, 1, 2], dtype=int)  # fire first 3 workers

    new_wf, new_opw, new_acpw = apply_firings(
        firm, fire_indices, t=5, output_per_worker=opw, aug_cost_per_worker=acpw
    )

    assert new_wf.K == firm.workforce.K - 3

    # Trailing inactive columns [new_wf.K : K_max] must be all-NaN in both arrays
    assert np.all(np.isnan(new_opw[:, new_wf.K:])), (
        f"Expected columns [{new_wf.K}:{K_max}] to be all-NaN in output_per_worker"
    )
    assert np.all(np.isnan(new_acpw[:, new_wf.K:])), (
        f"Expected columns [{new_wf.K}:{K_max}] to be all-NaN in aug_cost_per_worker"
    )


# ---------------------------------------------------------------------------
# T-16: numeraire invariance with firing review ACTIVE
# ---------------------------------------------------------------------------


def test_T16_numeraire_invariance_with_firing_active():
    """Multiplying all monetary params by 2 scales pi by 2 even with firing review active."""
    # With sigma_theta=0, sigma_w=0, firing_threshold=0.0: surplus = mean_output - wage
    # Both output and wage scale by 2x, so surplus × 2 means SAME workers are fired.
    # Hence n_review_fired is identical in both runs, and pi scales by exactly 2.
    params_base = FirmParams(
        seed=0,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=10.0,
        firing_threshold=0.0,
        tasks_per_worker=10,
        p=1.0,
    )

    SCALED = ("w", "c_aug", "c_auto", "c_fire", "c_hire", "c_train", "F", "p")
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED}
    params_scaled = replace(params_base, **scaled_kwargs)

    strategies = [all_H, all_A]  # two strategies is sufficient for numeraire

    for strat in strategies:
        name = strat.__name__

        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, strat)

        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, strat)

        # pi_scaled == 2 * pi_base for every period
        assert np.allclose(
            df_scaled["pi"].values,
            2.0 * df_base["pi"].values,
            rtol=1e-10,
            atol=1e-9,
        ), (
            f"Numeraire invariance failed for {name}: "
            f"max dev = {np.max(np.abs(df_scaled['pi'].values - 2.0 * df_base['pi'].values))}"
        )

        # SAME workers fired in both runs (invariance of who gets fired)
        assert np.array_equal(
            df_scaled["n_review_fired"].values,
            df_base["n_review_fired"].values,
        ), (
            f"n_review_fired differs between base and scaled for {name}: "
            f"base={df_base['n_review_fired'].values}, scaled={df_scaled['n_review_fired'].values}"
        )


# ---------------------------------------------------------------------------
# T-17: greedy gaming smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Stage 5 D-06: gaming test deferred to Phase 2 — no-replace semantics saturates totals at K_initial, eliminating discriminating power; reframe with strategy-driven rehire in Phase 2.")
def test_T17_greedy_gaming_smoke_no_firing_cascade():
    """Greedy gaming smoke: greedy_with_switching does not massively over-fire vs greedy_profit."""
    params = FirmParams(
        seed=0,
        T=60,
        T_review=10.0,
        firing_threshold=0.0,
        sigma_theta=0.2,
        sigma_w=0.05,
        tasks_per_worker=10,
        p=1.0,
    )
    K = params.N // params.tasks_per_worker  # 10

    firm_gp = make_firm(params)
    df_gp = run_simulation(firm_gp, greedy_profit)

    firm_gs = make_firm(params)
    df_gs = run_simulation(firm_gs, greedy_with_switching)

    total_fired_gp = int(df_gp["n_review_fired"].sum())
    total_fired_gs = int(df_gs["n_review_fired"].sum())

    # Discriminating comparison (MAJ-1, round 3): gaming would cause greedy_with_switching to
    # fire substantially more than greedy_profit. The bound: switching total ≤ no-switching + K.
    assert total_fired_gs <= total_fired_gp + K, (
        f"Greedy gaming detected: greedy_with_switching fired {total_fired_gs} workers "
        f"vs greedy_profit {total_fired_gp} (excess > K={K}). "
        f"R-10 escalates to High/Medium — mitigation required."
    )

    # Firm remains profitable under both strategies (no death spiral)
    assert float(df_gp["pi"].iloc[-1]) > 0, (
        f"greedy_profit unprofitable at final period: pi={df_gp['pi'].iloc[-1]}"
    )
    assert float(df_gs["pi"].iloc[-1]) > 0, (
        f"greedy_with_switching unprofitable at final period: pi={df_gs['pi'].iloc[-1]}"
    )


# ---------------------------------------------------------------------------
# T-04: Regression test — low-p scenario fires workers post-fix (BUG-1)
# ---------------------------------------------------------------------------


def test_low_price_triggers_firings():
    """BUG-1 regression: post-fix surplus = p * mean_output - wage fires workers at low p.

    Uses all_H strategy (not greedy_profit — greedy switches to augmented mode at p=0.22,
    yielding p * output >> wage even post-fix, so no firings would occur with greedy).
    With all_H: H-mode output per worker ≈ q_h * tpw * theta = 5 * theta.
    Post-fix surplus = p * 5 * theta - wage = 1.1 * theta - wage.
    For low-theta workers (theta < ~0.9), wage > 1.1 * theta → surplus < 0 → fired.

    Pre-fix (bug): surplus = mean_output - wage = 5*theta - wage > 0 for all workers → 0 firings.
    Post-fix: firings > 0 because some workers have p * output < wage.
    """
    # Post-fix: firings occur at p=0.22 all_H because low-theta workers have p*output < wage
    params = FirmParams(
        N=100, T=60, tasks_per_worker=5,
        T_review=10.0, firing_threshold=0.0,
        p=0.22, seed=0,
        sigma_theta=0.2, sigma_w=0.05,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    total_fired = int(df["n_review_fired"].sum())

    assert total_fired > 0, (
        f"BUG-1 regression: expected firings > 0 at p=0.22 all_H with sigma_theta=0.2, "
        f"got {total_fired}. The price factor (params.p) may be missing from surplus formula."
    )


# ---------------------------------------------------------------------------
# T-06: Hiring-enabled smoke test
# ---------------------------------------------------------------------------


def test_enable_hiring_restores_headcount():
    """Smoke test: enable_hiring=True restores headcount after firings, with hire_delay_periods=1.

    After D-02, hires are queued at the fire period and drained at t+hire_delay_periods.
    n_hired[t_fire] == 0; n_hired[t_fire+1] > 0 (delay realized).

    With the K*-based hire target, under these params (p=0.22, q_h=1.0, w=1.0,
    F=5.0, firing_threshold=0.0, all_H) the denominator is 0.22*5 - 1 - 0 - 0 = 0.10,
    giving K* = ceil(5/0.10) = 50, which is capped at K_max = N // tasks_per_worker
    = 100 // 5 = 20. K_max coincides with K0 by construction.

    Recipe: same as test_low_price_triggers_firings plus enable_hiring=True.
    Assertions:
      - n_hired sum > 0 (hires occurred, just delayed by 1 period)
      - at first fire period t_fire: n_hired[t_fire] == 0 (delay not yet realized)
      - at t_fire+1: n_hired[t_fire+1] > 0 (drain realized)
      - firm.workforce.K <= K0 after run
      - pi shows no runaway negative cost (sanity)
    """
    # Pin w=1.0 (pre-99ddaea default) so hiring is profitable at p=0.22:
    # p*tpw - w = 0.22*5 - 1.0 = 0.1 > 0 → optimal_hire_target returns K* > 0.
    # With w=2.0 (new default): p*tpw - w = -0.9 < 0 → denominator negative → no hires.
    params = FirmParams(
        N=100, T=60, tasks_per_worker=5,
        T_review=10.0, firing_threshold=0.0,
        p=0.22, w=1.0, seed=0,
        sigma_theta=0.2, sigma_w=0.05,
        enable_hiring=True,
        hire_delay_periods=1,
    )
    firm = make_firm(params)
    K0 = firm.K0
    df = run_simulation(firm, all_H)

    assert int(df["n_hired"].sum()) > 0, (
        "enable_hiring=True with low p should produce hires; got 0"
    )

    # Verify delay semantics: at first fire period, no same-period hires.
    fire_periods = df[df["n_review_fired"] > 0]["t"].values
    if len(fire_periods) > 0:
        t_fire = int(fire_periods[0])
        n_hired_at_fire = int(df[df["t"] == t_fire]["n_hired"].iloc[0])
        assert n_hired_at_fire == 0, (
            f"Expected n_hired=0 at fire period t={t_fire} (delay not yet realized); "
            f"got {n_hired_at_fire}. hire_delay_periods={params.hire_delay_periods}"
        )
        drain_rows = df[df["t"] == t_fire + 1]
        if len(drain_rows) > 0:
            n_hired_at_drain = int(drain_rows["n_hired"].iloc[0])
            assert n_hired_at_drain > 0, (
                f"Expected n_hired>0 at drain period t={t_fire+1}; "
                f"got {n_hired_at_drain}."
            )

    # Under K*-based hiring, K settles at the optimal hire target.
    # The bound that always holds: K <= K_max (== K0 by construction in make_firm).
    assert firm.workforce.K > 0, (
        f"Firm should still have workers after run; got K={firm.workforce.K}"
    )
    assert firm.workforce.K <= K0, (
        f"workforce.K={firm.workforce.K} exceeds K_max=K0={K0}"
    )

    # Sanity: pi shows no runaway negative cost (very loose bound)
    assert df["pi"].min() > -10 * params.c_hire * K0, (
        f"pi min {df['pi'].min():.2f} too negative for hire costs"
    )


# ---------------------------------------------------------------------------
# T-06b: optimal_hire_target — K* strictly below K_max
# ---------------------------------------------------------------------------


def test_optimal_hire_target_K_star_between_current_and_K_max():
    """optimal_hire_target returns K* in (K_current, K_max) for mid-range denom.

    Directly tests the formula without running a simulation, bypassing the
    all_H homogeneous-output regime where K* and K_post_fire are co-determined.

    With q_h=0.5, tasks_per_worker=5: e_output = 2.5 per worker.
    denom = p * 2.5 - w - 0 - threshold = 1.0*2.5 - 1.0 - 0 - 0 = 1.5.
    K* = ceil(F / denom) = ceil(10 / 1.5) = 7.
    K_current=5 < K*=7 < K_max=20 → function should return 7.
    """
    from firm_ai_abm.review import optimal_hire_target
    from firm_ai_abm.workers import Workforce

    params = FirmParams(
        N=100, T=30, tasks_per_worker=5,
        T_review=10.0,
        firing_threshold=0.0,
        p=1.0, q_h=0.5, w=1.0, F=10.0,
        c_aug=0.0,
        seed=0,
    )
    firm = make_firm(params)
    K_max = params.N // params.tasks_per_worker  # 20
    K_current = 5

    # Trim workforce to K_current survivors
    wf = firm.workforce
    firm.workforce = Workforce(
        theta=wf.theta[:K_current],
        wage=wf.wage[:K_current],
        a_trained=wf.a_trained[:K_current],
        tenure=wf.tenure[:K_current],
        hire_t=wf.hire_t[:K_current],
        a_training_in_progress=(
            wf.a_training_in_progress[:K_current]
            if wf.a_training_in_progress is not None else None
        ),
    )

    # Mock trailing window: K_current survivors with output = q_h * tpw = 2.5
    t = 10
    e_output = params.q_h * params.tasks_per_worker  # 2.5
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    opw[0:t, :K_current] = e_output
    acpw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    acpw[0:t, :K_current] = 0.0  # H-mode, no aug cost

    denom = params.p * e_output - params.w - 0.0 - params.firing_threshold  # 1.5
    K_star_expected = math.ceil(params.F / denom)  # 7
    assert K_current < K_star_expected < K_max, (
        f"Test misconfigured: need K_current={K_current} < K*={K_star_expected} < K_max={K_max}"
    )

    result = optimal_hire_target(firm, t, opw, acpw, params)
    assert result == K_star_expected, (
        f"Expected K*={K_star_expected}, got {result}. "
        f"optimal_hire_target should return K_star when K_current < K_star < K_max."
    )


# ---------------------------------------------------------------------------
# T-06c: optimal_hire_target — unprofitable-hire branch (denominator <= 0)
# ---------------------------------------------------------------------------


def test_enable_hiring_skips_when_new_hires_unprofitable():
    """When E[surplus] of a new hire is non-positive, optimal_hire_target returns
    wf.K → no hiring even though firings occurred.

    p=0.15, tpw=5, q_h=1 → E[output]=5, denom = 0.15*5 - 1 - 0 - 0 = -0.25 < 0.
    Firings still occur (surplus negative) but no rehires happen.
    """
    params = FirmParams(
        N=100, T=20, tasks_per_worker=5,
        T_review=5.0, firing_threshold=0.0,
        p=0.15, w=1.0, F=5.0,
        c_aug=0.0,
        seed=0,
        sigma_theta=0.05, sigma_w=0.05,
        enable_hiring=True,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    assert int(df["n_review_fired"].sum()) > 0, (
        "Expected at least one firing under low-p params"
    )
    assert int(df["n_hired"].sum()) == 0, (
        f"Expected n_hired == 0 when new hires unprofitable; got {int(df['n_hired'].sum())}"
    )


# ---------------------------------------------------------------------------
# T-06d: optimal_hire_target — no-trailing-data fallback (unit test)
# ---------------------------------------------------------------------------


def test_optimal_hire_target_no_data_fallback():
    """optimal_hire_target returns wf.K when no trailing data is available.

    Branches verified:
      - t == 0 → wf.K
      - all-NaN window at t > 0 → wf.K
      - T_review = inf (defensive guard) → wf.K
    """
    from firm_ai_abm.review import optimal_hire_target

    params = FirmParams(
        N=100, T=20, tasks_per_worker=5,
        T_review=5.0, firing_threshold=0.0,
        p=0.5, w=1.0, F=5.0,
        seed=0,
    )
    firm = make_firm(params)
    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    acpw = np.full((params.T, K_max), np.nan, dtype=np.float64)

    assert optimal_hire_target(firm, 0, opw, acpw, params) == firm.workforce.K
    assert optimal_hire_target(firm, 3, opw, acpw, params) == firm.workforce.K

    params_inf = replace(params, T_review=math.inf)
    assert optimal_hire_target(firm, 5, opw, acpw, params_inf) == firm.workforce.K


# ---------------------------------------------------------------------------
# T-07: Numeraire-invariance with low p (T-16b)
# ---------------------------------------------------------------------------


def test_T16b_numeraire_with_low_p():
    """Numeraire invariance holds with p=0.5 (and 2*p=1.0 for the scaled side).

    Confirms params.p * mean_output - wage scales linearly with monetary doubling.
    Post-fix: surplus = p * mean_output - wage. Under monetary doubling (all monetary
    params × 2, including p), surplus × 2 → same fire_mask → pi × 2.
    """
    params_base = FirmParams(
        seed=0,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=10.0,
        firing_threshold=0.0,
        tasks_per_worker=10,
        p=0.5,
    )

    SCALED = ("w", "c_aug", "c_auto", "c_fire", "c_hire", "c_train", "F", "p")
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED}
    params_scaled = replace(params_base, **scaled_kwargs)

    strategies = [all_H, all_A]

    for strat in strategies:
        name = strat.__name__

        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, strat)

        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, strat)

        assert np.allclose(
            df_scaled["pi"].values,
            2.0 * df_base["pi"].values,
            rtol=1e-10,
            atol=1e-9,
        ), (
            f"T-16b numeraire invariance failed for {name} at p=0.5: "
            f"max dev = {np.max(np.abs(df_scaled['pi'].values - 2.0 * df_base['pi'].values))}"
        )

        assert np.array_equal(
            df_scaled["n_review_fired"].values,
            df_base["n_review_fired"].values,
        ), (
            f"T-16b n_review_fired differs at p=0.5 for {name}"
        )


# ---------------------------------------------------------------------------
# T-09: F-share numeraire invariance (adaptive-firing-surplus regression fence)
# ---------------------------------------------------------------------------


def test_F_share_numeraire():
    """F_share numeraire invariance: pi_scaled == 2 * pi_base AND fire counts identical.

    Scenario: F=10.0, K0=10, F_share=1.0 per worker. With all_H and sigma=0:
    effective_surplus = 1.0 * 10 * 1.0 - 1.0 - 0 - 1.0 = 9.0 > 0 → no firings.
    Under 2× monetary scaling, F_share = 2.0, surplus = 2*10 - 2 - 0 - 2 = 18.0 > 0 → still no firings.
    pi_scaled must equal exactly 2 * pi_base (R-03 regression fence for the F_share formula).
    """
    params_base = FirmParams(
        seed=0,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=10.0,
        firing_threshold=0.0,
        tasks_per_worker=10,
        p=1.0,
        F=10.0,
        c_aug=0.05,
    )

    SCALED = ("w", "c_aug", "c_auto", "c_fire", "c_hire", "c_train", "F", "p", "firing_threshold")
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED}
    params_scaled = replace(params_base, **scaled_kwargs)

    for strat, name in [(all_H, "all_H"), (all_A, "all_A")]:
        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, strat)

        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, strat)

        assert np.allclose(
            df_scaled["pi"].values,
            2.0 * df_base["pi"].values,
            rtol=1e-10,
            atol=1e-9,
        ), (
            f"T-09 F_share numeraire failed for {name}: "
            f"max dev = {np.max(np.abs(df_scaled['pi'].values - 2.0 * df_base['pi'].values))}"
        )

        assert np.array_equal(
            df_scaled["n_review_fired"].values,
            df_base["n_review_fired"].values,
        ), (
            f"T-09 n_review_fired differs between base and scaled for {name}"
        )


# ---------------------------------------------------------------------------
# T-10: Single-tick cascade damping (stale-K discrimination test)
# ---------------------------------------------------------------------------


def test_F_share_stale_K_within_tick():
    """Within one review tick, F_share uses pre-fire K (stale). Discriminating recipe.

    4 workers, F=8.0, K=4, F_share=2.0 (stale K).
    Worker 0: output=5.0, surplus = 5 - 1 - 0 - 2 = 2.0 (keep)
    Workers 1/2/3: output=2.5, surplus = 2.5 - 1 - 0 - 2 = -0.5 (fire)
    Expected fire_indices = [1, 2, 3] (3 workers only).

    np.where returns column-order (ascending-index) results so the assertion is
    deterministic regardless of mask evaluation order (MIN-2 note).

    If F_share were recomputed after each firing (recursive cascade):
      after firing worker 1: K=3, F_share=2.67; worker 0 surplus = 5-1-0-2.67 = 1.33 (still keep)
      after firing worker 2: K=2, F_share=4.0; worker 0 surplus = 5-1-0-4 = 0 (still keep)
      after firing worker 3: K=1, F_share=8.0; worker 0 surplus = 5-1-0-8 = -4 (fire!)
    Recursive cascade would fire all 4; stale-K stops at 3.
    """
    params = _default_params(T_review=10.0, firing_threshold=0.0, F=8.0)
    wf = _make_fake_workforce(
        theta=[1.0, 1.0, 1.0, 1.0],
        wage=[1.0, 1.0, 1.0, 1.0],
    )
    opw = np.full((20, 4), np.nan)
    opw[0:10, 0] = 5.0   # surplus = 5 - 1 - 0 - (8/4=2) = 2 (keep)
    opw[0:10, 1] = 2.5   # surplus = 2.5 - 1 - 0 - 2 = -0.5 (fire)
    opw[0:10, 2] = 2.5   # fire
    opw[0:10, 3] = 2.5   # fire
    acpw = np.full_like(opw, 0.0)
    acpw[np.isnan(opw)] = np.nan

    fire_indices, _ = firing_review(
        wf, t=10, output_per_worker=opw, aug_cost_per_worker=acpw, params=params
    )
    assert np.array_equal(fire_indices, np.array([1, 2, 3])), (
        f"Expected stale-K firing [1, 2, 3] (3 workers); got {fire_indices}. "
        f"Per-tick recursive cascade would have fired all 4."
    )


# ---------------------------------------------------------------------------
# T-11: Hire/fire oscillation guard (enable_hiring + F-share)
# ---------------------------------------------------------------------------


def test_hiring_oscillation_bounded():
    """Hiring + F-share oscillation is bounded by the tight theoretical maximum.

    With enable_hiring=True, firing can trigger same-period hire-back (Step 0.5).
    Total hires must not exceed K0 * num_review_ticks where num_review_ticks
    = (T-1) // T_review (t=0 excluded by firing_review's short-circuit).

    For T=30, T_review=5: actual firing ticks t ∈ {5,10,15,20,25} = 5 ticks.
    bound = 20 * 5 = 100.

    Also detects sustained full-fire flip-flop: no 3+ consecutive review ticks
    with n_review_fired == K0 (structural oscillation detector).
    """
    # Pin w=1.0 so p*tpw - w = 0.22*5 - 1.0 = 0.1 > 0 → hiring profitable (new default w=2.0 breaks this).
    # Pin max_hire_period=0 (drain full backlog in one period) to match pre-99ddaea behavior.
    # With max_hire_period=3 (new default), slow backlog drain can leave K low enough that all
    # workers are fired at the next review tick before recovery, causing K=0 indefinitely.
    params = FirmParams(
        N=100,
        T=30,
        tasks_per_worker=5,
        T_review=5.0,
        firing_threshold=0.0,
        p=0.22,
        w=1.0,
        F=5.0,
        seed=0,
        sigma_theta=0.2,
        sigma_w=0.05,
        enable_hiring=True,
        hire_delay_periods=1,  # pin explicitly; delay semantics: hires drain at t+1
        max_hire_period=0,     # drain full backlog in one period (pre-99ddaea default)
    )
    K0 = params.N // params.tasks_per_worker  # 20
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    total_hired = int(df["n_hired"].sum())

    # t=0 excluded by firing_review's short-circuit (review.py:95)
    # Actual review ticks with firing = (T-1)//T_review, not T//T_review.
    num_reviews = (params.T - 1) // int(params.T_review)  # 5 (not 6)
    bound = K0 * num_reviews  # 20 * 5 = 100
    assert total_hired <= bound, (
        f"Hiring exceeds theoretical max: total_hired={total_hired} > bound={bound}. "
        f"T={params.T}, T_review={params.T_review}, num_reviews={num_reviews}, K0={K0}"
    )

    # Structural flip-flop detector: no 3+ consecutive review ticks with full-workforce fire.
    # Review rows: t % T_review == 0 (and t > 0 since t=0 is excluded by short-circuit).
    review_rows = df[(df["t"] % int(params.T_review) == 0) & (df["t"] > 0)]
    full_fire_ticks = (review_rows["n_review_fired"].values == K0).astype(int)
    max_run = cur = 0
    for v in full_fire_ticks:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    assert max_run <= 2, (
        f"Sustained full-workforce fire-then-hire-then-fire cycle detected: "
        f"{max_run} consecutive review ticks with n_review_fired == K0. "
        f"R-04 oscillation may be unbounded."
    )

    # Zero-workforce limbo check: no K_active == 0 for >1 consecutive period.
    zero_k_periods = (df["K_active"] == 0).astype(int).values
    max_zero_run = cur = 0
    for v in zero_k_periods:
        cur = cur + 1 if v else 0
        max_zero_run = max(max_zero_run, cur)
    # With hire_delay_periods=1, K_active==0 is allowed for 1 period (the fire
    # period itself). Hires drain at t+1, so K_active recovers by the next period.
    assert max_zero_run <= 1, (
        f"Workforce stuck at K_active=0 for {max_zero_run} consecutive periods. "
        f"enable_hiring with hire_delay_periods=1 should restore headcount within 1 period delay."
    )


# ---------------------------------------------------------------------------
# T-12: Regression neutrality — aug_cost_per_worker dormant under T_review=inf
# ---------------------------------------------------------------------------


def test_aug_cost_neutrality_when_dormant():
    """With T_review=inf, aug_cost_per_worker is populated but never read by firing_review.

    pi/Y/C columns must be byte-equal to the pre-change baseline captured in
    check8-pretip.json (before any Stage-7 code edits). This is the dormancy-guarantee
    regression fence: two cost_vec calls per period (D-07) but Step 7 is unchanged,
    so task_costs, C, and pi are bit-for-bit identical.

    Falls back to run_tier_a() check8 pass assertion if pretip.json is absent.
    """
    import json
    import pathlib

    pretip_path = pathlib.Path(".workflow_artifacts/adaptive-firing-surplus/check8-pretip.json")

    from firm_ai_abm.validate import run_tier_a

    result = run_tier_a()
    assert result["check8"]["passed"], (
        f"check8_stage3_neutrality FAILED post-change (dormancy guarantee broken): "
        f"{result['check8']['details']}"
    )

    if pretip_path.exists():
        # pretip structure: {"passed": bool, "details": {"per_strategy": {...}}}
        # written by: json.dumps(results.get('check8', {}), default=str)
        pretip = json.loads(pretip_path.read_text())
        assert pretip.get("passed", False), (
            "check8 was already failing before Stage-7 changes (pretip baseline shows failed)"
        )
        pre_details = pretip.get("details", {})
        post_details = result["check8"].get("details", {})
        for strat_name in post_details.get("per_strategy", {}):
            if strat_name in pre_details.get("per_strategy", {}):
                pre_passed = pre_details["per_strategy"][strat_name]["passed"]
                post_passed = post_details["per_strategy"][strat_name]["passed"]
                assert post_passed == pre_passed, (
                    f"check8 strategy '{strat_name}' passed={post_passed} post-change "
                    f"but was passed={pre_passed} in pretip baseline"
                )
    else:
        # Pretip baseline absent (T-01 pre-step was skipped); fall back to pass-only assertion
        import warnings
        warnings.warn(
            "check8-pretip.json absent — falling back to pass-only assertion. "
            "Re-run T-01 pre-step to capture the byte-identity baseline.",
            stacklevel=2,
        )
