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


# ---------------------------------------------------------------------------
# T-08: fire negative-surplus, keep positive-surplus; c_train_lost accounting
# ---------------------------------------------------------------------------


def test_T08_firing_review_fires_negative_surplus():
    """firing_review fires only negative-surplus workers (surplus < 0)."""
    params = _default_params(T_review=10.0, firing_threshold=0.0)

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

    fire_indices, c_train_lost = firing_review(wf, t=10, output_per_worker=opw, params=params)

    assert np.array_equal(fire_indices, np.array([2])), (
        f"Expected [2], got {fire_indices}"
    )
    assert c_train_lost == 0.0, f"Expected 0.0 (untrained), got {c_train_lost}"


def test_T08_firing_review_c_train_lost_when_trained():
    """c_train_lost reflects trained-worker count when a_trained=True for fired worker."""
    params = _default_params(T_review=10.0, firing_threshold=0.0, c_train=5.0)

    wf = _make_fake_workforce(
        theta=[1.5, 1.0, 0.5],
        wage=[1.0, 1.0, 1.0],
        a_trained=[False, False, True],  # worker 2 is trained
    )
    opw = np.full((20, 3), np.nan)
    opw[0:10, 0] = 1.5
    opw[0:10, 1] = 1.0
    opw[0:10, 2] = 0.4   # fired

    fire_indices, c_train_lost = firing_review(wf, t=10, output_per_worker=opw, params=params)

    assert np.array_equal(fire_indices, np.array([2]))
    # 1 trained worker fired × c_train=5.0
    assert np.isclose(c_train_lost, 5.0), f"Expected 5.0, got {c_train_lost}"


def test_T08_firing_review_T_review_inf_returns_empty():
    """T_review=math.inf → no review, returns empty regardless of surplus."""
    params = _default_params(T_review=math.inf)
    wf = _make_fake_workforce(theta=[0.1], wage=[10.0])
    opw = np.full((20, 1), -999.0)  # absurdly negative

    fire_indices, c_train_lost = firing_review(wf, t=10, output_per_worker=opw, params=params)
    assert fire_indices.size == 0
    assert c_train_lost == 0.0


# ---------------------------------------------------------------------------
# T-09: surplus calculation correctness
# ---------------------------------------------------------------------------


def test_T09_surplus_calculation_correctness():
    """Surplus = p * mean_output − wage (test uses p=1.0 so numerically equivalent to old formula, but verifies price-scaled formula is in place); firing depends on threshold."""
    params_base = _default_params(T_review=10.0)
    wf = _make_fake_workforce(theta=[1.0, 1.0], wage=[2.0, 0.5])

    opw = np.full((20, 2), np.nan)
    opw[0:10, 0] = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0])  # mean=1.8
    opw[0:10, 1] = 0.6   # mean=0.6

    # surplus[0] = 1.8 - 2.0 = -0.2; surplus[1] = 0.6 - 0.5 = 0.1
    # threshold=0.0: fire worker 0 (surplus=-0.2 < 0)
    params_0 = replace(params_base, firing_threshold=0.0)
    fire_0, _ = firing_review(wf, t=10, output_per_worker=opw, params=params_0)
    assert np.array_equal(fire_0, np.array([0])), f"Expected [0], got {fire_0}"

    # threshold=-0.5: both surplus above -0.5 → no firing
    params_neg = replace(params_base, firing_threshold=-0.5)
    fire_neg, _ = firing_review(wf, t=10, output_per_worker=opw, params=params_neg)
    assert fire_neg.size == 0, f"Expected empty, got {fire_neg}"

    # threshold=0.05: worker 0 still fired (surplus=-0.2 < 0.05); worker 1 not (surplus=0.1 >= 0.05)
    params_pos = replace(params_base, firing_threshold=0.05)
    fire_pos, _ = firing_review(wf, t=10, output_per_worker=opw, params=params_pos)
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
    fire_indices = np.array([1, 3], dtype=int)

    new_wf, new_opw = apply_firings(firm, fire_indices, t=10, output_per_worker=opw)

    # Stage 5 D-03: K DROPS (was 5, fired 2 → K=3)
    assert new_wf.K == 5 - len(fire_indices), (
        f"Expected K={5 - len(fire_indices)}, got {new_wf.K}"
    )

    # Survivors sorted by descending tenure (D-05)
    assert np.all(np.diff(new_wf.tenure) <= 0), (
        f"Expected non-increasing tenure, got {new_wf.tenure}"
    )

    # Trailing inactive columns [new_wf.K : K_max] must be all-NaN
    assert np.all(np.isnan(new_opw[:, new_wf.K:])), (
        "Trailing columns after K_new must be all-NaN"
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
    fire_indices = np.array([1, 3], dtype=int)

    new_wf, new_opw = apply_firings(firm, fire_indices, t=10, output_per_worker=opw)
    assert new_wf.K == 3

    # replace_to_target restores K to 5
    firm.workforce = new_wf
    restored_wf, restored_opw = replace_to_target(firm, K_target=5, t=10, output_per_worker=new_opw)
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
    ret_wf, ret_opw = apply_firings(firm2, empty_fire, t=5, output_per_worker=opw)
    assert ret_wf is firm2.workforce, "Empty fire_indices must return same workforce (identity)"
    assert ret_opw is opw, "Empty fire_indices must return same opw (identity)"


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
    """
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), np.nan, dtype=np.float64)
    # Fill first 10 rows with sub-threshold output so all workers get fired
    opw[0:10, :K_max] = 0.0  # output=0, wage=1.0, surplus=-1 < 10 threshold → all fired

    fire_indices, _ = firing_review(firm.workforce, t=10, output_per_worker=opw, params=params)
    assert fire_indices.size == K_max, f"Expected all {K_max} workers fired, got {fire_indices.size}"

    # Stage 5: apply_firings (K drops to 0) then replace_to_target (restore K)
    new_wf, new_opw = apply_firings(firm, fire_indices, t=10, output_per_worker=opw)
    assert new_wf.K == 0, f"After firing all, K should be 0; got {new_wf.K}"
    firm.workforce = new_wf  # update in-place for replace_to_target
    restored_wf, _ = replace_to_target(firm, K_target=K_max, t=10, output_per_worker=new_opw)

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
    """T_review=inf and T_review=999 (never fires in T=60 window) produce identical history."""
    parity_cols = ["t", "Y", "C", "pi", "K", "adj_cost"]
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
    """Sub-test 1: all-NaN worker column does not produce RuntimeWarning and is not fired."""
    params = _default_params(T_review=10.0, firing_threshold=0.0)

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

    # Assert no RuntimeWarning from the per-column mask approach (CRIT-1 pin)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fire_indices, _ = firing_review(wf, t=10, output_per_worker=opw, params=params)
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
    """Sub-test 3 (Stage 5 NEW): after bare apply_firings (no replace), trailing cols are NaN."""
    params = _default_params(T=20, T_review=10.0, firing_threshold=10.0)
    firm = make_firm(params)

    K_max = params.N // params.tasks_per_worker
    opw = np.full((params.T, K_max), 1.0, dtype=np.float64)  # non-NaN to show the wipe
    fire_indices = np.array([0, 1, 2], dtype=int)  # fire first 3 workers

    new_wf, new_opw = apply_firings(firm, fire_indices, t=5, output_per_worker=opw)

    assert new_wf.K == firm.workforce.K - 3

    # Trailing inactive columns [new_wf.K : K_max] must be all-NaN
    assert np.all(np.isnan(new_opw[:, new_wf.K:])), (
        f"Expected columns [{new_wf.K}:{K_max}] to be all-NaN after apply_firings; "
        f"got non-NaN values"
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
    """Smoke test: enable_hiring=True restores firm.workforce.K to K0 after firings.

    Uses all_H strategy (same reasoning as test_low_price_triggers_firings):
    greedy_profit picks augmented mode at p=0.22, giving surplus > 0 → no firings.
    Recipe: same as test_low_price_triggers_firings plus enable_hiring=True.
    Assertions:
      - n_hired sum > 0 (hires occurred)
      - firm.workforce.K == firm.K0 after run (headcount restored)
      - pi shows no runaway negative cost (sanity)
    """
    params = FirmParams(
        N=100, T=60, tasks_per_worker=5,
        T_review=10.0, firing_threshold=0.0,
        p=0.22, seed=0,
        sigma_theta=0.2, sigma_w=0.05,
        enable_hiring=True,
    )
    firm = make_firm(params)
    K0 = firm.K0
    df = run_simulation(firm, all_H)

    assert int(df["n_hired"].sum()) > 0, (
        "enable_hiring=True with low p should produce hires; got 0"
    )

    # Headcount is restored: firm.workforce.K should equal K0 at end of run
    assert firm.workforce.K == K0, (
        f"workforce.K={firm.workforce.K} after run with enable_hiring, expected K0={K0}"
    )

    # Sanity: pi shows no runaway negative cost (very loose bound)
    assert df["pi"].min() > -10 * params.c_hire * K0, (
        f"pi min {df['pi'].min():.2f} too negative for hire costs"
    )


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
