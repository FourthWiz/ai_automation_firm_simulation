"""Tests for forward_sim.py — T-01 acceptance criteria (a)–(g).

Criteria:
  (a) forward_simulate_action_path returns a float.
  (b) Function does not mutate the live firm.
  (c) Result uses TRUE per-task theta: firms with heterogeneous theta produce
      a different result than the sigma_theta=0 (theta=1.0) baseline when H/A
      tasks are assigned.
  (d) Union fire cost: when threshold-rule fires workers AND the action plans
      additional fires, the deducted cost is c_fire * |union|, not c_fire * (sum).
  (e) No-action path (all zeros) yields a finite float without error.
  (f) Deterministic: two calls with identical state return the same value.
  (g) Hire-cost timing: c_hire charged at drain period (s + hire_delay_periods),
      NOT at queue time.
"""
import copy
import math

import numpy as np
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.forward_sim import Action, forward_simulate_action_path


def _make_firm(seed=42, N=50, T_review=math.inf, enable_replenish_hiring=False,
               hire_delay_periods=1, sigma_theta=0.0, **extra):
    """Helper: small firm ready for forward simulation."""
    p = FirmParams(
        N=N, T=20, seed=seed,
        T_review=T_review,
        enable_replenish_hiring=enable_replenish_hiring,
        hire_delay_periods=hire_delay_periods,
        sigma_theta=sigma_theta,
        **extra,
    )
    firm = make_firm(p)
    # Seed the kernel seams so replay helper has arrays to work with.
    K = firm.workforce.K
    firm._output_per_worker_so_far = np.zeros((0, K), dtype=np.float64)
    firm._aug_cost_per_worker_so_far = np.zeros((0, K), dtype=np.float64)
    return firm


# ---------------------------------------------------------------------------
# (a) returns float
# ---------------------------------------------------------------------------

def test_returns_float():
    firm = _make_firm()
    path = [Action(n_fire=0, n_aug=0, n_hire=0)] * 3
    result = forward_simulate_action_path(firm, t=0, action_path=path, horizon=3)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# (b) does not mutate firm
# ---------------------------------------------------------------------------

def test_does_not_mutate_firm():
    firm = _make_firm()
    firm_snapshot_modes = firm.modes.copy()
    firm_snapshot_K = firm.workforce.K
    firm_snapshot_theta = firm.workforce.theta.copy()

    path = [Action(n_fire=0, n_aug=2, n_hire=0)] * 3
    forward_simulate_action_path(firm, t=0, action_path=path, horizon=3)

    np.testing.assert_array_equal(firm.modes, firm_snapshot_modes)
    assert firm.workforce.K == firm_snapshot_K
    np.testing.assert_array_equal(firm.workforce.theta, firm_snapshot_theta)


# ---------------------------------------------------------------------------
# (c) uses TRUE per-task theta
# ---------------------------------------------------------------------------

def test_uses_true_theta():
    """Result differs when workers have heterogeneous theta vs theta=1.0.

    If the forward sim were using theta_flat=1.0, both firms would produce the
    same profit trajectory. With true theta, the heterogeneous firm's all-A
    path yields a measurably different cumulative profit.
    """
    # Baseline: sigma_theta=0 → theta=1.0 for all workers
    firm_flat = _make_firm(sigma_theta=0.0, seed=7, N=50)
    # Heterogeneous: sigma_theta=0.3 → workers differ in productivity
    firm_het = _make_firm(sigma_theta=0.3, seed=7, N=50)

    horizon = 5
    # Use all-A path so H/A workers (with non-trivial theta) dominate
    path = [Action(n_fire=0, n_aug=firm_flat.params.N, n_hire=0)] + \
           [Action(n_fire=0, n_aug=0, n_hire=0)] * (horizon - 1)

    pi_flat = forward_simulate_action_path(firm_flat, t=0, action_path=path, horizon=horizon)
    pi_het = forward_simulate_action_path(firm_het, t=0, action_path=path, horizon=horizon)

    assert pi_flat != pi_het, (
        f"Expected different profits when sigma_theta differs, got pi_flat={pi_flat}, "
        f"pi_het={pi_het}. True theta may not be used."
    )


# ---------------------------------------------------------------------------
# (d) union fire cost
# ---------------------------------------------------------------------------

def test_union_fire_cost():
    """When threshold fires K_t workers AND action.n_fire>0 plans additional fires,
    the deducted cost should be c_fire * |union|, not double-count.

    Strategy: use a firm where all workers have low output (all-T mode) so the
    threshold-rule fires the entire workforce at t=T_review. We pass n_fire=0 in
    that step and compare to n_fire>0 (which should union-merge with the same set,
    not add extra cost).

    Under T_review=10 at step t=10, firing_review runs. With all-T mode and
    sigma_theta=0, all workers produce 0 H/A output → all are below the threshold.
    n_fire from the planned set and threshold set overlap completely → union cost
    equals threshold-only cost.
    """
    T_review = 10
    c_fire = 3.0
    firm = _make_firm(T_review=float(T_review), c_fire=c_fire, seed=1, N=50)

    # Force all tasks into T (automated) mode — no H/A workers
    from firm_ai_abm.production import Mode
    firm.modes[:] = int(Mode.T)

    # Populate output history with 10 zero-output rows so firing_review triggers
    K = firm.workforce.K
    firm._output_per_worker_so_far = np.zeros((T_review, K), dtype=np.float64)
    firm._aug_cost_per_worker_so_far = np.zeros((T_review, K), dtype=np.float64)

    # At step t=T_review: threshold fires all K workers (surplus all negative)
    # Case 1: n_fire=0 (no planned additional fires)
    path_no_fire = [Action(n_fire=0, n_aug=0, n_hire=0)]
    pi_no_fire = forward_simulate_action_path(firm, t=T_review, action_path=path_no_fire, horizon=1)

    # Case 2: n_fire=K — plans to fire entire workforce (same workers as threshold)
    path_with_fire = [Action(n_fire=K, n_aug=0, n_hire=0)]
    pi_with_fire = forward_simulate_action_path(firm, t=T_review, action_path=path_with_fire, horizon=1)

    # Union of two identical sets == the same set → both costs should be equal
    # (not 2× c_fire × K)
    assert abs(pi_no_fire - pi_with_fire) < 1e-9, (
        f"Union fire cost violated: pi_no_fire={pi_no_fire:.4f}, "
        f"pi_with_fire={pi_with_fire:.4f}. Double-count suspected."
    )


# ---------------------------------------------------------------------------
# (e) no-action path yields finite float
# ---------------------------------------------------------------------------

def test_no_action_path_finite():
    firm = _make_firm()
    horizon = 10
    path = [Action(n_fire=0, n_aug=0, n_hire=0)] * horizon
    result = forward_simulate_action_path(firm, t=0, action_path=path, horizon=horizon)
    assert math.isfinite(result), f"Expected finite result from no-action path, got {result}"


# ---------------------------------------------------------------------------
# (f) deterministic
# ---------------------------------------------------------------------------

def test_deterministic():
    firm = _make_firm()
    path = [Action(n_fire=0, n_aug=3, n_hire=0)] * 5
    r1 = forward_simulate_action_path(firm, t=0, action_path=path, horizon=5)
    r2 = forward_simulate_action_path(firm, t=0, action_path=path, horizon=5)
    assert r1 == r2, f"Expected deterministic result, got {r1} vs {r2}"


# ---------------------------------------------------------------------------
# (g) hire-cost timing: charged at drain period, not queue time
# ---------------------------------------------------------------------------

def test_hire_cost_timing():
    """c_hire is charged at drain period (s + hire_delay_periods), not at queue time.

    Setup: enable_replenish_hiring=True, hire_delay_periods=2.
    Action path: hire 2 workers at s=0 (n_hire=2), no-ops at s=1 and s=2.

    We measure the cost contribution by comparing:
      - path with hire at s=0 vs path with no hire at all.
    The difference should appear only at s>=2 (the drain step), not at s=0 or s=1.

    We proxy this by comparing partial-horizon profit sums:
      horizon=1 and horizon=2 should NOT yet show the hire cost difference;
      horizon=3 (includes s=2) SHOULD show it.
    """
    hire_delay = 2
    c_hire = 1.5
    firm = _make_firm(
        enable_replenish_hiring=True,
        hire_delay_periods=hire_delay,
        c_hire=c_hire,
        seed=3, N=50,
    )
    firm_copy = copy.deepcopy(firm)

    n_hire = 2

    # No-hire baseline
    path_nohire_h1 = [Action(0, 0, 0)]
    path_nohire_h2 = [Action(0, 0, 0)] * 2
    path_nohire_h3 = [Action(0, 0, 0)] * 3

    # Hire at s=0, no-ops after
    path_hire_h1 = [Action(0, 0, n_hire)]
    path_hire_h2 = [Action(0, 0, n_hire), Action(0, 0, 0)]
    path_hire_h3 = [Action(0, 0, n_hire), Action(0, 0, 0), Action(0, 0, 0)]

    pi_nohire_h1 = forward_simulate_action_path(firm, t=0, action_path=path_nohire_h1, horizon=1)
    pi_hire_h1   = forward_simulate_action_path(firm_copy, t=0, action_path=path_hire_h1, horizon=1)

    firm2 = copy.deepcopy(firm)
    firm3 = copy.deepcopy(firm)
    firm4 = copy.deepcopy(firm)

    pi_nohire_h2 = forward_simulate_action_path(firm2, t=0, action_path=path_nohire_h2, horizon=2)
    pi_hire_h2   = forward_simulate_action_path(firm3, t=0, action_path=path_hire_h2, horizon=2)

    firm5 = copy.deepcopy(firm)
    firm6 = copy.deepcopy(firm)

    pi_nohire_h3 = forward_simulate_action_path(firm5, t=0, action_path=path_nohire_h3, horizon=3)
    pi_hire_h3   = forward_simulate_action_path(firm6, t=0, action_path=path_hire_h3, horizon=3)

    # At horizon=1 and horizon=2: drain period (s=2) not yet reached → no cost difference
    assert abs(pi_hire_h1 - pi_nohire_h1) < 1e-9, (
        f"Hire cost should NOT appear at horizon=1 (delay={hire_delay}), "
        f"but got diff={pi_hire_h1 - pi_nohire_h1:.6f}"
    )
    assert abs(pi_hire_h2 - pi_nohire_h2) < 1e-9, (
        f"Hire cost should NOT appear at horizon=2 (delay={hire_delay}), "
        f"but got diff={pi_hire_h2 - pi_nohire_h2:.6f}"
    )

    # At horizon=3: s=2 is the drain step → cost difference = c_hire * n_hire
    diff_h3 = pi_nohire_h3 - pi_hire_h3
    expected_cost = c_hire * n_hire
    assert abs(diff_h3 - expected_cost) < 1e-9, (
        f"Expected hire cost {expected_cost:.4f} at horizon=3, got diff={diff_h3:.6f}"
    )
