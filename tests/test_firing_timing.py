"""Firing-timing semantics: T-06 and T-07 from firing-timing-and-horizon-fix plan.

Four assertions covering:
  - adj_cost zeros c_fire under finite T_review (H→T transition)
  - adj_cost unchanged under T_review=inf (H→T still charges c_fire)
  - wage_bill covers all K workers between reviews (not just assigned workers)
  - forward_simulate_action_path / run_horizon wage-bill parity under finite T_review

Fixture: N=20, tpw=5, K=4, c_fire=2.0, c_hire=0.5, c_train=0.10, T_review=5.
"""
import copy
import math

import numpy as np
import pytest
from dataclasses import replace

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.adjustment import adj_cost
from firm_ai_abm.production import Mode, compute_K
from firm_ai_abm.simulate import run_horizon
from firm_ai_abm.forward_sim import forward_simulate_action_path, Action
from firm_ai_abm.strategy import all_T


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------

def _make_params(**overrides) -> FirmParams:
    """Return FirmParams for the standard N=20, tpw=5, K=4 firing-timing fixture."""
    defaults = dict(
        seed=42,
        N=20,
        tasks_per_worker=5,
        c_fire=2.0,
        c_hire=0.5,
        c_train=0.10,
        T_review=5.0,
        sigma_theta=0.0,
        sigma_w=0.0,
        p=1.0,
        w=1.0,
    )
    defaults.update(overrides)
    return FirmParams(**defaults)


# ---------------------------------------------------------------------------
# T-06a: adj_cost zeros c_fire under finite T_review for H→T transition
#
# T-07 parametrize row: all-T → all-T with K_prev=K_new=0 is also zero
# regardless of T_review — both branches converge to 0.0 here.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prev_modes_mode,new_modes_mode,N,expected_cost", [
    # Main H→T case (Fix 2a): fire_cost gated to 0 under finite T_review
    (Mode.H, Mode.T, 20, 0.0),
    # T-07: all-T → all-T with N=4 < tpw=5 → K_prev=K_new=0 → cost=0 always
    (Mode.T, Mode.T, 4, 0.0),
])
def test_adj_cost_zero_HT_under_finite_T_review(prev_modes_mode, new_modes_mode, N, expected_cost):
    """adj_cost returns 0.0 for H→T (and T→T) transitions under finite T_review.

    Under finite T_review, c_fire is gated to 0 in adj_cost (Fix 2a / T-02):
    periodic-review firings are handled by the kernel Step 0 gate, not adj_cost.
    For T→T: K_prev = compute_K([T]*N) = 0, K_new = 0 → delta = 0 independently
    of the T_review conditional (T-07 verification).
    """
    params = _make_params(N=N, T_review=5.0)
    firm = make_firm(params)

    prev = np.full(N, int(prev_modes_mode), dtype=int)
    new = np.full(N, int(new_modes_mode), dtype=int)

    cost = adj_cost(prev, new, params, workforce=firm.workforce)
    assert cost == pytest.approx(expected_cost, abs=1e-9)


# ---------------------------------------------------------------------------
# T-06b: adj_cost unchanged under T_review=inf (H→T still charges c_fire)
# ---------------------------------------------------------------------------

def test_adj_cost_unchanged_HT_under_inf_T_review():
    """adj_cost charges c_fire * K_prev for H→T transition under T_review=inf.

    With N=20, tpw=5: K_prev = compute_K([H]*20) = ceil(20/5) = 4.
    Expected fire_cost = c_fire * K_prev = 2.0 * 4 = 8.0.
    """
    params = _make_params(T_review=math.inf)
    firm = make_firm(params)

    prev = np.full(20, int(Mode.H), dtype=int)
    new = np.full(20, int(Mode.T), dtype=int)

    K_prev = compute_K(prev, params)  # = 4
    expected_fire_cost = params.c_fire * K_prev  # 2.0 * 4 = 8.0

    cost = adj_cost(prev, new, params, workforce=firm.workforce)
    assert cost == pytest.approx(expected_fire_cost, abs=1e-9), (
        f"Expected fire_cost={expected_fire_cost} under T_review=inf, got {cost}"
    )


# ---------------------------------------------------------------------------
# T-06c: wage_bill covers all K workers between reviews
# ---------------------------------------------------------------------------

def test_wage_bill_all_K_between_reviews():
    """run_horizon for 6 periods with all_T strategy and T_review=5.

    Between t=0..4 (pre-review): wage_bill = K * w = 4 * 1.0 = 4.0 per row.
    At t=5 (review period): Step 0 fires all K=4 workers (output=0 → negative
    surplus). After firing, workforce.K drops to 0. The K>0 guard in the finite
    T_review branch returns 0.0. wage_bill at t=5 == 0.0.

    This combined effect (firing + K>0 guard) means both a regression that
    stops firings AND one that removes the K>0 guard would surface as
    wage_bill > 0 at t=5.
    """
    params = _make_params(T_review=5.0)
    firm = make_firm(params)

    firm_copy = copy.deepcopy(firm)
    df = run_horizon(firm_copy, all_T, 6)

    K = 4  # N=20, tpw=5 → K=4
    w = 1.0

    # t=0..4: wage_bill = K * w
    pre_review = df.loc[df["t"].isin(range(0, 5)), "wage_bill"]
    expected_wage = K * w
    assert np.allclose(pre_review.values, expected_wage, atol=1e-9), (
        f"Expected wage_bill={expected_wage} for t=0..4, got:\n{pre_review.to_string()}"
    )

    # t=5: all K workers fired at Step 0 → wage_bill=0
    review_wage = float(df.loc[df["t"] == 5, "wage_bill"].iloc[0])
    assert review_wage == pytest.approx(0.0, abs=1e-9), (
        f"Expected wage_bill=0.0 at t=5 after full firing, got {review_wage}"
    )


# ---------------------------------------------------------------------------
# T-06d: forward_simulate_action_path / run_horizon wage-bill parity
#
# Verifies kernel-symmetry principle (D-03): planner's Step 6 wage bill uses
# the same conditional as the kernel's Step 8, so both charge wages for all K
# workers under finite T_review.
#
# Setup: firm starts in all-T modes (workers employed but unassigned), horizon=4
# (t=0..3, before the review at t=5). alpha_hat set equal to alpha so planner
# uses truth — removes productivity discrepancy as a confound.
# ---------------------------------------------------------------------------

def test_forward_sim_planner_kernel_wage_parity():
    """forward_simulate_action_path and run_horizon produce equal cumulative pi.

    Parity scenario: all-T modes, K=4 employed workers, T_review=5, horizon=4
    (before the first review at t=5). No adj_cost (modes unchanged). Both
    kernel and planner must charge K * w per step — the fix ensures planner
    uses `f_workforce.wage.sum()` (not assigned-only) when T_review is finite.

    Pre-condition: alpha_hat == alpha (posteriors set to truth to eliminate
    productivity discrepancy as a confound). Tolerance: 1e-6.
    """
    params = _make_params(T_review=5.0)
    firm = make_firm(params)

    # Pre-set modes to all-T (workers employed but unassigned)
    firm.modes = np.full(params.N, int(Mode.T), dtype=int)

    # Set posteriors == truth to eliminate planner/kernel alpha discrepancy
    firm.alpha_hat = firm.alpha.copy()
    firm.beta_hat = firm.beta.copy()

    # No-op strategy: keep modes at all-T (no mode changes, so adj_cost=0)
    def stay_T(firm_inner, t):
        return np.full(firm_inner.params.N, int(Mode.T), dtype=int)

    horizon = 4  # t=0..3, before review at t=5

    # Kernel path
    firm_kernel = copy.deepcopy(firm)
    df = run_horizon(firm_kernel, stay_T, horizon)
    kernel_pi = float(df["pi"].sum())

    # Planner path: n_fire=0, n_aug=0, n_hire=0 (hold all-T)
    firm_planner = copy.deepcopy(firm)
    path = [Action(n_fire=0, n_aug=0, n_hire=0)] * horizon
    planner_pi = forward_simulate_action_path(firm_planner, t=0, action_path=path, horizon=horizon)

    assert planner_pi == pytest.approx(kernel_pi, abs=1e-6), (
        f"Planner/kernel wage-bill parity failed: "
        f"planner={planner_pi:.6f}, kernel={kernel_pi:.6f}, "
        f"diff={abs(planner_pi - kernel_pi):.2e}"
    )
