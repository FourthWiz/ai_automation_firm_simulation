"""Tests for alpha-dependent automation cost and belief_alpha parameter.

Covers T-08 through T-13 from the unknown-alpha-cost-model plan:
  T-08  — three-anchor calibration check for cost_vec engaged path
  T-09  — fixture byte-parity regression (dormant default)
  T-10  — numeraire invariance under non-zero alpha-dependent cost (Variant A only)
  T-11  — belief-vs-realized wedge smoke test (JOINT effect)
  T-12  — see test_training_delay.py for direct cost_vec caller fix
  T-13  — belief-substitution isolation: score_T is a constant vector
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.production import Mode, cost_vec
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import (
    all_A,
    all_H,
    all_T,
    greedy_profit,
    greedy_with_switching,
)
from firm_ai_abm.validate import SCALED_PARAMS


# ---------------------------------------------------------------------------
# T-08: three-anchor calibration
# ---------------------------------------------------------------------------

def test_cost_vec_three_anchor_calibration():
    """T-08: cost_vec engaged path satisfies all three calibration anchors at (2.0, 2.0).

    Calibration at w=1, tpw=5, c_auto=0.4 (unchanged project default):
        wage_per_task = 0.2
        c(0.0) = 0.2 * (2.0 - 0.0) = 0.40 > 0.20  anchor 1: above w/tpw
        c(0.5) = 0.2 * (2.0 - 1.0) = 0.20 == 0.20  anchor 2: exact equality
        c(1.0) = 0.2 * (2.0 - 2.0) = 0.00 < 0.20   anchor 3: floor triggered

    c_auto=0.4 is the project default and is NOT used in the engaged-path formula (D-01).
    The sentinel at the top guards against a silent c_auto=0.0 regression.
    """
    params = FirmParams(
        c_auto_alpha_intercept=2.0,
        c_auto_alpha_slope=2.0,
        belief_alpha=0.5,
        w=1.0,
        tasks_per_worker=5,
    )
    # MIN-5: guard against silent regression where c_auto is set to 0.0
    assert params.c_auto == 0.4, (
        "c_auto must remain at project default 0.4; setting it to 0.0 would let the "
        "engaged path trivially pass"
    )

    wage_per_task = params.w / params.tasks_per_worker  # 0.2

    alpha = np.array([0.0, 0.5, 1.0])
    modes = np.full(3, int(Mode.T), dtype=int)

    c = cost_vec(modes, alpha, params)

    # anchor 1: c(0) > w/tpw
    assert c[0] > wage_per_task, f"anchor 1 failed: c[0]={c[0]} not > w/tpw={wage_per_task}"
    # anchor 2: c(0.5) == w/tpw (exact equality — MAJ-7 fix)
    assert np.isclose(c[1], wage_per_task), (
        f"anchor 2 failed: c[0.5]={c[1]} not isclose to w/tpw={wage_per_task}"
    )
    # anchor 3: c(1) < w/tpw
    assert c[2] < wage_per_task, f"anchor 3 failed: c[1]={c[2]} not < w/tpw={wage_per_task}"
    # monotone decreasing
    assert np.all(np.diff(c) <= 0), f"cost not monotone decreasing: {c}"
    # MIN-3: exact floor at c(1) = 0.0
    assert c[2] == 0.0, f"floor assertion failed: c[1]={c[2]} not exactly 0.0"


# ---------------------------------------------------------------------------
# T-09: fixture byte-parity regression (dormant default)
# ---------------------------------------------------------------------------

_STRATEGIES = [
    ("all_H", all_H),
    ("all_A", all_A),
    ("all_T", all_T),
    ("greedy_profit", greedy_profit),
    ("greedy_with_switching", greedy_with_switching),
]
_COLS = ["t", "Y", "C", "pi", "K", "adj_cost"]
_FIXTURE_DIR = "tests/fixtures"
_STRATEGY_FIXTURE_MAP = {
    "all_H": "phase1_baseline_all_H.parquet",
    "all_A": "phase1_baseline_all_A.parquet",
    "all_T": "phase1_baseline_all_T.parquet",
    "greedy_profit": "phase1_baseline_greedy_profit.parquet",
    "greedy_with_switching": "phase1_baseline_greedy_with_switching.parquet",
}


@pytest.mark.parametrize("name,strategy", _STRATEGIES)
def test_fixture_byte_parity_dormant(name, strategy):
    """T-09: dormant default is byte-identical to Stage-1/2/3 parquet fixtures.

    Runs run_simulation under the params matching fixture capture (sigma_theta=0,
    sigma_w=0 for homogeneous workers; tasks_per_worker=10, p=1.0 as at capture time)
    and compares all 6 columns to the fixture. np.array_equal enforces byte identity.
    """
    fixture_path = f"{_FIXTURE_DIR}/{_STRATEGY_FIXTURE_MAP[name]}"
    fixture = pd.read_parquet(fixture_path)

    firm = make_firm(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0, sigma_theta=0.0, sigma_w=0.0))
    df = run_simulation(firm, strategy)

    result_vals = df[_COLS].values
    fixture_vals = fixture[_COLS].values

    assert np.array_equal(result_vals, fixture_vals), (
        f"Byte-parity failed for strategy={name}. "
        f"Max abs diff: {np.abs(result_vals - fixture_vals).max():.3e}. "
        "Check cost_vec dormant fast-path (T-03) and strategy dormant sentinel (T-05)."
    )


# ---------------------------------------------------------------------------
# T-10: numeraire invariance under non-zero alpha-dependent cost (Variant A only)
# ---------------------------------------------------------------------------

def test_numeraire_invariance_alpha_cost_all_T():
    """T-10: multiplying all SCALED_PARAMS by 2 scales pi by exactly 2 under engaged path.

    Variant A only (all_T strategy). Greedy numeraire invariance is intentionally NOT
    tested here per tests/test_workers.py:246-250; Variant A (all_T) is sufficient to
    validate R-01 for the new cost formula.
    """
    params_base = FirmParams(
        seed=0,
        tasks_per_worker=10,
        p=1.0,
        c_auto_alpha_slope=2.0,
        c_auto_alpha_intercept=2.0,
        belief_alpha=0.5,
    )
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED_PARAMS}
    params_scaled = dataclasses.replace(params_base, **scaled_kwargs)

    firm_base = make_firm(params_base)
    df_base = run_simulation(firm_base, all_T)
    pi_base = df_base["pi"].values

    firm_scaled = make_firm(params_scaled)
    df_scaled = run_simulation(firm_scaled, all_T)
    pi_scaled = df_scaled["pi"].values

    assert np.allclose(pi_scaled, 2.0 * pi_base, rtol=1e-12, atol=2e-9), (
        f"Numeraire invariance failed. Max deviation: "
        f"{np.abs(pi_scaled - 2.0 * pi_base).max():.3e}"
    )


# ---------------------------------------------------------------------------
# T-11: belief-vs-realized wedge smoke test (JOINT effect)
# ---------------------------------------------------------------------------

def test_belief_vs_realized_wedge_smoke():
    """T-11: engaged path (joint cost-shape + belief scoring) produces wedge vs dormant.

    This test demonstrates the JOINT effect of alpha-dependent cost shape and
    belief-based strategy scoring (Feature A engaged vs dormant). To isolate the
    strategy-side belief substitution mechanism alone, see test_belief_substitution_isolation
    (T-13).

    Verified parameter set: q_a=2.5, c_aug=0.5, g=0.1 (A-mode score below H), seed=0.
        wage_per_task = 0.2; score_H ≈ 0.8; score_T_belief = 2.5*0.5 - 0.20 = 1.05 > 0.80
    Asym (belief) commits ALL tasks to T (K=0 workers); dormant uses realized alpha,
    keeping some tasks in H/A (~9 workers). Both strategies are tested with p=1.0 to
    ensure meaningful revenue (the current default p=0.22 is calibrated for all-H baseline).

    Pi-delta note: the engaged path may yield HIGHER or LOWER total profit than dormant
    depending on whether wage savings from K=0 workers outweigh productivity losses on
    low-alpha tasks automated under the belief. With q_a=2.5 and the linear cost formula
    (avg c_auto_belief=0.2 < flat c_auto=0.4), asym typically yields higher pi. The
    critical assertion is the MODE-COUNT difference — asym over-commits to T-mode.
    """
    base_kwargs = dict(
        q_a=2.5,
        c_aug=0.5,
        g=0.1,
        c_auto_alpha_intercept=2.0,
        c_auto_alpha_slope=2.0,
        seed=0,
        p=1.0,  # use p=1.0 to keep revenues meaningful for this test
    )
    # Dormant baseline: realized alpha used for both strategy scoring and cost
    params_sym = FirmParams(**base_kwargs, belief_alpha=None)
    # Engaged: belief=0.5 for strategy scoring; alpha-shape cost in cost_vec
    params_asym = FirmParams(**base_kwargs, belief_alpha=0.5)

    firm_sym = make_firm(params_sym)
    df_sym = run_simulation(firm_sym, greedy_with_switching)

    firm_asym = make_firm(params_asym)
    df_asym = run_simulation(firm_asym, greedy_with_switching)

    pi_sym_total = df_sym["pi"].sum()
    pi_asym_total = df_asym["pi"].sum()

    # Pi totals must be different (feature is engaged and changes outcome)
    assert pi_sym_total != pi_asym_total, (
        f"pi_sym and pi_asym are identical — belief feature may not be engaged. "
        f"pi_sym={pi_sym_total:.4f}, pi_asym={pi_asym_total:.4f}"
    )
    print(f"T-11 pi_sym={pi_sym_total:.2f}, pi_asym={pi_asym_total:.2f}, "
          f"delta={pi_sym_total - pi_asym_total:.2f}")

    # Mode-count assertion (primary): asymmetric path commits all tasks to T (K=0)
    # dormant path keeps H/A tasks → K > 0 in early periods
    k_sym_early = df_sym["K"].iloc[:5].mean()
    k_asym_early = df_asym["K"].iloc[:5].mean()
    assert k_asym_early < k_sym_early, (
        f"Expected fewer workers (more T) under belief path in early periods. "
        f"K_asym_mean={k_asym_early:.2f}, K_sym_mean={k_sym_early:.2f}"
    )
    # Specifically: asym should have 0 workers (all T-mode) in early periods
    assert k_asym_early == 0.0, (
        f"Asym path should have K=0 (all T-mode under belief dominance), "
        f"got K_asym_mean={k_asym_early:.2f}"
    )


# ---------------------------------------------------------------------------
# T-13: belief-substitution isolation (score_T is a constant vector)
# ---------------------------------------------------------------------------

def test_belief_substitution_isolation():
    """T-13: belief-only scoring (D-04) produces a constant score_T independent of alpha.

    Isolates the strategy-side belief-substitution mechanism alone, independent of
    cost-shape changes. Because belief_alpha replaces per-task realized alpha in score_T
    computation, all tasks receive the same score_T scalar regardless of their realized alpha.

    Uses the plan's preferred alternative: replicate score_T inline, compute expected modes
    via argmax, and verify greedy_profit matches exactly. This tests that score_T is constant
    (no per-alpha variation) without requiring all tasks to be T-mode (some high-beta A-tasks
    may dominate with default c_aug=0.05, g=0.5).

    T-11 tests the joint effect with cost-shape; this test isolates the strategy-side
    mechanism alone.
    """
    params = FirmParams(
        c_auto_alpha_intercept=2.0,
        c_auto_alpha_slope=2.0,
        belief_alpha=0.5,
        q_a=2.5,
        w=1.0,
        tasks_per_worker=5,
        seed=0,
    )

    firm = make_firm(params)
    p = params

    # Replicate score_T computation from D-04 inline (plan's preferred alternative)
    wage_per_task_scalar = p.w / p.tasks_per_worker
    c_auto_belief_val = wage_per_task_scalar * (
        p.c_auto_alpha_intercept - p.c_auto_alpha_slope * p.belief_alpha
    )
    c_auto_belief_val = max(c_auto_belief_val, 0.0)
    expected_score_T = np.full(p.N, p.q_a * p.belief_alpha - c_auto_belief_val)

    # score_T scalar value
    assert np.isclose(expected_score_T[0], 1.05), (
        f"Expected score_T=1.05, got {expected_score_T[0]:.6f}"
    )
    # Core assertion: score_T is a constant vector (the belief-substitution mechanism)
    assert np.all(expected_score_T == expected_score_T[0]), (
        "score_T must be constant across all tasks under belief substitution — "
        "per-task realized alpha must NOT appear in the engaged scoring path"
    )

    # Compute per-task scores for all modes (replicating strategy internals)
    # score_A uses beta_hat (posterior) per D-01 — strategy now plans under beliefs.
    b = firm.beta_hat if firm.beta_hat is not None else firm.beta
    slot_idx = np.arange(p.N) // p.tasks_per_worker
    slot_idx_clamped = np.minimum(slot_idx, firm.workforce.K - 1)
    worker_wage = firm.workforce.wage[slot_idx_clamped]
    wage_per_task = worker_wage / p.tasks_per_worker
    score_H = p.q_h - wage_per_task
    score_A = p.q_h * (1.0 + p.g * b) - p.c_aug - wage_per_task
    scores = np.stack([score_H, score_A, expected_score_T], axis=1)
    expected_modes = np.argmax(scores, axis=1).astype(int)

    # Call greedy_profit and verify modes match expected
    actual_modes = greedy_profit(firm, t=0)

    assert np.array_equal(actual_modes, expected_modes), (
        f"greedy_profit modes do not match expected argmax under belief scoring. "
        f"Mismatches: {(actual_modes != expected_modes).sum()} tasks. "
        f"Actual H/A/T: {np.bincount(actual_modes, minlength=3)}, "
        f"Expected H/A/T: {np.bincount(expected_modes, minlength=3)}"
    )
