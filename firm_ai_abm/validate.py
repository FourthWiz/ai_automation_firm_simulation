"""Tier A kernel-only validation for Phase 1 single-firm simulation.

Checks implemented here: one (constant baseline), three (monotonicity in q_a),
four (monotonicity in w), five (numeraire invariance). These correspond to
checks 1, 3, 4, 5 from 01_phase1_single_firm.md §10. Checks 2 (greedy
dominance) and 6 (adjustment-cost integration) are Tier B and land in a later
stage.

Wage-charging convention: wages are charged per worker per period in
simulate.run_simulation as ``w * K`` (architecture D-01). cost_vec does NOT
include wages; this is load-bearing for check3, check4, and check5 correctness.

Numeraire scope:
- SCALED_PARAMS (9 fields): the monetary parameters that scale profit linearly.
  firing_threshold added in Phase 1.5 Stage 7 (adaptive-firing-surplus).
- UNSCALED_PARAMS (29 fields): productivity scalars, counts, flags, and the seed.
  Three fields added in Phase 1.5 Stage X (augment-replenish-hiring):
  enable_replenish_hiring, max_hire_period, hire_delay_periods.
  Four fields added in beta-dist-task-attrs:
  alpha_mean, alpha_concentration, beta_mean, beta_concentration.
  These two tuples partition dataclasses.fields(FirmParams) exactly (no overlap,
  no missing field). An in-code assertion in check5_numeraire enforces this
  contract; any future FirmParams field addition will raise immediately on
  first run.

Float-tolerance choices:
- RTOL = 1e-12 (below float64 epsilon × typical scale; discriminates real drift)
- ATOL = 1e-9  (absolute floor for near-zero pi values, e.g. q_a=0 all-T case
  where pi = -F = -5/period; guards against spurious float failures on zero
  crossings in monotonicity diffs)
- check5 uses ``2 * ATOL`` because comparisons are against ``2 * pi_base``,
  doubling the magnitude scale while keeping relative discrimination consistent.

Monotonicity convention: weakly monotone (``>= -ATOL`` / ``<= ATOL`` on
np.diff) rather than strict, because random alpha/beta arrays can produce a
flat segment at boundary grid points. The phase-doc wording "monotonically
increases" is interpreted as weakly monotone per stage-three plan §Decisions.
"""

import os
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_A, all_H, all_T, greedy_profit, greedy_with_switching

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

RTOL: float = 1e-12
ATOL: float = 1e-9

# Monetary parameters: scale profit linearly under numeraire change.
# Eight params (matches check5 acceptance criterion).
SCALED_PARAMS: tuple[str, ...] = (
    "w",
    "c_aug",
    "c_auto",
    "c_fire",
    "c_hire",
    "c_train",
    "F",
    "p",
    "firing_threshold",
)

# Non-monetary parameters: productivity scalars, counts, seed, heterogeneity shape.
# Fifteen params (8 original + 5 from Phase 1.5 Stage 1 + 2 from Phase 1.5 Stage 3).
UNSCALED_PARAMS: tuple[str, ...] = (
    "q_h",
    "q_a",
    "g",
    "N",
    "T",
    "tasks_per_worker",
    "n_amortize",
    "seed",
    # Phase 1.5 Stage 1: worker heterogeneity shape parameters (not monetary)
    "sigma_theta",
    "theta_min",
    "theta_max",
    "corr_w_theta",
    "sigma_w",
    # Phase 1.5 Stage 3: periodic firing review shape parameters (not monetary)
    # T_review: period count (unscaled).
    # firing_threshold: monetary threshold (SCALED — moved to SCALED_PARAMS in Phase 1.5
    # Stage 7 / adaptive-firing-surplus, because the new formula makes surplus genuinely
    # monetary: effective_surplus = p*mean_output - wage - mean_aug_cost - F/K_review).
    "T_review",
    # Phase 1.5 Stage 6: training delay and margin scenario (not monetary scalars)
    "enable_training_delay",
    "scenario_mode",
    "margin_horizon",
    # target_margin is a ratio (dimensionless), not monetary — stays UNSCALED
    "target_margin",
    # Phase 1.5 Stage 6: enable_hiring is a boolean flag (not monetary)
    "enable_hiring",
    # Alpha-dependent automation cost (unknown-alpha-cost-model): dimensionless ratios (D-01)
    # and belief sentinel (D-02). Monetary scaling comes through the w/tpw factor in cost_vec.
    # Phase 1.5 Stage X (augment-replenish-hiring): three new non-monetary fields.
    # enable_replenish_hiring/enable_horizon_brute_action_grid: boolean flags.
    # max_hire_period/max_hire_per_step: counts; hire_delay_periods: count.
    # beta-dist-task-attrs: four new non-monetary distribution-shape fields.
    # Total: 9 SCALED + 33 UNSCALED = 42 FirmParams fields.
    "c_auto_alpha_slope",
    "c_auto_alpha_intercept",
    "belief_alpha",
    "enable_replenish_hiring",
    "max_hire_period",
    "hire_delay_periods",
    "max_hire_per_step",
    "enable_horizon_brute_action_grid",
    "alpha_mean",
    "alpha_concentration",
    "beta_mean",
    "beta_concentration",
    "dp_prior_alpha",
    "dp_prior_beta",
)


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def firm_factory_default():
    """Return a default Firm built from FirmParams(seed=0).

    Used as the default factory in run_tier_a when no factory is provided.
    """
    return make_firm(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0))


# ---------------------------------------------------------------------------
# Check 1: constant baseline
# ---------------------------------------------------------------------------


def check1_constant_baseline(firm_factory) -> tuple[bool, dict]:
    """Check 1: All-H with q_a=0, g=0, c_aug=0 produces constant per-period profit.

    Procedure:
      - Build params with q_a=0.0, g=0.0, c_aug=0.0 (everything else default
        from FirmParams(seed=0)).
      - Construct firm directly via make_firm(params) — NOT through firm_factory
        (which is reserved as a Phase-3 sweep hook; this check overrides params
        for axis isolation).
      - Run run_simulation(firm, all_H).
      - Closed-form expected: p * q_h * N - w * K - F where K = N // tasks_per_worker.
        With axis-isolated params (p=1, q_h=1, N=100, w=1, tasks_per_worker=10, F=5):
        expected_pi = 1.0 * 1.0 * 100 - 1.0 * 10 - 5.0 = 85.0 per period.
      - Also assert K is constant (worker-integerization regression).

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check
    builds its own params for axis isolation.

    Returns:
        (passed, details) where details = {
            "expected_pi": float,
            "actual_pi_min": float,
            "actual_pi_max": float,
            "max_abs_dev": float,
            "K_constant": bool,
        }
    """
    # sigma_theta=sigma_w=0: homogeneous workers so that expected_pi formula holds exactly
    params = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), q_a=0.0, g=0.0, c_aug=0.0, sigma_theta=0.0, sigma_w=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    # Closed-form constant pi per period (no noise at q_a=0, g=0, c_aug=0)
    K_expected = params.N // params.tasks_per_worker
    expected_pi = params.p * params.q_h * params.N - params.w * K_expected - params.F

    pi_vals = df["pi"].values
    max_abs_dev = float(np.max(np.abs(pi_vals - expected_pi)))
    passed = bool(np.allclose(pi_vals, expected_pi, rtol=RTOL, atol=ATOL))

    K_constant = bool((df["K"] == K_expected).all())
    passed = passed and K_constant

    return passed, {
        "expected_pi": float(expected_pi),
        "actual_pi_min": float(pi_vals.min()),
        "actual_pi_max": float(pi_vals.max()),
        "max_abs_dev": max_abs_dev,
        "K_constant": K_constant,
    }


# ---------------------------------------------------------------------------
# Check 3: monotonicity in q_a (all_T only)
# ---------------------------------------------------------------------------


def check3_monotone_q_a(firm_factory) -> tuple[bool, dict]:
    """Check 3: Higher q_a monotonically increases all_T total profit.

    Grid: q_a_grid = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0] (6 points; spans
    default q_a=1.2; includes zero to anchor).

    Procedure for each q_a in grid:
      - params = replace(FirmParams(seed=0), q_a=q_a)
      - firm = make_firm(params)
      - df = run_simulation(firm, all_T)
      - record total_pi = sum of df["pi"]

    Monotonicity (weakly increasing): diffs = np.diff(total_pis); pass iff
    (diffs > -ATOL).all(). Weakly monotone because a flat segment is possible
    with random alpha/beta (strict would produce spurious failures at boundary
    points). Phase-doc wording "monotonically increases" interpreted as weakly
    monotone per stage-three plan §Decisions.

    All-H is q_a-invariant by construction (productivity_vec does not use q_a
    for H tasks) so checking it here would be vacuous; omitted per stage-three
    scope.

    Sanity: at q_a=0, all tasks have zero productivity, K=0 (all T → no
    workers). So pi/period = 0 - 0 - F = -5, total_pi = -T*F = -300 with
    defaults (T=60, F=5).

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check
    builds its own params for axis isolation.

    Returns:
        (passed, details) where details = {
            "q_a_grid": list,
            "total_pi_per_q_a": list,
            "diffs": list,
            "min_diff": float,
        }
    """
    q_a_grid = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    total_pis = []

    for q_a in q_a_grid:
        params = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), q_a=float(q_a), sigma_theta=0.0, sigma_w=0.0)
        firm = make_firm(params)
        df = run_simulation(firm, all_T)
        total_pis.append(float(df["pi"].sum()))

    diffs = np.diff(total_pis)
    min_diff = float(diffs.min())
    passed = bool((diffs > -ATOL).all())

    return passed, {
        "q_a_grid": q_a_grid.tolist(),
        "total_pi_per_q_a": total_pis,
        "diffs": diffs.tolist(),
        "min_diff": min_diff,
    }


# ---------------------------------------------------------------------------
# Check 4: monotonicity in w (all_H only)
# ---------------------------------------------------------------------------


def check4_monotone_w(firm_factory) -> tuple[bool, dict]:
    """Check 4: Higher w monotonically decreases all_H total profit.

    Grid: w_grid = [0.5, 1.0, 1.5, 2.0, 2.5] (5 points; spans default w=1.0).

    Procedure for each w in grid:
      - params = replace(FirmParams(seed=0), w=w)
      - firm = make_firm(params)
      - df = run_simulation(firm, all_H)
      - record total_pi = sum of df["pi"]

    Monotonicity (weakly decreasing): diffs = np.diff(total_pis); pass iff
    (diffs < ATOL).all(). Per stage-three plan §Decisions: pi can be negative;
    np.diff is sign-preserving, so "more negative" produces a negative diff and
    the comparison is correct even when both endpoints are negative.

    All-T is w-invariant (K=0 when all tasks are automated → wage_cost=0) so
    checking it here would be vacuous; omitted per stage-three scope.

    Hand-check at w=2.5: K=10 (100 tasks / 10 tasks_per_worker), T=60.
      pi_per_period = p*q_h*N - w*K - F = 1*1*100 - 2.5*10 - 5 = 70
      total_pi = 60 * 70 = 4200
    (Both endpoints positive for the w-grid used here.)

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check
    builds its own params for axis isolation.

    Returns:
        (passed, details) where details = {
            "w_grid": list,
            "total_pi_per_w": list,
            "diffs": list,
            "max_diff": float,
        }
    """
    w_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    total_pis = []

    for w in w_grid:
        params = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), w=float(w), sigma_theta=0.0, sigma_w=0.0)
        firm = make_firm(params)
        df = run_simulation(firm, all_H)
        total_pis.append(float(df["pi"].sum()))

    diffs = np.diff(total_pis)
    max_diff = float(diffs.max())
    passed = bool((diffs < ATOL).all())

    return passed, {
        "w_grid": w_grid.tolist(),
        "total_pi_per_w": total_pis,
        "diffs": diffs.tolist(),
        "max_diff": max_diff,
    }


# ---------------------------------------------------------------------------
# Check 5: numeraire invariance (all 3 strategies)
# ---------------------------------------------------------------------------


def check5_numeraire(firm_factory) -> tuple[bool, dict]:
    """Check 5: Multiplying all monetary params by 2 scales all profits by exactly 2.

    Numeraire-scope contract enforced by runtime assertions:
      - SCALED_PARAMS and UNSCALED_PARAMS are disjoint.
      - Their union covers every field in FirmParams exactly.
    These assertions fire at the first check5 call; any future FirmParams
    field addition (without updating the tuples) raises immediately.

    Procedure for each strategy in (all_H, all_A, all_T):
      - Base: params_base = FirmParams(seed=0); pi_base = run_simulation(...)["pi"]
      - Scaled: params_scaled via dataclasses.replace with each SCALED_PARAMS
        field multiplied by 2.0. UNSCALED_PARAMS (q_h, q_a, g, N, T,
        tasks_per_worker, n_amortize, seed) are NOT passed to replace.
        Same seed → same alpha/beta → same task structure.
      - Pass iff np.allclose(pi_scaled, 2.0 * pi_base, rtol=RTOL, atol=2*ATOL).
        atol=2*ATOL because comparisons are against 2*pi_base (doubled magnitude
        scale → keep relative discrimination consistent).

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check
    builds its own params for axis isolation.

    Returns:
        (passed, details) where details = {
            "per_strategy": {
                "all_H": {"max_abs_dev": float, "passed": bool},
                "all_A": {"max_abs_dev": float, "passed": bool},
                "all_T": {"max_abs_dev": float, "passed": bool},
            },
            "scaled_params": list,
            "unscaled_params": list,
        }
    """
    # Runtime enforcement of the numeraire-scope contract
    all_field_names = {f.name for f in fields(FirmParams)}
    assert set(SCALED_PARAMS).isdisjoint(UNSCALED_PARAMS), (
        "SCALED_PARAMS and UNSCALED_PARAMS overlap — numeraire-scope contract violated"
    )
    assert set(SCALED_PARAMS) | set(UNSCALED_PARAMS) == all_field_names, (
        f"SCALED_PARAMS | UNSCALED_PARAMS does not cover all FirmParams fields. "
        f"Missing: {all_field_names - (set(SCALED_PARAMS) | set(UNSCALED_PARAMS))}, "
        f"Extra: {(set(SCALED_PARAMS) | set(UNSCALED_PARAMS)) - all_field_names}"
    )

    strategies = [all_H, all_A, all_T]
    per_strategy: dict[str, dict] = {}
    all_passed = True

    for strategy in strategies:
        name = strategy.__name__

        params_base = FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0)
        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, strategy)
        pi_base = df_base["pi"].values

        scaled_kwargs = {
            field_name: getattr(params_base, field_name) * 2.0
            for field_name in SCALED_PARAMS
        }
        params_scaled = replace(params_base, **scaled_kwargs)
        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, strategy)
        pi_scaled = df_scaled["pi"].values

        max_abs_dev = float(np.max(np.abs(pi_scaled - 2.0 * pi_base)))
        strategy_passed = bool(
            np.allclose(pi_scaled, 2.0 * pi_base, rtol=RTOL, atol=2 * ATOL)
        )

        per_strategy[name] = {
            "max_abs_dev": max_abs_dev,
            "passed": strategy_passed,
        }
        all_passed = all_passed and strategy_passed

    return all_passed, {
        "per_strategy": per_strategy,
        "scaled_params": list(SCALED_PARAMS),
        "unscaled_params": list(UNSCALED_PARAMS),
    }


# ---------------------------------------------------------------------------
# Check 7: Phase 1 degenerate parity (Tier A — Phase 1.5 Stage 1)
# ---------------------------------------------------------------------------


def check7_phase1_parity(firm_factory) -> tuple[bool, dict]:
    """Check 7: With sigma_theta=sigma_w=0, Phase 1.5 history is byte-identical to Phase 1.

    Reads parquet fixtures from tests/fixtures/ (captured before kernel edits,
    see D-07 provenance sentinel). Verifies columns t, Y, C, pi, K, adj_cost
    are np.array_equal to the Phase 1 baseline for all 5 strategies.

    D-07 provenance sentinel: reads tests/fixtures/_provenance.txt and raises
    RuntimeError if git_dirty_files != 0 or the file is absent.

    Returns:
        (passed, details) where details = {
            "per_strategy": {<name>: {"passed": bool, "max_dev_per_col": dict}},
            "provenance_ok": bool,
            "git_commit": str,
        }
    """
    import pandas as pd

    fixtures_dir = Path("tests/fixtures")
    provenance_path = fixtures_dir / "_provenance.txt"

    # D-07: provenance sentinel check
    if not provenance_path.exists():
        raise RuntimeError(
            f"Parity fixtures missing: {provenance_path} not found. "
            "Run T-08 fixture-capture procedure before kernel edits."
        )
    provenance = {}
    for line in provenance_path.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            provenance[k.strip()] = v.strip()
    if provenance.get("git_dirty_files", "1") != "0":
        raise RuntimeError(
            f"Provenance sentinel reports dirty tree at capture time: {provenance}. "
            "Fixtures may be contaminated — re-capture on a clean working tree."
        )
    git_commit = provenance.get("git_commit", "unknown")

    strategies = [
        ("all_H", all_H),
        ("all_A", all_A),
        ("all_T", all_T),
        ("greedy_profit", greedy_profit),
        ("greedy_with_switching", greedy_with_switching),
    ]
    parity_cols = ["t", "Y", "C", "pi", "K", "adj_cost"]

    per_strategy: dict[str, dict] = {}
    all_passed = True

    for strat_name, strat in strategies:
        fixture_path = fixtures_dir / f"phase1_baseline_{strat_name}.parquet"
        if not fixture_path.exists():
            per_strategy[strat_name] = {"passed": False, "error": f"fixture missing: {fixture_path}"}
            all_passed = False
            continue

        df1 = pd.read_parquet(fixture_path)
        # Pin fixture-capture params (w, c_auto, alpha/beta_mean changed in 99ddaea).
        params7 = FirmParams(
            seed=0, N=100, sigma_theta=0.0, sigma_w=0.0,
            tasks_per_worker=10, p=1.0,
            w=1.0, c_aug=0.05, c_auto=0.4, enable_hiring=False,
            alpha_mean=0.5, alpha_concentration=2.0,
            beta_mean=0.5, beta_concentration=2.0,
        )
        firm = make_firm(params7)
        df15 = run_simulation(firm, strat)

        max_dev_per_col: dict[str, float] = {}
        strat_passed = True
        for col in parity_cols:
            eq = np.array_equal(df15[col].values, df1[col].values)
            if not eq:
                dev = float(np.max(np.abs(df15[col].values.astype(float) - df1[col].values.astype(float))))
                max_dev_per_col[col] = dev
                strat_passed = False
            else:
                max_dev_per_col[col] = 0.0

        per_strategy[strat_name] = {"passed": strat_passed, "max_dev_per_col": max_dev_per_col}
        all_passed = all_passed and strat_passed

    return all_passed, {
        "per_strategy": per_strategy,
        "provenance_ok": True,
        "git_commit": git_commit,
    }


# ---------------------------------------------------------------------------
# Check 8: Stage 3 neutrality — T_review=inf produces byte-identical output
# to Stage 2 fixtures (Tier A — Phase 1.5 Stage 3)
# ---------------------------------------------------------------------------


def check8_stage3_neutrality(firm_factory) -> tuple[bool, dict]:
    """Check 8: Stage 3 with T_review=math.inf is byte-identical to Stage 2 fixtures.

    The periodic firing review path is dormant when T_review=math.inf (D-11 default).
    This check verifies that Stage 3 code with the dormant path produces the SAME
    output as the Stage 2 fixtures — confirming the firing-review wire-in introduced
    no numerical drift.

    D-07 revision: re-capturing fixtures at Stage 3 tip would test a tautology.
    The same Stage 2 parquet files (captured at 828db38) serve as the parity baseline.

    D-11: since default T_review=math.inf, NO axis-isolation overrides are needed in
    any other check. This check may still pass T_review=math.inf explicitly for
    documentation intent (it is redundant with the default).

    Procedure: mirrors check7_phase1_parity.
      - params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0)
        (default T_review=math.inf — firing review dormant)
      - Run all 5 strategies; assert np.array_equal on parity_cols vs Stage 2 fixtures.

    Returns:
        (passed, details) where details = {
            "per_strategy": {<name>: {"passed": bool, "max_dev_per_col": dict}},
            "provenance_ok": bool,
            "git_commit": str,
        }
    """
    import math as _math

    import pandas as pd

    fixtures_dir = Path("tests/fixtures")
    provenance_path = fixtures_dir / "_provenance.txt"

    # D-07: provenance sentinel check (same as check7)
    if not provenance_path.exists():
        raise RuntimeError(
            f"Parity fixtures missing: {provenance_path} not found. "
            "Run T-08 fixture-capture procedure before kernel edits."
        )
    provenance = {}
    for line in provenance_path.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            provenance[k.strip()] = v.strip()
    if provenance.get("git_dirty_files", "1") != "0":
        raise RuntimeError(
            f"Provenance sentinel reports dirty tree at capture time: {provenance}. "
            "Fixtures may be contaminated — re-capture on a clean working tree."
        )
    git_commit = provenance.get("git_commit", "unknown")

    strategies = [
        ("all_H", all_H),
        ("all_A", all_A),
        ("all_T", all_T),
        ("greedy_profit", greedy_profit),
        ("greedy_with_switching", greedy_with_switching),
    ]
    parity_cols = ["t", "Y", "C", "pi", "K", "adj_cost"]

    per_strategy: dict[str, dict] = {}
    all_passed = True

    for strat_name, strat in strategies:
        fixture_path = fixtures_dir / f"phase1_baseline_{strat_name}.parquet"
        if not fixture_path.exists():
            per_strategy[strat_name] = {"passed": False, "error": f"fixture missing: {fixture_path}"}
            all_passed = False
            continue

        df_fixture = pd.read_parquet(fixture_path)
        # D-11: T_review=math.inf is the default — explicit here for documentation intent.
        # Pin fixture-capture params (w, c_auto, alpha/beta_mean changed in 99ddaea).
        params = FirmParams(
            seed=0, N=100, sigma_theta=0.0, sigma_w=0.0,
            T_review=_math.inf, tasks_per_worker=10, p=1.0,
            w=1.0, c_aug=0.05, c_auto=0.4, enable_hiring=False,
            alpha_mean=0.5, alpha_concentration=2.0,
            beta_mean=0.5, beta_concentration=2.0,
        )
        firm = make_firm(params)
        df_s3 = run_simulation(firm, strat)

        max_dev_per_col: dict[str, float] = {}
        strat_passed = True
        for col in parity_cols:
            eq = np.array_equal(df_s3[col].values, df_fixture[col].values)
            if not eq:
                dev = float(np.max(np.abs(
                    df_s3[col].values.astype(float) - df_fixture[col].values.astype(float)
                )))
                max_dev_per_col[col] = dev
                strat_passed = False
            else:
                max_dev_per_col[col] = 0.0

        per_strategy[strat_name] = {"passed": strat_passed, "max_dev_per_col": max_dev_per_col}
        all_passed = all_passed and strat_passed

    return all_passed, {
        "per_strategy": per_strategy,
        "provenance_ok": True,
        "git_commit": git_commit,
    }


# ---------------------------------------------------------------------------
# Check 9: numeraire invariance with firing review ACTIVE (Stage 5 T-11)
# ---------------------------------------------------------------------------


def check9_numeraire_with_firing_active(firm_factory) -> tuple[bool, dict]:
    """Check 9: numeraire invariance holds even when firing review is active.

    Uses sigma_theta=0, sigma_w=0, T_review=10, firing_threshold=0.0.
    Per D-11: at these params, surplus = mean_output - wage. With all_H,
    output = q_h * tasks_per_worker = 10.0, wage = w = 1.0 → surplus = 9.0 > 0.
    No workers fire. Numeraire holds trivially (no firing-set change under scaling).
    Additionally asserts adj_cost_base == 0 for the first period (no mode transitions
    at t=0 — strategy proposes same all_H modes as initial state; clamp is never active).

    Returns:
        (passed, details) where details contains per-strategy results and the adj_cost=0 assertion.
    """
    from dataclasses import replace as _replace

    params_base = FirmParams(
        seed=0,
        N=100,
        sigma_theta=0.0,
        sigma_w=0.0,
        T=20,
        T_review=10.0,
        firing_threshold=0.0,
        tasks_per_worker=10,
        p=1.0,
    )
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED_PARAMS}
    params_scaled = _replace(params_base, **scaled_kwargs)

    strategies = [
        ("all_H", all_H),
        ("all_A", all_A),
    ]

    per_strategy: dict[str, dict] = {}
    all_passed = True

    for strat_name, strat in strategies:
        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, strat)
        pi_base = df_base["pi"].values

        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, strat)
        pi_scaled = df_scaled["pi"].values

        numeraire_ok = bool(np.allclose(pi_scaled, 2.0 * pi_base, rtol=1e-10, atol=1e-9))
        max_dev = float(np.max(np.abs(pi_scaled - 2.0 * pi_base)))

        # adj_cost=0 at t=0 for all_H only: all_H proposes H from initial all-H state → no
        # mode transition → adj_cost=0. all_A switches H→A at t=0 → adj_cost > 0 (expected).
        adj_cost_base_t0 = float(df_base["adj_cost"].iloc[0])
        adj_cost_ok = (strat_name != "all_H") or (adj_cost_base_t0 == 0.0)

        # n_review_fired should be 0 for all periods (surplus > 0 at sigma=0, firing_threshold=0)
        no_firings = bool(df_base["n_review_fired"].sum() == 0)

        strat_passed = numeraire_ok and adj_cost_ok and no_firings
        all_passed = all_passed and strat_passed
        per_strategy[strat_name] = {
            "passed": strat_passed,
            "numeraire_ok": numeraire_ok,
            "max_pi_dev": max_dev,
            "adj_cost_base_t0": adj_cost_base_t0,
            "adj_cost_ok": adj_cost_ok,
            "no_firings": no_firings,
            "n_review_fired_sum": int(df_base["n_review_fired"].sum()),
        }

    return all_passed, {"per_strategy": per_strategy}


# ---------------------------------------------------------------------------
# Check 10: adaptive-firing surplus numeraire invariance (Stage 7)
# ---------------------------------------------------------------------------


def check10_adaptive_firing_numeraire(firm_factory) -> tuple[bool, dict]:
    """Check 10: numeraire invariance holds with four-term surplus and active firing.

    Tests two scenarios:
    (a) firing_threshold=0.0 — at default-ish params (p=0.22, tpw=5, sigma=0),
        all H-mode workers have surplus = 0.22*5 - 1.0 - 0 - 5/20 = -0.15 < 0 →
        all workers fire at every review tick. Verifies that pi_scaled == 2*pi_base
        and n_review_fired is identical between base and scaled run.
    (b) firing_threshold=0.05 — same structure; scaled threshold = 0.10 (firing_threshold
        now in SCALED_PARAMS after Stage 7). Fire mask is identical because
        surplus × 2 and threshold × 2 preserve the inequality direction.

    Validates both: (i) the four-term surplus formula scales linearly under monetary
    doubling; (ii) reclassifying firing_threshold as SCALED doesn't break invariance.
    """
    from dataclasses import replace as _replace

    all_passed = True
    per_variant: dict[str, dict] = {}

    for label, threshold in [("threshold_0", 0.0), ("threshold_005", 0.05)]:
        params_base = FirmParams(
            seed=0,
            N=100,  # AI-era recipe at small scale — N=100 by intent
            sigma_theta=0.0,
            sigma_w=0.0,
            T=20,
            T_review=10.0,
            firing_threshold=threshold,
            tasks_per_worker=5,
            p=0.22,
        )
        scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED_PARAMS}
        params_scaled = _replace(params_base, **scaled_kwargs)

        firm_base = make_firm(params_base)
        df_base = run_simulation(firm_base, all_H)
        pi_base = df_base["pi"].values

        firm_scaled = make_firm(params_scaled)
        df_scaled = run_simulation(firm_scaled, all_H)
        pi_scaled = df_scaled["pi"].values

        numeraire_ok = bool(np.allclose(pi_scaled, 2.0 * pi_base, rtol=1e-10, atol=1e-9))
        max_dev = float(np.max(np.abs(pi_scaled - 2.0 * pi_base)))

        fired_base = df_base["n_review_fired"].values
        fired_scaled = df_scaled["n_review_fired"].values
        fire_mask_ok = bool(np.array_equal(fired_base, fired_scaled))

        # Verify firings actually occurred (formula is exercised, not dormant)
        any_firings = bool(df_base["n_review_fired"].sum() > 0)

        variant_passed = numeraire_ok and fire_mask_ok and any_firings
        all_passed = all_passed and variant_passed
        per_variant[label] = {
            "passed": variant_passed,
            "numeraire_ok": numeraire_ok,
            "fire_mask_ok": fire_mask_ok,
            "any_firings": any_firings,
            "max_pi_dev": max_dev,
            "n_review_fired_base": int(df_base["n_review_fired"].sum()),
        }

    return all_passed, {"per_variant": per_variant}


# ---------------------------------------------------------------------------
# Check 11: numeraire invariance with replenish-hiring active (Stage X)
# ---------------------------------------------------------------------------


def check11_replenish_numeraire(firm_factory) -> tuple[bool, dict]:
    """Check 11: numeraire invariance holds when enable_replenish_hiring=True.

    Uses params that guarantee firings occur (p=0.22, tpw=5, firing_threshold=0.0,
    sigma=0) so the replenish path is non-vacuously exercised.

    Verifies:
    - pi_scaled == 2 * pi_base (rtol=1e-10, atol=1e-9)
    - n_review_fired identical between base and scaled run (fire mask unchanged)
    - n_hired identical between base and scaled run (hire decisions monetary-invariant)
    - At least one firing AND one rehire actually occurred (non-vacuous)

    The replenish path only adds: period_hire_cost = c_hire * n_hire. c_hire is in
    SCALED_PARAMS → doubles linearly. n_hire is determined by K counts (UNSCALED) →
    identical across base/scaled runs. Invariance holds.
    """
    from dataclasses import replace as _replace

    params_base = FirmParams(
        seed=0,
        N=100,  # AI-era recipe at small scale — N=100 by intent
        tasks_per_worker=5, p=0.22,
        sigma_theta=0.0, sigma_w=0.0,
        T=20, T_review=10.0,
        firing_threshold=0.0,
        enable_hiring=False, enable_replenish_hiring=True,
        max_hire_period=0,
        hire_delay_periods=1,
    )
    scaled_kwargs = {f: getattr(params_base, f) * 2.0 for f in SCALED_PARAMS}
    params_scaled = _replace(params_base, **scaled_kwargs)

    firm_base = make_firm(params_base)
    df_base = run_simulation(firm_base, all_H)
    pi_base = df_base["pi"].values

    firm_scaled = make_firm(params_scaled)
    df_scaled = run_simulation(firm_scaled, all_H)
    pi_scaled = df_scaled["pi"].values

    numeraire_ok = bool(np.allclose(pi_scaled, 2.0 * pi_base, rtol=1e-10, atol=1e-9))
    max_dev = float(np.max(np.abs(pi_scaled - 2.0 * pi_base)))

    n_review_fired_base = df_base["n_review_fired"].values
    n_review_fired_scaled = df_scaled["n_review_fired"].values
    fire_mask_ok = bool(np.array_equal(n_review_fired_base, n_review_fired_scaled))

    n_hired_base = df_base["n_hired"].values
    n_hired_scaled = df_scaled["n_hired"].values
    hire_mask_ok = bool(np.array_equal(n_hired_base, n_hired_scaled))

    any_firings = bool(df_base["n_review_fired"].sum() > 0)
    any_hires = bool(df_base["n_hired"].sum() > 0)

    passed = numeraire_ok and fire_mask_ok and hire_mask_ok and any_firings and any_hires

    return passed, {
        "numeraire_ok": numeraire_ok,
        "fire_mask_ok": fire_mask_ok,
        "hire_mask_ok": hire_mask_ok,
        "any_firings": any_firings,
        "any_hires": any_hires,
        "max_pi_dev": max_dev,
        "n_review_fired_base": int(df_base["n_review_fired"].sum()),
        "n_hired_base": int(df_base["n_hired"].sum()),
    }


# ---------------------------------------------------------------------------
# Tier A aggregator
# ---------------------------------------------------------------------------


def run_tier_a(firm_factory=None) -> dict:
    """Run all Tier A kernel checks and return a combined results dict.

    Tier A checks: check1 (constant baseline), check3 (monotonicity in q_a),
    check4 (monotonicity in w), check5 (numeraire invariance), check7 (Phase 1
    degenerate parity), check8 (Stage 3 T_review=inf neutrality), check9
    (numeraire with firing active, Stage 5), check10 (adaptive-firing surplus
    numeraire, Stage 7 — four-term formula + firing_threshold in SCALED_PARAMS),
    check11 (replenish-hiring numeraire, Stage X — enable_replenish_hiring path).
    Checks 2 and 6 (greedy dominance and adjustment-cost integration) are Tier B.

    Args:
        firm_factory: Callable returning a Firm, or None. If None, uses
            firm_factory_default (FirmParams(seed=0)). The factory provides
            the "shape baseline" for Phase-3 sweep reuse; individual checks
            override specific params via dataclasses.replace for axis isolation.

    Returns:
        dict with keys: "check1", "check3", "check4", "check5", "check7", "check8",
        "check9", "check10", "check11", "all_passed". Each check value is
        {"passed": bool, "details": dict}. Does NOT raise on failure — returns the
        dict so the driver can render a full report showing which checks failed and why.

    Deterministic: same seed → same alpha/beta → identical outputs across
    consecutive calls (no global state mutation between calls).
    """
    if firm_factory is None:
        firm_factory = firm_factory_default

    results: dict = {}

    for check_name, check_fn in [
        ("check1", check1_constant_baseline),
        ("check3", check3_monotone_q_a),
        ("check4", check4_monotone_w),
        ("check5", check5_numeraire),
        ("check7", check7_phase1_parity),
        ("check8", check8_stage3_neutrality),
        ("check9", check9_numeraire_with_firing_active),
        ("check10", check10_adaptive_firing_numeraire),
        ("check11", check11_replenish_numeraire),
    ]:
        passed, details = check_fn(firm_factory)
        results[check_name] = {"passed": passed, "details": details}

    results["all_passed"] = all(
        results[k]["passed"] for k in (
            "check1", "check3", "check4", "check5", "check7", "check8",
            "check9", "check10", "check11"
        )
    )

    return results


# ---------------------------------------------------------------------------
# Check 2: greedy dominance (Tier B)
# ---------------------------------------------------------------------------


def check2_greedy_dominance(firm_factory) -> tuple[bool, dict]:
    """Check 2: greedy_profit cumulative profit >= each baseline with zero switching costs.

    With all switching costs zero (c_train=c_fire=c_hire=0), greedy_profit must
    achieve cumulative profit >= each of {all_H, all_A, all_T, greedy_with_switching}
    at every period, using weak >= with atol=ATOL=1e-9.

    Rationale for weak inequality (parent architecture tie-break risk):
    When q_a * alpha_i == q_h for some tasks, greedy_profit and other strategies
    may produce identical per-task scores, leading to identical cumulative profits.
    The np.argmax first-index-wins tie-break (H < A < T) means greedy_profit and
    greedy_with_switching (which is identical to greedy_profit when all switching
    costs are zero) may choose the same modes, producing exactly equal cumulative
    profits. Strict > would produce false failures on these knife-edge configurations.

    Single-firm-instance constraint (parent architecture shared-firm-instance risk):
    All strategies run on the SAME firm instance (same alpha/beta arrays) via
    run_simulation, which calls firm.reset() before each run and preserves alpha/beta.
    This ensures greedy_profit's advantage is measured on an identical task landscape,
    not a different random draw.

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check builds
    its own params for axis isolation (mirrors check1/check3/check4/check5 contract).

    Holds in the dormant regime (belief_alpha=None). Under the engaged path
    (belief_alpha is not None), greedy-under-belief is no longer optimal in hindsight
    by design — the belief wedge IS the feature. Use check11_greedy_dominance_under_belief
    (future work, Q-04) for the engaged-path invariant.

    Returns:
        (passed, details) where details = {
            "per_baseline": {
                "all_H": {"min_gap": float, "passed": bool},
                "all_A": {"min_gap": float, "passed": bool},
                "all_T": {"min_gap": float, "passed": bool},
                "greedy_with_switching": {"min_gap": float, "passed": bool},
            },
            "atol_used": 1e-9,
            "tolerance_form": "weak (>=) per parent architecture tie-break risk",
        }
    """
    params = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), c_train=0.0, c_fire=0.0, c_hire=0.0, sigma_theta=0.0, sigma_w=0.0)
    firm = make_firm(params)

    # Run greedy_profit on the shared firm instance
    df_g = run_simulation(firm, greedy_profit)
    cum_g = df_g["pi"].cumsum().values

    baselines = [
        ("all_H", all_H),
        ("all_A", all_A),
        ("all_T", all_T),
        ("greedy_with_switching", greedy_with_switching),
    ]

    per_baseline: dict[str, dict] = {}
    all_passed = True

    for name, strategy in baselines:
        # run_simulation calls firm.reset() — alpha/beta preserved (R-07)
        df_s = run_simulation(firm, strategy)
        cum_s = df_s["pi"].cumsum().values
        gap = cum_g - cum_s
        s_passed = bool((gap >= -ATOL).all())
        per_baseline[name] = {
            "min_gap": float(gap.min()),
            "passed": s_passed,
        }
        all_passed = all_passed and s_passed

    return all_passed, {
        "per_baseline": per_baseline,
        "atol_used": ATOL,
        "tolerance_form": "weak (>=) per parent architecture tie-break risk",
    }


# ---------------------------------------------------------------------------
# Check 6: no switching under high adjustment costs (Tier B)
# ---------------------------------------------------------------------------


def check6_no_switching_under_high_costs(firm_factory) -> tuple[bool, dict]:
    """Check 6: greedy_with_switching stays all-H when c_train=c_fire=c_hire=100.

    Joint switching cost recipe (parent architecture joint-cost-recipe risk, R-15):
    Only setting c_train=100 blocks H->A but NOT H->T or A->T under the smooth
    amortized greedy decision rule (parent architecture smooth-vs-lumpy risk, R-14).
    The smooth per-task amortization for H->T is c_fire/tasks_per_worker/n_amortize
    = 2/10/6 ≈ 0.033 — small relative to plausible per-task gains. Setting ALL three
    (c_train=c_fire=c_hire=100) drives every per-task switching cost well above any
    plausible per-task gain, making "no switching" deterministic.

    Verification:
    - Primary: adj_cost == 0 every period (load-bearing; reads simulate.py's
      adj_cost column — D-01 from current plan ensures this column exists).
    - Secondary: all modes stay at Mode.H (firm initialises all-H via firm.reset()).

    Differential sanity (parent architecture adj-cost-timing risk, R-03):
    Also run the SAME scenario with c_train=c_fire=c_hire=0 and assert that
    greedy_with_switching DOES switch (adj_cost.sum() > 0). This pins the check
    in both directions: "stays put under high costs" and "does switch when free",
    preventing a vacuously-passing check from a broken greedy that always picks H.

    Note: firm_factory parameter is reserved for Phase-3 sweep reuse; check builds
    its own params for axis isolation.

    Returns:
        (passed, details) where details = {
            "max_adj_cost": float,
            "no_switching_under_high_costs": bool,
            "differential_switches_under_zero_costs": bool,
            "joint_cost_recipe": "c_train=c_fire=c_hire=100",
        }
    """
    # High-cost scenario: joint recipe per R-15; sigma=0 for axis isolation
    params_high = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), c_train=100.0, c_fire=100.0, c_hire=100.0,
                          sigma_theta=0.0, sigma_w=0.0)
    firm_high = make_firm(params_high)
    df_high = run_simulation(firm_high, greedy_with_switching)

    max_adj_cost = float(np.abs(df_high["adj_cost"].values).max())
    no_switching = bool(np.allclose(df_high["adj_cost"].values, 0.0, atol=ATOL))

    # Differential sanity: with zero switching costs, greedy_with_switching must
    # move AWAY from the all-H initial state on at least one task at t=0.
    # NOTE: adj_cost is zero-by-construction when c_train=c_fire=c_hire=0,
    # even if modes change — so we check modes directly, not adj_cost.sum().
    params_zero = replace(FirmParams(seed=0, N=100, tasks_per_worker=10, p=1.0), c_train=0.0, c_fire=0.0, c_hire=0.0,
                          sigma_theta=0.0, sigma_w=0.0)
    firm_zero = make_firm(params_zero)
    df_zero = run_simulation(firm_zero, greedy_with_switching)
    # The firm starts all-H (firm.reset() sets all modes to Mode.H = 0).
    # At t=0, greedy_with_switching (with zero switching costs) is identical
    # to greedy_profit and picks the globally optimal mode per task, typically
    # moving many tasks to A or T. If even one task moves non-H, this fires.
    modes_t0_zero = df_zero["modes"].iloc[0]
    differential_switches = bool((modes_t0_zero != 0).any())

    passed = no_switching and differential_switches

    return passed, {
        "max_adj_cost": max_adj_cost,
        "no_switching_under_high_costs": no_switching,
        "differential_switches_under_zero_costs": differential_switches,
        "joint_cost_recipe": "c_train=c_fire=c_hire=100",
    }


# ---------------------------------------------------------------------------
# Tier B aggregator
# ---------------------------------------------------------------------------


def run_tier_b(firm_factory=None) -> dict:
    """Run all Tier B greedy-strategy checks and return a combined results dict.

    Tier B checks: check2 (greedy dominance over baselines with zero switching
    costs) and check6 (no switching under joint high costs). These checks require
    the greedy strategies and adjustment-cost code from Stage 4.

    Args:
        firm_factory: Callable returning a Firm, or None. If None, uses
            firm_factory_default. Reserved for Phase-3 sweep reuse; individual
            checks build their own params for axis isolation.

    Returns:
        dict with keys: "check2", "check6", "all_passed".
        Each check value is {"passed": bool, "details": dict}.
        Does NOT raise on failure — returns the dict for full reporting.
    """
    if firm_factory is None:
        firm_factory = firm_factory_default

    results: dict = {}

    for check_name, check_fn in [
        ("check2", check2_greedy_dominance),
        ("check6", check6_no_switching_under_high_costs),
    ]:
        passed, details = check_fn(firm_factory)
        results[check_name] = {"passed": passed, "details": details}

    results["all_passed"] = all(
        results[k]["passed"] for k in ("check2", "check6")
    )

    return results


# ---------------------------------------------------------------------------
# Full-suite aggregator (all six checks)
# ---------------------------------------------------------------------------


def run_all_checks(firm_factory=None) -> dict:
    """Run all six validation checks (Tier A + Tier B) and return a combined dict.

    This is the single function the gate reviewer runs to validate Phase 1.
    Covers: check1 (constant baseline), check2 (greedy dominance), check3
    (monotonicity q_a), check4 (monotonicity w), check5 (numeraire invariance),
    check6 (no switching under high costs).

    Args:
        firm_factory: Callable returning a Firm, or None. Passed through to
            run_tier_a and run_tier_b; individual checks override params for
            axis isolation.

    Returns:
        dict with keys: "tier_a", "tier_b", "all_passed".
        "tier_a" and "tier_b" are the dicts returned by run_tier_a/run_tier_b.
        "all_passed" is True iff both tier_a["all_passed"] and tier_b["all_passed"].
    """
    tier_a = run_tier_a(firm_factory)
    tier_b = run_tier_b(firm_factory)
    return {
        "tier_a": tier_a,
        "tier_b": tier_b,
        "all_passed": tier_a["all_passed"] and tier_b["all_passed"],
    }
