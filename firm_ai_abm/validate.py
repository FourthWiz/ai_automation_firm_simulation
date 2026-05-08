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
- SCALED_PARAMS (8 fields): the monetary parameters that scale profit linearly.
- UNSCALED_PARAMS (8 fields): productivity scalars, counts, and the seed.
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

from dataclasses import fields, replace

import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_A, all_H, all_T

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
)

# Non-monetary parameters: productivity scalars, counts, seed.
# Eight params.
UNSCALED_PARAMS: tuple[str, ...] = (
    "q_h",
    "q_a",
    "g",
    "N",
    "T",
    "tasks_per_worker",
    "n_amortize",
    "seed",
)


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def firm_factory_default():
    """Return a default Firm built from FirmParams(seed=0).

    Used as the default factory in run_tier_a when no factory is provided.
    """
    return make_firm(FirmParams(seed=0))


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
        With defaults (p=1, q_h=1, N=100, w=1, tasks_per_worker=10, F=5):
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
    params = replace(FirmParams(seed=0), q_a=0.0, g=0.0, c_aug=0.0)
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
        params = replace(FirmParams(seed=0), q_a=float(q_a))
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
        params = replace(FirmParams(seed=0), w=float(w))
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

        params_base = FirmParams(seed=0)
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
# Tier A aggregator
# ---------------------------------------------------------------------------


def run_tier_a(firm_factory=None) -> dict:
    """Run all four Tier A kernel checks and return a combined results dict.

    Tier A checks: check1 (constant baseline), check3 (monotonicity in q_a),
    check4 (monotonicity in w), check5 (numeraire invariance). Checks 2 and 6
    (greedy dominance and adjustment-cost integration) are Tier B.

    Args:
        firm_factory: Callable returning a Firm, or None. If None, uses
            firm_factory_default (FirmParams(seed=0)). The factory provides
            the "shape baseline" for Phase-3 sweep reuse; individual checks
            override specific params via dataclasses.replace for axis isolation.

    Returns:
        dict with keys: "check1", "check3", "check4", "check5", "all_passed".
        Each check value is {"passed": bool, "details": dict}.
        Does NOT raise on failure — returns the dict so the driver can render a
        full report showing which checks failed and why.

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
    ]:
        passed, details = check_fn(firm_factory)
        results[check_name] = {"passed": passed, "details": details}

    results["all_passed"] = all(
        results[k]["passed"] for k in ("check1", "check3", "check4", "check5")
    )

    return results
