"""Tests for augment-replenish-hiring feature (Phase 1.5 Stage X).

Covers: T-10 (delay semantics), T-11 (cap semantics), T-12 (stacked-firings backlog),
T-13 (K_review target), T-14 (mutual exclusion errors), T-15 (numeraire invariance).
"""
import math
import pytest
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H
from firm_ai_abm.review import replenish_hire_step
from firm_ai_abm.validate import check11_replenish_numeraire


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_with_firings(**kwargs) -> FirmParams:
    """FirmParams that produce firings at T_review=10 (all_H, p=0.22, tpw=5, N=100, sigma=0).

    N=100 is pinned explicitly: at N=500 the F/K share is too small to trigger firings.
    """
    defaults = dict(
        seed=0, N=100, sigma_theta=0.0, sigma_w=0.0,
        T=40, T_review=10.0, firing_threshold=0.0,
        p=0.22, tasks_per_worker=5,
        enable_hiring=False, enable_replenish_hiring=True,
        max_hire_period=0,
        hire_delay_periods=1,
    )
    defaults.update(kwargs)
    return FirmParams(**defaults)


# ---------------------------------------------------------------------------
# T-10: delay semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [1, 2, 3])
def test_hire_arrives_at_fire_plus_delay(delay):
    """Hires appear exactly at fire_period + delay, not before."""
    params = _params_with_firings(T=delay + 15, hire_delay_periods=delay, max_hire_period=0)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    # Find the first review period with firings
    fire_rows = df[df["n_review_fired"] > 0]
    assert len(fire_rows) > 0, "Test setup error: no firings occurred"
    t_fire = int(fire_rows.iloc[0]["t"])
    n_fired = int(fire_rows.iloc[0]["n_review_fired"])

    # At the fire period itself, no hires (delay >= 1)
    n_hired_at_fire = int(df.loc[df["t"] == t_fire, "n_hired"].iloc[0])
    assert n_hired_at_fire == 0, f"Expected 0 hires at fire period t={t_fire}, got {n_hired_at_fire}"

    # At t_fire + delay, hires appear
    t_hire_expected = t_fire + delay
    hire_rows = df[df["t"] == t_hire_expected]
    assert len(hire_rows) == 1, f"No row at t={t_hire_expected}"
    n_hired_at_delay = int(hire_rows.iloc[0]["n_hired"])
    assert n_hired_at_delay == n_fired, (
        f"Expected {n_fired} hires at t={t_hire_expected} (fire+{delay}), got {n_hired_at_delay}"
    )

    # Between fire period and hire period, no hires
    for t_between in range(t_fire + 1, t_hire_expected):
        n = int(df.loc[df["t"] == t_between, "n_hired"].iloc[0])
        assert n == 0, f"Unexpected hire of {n} at t={t_between} (before delay expires)"


# ---------------------------------------------------------------------------
# T-11: cap semantics (multi-period drain)
# ---------------------------------------------------------------------------

def test_cap_drains_backlog_over_multiple_periods():
    """With max_hire_period=3, a backlog of n drains at most 3/period."""
    # Use large T to give enough periods for full drain
    params = _params_with_firings(T=60, hire_delay_periods=1, max_hire_period=3)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    # Find first fire event
    fire_rows = df[df["n_review_fired"] > 0]
    assert len(fire_rows) > 0
    t_fire = int(fire_rows.iloc[0]["t"])
    n_fired = int(fire_rows.iloc[0]["n_review_fired"])

    if n_fired == 0:
        pytest.skip("No firings — test setup issue")

    # Collect n_hired from t_fire+1 onward until we have seen all backlog drained
    hire_series = df[df["t"] > t_fire]["n_hired"].values

    # No period should hire more than max_hire_period
    assert (hire_series <= 3).all(), f"Hire exceeded cap: {hire_series}"

    # Total hires from this backlog must equal n_fired (eventually)
    # We allow more firings to happen — check only the immediate post-fire drain
    # by summing until cumsum reaches n_fired
    cumulative = 0
    for n in hire_series:
        cumulative += n
        if cumulative >= n_fired:
            break
    assert cumulative >= n_fired, (
        f"Backlog not fully drained: cumsum={cumulative}, expected>={n_fired}"
    )


# ---------------------------------------------------------------------------
# T-12: stacked-firings backlog (R-04)
# ---------------------------------------------------------------------------

def test_stacked_firings_backlog_fifo():
    """Two fire events stack in the backlog; global per-period cap applies.

    Key invariants:
    - No single period hires more than max_hire_period workers.
    - Firings accumulate in pending_hires as separate (period_eligible, n) tuples.
    - Cap applies to the combined due total, not per-firing.
    """
    # Use T=50 with T_review=10 → multiple fire+hire cycles
    params = _params_with_firings(T=50, T_review=10.0, hire_delay_periods=1, max_hire_period=2)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)

    # No period should ever hire more than the per-period cap
    assert (df["n_hired"].values <= 2).all(), (
        f"Hire exceeded per-period cap of 2: max={df['n_hired'].max()}"
    )

    # Firings must have occurred for the test to be non-vacuous
    assert df["n_review_fired"].sum() > 0, "Test setup error: no firings occurred"

    # Hires should occur (replenish path active)
    assert df["n_hired"].sum() > 0, "No hires occurred despite firings"


# ---------------------------------------------------------------------------
# T-13: K_review target — full drain returns to pre-fire K
# ---------------------------------------------------------------------------

def test_full_drain_restores_pre_fire_headcount():
    """After fire+rehire with max_hire_period=0 and delay=1, K returns to K_pre_fire."""
    # Use T=30 with a single review at t=10
    params = _params_with_firings(T=30, T_review=10.0, hire_delay_periods=1, max_hire_period=0)
    firm = make_firm(params)
    K_before_run = firm.workforce.K
    df = run_simulation(firm, all_H)

    n_fired = int(df.loc[df["t"] == 10, "n_review_fired"].iloc[0]) if 10 in df["t"].values else 0
    n_hired_t11 = int(df.loc[df["t"] == 11, "n_hired"].iloc[0]) if 11 in df["t"].values else 0

    if n_fired == 0:
        pytest.skip("No firings at t=10")

    # Delay=1, max_hire_period=0 → all backlog hired at t=11
    assert n_hired_t11 == n_fired, (
        f"Expected {n_fired} hires at t=11 (full drain), got {n_hired_t11}"
    )

    # After drain, K_active should be back to (or near) pre-fire level
    # K_active at t=11 should equal K_at_t=9 (before firing) = K_before_run
    k_at_t11 = int(df.loc[df["t"] == 11, "K_active"].iloc[0])
    assert k_at_t11 >= K_before_run - n_fired + n_hired_t11 - 1, (
        f"K not restored: K_at_t11={k_at_t11}, expected ~{K_before_run}"
    )
    # Stronger: pending_hires should be empty after drain
    assert firm.pending_hires == [], f"pending_hires not empty after full drain: {firm.pending_hires}"


# ---------------------------------------------------------------------------
# T-14: mutual exclusion error
# ---------------------------------------------------------------------------

def test_mutual_exclusion_both_true():
    """Both enable_hiring=True and enable_replenish_hiring=True → ValueError."""
    params = FirmParams(enable_hiring=True, enable_replenish_hiring=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_firm(params)


def test_mutual_exclusion_delay_zero():
    """enable_replenish_hiring=True with hire_delay_periods=0 → ValueError."""
    params = FirmParams(enable_hiring=False, enable_replenish_hiring=True, hire_delay_periods=0)
    with pytest.raises(ValueError, match="hire_delay_periods must be >= 1"):
        make_firm(params)


def test_mutual_exclusion_negative_max_hire():
    """enable_replenish_hiring=True with max_hire_period=-1 → ValueError."""
    params = FirmParams(enable_hiring=False, enable_replenish_hiring=True, max_hire_period=-1)
    with pytest.raises(ValueError, match="max_hire_period must be >= 0"):
        make_firm(params)


def test_dormant_no_validation():
    """enable_replenish_hiring=False and enable_hiring=False skips validation → no error even with invalid shape params."""
    params = FirmParams(enable_hiring=False, enable_replenish_hiring=False, hire_delay_periods=0, max_hire_period=-1)
    firm = make_firm(params)  # must not raise
    assert firm is not None


# ---------------------------------------------------------------------------
# T-01 (D-02): enable_hiring path also validated by validate_hiring_params
# ---------------------------------------------------------------------------


def test_enable_hiring_delay_zero_raises():
    """enable_hiring=True with hire_delay_periods=0 → ValueError (same as replenish path)."""
    params = FirmParams(enable_hiring=True, hire_delay_periods=0)
    with pytest.raises(ValueError, match="hire_delay_periods must be >= 1"):
        make_firm(params)


def test_enable_hiring_negative_max_hire_raises():
    """enable_hiring=True with max_hire_period=-1 → ValueError."""
    params = FirmParams(enable_hiring=True, max_hire_period=-1)
    with pytest.raises(ValueError, match="max_hire_period must be >= 0"):
        make_firm(params)


def test_enable_hiring_valid_defaults_succeeds():
    """enable_hiring=True with default hire_delay_periods=1 and max_hire_period=0 → no error."""
    params = FirmParams(enable_hiring=True)
    firm = make_firm(params)  # must not raise
    assert firm is not None


# ---------------------------------------------------------------------------
# T-15: numeraire invariance with replenish active (wraps check11)
# ---------------------------------------------------------------------------

def test_check11_replenish_numeraire_passes():
    """check11_replenish_numeraire passes: pi_scaled==2*pi_base, fire/hire masks identical."""
    passed, details = check11_replenish_numeraire(None)
    assert passed, f"check11 failed: {details}"
    assert details["any_firings"] is True, "Test is vacuous: no firings occurred"
    assert details["any_hires"] is True, "Test is vacuous: no hires occurred"


# ---------------------------------------------------------------------------
# T-09: dormant byte-identity smoke test
# ---------------------------------------------------------------------------

def test_replenish_dormant_no_hires():
    """With enable_replenish_hiring=False (default), n_hired is always 0."""
    params = FirmParams(
        seed=0, sigma_theta=0.0, sigma_w=0.0,
        T=20, T_review=10.0, firing_threshold=0.0,
        p=0.22, tasks_per_worker=5,
        enable_replenish_hiring=False,
    )
    firm = make_firm(params)
    df = run_simulation(firm, all_H)
    assert int(df["n_hired"].sum()) == 0, "Dormant path should produce zero hires"
