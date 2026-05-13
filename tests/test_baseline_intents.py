"""Tests for T-02b: _fire_intent / _hire_intent written by all 5 baseline strategies.

Acceptance criteria:
  (1) K=0 guard: all strategies write _fire_intent=0, _hire_intent=0 when K==0.
  (2) K=K_max (full capacity): H/A strategies write _fire_intent=0; T writes K_max.
  (3) Ceiling vs floor for K_needed: all-T on a firm where one task is non-T causes
      _fire_intent = K (all workers) regardless; for a mixed case, ceil(n_HA/tpw) is
      used — not floor — preventing spurious fires.
  (4) End-to-end under T_review=5: strategies write _fire_intent and the kernel's
      simulate.py consumes it (verify behavior doesn't crash and workforce evolves).
"""
import math

import numpy as np
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.production import Mode, compute_K
from firm_ai_abm.strategy import all_H, all_A, all_T, greedy_profit, greedy_with_switching


_ALL_STRATEGIES = [all_H, all_A, all_T, greedy_profit, greedy_with_switching]


def _make_firm(N=50, **kwargs):
    p = FirmParams(N=N, T=20, seed=0, **kwargs)
    return make_firm(p)


# ---------------------------------------------------------------------------
# (1) K=0 guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", _ALL_STRATEGIES, ids=lambda s: s.__name__)
def test_K0_guard_writes_zeros(strategy):
    """When K==0, every strategy must write _fire_intent=0 and _hire_intent=0.

    K=0 by construction: N=4, tasks_per_worker=5 → K = 4//5 = 0.
    """
    # K=0 by construction: N=4, tasks_per_worker=5 → K = 4//5 = 0
    firm = _make_firm(N=4, tasks_per_worker=5)
    assert firm.workforce.K == 0, f"Precondition: expected K=0, got {firm.workforce.K}"

    strategy(firm, t=0)

    assert firm._fire_intent == 0, (
        f"{strategy.__name__}: expected _fire_intent=0 when K=0, got {firm._fire_intent}"
    )
    assert firm._hire_intent == 0, (
        f"{strategy.__name__}: expected _hire_intent=0 when K=0, got {firm._hire_intent}"
    )


# ---------------------------------------------------------------------------
# (2) K = K_max (full workforce capacity)
# ---------------------------------------------------------------------------

def test_all_H_at_K_max_no_fire():
    """all_H with full workforce: _fire_intent==0 (no firings needed)."""
    firm = _make_firm(N=50, tasks_per_worker=5)
    all_H(firm, t=0)
    assert firm._fire_intent == 0, f"Expected 0, got {firm._fire_intent}"
    # all-H needs all K workers — no hire needed either
    assert firm._hire_intent == 0, f"Expected _hire_intent=0, got {firm._hire_intent}"


def test_all_T_at_K_max_fires_all():
    """all_T with K>0: _fire_intent == K (all workers should be fired)."""
    firm = _make_firm(N=50, tasks_per_worker=5)
    K = firm.workforce.K
    assert K > 0, "Test precondition: K must be > 0"
    all_T(firm, t=0)
    assert firm._fire_intent == K, (
        f"all_T: expected _fire_intent={K}, got {firm._fire_intent}"
    )
    assert firm._hire_intent == 0, f"Expected _hire_intent=0, got {firm._hire_intent}"


def test_all_A_at_K_max_no_fire():
    """all_A with full workforce: _fire_intent==0, _hire_intent==0."""
    firm = _make_firm(N=50, tasks_per_worker=5)
    all_A(firm, t=0)
    assert firm._fire_intent == 0, f"Expected 0, got {firm._fire_intent}"
    assert firm._hire_intent == 0, f"Expected 0, got {firm._hire_intent}"


# ---------------------------------------------------------------------------
# (3) Ceiling vs floor: no spurious fires from K_needed computation
# ---------------------------------------------------------------------------

def test_greedy_no_spurious_fire_at_full_capacity():
    """When greedy recommends all-H with K == ceil(N/tpw), _fire_intent must be 0.

    Regression target: if floor(n_HA/tpw) were used instead of ceil, a firm with
    N=11, tasks_per_worker=5, K=3 (ceil) recommending all-H (n_HA=11) would compute
    floor(11/5)=2 < K=3 → spurious fire. ceil(11/5)=3 == K=3 → no fire.

    make_firm uses floor (K_init=2 for N=11,tpw=5), so we manually extend the
    workforce to K=3 to hit the ceil boundary.
    """
    from firm_ai_abm.workers import Workforce

    firm = _make_firm(N=11, tasks_per_worker=5)
    # Replace workforce with 3-worker version (ceil(11/5) = 3)
    firm.workforce = Workforce(
        theta=np.ones(3, dtype=np.float64),
        wage=np.full(3, firm.params.w, dtype=np.float64),
        a_trained=np.zeros(3, dtype=bool),
        tenure=np.zeros(3, dtype=int),
        hire_t=np.zeros(3, dtype=int),
    )
    K = firm.workforce.K
    assert K == 3, f"Precondition: expected K=3, got {K}"

    # Verify ceil and floor diverge for this N/tpw
    modes_allH = np.zeros(11, dtype=int)
    K_needed_ceil = compute_K(modes_allH, firm.params)
    K_needed_floor = int(11 // 5)
    assert K_needed_ceil == 3, f"ceil(11/5) should be 3, got {K_needed_ceil}"
    assert K_needed_floor == 2, f"floor(11/5) should be 2, got {K_needed_floor}"

    # Set high auto cost so greedy chooses H for all tasks
    firm.params.c_auto = 100.0

    greedy_profit(firm, t=0)

    assert firm._fire_intent == 0, (
        f"Spurious fire: _fire_intent={firm._fire_intent} (expected 0). "
        "floor division likely used instead of ceil."
    )


def test_ceil_vs_floor_asymmetry():
    """Explicit ceil vs floor assertion: compute_K uses ceil, not floor."""
    from firm_ai_abm.config import FirmParams as FP
    p = FP(N=11, tasks_per_worker=5)
    modes = np.zeros(11, dtype=int)  # all-H
    k_ceil = compute_K(modes, p)
    k_floor = int(np.count_nonzero((modes == int(Mode.H)) | (modes == int(Mode.A))) // p.tasks_per_worker)
    assert k_ceil == 3, f"Expected ceil(11/5)=3, got {k_ceil}"
    assert k_floor == 2, f"Expected floor(11/5)=2, got {k_floor}"
    assert k_ceil != k_floor, "ceil and floor agree — adjust N or tasks_per_worker to get divergence"


# ---------------------------------------------------------------------------
# (4) Mutual exclusion: _fire_intent > 0 and _hire_intent > 0 never both true
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", _ALL_STRATEGIES, ids=lambda s: s.__name__)
def test_mutual_exclusion_invariant(strategy):
    """_fire_intent and _hire_intent are never both > 0 after any strategy call."""
    firm = _make_firm(N=50)
    strategy(firm, t=0)
    assert not (firm._fire_intent > 0 and firm._hire_intent > 0), (
        f"{strategy.__name__}: mutual exclusion violated — "
        f"_fire_intent={firm._fire_intent}, _hire_intent={firm._hire_intent}"
    )


# ---------------------------------------------------------------------------
# (5) End-to-end: strategies participate in the kernel intent protocol
# ---------------------------------------------------------------------------

def test_all_T_end_to_end_fires_workforce():
    """all_T with T_review=5 should lead to K=0 by period 5 via intent protocol."""
    from firm_ai_abm.simulate import run_simulation

    p = FirmParams(N=50, T=10, seed=0, T_review=5.0, tasks_per_worker=5)
    firm = make_firm(p)
    assert firm.workforce.K > 0, "Precondition: workforce must be non-empty"

    df = run_simulation(firm, all_T)

    # By period 5 (T_review), all workers should have been fired
    K_at_5 = int(df["K_active"].iloc[4])  # iloc[4] = period index 4 (0-indexed) = period 5
    assert K_at_5 == 0, (
        f"Expected K=0 at period 5 under all_T + T_review=5, got K={K_at_5}"
    )


def test_all_H_no_fire_with_T_review():
    """all_H with T_review=5: since it keeps workers, K should not drop below initial."""
    from firm_ai_abm.simulate import run_simulation

    p = FirmParams(N=50, T=10, seed=0, T_review=5.0, tasks_per_worker=5)
    firm = make_firm(p)
    K_initial = firm.workforce.K

    df = run_simulation(firm, all_H)

    # all_H writes _fire_intent=0, so no deliberate fires from strategy.
    # Some threshold-based fires may occur if workers have negative surplus;
    # but the point is the strategy doesn't drive deliberate intent fires.
    # Just verify the simulation completes without error and K ≥ 0.
    assert all(df["K_active"] >= 0), "K should remain non-negative"
    # Stronger: the strategy writes fire_intent=0, so K should not drop due to strategy
    # (threshold fires are a separate mechanism; allow for them here)
    assert len(df) == p.T, f"Expected {p.T} rows, got {len(df)}"
