"""Stage 2 training-memory tests: T-04..T-10.

Tests the per-worker training gate introduced in Stage 2:
  - count_workers_entering_a_first_time helper (T-05)
  - adj_cost per-worker idempotence (T-06)
  - oscillating H/A/H/A charges once (T-07)
  - high-cost blocking (T-08)
  - full Tier-A + Tier-B regression (T-09)
  - n_a_trained history column monotonicity (T-10)
  - legacy 3-arg byte-parity (T-04)

Run with: .venv/bin/python -m pytest tests/test_training_memory.py -v
"""
import math

import numpy as np
import pytest
from dataclasses import replace

from firm_ai_abm.adjustment import adj_cost, count_workers_entering_a_first_time
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.production import Mode
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import greedy_with_switching
from firm_ai_abm.workers import Workforce, sample_workforce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workforce(K: int, a_trained_mask: np.ndarray | None = None) -> Workforce:
    """Build a minimal Workforce with a_trained all-False (or given mask)."""
    theta = np.ones(K, dtype=np.float64)
    wage = np.ones(K, dtype=np.float64)
    if a_trained_mask is None:
        a_trained = np.zeros(K, dtype=bool)
    else:
        a_trained = a_trained_mask.astype(bool).copy()
    tenure = np.zeros(K, dtype=int)
    hire_t = np.zeros(K, dtype=int)
    return Workforce(theta=theta, wage=wage, a_trained=a_trained, tenure=tenure, hire_t=hire_t)


def _make_params(N: int = 100, K_implicit: int = 10, **kwargs) -> FirmParams:
    """Return FirmParams with tasks_per_worker = N // K_implicit and given overrides.

    Pins p=1.0 and T_review=math.inf by default to preserve Stage 1–2 fixtures.
    T_review=math.inf: adj_cost charges fire_cost only when T_review=inf (finite T_review
    delegates firing to the periodic review gate — adj_cost would double-charge otherwise).
    """
    tpw = N // K_implicit
    defaults = {"p": 1.0, "T_review": math.inf}
    defaults.update(kwargs)
    return FirmParams(N=N, tasks_per_worker=tpw, sigma_theta=0.0, sigma_w=0.0, **defaults)


# ---------------------------------------------------------------------------
# T-05: count_workers_entering_a_first_time helper correctness
# ---------------------------------------------------------------------------

class TestCountWorkersEnteringAFirstTime:
    """T-05: five fixture cases from the plan acceptance criteria."""

    def test_all_h_to_all_h_returns_zero(self):
        """all-H → all-H: no worker enters A, returns (0, empty)."""
        params = _make_params()
        wf = _make_workforce(10)
        prev = np.zeros(100, dtype=int)
        new = np.zeros(100, dtype=int)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 0
        assert indices.size == 0

    def test_all_h_to_all_a_all_untrained(self):
        """all-H → all-A with 10 untrained workers: returns (10, [0..9])."""
        params = _make_params()
        wf = _make_workforce(10)
        prev = np.zeros(100, dtype=int)
        new = np.full(100, int(Mode.A), dtype=int)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 10
        assert np.array_equal(indices, np.arange(10))

    def test_all_h_to_all_a_half_trained(self):
        """all-H → all-A with workers 0-4 already trained: returns (5, [5..9])."""
        params = _make_params()
        a_trained = np.zeros(10, dtype=bool)
        a_trained[:5] = True
        wf = _make_workforce(10, a_trained_mask=a_trained)
        prev = np.zeros(100, dtype=int)
        new = np.full(100, int(Mode.A), dtype=int)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 5
        assert np.array_equal(indices, np.arange(5, 10))

    def test_mixed_single_worker_untrained(self):
        """Only worker 3 has an A-task in new_modes and is untrained: returns (1, [3])."""
        params = _make_params()
        wf = _make_workforce(10)
        prev = np.zeros(100, dtype=int)
        new = np.zeros(100, dtype=int)
        # Worker 3 covers tasks [30, 40) — set them all to A
        new[30:40] = int(Mode.A)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 1
        assert np.array_equal(indices, np.array([3]))

    def test_mixed_single_worker_already_trained(self):
        """Only worker 3 has an A-task, but already trained: returns (0, empty)."""
        params = _make_params()
        a_trained = np.zeros(10, dtype=bool)
        a_trained[3] = True
        wf = _make_workforce(10, a_trained_mask=a_trained)
        prev = np.zeros(100, dtype=int)
        new = np.zeros(100, dtype=int)
        new[30:40] = int(Mode.A)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 0
        assert indices.size == 0

    def test_k0_all_t_returns_zero(self):
        """K=0, all-T modes: returns (0, empty) without AssertionError."""
        params = FirmParams(N=100, tasks_per_worker=10, sigma_theta=0.0, sigma_w=0.0, p=1.0)
        wf = _make_workforce(0)
        prev = np.full(100, int(Mode.T), dtype=int)
        new = np.full(100, int(Mode.T), dtype=int)
        n, indices = count_workers_entering_a_first_time(prev, new, wf, params)
        assert n == 0
        assert indices.size == 0


# ---------------------------------------------------------------------------
# T-06: adj_cost idempotence per worker (D-02)
# ---------------------------------------------------------------------------

class TestAdjCostIdempotentPerWorker:
    """T-06: second call with already-trained workforce returns 0 training cost."""

    def test_idempotence_training_cost(self):
        """First call charges c_train * K; second call charges 0 for training."""
        params = _make_params(c_train=0.1, c_fire=0.0, c_hire=0.0)
        wf = _make_workforce(10)
        prev = np.zeros(100, dtype=int)
        new = np.full(100, int(Mode.A), dtype=int)

        # First call: charges training for all 10 workers
        cost1 = adj_cost(prev, new, params, workforce=wf)
        # K_prev (all-H) = 10, K_new (all-A) = 10 → no hire/fire
        assert cost1 == pytest.approx(0.1 * 10, rel=1e-12)

        # After first call, all workers are trained
        assert wf.a_trained.all(), "a_trained should be all True after first call"

        # Second call: no new workers enter A → training cost = 0
        cost2 = adj_cost(prev, new, params, workforce=wf)
        assert cost2 == pytest.approx(0.0, abs=1e-12), (
            f"Second call training cost should be 0.0, got {cost2}"
        )

    def test_idempotence_does_not_affect_hire_fire(self):
        """Lumpy hire/fire portion is unaffected by a_trained state."""
        params = _make_params(c_train=0.1, c_fire=5.0, c_hire=5.0)
        wf = _make_workforce(10)
        all_H_modes = np.zeros(100, dtype=int)
        all_T_modes = np.full(100, int(Mode.T), dtype=int)

        # H → T: K drops from 10 to 0, fire_cost = c_fire * 10 = 50.0; no training (no A)
        cost = adj_cost(all_H_modes, all_T_modes, params, workforce=wf)
        assert cost == pytest.approx(5.0 * 10, rel=1e-12)


# ---------------------------------------------------------------------------
# T-07: oscillating H→A→H→A charges training only once
# ---------------------------------------------------------------------------

class TestAdjCostOscillationChargesOnce:
    """T-07: 4-period H/A/H/A with K=1 charges c_train only on the first H→A."""

    def test_oscillation_charges_once(self):
        """
        Period 0: all-H (start)
        Period 1: all-H → all-A  (first H→A — charges c_train * 1)
        Period 2: all-A → all-H  (A→H — no training cost)
        Period 3: all-H → all-A  (H→A again, but worker already trained — 0 cost)
        """
        params = _make_params(N=10, K_implicit=1, c_train=0.1, c_fire=0.0, c_hire=0.0)
        wf = _make_workforce(1)

        modes_H = np.zeros(10, dtype=int)
        modes_A = np.full(10, int(Mode.A), dtype=int)

        # Period 1: H → A (first entry)
        cost1 = adj_cost(modes_H, modes_A, params, workforce=wf)
        assert cost1 == pytest.approx(0.1, rel=1e-12), f"Expected 0.1, got {cost1}"
        assert wf.a_trained[0] is True or wf.a_trained[0] == True, (
            "Worker 0 should be trained after first H→A"
        )

        # Period 2: A → H (un-training doesn't cost, a_trained stays True)
        cost2 = adj_cost(modes_A, modes_H, params, workforce=wf)
        training_portion2 = cost2  # c_fire=c_hire=0, so entire cost is training
        assert training_portion2 == pytest.approx(0.0, abs=1e-12), (
            f"A→H should have 0 training cost, got {cost2}"
        )
        assert wf.a_trained[0] == True, "a_trained should remain True after A→H"

        # Period 3: H → A again (worker already trained → 0 training cost)
        cost3 = adj_cost(modes_H, modes_A, params, workforce=wf)
        assert cost3 == pytest.approx(0.0, abs=1e-12), (
            f"Second H→A should have 0 training cost, got {cost3}"
        )


# ---------------------------------------------------------------------------
# T-08: check6 per-worker blocking (greedy stays all-H under high costs)
# ---------------------------------------------------------------------------

class TestCheck6PerWorkerBlocking:
    """T-08: greedy_with_switching stays all-H when c_train=c_fire=c_hire=100."""

    def test_greedy_stays_all_h_under_high_costs(self):
        """max(adj_cost) == 0 over T=60 when switching costs are extreme."""
        params = FirmParams(
            N=100,
            T=60,
            tasks_per_worker=10,
            c_train=100.0,
            c_fire=100.0,
            c_hire=100.0,
            sigma_theta=0.0,
            sigma_w=0.0,
            seed=42,
            p=1.0,
        )
        firm = make_firm(params)
        df = run_simulation(firm, greedy_with_switching)
        max_adj = df["adj_cost"].max()
        assert max_adj == 0.0, (
            f"Expected no switching under c_train=c_fire=c_hire=100, "
            f"but max adj_cost = {max_adj}"
        )


# ---------------------------------------------------------------------------
# T-09: full Tier-A + Tier-B regression via run_all_checks()
# ---------------------------------------------------------------------------

class TestTierAFullSuitePasses:
    """T-09: run_all_checks() returns all_passed=True under Stage 2 semantics."""

    def test_all_checks_pass(self):
        """All check1..check7 must pass (Tier-A and Tier-B) after Stage 2 changes."""
        from firm_ai_abm.validate import run_all_checks
        result = run_all_checks()
        # Report details on failure for easier diagnosis
        failures = []
        for tier_key, tier_val in result.items():
            if tier_key == "all_passed":
                continue
            if isinstance(tier_val, dict):
                for check_key, check_val in tier_val.items():
                    if check_key == "all_passed":
                        continue
                    if isinstance(check_val, dict) and not check_val.get("passed", True):
                        failures.append(f"{tier_key}/{check_key}: {check_val}")
        assert result["all_passed"] is True, (
            f"run_all_checks() returned all_passed=False. Failing checks: {failures}"
        )


# ---------------------------------------------------------------------------
# T-04: legacy 3-arg byte-parity (D-04 backward-compat)
# ---------------------------------------------------------------------------

class TestAdjCostLegacyThreeArgByteParity:
    """T-04: calling adj_cost without workforce returns Phase 1 per-task result."""

    def test_legacy_no_workforce_h_to_a(self):
        """Phase 1 path: c_train charged per H→A task, not per worker."""
        params = FirmParams(
            N=100,
            tasks_per_worker=10,
            c_train=0.1,
            c_fire=0.0,
            c_hire=0.0,
            sigma_theta=0.0,
            sigma_w=0.0,
            p=1.0,
        )
        prev = np.zeros(100, dtype=int)      # all H
        new = np.full(100, int(Mode.A), dtype=int)  # all A

        # Legacy 3-arg call (D-04)
        cost_legacy = adj_cost(prev, new, params)

        # Per-task: 100 tasks × c_train = 10.0
        assert cost_legacy == pytest.approx(0.1 * 100, rel=1e-12), (
            f"Legacy 3-arg cost expected {0.1 * 100}, got {cost_legacy}"
        )

    def test_legacy_no_workforce_no_switching_zero_cost(self):
        """Phase 1 path: same modes → 0 cost."""
        params = FirmParams(
            N=100, tasks_per_worker=10, c_train=0.5, c_fire=1.0, c_hire=1.0,
            sigma_theta=0.0, sigma_w=0.0, p=1.0,
        )
        modes = np.zeros(100, dtype=int)
        cost = adj_cost(modes, modes, params)
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_legacy_vs_per_worker_differ_on_repeated_transitions(self):
        """Legacy and per-worker paths agree on first call; differ on second (idempotence gap)."""
        params = _make_params(c_train=0.1, c_fire=0.0, c_hire=0.0)
        prev = np.zeros(100, dtype=int)
        new = np.full(100, int(Mode.A), dtype=int)

        # Legacy (Phase 1): always charges per task regardless of prior calls
        cost_legacy_1 = adj_cost(prev, new, params)
        cost_legacy_2 = adj_cost(prev, new, params)
        # Legacy charges c_train * n_HA every time (100 tasks × 0.1 = 10.0)
        assert cost_legacy_1 == cost_legacy_2 == pytest.approx(0.1 * 100, rel=1e-12)

        # Per-worker (Stage 2): second call charges 0 (workers already trained)
        wf = _make_workforce(10)
        cost_wf_1 = adj_cost(prev, new, params, workforce=wf)
        cost_wf_2 = adj_cost(prev, new, params, workforce=wf)
        assert cost_wf_1 == pytest.approx(0.1 * 10, rel=1e-12)   # 10 workers
        assert cost_wf_2 == pytest.approx(0.0, abs=1e-12)         # already trained


# ---------------------------------------------------------------------------
# T-10: n_a_trained history column weakly non-decreasing
# ---------------------------------------------------------------------------

class TestNATrainedMonotoneNondecreasing:
    """T-10: n_a_trained column in run_simulation history is weakly increasing."""

    def test_n_a_trained_monotone_under_free_training(self):
        """c_train=0 → greedy switches to A freely; n_a_trained is weakly non-decreasing."""
        params = FirmParams(
            N=100,
            T=60,
            tasks_per_worker=10,
            c_train=0.0,
            c_fire=0.0,
            c_hire=0.0,
            sigma_theta=0.0,
            sigma_w=0.0,
            seed=42,
            p=1.0,
        )
        firm = make_firm(params)
        df = run_simulation(firm, greedy_with_switching)

        assert "n_a_trained" in df.columns, "n_a_trained column missing from history"

        n_trained = df["n_a_trained"].values
        diffs = np.diff(n_trained)
        assert (diffs >= 0).all(), (
            f"n_a_trained decreased in some period: min diff = {diffs.min()}"
        )

    def test_n_a_trained_final_equals_k_when_all_switch(self):
        """When greedy switches all tasks to A, final n_a_trained == K."""
        # With very high q_a and zero switching costs, greedy picks A for all tasks
        params = FirmParams(
            N=100,
            T=60,
            tasks_per_worker=10,
            q_h=1.0,
            q_a=5.0,       # very high → T is best, but c_auto is high → A wins
            g=2.0,          # strong augmentation gain → all-A clearly best
            c_aug=0.0,
            c_auto=999.0,   # T is prohibitively expensive → greedy picks A
            c_train=0.0,
            c_fire=0.0,
            c_hire=0.0,
            sigma_theta=0.0,
            sigma_w=0.0,
            seed=42,
            p=1.0,
        )
        K = params.N // params.tasks_per_worker  # = 10
        firm = make_firm(params)
        df = run_simulation(firm, greedy_with_switching)

        final_n_a_trained = int(df["n_a_trained"].iloc[-1])
        # With free training + all-A best, by end all K workers should be trained
        assert final_n_a_trained == K, (
            f"Expected all {K} workers trained by end, got n_a_trained={final_n_a_trained}"
        )

    def test_n_a_trained_column_present_for_all_strategies(self):
        """n_a_trained column is populated for static strategies too (all_H, all_A, all_T)."""
        from firm_ai_abm.strategy import all_H, all_A, all_T
        params = FirmParams(
            N=100, T=10, tasks_per_worker=10,
            c_train=0.1, c_fire=0.0, c_hire=0.0,
            sigma_theta=0.0, sigma_w=0.0, seed=1, p=1.0,
        )
        for strategy, name in [(all_H, "all_H"), (all_A, "all_A"), (all_T, "all_T")]:
            firm = make_firm(params)
            df = run_simulation(firm, strategy)
            assert "n_a_trained" in df.columns, f"n_a_trained missing for {name}"
            assert len(df) == 10, f"Expected 10 rows for {name}"
