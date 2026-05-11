"""T-13: Worked-example unit tests for the training-delay feature.

All tests use enable_training_delay=True explicitly.
Dormant-path byte-parity test verifies enable_training_delay=False is unchanged.
"""
import numpy as np
import pytest

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.production import Mode, productivity_vec, cost_vec
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_A, all_H


def _make_delay_params(**kwargs) -> FirmParams:
    """Return FirmParams suitable for training-delay tests.

    Uses tpw=10, N=10, K=1, sigma=0 to give a deterministic single-worker setup.
    """
    defaults = dict(
        N=10,
        tasks_per_worker=10,
        sigma_theta=0.0,
        sigma_w=0.0,
        g=0.5,
        q_h=1.0,
        beta_override=None,  # ignored — passed to override beta after make_firm
        seed=0,
        enable_training_delay=True,
    )
    defaults.update(kwargs)
    defaults.pop("beta_override", None)
    return FirmParams(**defaults)


def test_period0_output_no_bonus():
    """T-13: With enable_training_delay=True, period-0 output of all_A == q_h * N (no bonus).

    Setup: tpw=10, N=10, K=1, sigma=0, g=0.5, beta=ones.
    Period 0: worker enters A for first time → a_training_in_progress=True.
    Productivity = q_h * N = 1.0 * 10 = 10.0 (no aug bonus).
    adj_cost = c_train * 1 (one worker trained).
    """
    params = _make_delay_params(c_train=1.0, c_fire=0.0, c_hire=0.0, c_aug=0.0, F=0.0, p=1.0)
    firm = make_firm(params)
    # Set beta=ones for deterministic calculation
    firm.beta[:] = 1.0

    df = run_simulation(firm, all_A)

    Y0 = float(df["Y"].iloc[0])
    assert abs(Y0 - 10.0) < 1e-10, (
        f"Period-0 output with delay should be q_h * N = 10.0, got {Y0}"
    )

    adj0 = float(df["adj_cost"].iloc[0])
    assert abs(adj0 - 1.0) < 1e-10, (
        f"Period-0 adj_cost should be c_train * 1 = 1.0, got {adj0}"
    )

    wage_bill0 = float(df["wage_bill"].iloc[0])
    assert abs(wage_bill0 - params.w) < 1e-10, (
        f"Period-0 wage_bill should be w * 1 = {params.w}, got {wage_bill0}"
    )


def test_period1_output_has_bonus():
    """T-13: Period 1 shows full augmentation output after step-11.5 flip.

    Period 1: a_trained=True, a_training_in_progress=False.
    Y[1] = q_h * (1 + g * 1.0) * N = 1.0 * 1.5 * 10 = 15.0.
    """
    params = _make_delay_params(c_train=1.0, c_fire=0.0, c_hire=0.0, c_aug=0.0, F=0.0, p=1.0, T=2)
    firm = make_firm(params)
    firm.beta[:] = 1.0

    df = run_simulation(firm, all_A)

    Y1 = float(df["Y"].iloc[1])
    expected = params.q_h * (1.0 + params.g * 1.0) * params.N  # 15.0
    assert abs(Y1 - expected) < 1e-10, (
        f"Period-1 output should be {expected}, got {Y1}"
    )


def test_flag_state_after_period0():
    """T-13: After period 0, a_trained=True and a_training_in_progress=False (step-11.5 flip)."""
    params = _make_delay_params(c_train=0.0, c_fire=0.0, c_hire=0.0, F=0.0, p=1.0, T=1)
    firm = make_firm(params)
    firm.beta[:] = 1.0

    run_simulation(firm, all_A)

    assert bool(firm.workforce.a_trained[0]) is True, (
        "a_trained[0] should be True after period 0 with delay"
    )
    assert bool(firm.workforce.a_training_in_progress[0]) is False, (
        "a_training_in_progress[0] should be False after step-11.5 flip"
    )


def test_dormant_path_byte_parity():
    """T-13: enable_training_delay=False is byte-identical to Stage 2 behavior."""
    params_delay = FirmParams(
        N=100, tasks_per_worker=10, sigma_theta=0.0, sigma_w=0.0,
        seed=0, enable_training_delay=True, p=1.0,
    )
    params_no_delay = FirmParams(
        N=100, tasks_per_worker=10, sigma_theta=0.0, sigma_w=0.0,
        seed=0, enable_training_delay=False, p=1.0,
    )

    firm_d = make_firm(params_delay)
    firm_nd = make_firm(params_no_delay)

    # Use all_H — no A transitions, so delay has no effect
    df_d = run_simulation(firm_d, all_H)
    df_nd = run_simulation(firm_nd, all_H)

    for col in ["Y", "C", "pi", "adj_cost", "wage_bill"]:
        assert np.array_equal(df_d[col].values, df_nd[col].values), (
            f"Dormant path parity failed for col={col}: delay vs no-delay differ"
        )


def test_cost_vec_no_aug_during_training():
    """T-13 / MAJ-4: Unit test — cost_vec with in-training A-task returns 0 c_aug."""
    params = FirmParams(
        N=10, tasks_per_worker=10, c_aug=0.05, c_auto=0.4, p=1.0,
        sigma_theta=0.0, sigma_w=0.0, seed=0,
    )
    modes = np.full(10, int(Mode.A), dtype=int)
    a_in_training = np.ones(10, dtype=bool)  # all tasks in training

    cost = cost_vec(modes, params, a_in_training_per_task=a_in_training)
    assert np.allclose(cost, 0.0, atol=1e-15), (
        f"In-training A-tasks should have 0 c_aug, got max={cost.max()}"
    )

    # Partial: half in training, half fully trained
    a_partial = np.zeros(10, dtype=bool)
    a_partial[:5] = True
    cost_partial = cost_vec(modes, params, a_in_training_per_task=a_partial)
    assert np.allclose(cost_partial[:5], 0.0, atol=1e-15), "First 5 tasks in training → 0"
    assert np.allclose(cost_partial[5:], params.c_aug, atol=1e-15), "Last 5 tasks trained → c_aug"
