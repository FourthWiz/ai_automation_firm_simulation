"""T-14: Default recalibration impact tests.

Verifies that the new defaults (tpw=5, p=0.22, F=5.0) produce the
intentional negative all-H baseline and that firing_threshold=0 works.
"""
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H, greedy_profit


def test_default_human_baseline_is_negative():
    """T-14: At new defaults (tpw=5, p=0.22), all-H mean profit is negative.

    At tpw=5, p=0.22, N=100, K=20, w=1, F=5.0:
    per-period all-H profit ≈ 0.22*100 − 1*20 − 5.0 = −3.0.
    The negative baseline is intentional — human-only strategy is loss-making
    under AI-era params, so augment/automate strategies are materially better.
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)
    mean_pi = float(df["pi"].mean())
    assert mean_pi < 0, (
        f"Default all-H baseline should be negative (intended economic framing). "
        f"mean_pi={mean_pi:.4f}. If this fails, recalibration may have drifted."
    )


def test_default_firing_threshold_fires_all_H_at_tpw5():
    """Canary: at default params (all_H strategy, sigma=0), the F/K share tips H-mode workers
    into negative-surplus territory. ALL workers fire at the first review.

    Closed-form surplus (decision-thirteen):
      effective_surplus = p * mean_output - wage - mean_aug_cost - F/K_review
                        = 0.22 * 5.0 - 1.0 - 0.0 - 5.0/20
                        = 1.1 - 1.0 - 0.25 = -0.15 < 0

    total_fired >= K0 confirms the F/K_review term is active. If this test fails
    post-change, the F_share term may be missing from the surplus formula.
    Uses all_H (not greedy_profit) for closed-form arithmetic.
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=10.0, firing_threshold=0.0)
    firm = make_firm(params)
    K0 = firm.K0  # 20 at default (N=100, tpw=5)
    df = run_simulation(firm, all_H)
    total_fired = int(df["n_review_fired"].sum())
    assert total_fired >= K0, (
        f"Expected all {K0} H-mode workers to fire (surplus=-0.15 per worker). "
        f"Got total_fired={total_fired}. The F/K_review term may be missing."
    )


def test_default_firing_threshold_greedy_fires():
    """Weak floor: greedy_profit at default params also produces firings due to F/K share.

    Exact count depends on mode assignment; >= 1 is the floor. For the closed-form
    canary, see test_default_firing_threshold_fires_all_H_at_tpw5.
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=10.0, firing_threshold=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)
    total_fired = int(df["n_review_fired"].sum())
    assert total_fired >= 1, (
        f"Expected at least 1 firing with greedy_profit at default params (F/K share active). "
        f"Got {total_fired}."
    )


def test_enable_hiring_default_is_false():
    """T-02: enable_hiring defaults to False (dormant per lessons-learned 2026-05-09)."""
    assert FirmParams().enable_hiring is False, (
        "enable_hiring must default to False to preserve fixture byte-parity and opt-in semantics"
    )
