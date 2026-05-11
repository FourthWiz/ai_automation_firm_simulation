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


def test_default_firing_threshold_works_at_tpw5():
    """T-14: firing_threshold=0.0 at default params produces no firings.

    Post-fix surplus formula: p × mean_output − wage = 0.22 × 5.0 − 1.0 = 0.1 per worker
    (using sigma_theta=0, sigma_w=0 so all workers have theta=1, wage=1.0, H-mode output=5.0).
    Margin from fire boundary = 0.1. Canary: if p, q_h, or w changes such that
    p × q_h × tpw − w ≤ 0, this test flips.
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=10.0, firing_threshold=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)
    total_fired = int(df["n_review_fired"].sum())
    expected_surplus_margin = 0.1  # p * q_h * tpw - w = 0.22 * 5 - 1
    assert total_fired == 0, (
        f"firing_threshold=0.0 should produce no firings at default params "
        f"(surplus margin={expected_surplus_margin}), but {total_fired} workers were fired."
    )


def test_enable_hiring_default_is_false():
    """T-02: enable_hiring defaults to False (dormant per lessons-learned 2026-05-09)."""
    assert FirmParams().enable_hiring is False, (
        "enable_hiring must default to False to preserve fixture byte-parity and opt-in semantics"
    )
