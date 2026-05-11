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
    """T-14: firing_threshold=0.0 at default params produces no firings."""
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=10.0, firing_threshold=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)
    total_fired = int(df["n_review_fired"].sum())
    assert total_fired == 0, (
        f"firing_threshold=0.0 should produce no firings at default params, "
        f"but {total_fired} workers were fired."
    )
