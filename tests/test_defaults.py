"""T-14: Default recalibration impact tests.

Verifies that the new defaults (tpw=5, p=0.22, F=5.0, N=500) produce the
expected baseline and that firing_threshold=0 works.

At N=500 (K=100), the all-H per-period profit is mildly positive:
  0.22*500 − 1*100 − 5.0 = +5.0/period
The previous "intentionally negative" framing (N=100/K=20 → -3/period) no
longer applies at the new default scale. See D-01 in defaults-n500-hire-delay.
"""
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H, greedy_profit


def test_default_human_baseline_is_mildly_positive():
    """At new defaults (N=500, tpw=5, p=0.22, K=100, F=5.0), all-H mean profit is ≈ +5.0.

    Closed-form: per-period pi = 0.22*500 − 1*100 − 5.0 = 5.0.
    atol=0.2 catches recalibration drift larger than ~4% (sigma>0 adds noise).
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0)
    firm = make_firm(params)
    df = run_simulation(firm, all_H)
    mean_pi = float(df["pi"].mean())
    assert np.isclose(mean_pi, 5.0, atol=0.2), (
        f"Default all-H baseline should be ≈ +5.0 at N=500. "
        f"mean_pi={mean_pi:.4f}. If this fails, N default or price may have changed."
    )


def test_default_firing_threshold_no_fires_all_H_at_tpw5():
    """At default params (N=500, all_H, sigma=0), per-worker surplus is positive → no firings.

    Closed-form surplus at N=500/K=100 (decision-thirteen):
      effective_surplus = p * mean_output - wage - mean_aug_cost - F/K_review
                        = 0.22 * 5.0 - 1.0 - 0.0 - 5.0/100
                        = 1.1 - 1.0 - 0.05 = +0.05 > 0

    All workers survive; total_fired == 0. If this test fails, the F/K_review
    term may be incorrectly scaled or N default has changed.
    """
    params = FirmParams(seed=0, sigma_theta=0.0, sigma_w=0.0, T_review=10.0, firing_threshold=0.0)
    firm = make_firm(params)
    K0 = firm.K0  # 100 at new default (N=500, tpw=5)
    df = run_simulation(firm, all_H)
    total_fired = int(df["n_review_fired"].sum())
    assert total_fired == 0, (
        f"At N=500, per-worker surplus=+0.05 > 0 → expected 0 firings for all_H. "
        f"Got total_fired={total_fired}. K0={K0}. "
        f"The F/K_review term may be miscalculated."
    )


# TODO: re-add positive-direction invariant for greedy_profit once greedy behavior
# at N=500 is measured and a meaningful canary count is established.
# Round-2 critic ruled out a vacuous ">= 0" stub (D-08 in defaults-n500-hire-delay).
# def test_default_firing_threshold_greedy_fires(): ...


def test_enable_hiring_default_is_false():
    """T-02: enable_hiring defaults to False (dormant per lessons-learned 2026-05-09)."""
    assert FirmParams().enable_hiring is False, (
        "enable_hiring must default to False to preserve fixture byte-parity and opt-in semantics"
    )
