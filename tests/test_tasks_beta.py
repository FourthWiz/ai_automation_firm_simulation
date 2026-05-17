"""Tests for Beta-distribution task-attribute sampling (beta-dist-task-attrs).

Covers T-08 acceptance criteria:
  1. Default (mu=0.5, kappa=2.0) is byte-identical to rng.uniform(0,1)
  2. Snapshot parity via make_firm — byte-identity at the make_firm boundary
  3. Beta translation correctness (mean and variance)
  4. Determinism under seed
  5. Clamp boundary safety (mu=0, kappa<2)
  6. J-shape reachability (mu=0.1, kappa=2)
  7. Non-default engages Beta path (exact-equality contract)
"""
import numpy as np
import pytest

from firm_ai_abm.tasks import sample_alpha, sample_beta
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm


def test_default_is_byte_identical_to_uniform():
    """Default params (mu=0.5, kappa=2.0) produce same values as rng.uniform(0,1)."""
    # alpha
    rng1 = np.random.default_rng(42)
    expected = rng1.uniform(0.0, 1.0, size=500)
    rng2 = np.random.default_rng(42)
    actual = sample_alpha(500, rng2)
    assert np.array_equal(expected, actual), "sample_alpha default not byte-identical to rng.uniform"

    # beta
    rng1 = np.random.default_rng(99)
    expected = rng1.uniform(0.0, 1.0, size=500)
    rng2 = np.random.default_rng(99)
    actual = sample_beta(500, rng2)
    assert np.array_equal(expected, actual), "sample_beta default not byte-identical to rng.uniform"


def test_default_via_make_firm_byte_identical():
    """Snapshot test: make_firm with defaults produces alpha/beta byte-identical to fixture.

    The fixture tests/fixtures/_alpha_beta_seed0_N100.npz was captured before this
    task's changes were made, using the default FirmParams (mu=0.5, kappa=2.0).
    """
    from pathlib import Path
    fixture_path = Path("tests/fixtures/_alpha_beta_seed0_N100.npz")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    fixture = np.load(fixture_path)
    # Pin pre-99ddaea alpha/beta defaults: old defaults were alpha_mean=0.5, alpha_conc=2.0,
    # beta_mean=0.5, beta_conc=2.0 (99ddaea changed these to 0.4/3.0 and 0.8/3.0).
    firm = make_firm(FirmParams(seed=0, N=100, alpha_mean=0.5, alpha_concentration=2.0,
                                beta_mean=0.5, beta_concentration=2.0))
    assert np.array_equal(firm.alpha, fixture["alpha"]), (
        "make_firm alpha not byte-identical to pre-change fixture"
    )
    assert np.array_equal(firm.beta, fixture["beta"]), (
        "make_firm beta not byte-identical to pre-change fixture"
    )


def test_beta_translation_mean_and_variance():
    """Non-default params produce samples with correct mean and variance."""
    rng = np.random.default_rng(7)
    mu, kappa = 0.3, 10
    samples = sample_alpha(100_000, rng, mu=mu, kappa=kappa)
    # Theoretical mean = mu = 0.3
    assert abs(samples.mean() - mu) < 0.005, (
        f"Mean mismatch: expected ~{mu}, got {samples.mean():.4f}"
    )
    # Theoretical variance = mu*(1-mu)/(kappa+1)
    expected_var = mu * (1 - mu) / (kappa + 1)
    assert abs(samples.var() - expected_var) < 0.001, (
        f"Variance mismatch: expected ~{expected_var:.4f}, got {samples.var():.4f}"
    )


def test_deterministic_under_seed():
    """Same seed produces identical draws; different seed produces different draws."""
    mu, kappa = 0.7, 5
    rng_a = np.random.default_rng(123)
    draw1 = sample_alpha(1000, rng_a, mu=mu, kappa=kappa)
    rng_b = np.random.default_rng(123)
    draw2 = sample_alpha(1000, rng_b, mu=mu, kappa=kappa)
    assert np.array_equal(draw1, draw2), "Same seed should produce identical draws"

    rng_c = np.random.default_rng(456)
    draw3 = sample_alpha(1000, rng_c, mu=mu, kappa=kappa)
    assert not np.array_equal(draw1, draw3), "Different seeds should produce different draws"


def test_clamp_boundary_safety():
    """Boundary inputs (mu=0.0, kappa=1.0) are clamped and don't raise."""
    rng = np.random.default_rng(0)
    # Both mu and kappa are below valid ranges — should be clamped, not raise
    result = sample_alpha(100, rng, mu=0.0, kappa=1.0)
    assert len(result) == 100, "Should return 100 values"
    assert np.all(np.isfinite(result)), "All values should be finite"
    assert np.all(result >= 0.0) and np.all(result <= 1.0), "All values should be in [0, 1]"
    # With mu clamped to eps (~1e-6), mean should be very close to 0
    assert result.mean() < 0.05, (
        f"Clamped mu=eps should produce near-zero mean, got {result.mean():.4f}"
    )


def test_j_shape_allowed():
    """J-shape distribution (mu=0.1, kappa=2) is reachable and has correct mean."""
    # a = 0.1*2 = 0.2, b = 0.9*2 = 1.8 — J-shaped at the left boundary
    rng = np.random.default_rng(42)
    samples = sample_alpha(10_000, rng, mu=0.1, kappa=2)
    # Mean should be ~0.1
    assert abs(samples.mean() - 0.1) < 0.01, (
        f"J-shape mean: expected ~0.1, got {samples.mean():.4f}"
    )
    # J-shape: at least 30% of samples should be below 0.05
    frac_below_005 = float(np.mean(samples < 0.05))
    assert frac_below_005 >= 0.30, (
        f"J-shape spike: expected >=30% below 0.05, got {frac_below_005:.2%}"
    )


def test_non_default_engages_beta_path():
    """kappa=2.0001 (just off default) engages Beta sampler, not Uniform fall-through.

    This locks D-03's exact-equality contract: any nudge from the default,
    even by a tiny amount, must engage the Beta sampler.
    """
    seed = 55
    rng1 = np.random.default_rng(seed)
    uniform_result = rng1.uniform(0.0, 1.0, size=200)

    rng2 = np.random.default_rng(seed)
    beta_result = sample_alpha(200, rng2, mu=0.5, kappa=2.0001)

    assert not np.array_equal(uniform_result, beta_result), (
        "kappa=2.0001 should engage Beta sampler, not return Uniform values"
    )


def test_sample_alpha_beta_accept_params():
    """Calling with non-default mu/kappa doesn't raise."""
    rng = np.random.default_rng(0)
    result_a = sample_alpha(100, rng, mu=0.4, kappa=5.0)
    result_b = sample_beta(100, rng, mu=0.6, kappa=10.0)
    assert len(result_a) == 100
    assert len(result_b) == 100


def test_cache_key_includes_new_params():
    """FirmParams with different alpha_mean produces different params_to_key() output."""
    from app import params_to_key
    key1 = params_to_key(FirmParams(), 0)
    key2 = params_to_key(FirmParams(alpha_mean=0.3), 0)
    assert key1 != key2, "Different alpha_mean should produce different cache keys"


def test_widget_keys_include_new_params():
    """ALL_WIDGET_KEYS contains the 4 new task-attribute keys."""
    from app import ALL_WIDGET_KEYS
    for key in ("alpha_mean", "alpha_concentration", "beta_mean", "beta_concentration"):
        assert key in ALL_WIDGET_KEYS, (
            f"'{key}' missing from ALL_WIDGET_KEYS"
        )
