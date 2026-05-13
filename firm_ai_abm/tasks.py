"""Task-attribute sampling. alpha, beta ~ Beta(a, b) parameterized by mean
and concentration; default (mean=0.5, concentration=2.0) short-circuits to
rng.uniform(0, 1) for byte-identity with Phase 1 fixtures.

The `rho` parameter (Phase-3 correlation placeholder) has been dropped. If
Phase-3 reintroduces correlation, use a new `sample_correlated(...)` API."""
import numpy as np

_EPS = 1e-6
_KAPPA_MIN = 2.0
_UNIFORM_MU = 0.5
_UNIFORM_KAPPA = 2.0


def _sample_beta_or_uniform(N, rng, mu, kappa):
    # Exact-equality fall-through to the historical Uniform sampler.
    # Load-bearing: preserves byte-identity with tests/fixtures/*.parquet
    # captured at git commit recorded in tests/fixtures/_provenance.txt.
    if mu == _UNIFORM_MU and kappa == _UNIFORM_KAPPA:
        return rng.uniform(0.0, 1.0, size=N)
    # Clamp defensively (UI/FirmParams should already enforce these bounds).
    mu_c = float(min(max(mu, _EPS), 1.0 - _EPS))
    kappa_c = float(max(kappa, _KAPPA_MIN))
    a = mu_c * kappa_c
    b = (1.0 - mu_c) * kappa_c
    return rng.beta(a, b, size=N)


def sample_alpha(N, rng, mu=_UNIFORM_MU, kappa=_UNIFORM_KAPPA):
    """Sample automatability alpha_i for N tasks. See module docstring."""
    return _sample_beta_or_uniform(N, rng, mu, kappa)


def sample_beta(N, rng, mu=_UNIFORM_MU, kappa=_UNIFORM_KAPPA):
    """Sample augmentability beta_i for N tasks. See module docstring."""
    return _sample_beta_or_uniform(N, rng, mu, kappa)
