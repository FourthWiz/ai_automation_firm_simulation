"""Task-attribute sampling. Phase 1 ignores `rho`; Phase 3 may use it."""
import numpy as np


def sample_alpha(N: int, rng: np.random.Generator, rho: float = 0.0) -> np.ndarray:
    """Sample automatability α_i ~ Uniform(0, 1) for N tasks.

    `rho` is accepted for API stability but is a no-op in Phase 1. In Phase 3
    a correlation sampler may replace this body without changing call sites.
    """
    del rho  # Phase 1 no-op
    return rng.uniform(0.0, 1.0, size=N)


def sample_beta(N: int, rng: np.random.Generator, rho: float = 0.0) -> np.ndarray:
    """Sample augmentability β_i ~ Uniform(0, 1) for N tasks.

    `rho` is accepted for API stability but is a no-op in Phase 1.
    """
    del rho  # Phase 1 no-op
    return rng.uniform(0.0, 1.0, size=N)
