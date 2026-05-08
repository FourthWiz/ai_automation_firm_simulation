"""Pure strategies for Phase 1 stage 2. Greedy strategies and tie-breaking rules land in Stage 4.
Each `decide()` returns a NEW array — never an alias of `firm.modes` (R-08).
"""
import numpy as np

from firm_ai_abm.firm import Firm
from firm_ai_abm.production import Mode


def all_H(firm: Firm, t: int) -> np.ndarray:
    """Return a new modes array with all tasks in Human mode (Mode.H = 0)."""
    return np.zeros(firm.params.N, dtype=int)


def all_A(firm: Firm, t: int) -> np.ndarray:
    """Return a new modes array with all tasks in Augmented mode (Mode.A = 1)."""
    return np.full(firm.params.N, int(Mode.A), dtype=int)


def all_T(firm: Firm, t: int) -> np.ndarray:
    """Return a new modes array with all tasks in Automated mode (Mode.T = 2)."""
    return np.full(firm.params.N, int(Mode.T), dtype=int)
