"""firm_ai_abm — agent-based simulation of firm responses to AI."""
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import Firm, make_firm
from firm_ai_abm.production import Mode, compute_K, productivity_vec, cost_vec
from firm_ai_abm.adjustment import adj_cost
from firm_ai_abm.strategy import (
    all_H, all_A, all_T, greedy_profit, greedy_with_switching,
)
from firm_ai_abm.simulate import run_simulation

__all__ = [
    "FirmParams",
    "Firm",
    "make_firm",
    "Mode",
    "compute_K",
    "productivity_vec",
    "cost_vec",
    "adj_cost",
    "all_H",
    "all_A",
    "all_T",
    "greedy_profit",
    "greedy_with_switching",
    "run_simulation",
]
