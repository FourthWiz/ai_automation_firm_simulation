"""firm_ai_abm — agent-based simulation of firm responses to AI."""
from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import Firm, make_firm
from firm_ai_abm.production import Mode, compute_K, productivity_vec, cost_vec
from firm_ai_abm.adjustment import adj_cost
from firm_ai_abm.strategy import (
    all_H, all_A, all_T, greedy_profit, greedy_with_switching,
)
from firm_ai_abm.simulate import run_simulation, run_horizon
from firm_ai_abm.workers import Workforce, sample_workforce, task_to_worker_map, _make_initial_workforce
from firm_ai_abm.review import firing_review, apply_firings, replace_to_target  # Stage 5 D-12: apply_firings_and_replace removed; split into apply_firings + replace_to_target
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
    "run_horizon",
    # Phase 1.5 Stage 1: worker heterogeneity
    "Workforce",
    "sample_workforce",
    "task_to_worker_map",
    "_make_initial_workforce",
    # Phase 1.5 Stage 3 / Stage 5: periodic firing review
    "firing_review",
    "apply_firings",
    "replace_to_target",
]
