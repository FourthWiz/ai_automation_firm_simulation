"""Phase 1 firm parameters. Defaults from 01_phase1_single_firm.md §7."""
from dataclasses import dataclass


@dataclass
class FirmParams:
    # Counts
    N: int = 100                 # number of tasks
    T: int = 60                  # number of periods
    tasks_per_worker: int = 10   # how many tasks one worker covers

    # Productivity
    q_h: float = 1.0             # human productivity per task (numeraire)
    q_a: float = 1.2             # AI productivity ceiling (key dial)
    g: float = 0.5               # augmentation gain (key dial)

    # Costs
    w: float = 1.0               # wage per worker per period (numeraire)
    c_aug: float = 0.05          # per-task augmentation cost
    c_auto: float = 0.4          # per-task automation cost (key dial)
    c_fire: float = 2.0          # firing cost per worker (lumpy, K-based)
    c_hire: float = 0.5          # hiring cost per worker (lumpy, K-based)
    c_train: float = 0.1         # training cost per H->A task (per-task)
    F: float = 5.0               # fixed cost per period

    # Prices
    p: float = 1.0               # output price (numeraire)

    # Strategy
    n_amortize: int = 6          # horizon for greedy-with-switching amortization

    # RNG
    seed: int | None = None      # if set, reproducible alpha/beta sampling
