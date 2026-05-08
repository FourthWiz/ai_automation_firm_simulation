"""Strategies for Phase 1 stages 2 and 4.

Stage 2: pure baseline strategies — all_H, all_A, all_T.
Stage 4: greedy myopic strategies — greedy_profit, greedy_with_switching.

Each strategy returns a NEW array — never an alias of firm.modes (R-08).
Tie-breaking is first-index-wins via np.argmax, encoding H < A < T preference
(architecture R-11, design doc §5). This gives a deterministic result when two
modes have equal per-task score.

greedy_profit: per-task argmax over instantaneous score, no switching costs.
  Wages enter as smooth per-task amortization w/tasks_per_worker for H and A
  modes; T tasks consume no worker slot and bear no wage (D-06).

greedy_with_switching: per-task argmax over (gross score - amortized switching
  cost). Uses smooth c_fire/tasks_per_worker and c_hire/tasks_per_worker per-task
  amortization for the decision rule (the parent architecture's decision-vs-payment
  asymmetry risk). The simulator pays lumpy delta-K-based hire/fire via
  adjustment.adj_cost; the strategy uses the smooth approximation. See D-04 and
  the parent architecture's decision-vs-payment asymmetry risk.
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


def greedy_profit(firm: Firm, t: int) -> np.ndarray:
    """Myopic per-task argmax over instantaneous profit score, ignoring switching costs.

    For each task, computes the instantaneous score for each mode and picks the best:
        score_H = q_h - wage_per_task
        score_A = q_h * (1 + g * beta_i) - c_aug - wage_per_task
        score_T = q_a * alpha_i - c_auto

    where wage_per_task = w / tasks_per_worker — smooth per-task amortization of the
    lumpy w*K firm payment (D-06). H and A tasks consume one worker slot each; T tasks
    consume no worker and bear no wage.

    No switching costs in the decision — for the with-switching-cost variant see
    `greedy_with_switching` and the parent architecture's decision-vs-payment asymmetry
    risk. Wages enter as smooth per-task amortization `w/tasks_per_worker` for H and A
    modes; T tasks consume no worker slot and bear no wage (D-06).

    Tie-breaking: np.argmax first-index-wins, encoding H < A < T (architecture R-11,
    design doc §5). When score_H == score_A, H is preferred (column 0 < column 1).

    Returns:
        Fresh np.ndarray of shape (N,), dtype int, values in {0, 1, 2}.
        NOT an alias of firm.modes (R-08).

    Risk citations:
        R-11: tie-breaking via np.argmax first-index-wins (H < A < T column order).
        R-08: returns fresh ndarray, not alias of firm.modes.
        D-06: wage_per_task = w/tasks_per_worker for H and A; zero for T.
    """
    assert firm.modes.dtype.kind == "i", (
        f"firm.modes must be integer dtype at strategy entry, got {firm.modes.dtype}"
    )

    p = firm.params
    wage_per_task = p.w / p.tasks_per_worker  # smooth amortization of lumpy w*K (D-06)

    score_H = p.q_h - wage_per_task                                     # scalar
    score_A = p.q_h * (1.0 + p.g * firm.beta) - p.c_aug - wage_per_task  # shape (N,)
    score_T = p.q_a * firm.alpha - p.c_auto                              # shape (N,); no wage

    scores = np.zeros((p.N, 3), dtype=np.float64)  # zeros not empty (defensive, MIN-2)
    scores[:, 0] = score_H  # broadcast scalar
    scores[:, 1] = score_A
    scores[:, 2] = score_T

    return np.argmax(scores, axis=1).astype(int)


def greedy_with_switching(firm: Firm, t: int) -> np.ndarray:
    """Myopic per-task argmax over (gross score - amortized switching cost).

    Per-task decision score:
        net_score[i, m] = gross_score[i, m] - S_amortized[prev_mode_i, m]

    where:
        gross_score — same instantaneous profit score as greedy_profit
        S_amortized = S / n_amortize  (amortize switching cost over n_amortize periods)

    The 3x3 switching cost matrix S[from_mode, to_mode]:
        S[H, A] = c_train          (per-task training cost, H->A only)
        S[H, T] = c_fire / tasks_per_worker   (H->T fires the worker, smooth)
        S[A, T] = c_fire / tasks_per_worker   (A->T fires the worker, smooth)
        S[T, H] = c_hire / tasks_per_worker   (T->H hires the worker, smooth)
        S[T, A] = c_hire / tasks_per_worker   (T->A hires the worker, smooth)
        S[A, H] = 0  (A and H share the same worker class — no fire; architecture
                       also declines to refund c_train — no untraining cost; D-02)
        diagonal = 0  (no cost to stay in same mode)

    Amortization: S_amortized = S / n_amortize (default n_amortize=6, design doc §5).
    Formulation is "switch iff gross gain exceeds amortized cost" — equivalent to
    argmax of score - S_amortized[prev_mode, candidate_mode] (R-05).

    Decision-vs-payment asymmetry (parent architecture's smooth-vs-lumpy risk):
    This decision rule uses smooth c_fire/tasks_per_worker and c_hire/tasks_per_worker
    per-task amortization. The simulator pays lumpy c_fire * delta_K + c_hire * delta_K
    via adjustment.adj_cost. Decision-cost != payment-cost is intentional and pinned by
    the parent architecture's decision-vs-payment asymmetry risk. The asymmetry also
    applies to wages: the decision charges smooth w/tasks_per_worker per H/A task while
    the firm pays lumpy w*K (D-06).

    A-to-H zero by enumeration: A and H tasks both covered by same human-worker class
    (no fire on A->H transition) AND architecture declines to refund c_train (training
    is a one-way investment). Both compounding reasons yield zero cost. Revisit in Phase
    2 if untraining matters.

    Worked example at default params (q_h=1, q_a=1.2, g=0.5, c_aug=0.05, c_auto=0.4,
    c_train=0.1, c_fire=2, c_hire=0.5, w=1, tasks_per_worker=10, n_amortize=6):
        wage_per_task = w/tasks_per_worker = 0.1
        For a task at H with alpha=0.9, beta=0.5:
          score_H = 1.0 - 0.1 = 0.9
          score_A = 1.0*(1+0.5*0.5) - 0.05 - 0.1 = 1.10
          score_T = 1.2*0.9 - 0.4 = 0.68  (no wage; T tasks consume no worker)
          Switch from H: cost_H=0, cost_A = c_train/n_amortize = 0.1/6 ≈ 0.0167,
                         cost_T = c_fire/tasks_per_worker/n_amortize = 2/10/6 ≈ 0.0333
          Net: H = 0.9, A ≈ 1.0833, T ≈ 0.6467. Argmax = A (column 1). Decision: H->A.

    Tie-breaking: np.argmax first-index-wins (H < A < T). When prev_mode == m,
    switch_cost[i, m] == 0, so the diagonal is favored over a tied off-diagonal —
    this gives "stay put on tie" semantics (R-11).

    Returns:
        Fresh np.ndarray of shape (N,), dtype int, values in {0, 1, 2}.
        NOT an alias of firm.modes (R-08).

    Risk citations:
        R-05: amortization direction — divide cost by n_amortize (not multiply gain).
        R-11: tie-breaking via first-index argmax; stay-put on tie from diagonal-zero.
        R-08: returns fresh ndarray, not alias.
        D-02: A->H zero by enumeration (same worker class + no c_train refund).
        D-06: wage_per_task = w/tasks_per_worker for H and A; zero for T.
        parent architecture's smooth-vs-lumpy decision/payment asymmetry risk.
    """
    assert firm.modes.dtype.kind == "i", (
        f"firm.modes must be integer dtype at strategy entry, got {firm.modes.dtype}"
    )

    p = firm.params
    wage_per_task = p.w / p.tasks_per_worker  # smooth amortization of lumpy w*K (D-06)

    score_H = p.q_h - wage_per_task                                     # scalar
    score_A = p.q_h * (1.0 + p.g * firm.beta) - p.c_aug - wage_per_task  # shape (N,)
    score_T = p.q_a * firm.alpha - p.c_auto                              # shape (N,); no wage

    gross_scores = np.zeros((p.N, 3), dtype=np.float64)  # zeros not empty (MIN-2)
    gross_scores[:, 0] = score_H
    gross_scores[:, 1] = score_A
    gross_scores[:, 2] = score_T

    # Defensive copy of prev modes (MIN-1); eager dtype check pins the contract (MIN-3)
    prev = firm.modes.copy()
    assert prev.dtype.kind == "i", (
        f"prev (firm.modes) must be integer dtype, got {prev.dtype}"
    )

    # Build 3x3 switching cost matrix S[from_mode, to_mode]
    S = np.zeros((3, 3), dtype=np.float64)
    S[0, 1] = p.c_train                        # H -> A: per-task training cost
    S[0, 2] = p.c_fire / p.tasks_per_worker    # H -> T: fire worker (smooth)
    S[1, 2] = p.c_fire / p.tasks_per_worker    # A -> T: fire worker (smooth)
    S[2, 0] = p.c_hire / p.tasks_per_worker    # T -> H: hire worker (smooth)
    S[2, 1] = p.c_hire / p.tasks_per_worker    # T -> A: hire worker (smooth)
    # S[1, 0] (A -> H) = 0: same worker class (no fire) + no c_train refund (D-02)
    # Diagonal = 0: no cost to stay in same mode

    S_amort = S / p.n_amortize  # amortize over n_amortize periods (R-05)

    # Fancy indexing: switch_cost[i] = S_amort[prev[i], :] -> shape (N, 3)
    switch_cost = S_amort[prev]

    net_scores = gross_scores - switch_cost
    return np.argmax(net_scores, axis=1).astype(int)
