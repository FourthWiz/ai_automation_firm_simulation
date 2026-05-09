"""Visualization for Phase 1 single-firm simulation.

Three figures:
  fig1_primary_lines       — MANDATORY (§14): cumulative profit for all 5 strategies.
  fig2_small_multiples_q_a — MANDATORY (§14): 3×1 small-multiples by q_a ∈ {0.8, 1.2, 1.6}.
  fig3_mode_mix_greedy     — OPTIONAL (§11): stacked bar of mode mix for greedy_profit.

Headless safety (R-04): matplotlib.use("Agg") is called before any pyplot import so
that file output works in environments without a display. Callers running in an
interactive Jupyter session should call ``matplotlib.use("TkAgg")`` (or similar) BEFORE
importing this module if they want inline display.

Y-axis clamping policy (parent architecture negative-profit-handling risk, R-10):
Do NOT clamp the y-axis at zero. Cumulative profit can legitimately go negative at
low q_a (no bankruptcy in Phase 1, per §13). All axes use matplotlib auto-range.
A dashed zero-reference line (``ax.axhline(0, ...)``) is added for readability only.

Shared-firm-instance policy (parent architecture shared-firm-instance risk, R-08):
All 5 strategies in a single panel run on the SAME firm instance. run_simulation()
calls firm.reset() before each strategy, preserving alpha/beta while resetting modes,
K, and history. This ensures all strategies see the same task landscape.
"""
import dataclasses
import pathlib

import matplotlib
matplotlib.use("Agg")  # headless safety (R-04): must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.production import Mode
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import (
    all_H, all_A, all_T, greedy_profit, greedy_with_switching,
)

# Strategy color and marker assignments (consistent across all figures)
_STRATEGIES = [all_H, all_A, all_T, greedy_profit, greedy_with_switching]
_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
_MARKERS = ["o", "s", "^", "D", "v"]


def fig1_primary_lines(
    params: FirmParams | None = None,
    save_dir: pathlib.Path | None = None,
) -> plt.Figure:
    """PRIMARY figure (§11, mandatory §14): cumulative profit for all 5 strategies.

    Plots cumulative period profit (df["pi"].cumsum()) over T periods for:
    all_H, all_A, all_T, greedy_profit, greedy_with_switching.

    All 5 strategies run on the SAME firm instance (parent architecture
    shared-firm-instance risk, R-08). run_simulation() calls firm.reset() before
    each strategy, preserving alpha/beta.

    Y-axis: matplotlib auto-range — NOT clamped at zero (parent architecture
    negative-profit-handling risk, R-10). A dashed gray zero-reference line is
    added for readability but does NOT clamp.

    Saved as:
      save_dir/fig1_primary_lines.png  (dpi=300)
      save_dir/fig1_primary_lines.svg  (vector)

    Args:
        params: FirmParams to use. Default: FirmParams(seed=0).
        save_dir: Directory for output files. Default: pathlib.Path("results/figures").
            Created if absent.

    Returns:
        matplotlib.figure.Figure (caller decides when to close).
    """
    if params is None:
        params = FirmParams(seed=0)
    if save_dir is None:
        save_dir = pathlib.Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    firm = make_firm(params)

    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, color, marker in zip(_STRATEGIES, _COLORS, _MARKERS):
        df = run_simulation(firm, strategy)
        ax.plot(
            df["t"],
            df["pi"].cumsum(),
            label=strategy.__name__,
            color=color,
            marker=marker,
            markevery=10,
            linewidth=1.2,
        )

    ax.set_xlabel("period")
    ax.set_ylabel("cumulative profit")
    ax.set_title("Phase 1: Cumulative Profit by Strategy")
    # Zero reference line — does NOT clamp (R-10)
    ax.axhline(0, color="lightgray", linewidth=0.5, linestyle="--")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)

    # Key params annotation
    ax.text(
        0.02, 0.98,
        f"q_a={params.q_a}, g={params.g}, c_auto={params.c_auto}, w={params.w}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(save_dir / "fig1_primary_lines.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig1_primary_lines.svg", bbox_inches="tight")
    return fig


def fig2_small_multiples_q_a(
    params: FirmParams | None = None,
    q_a_grid: tuple[float, ...] = (0.8, 1.2, 1.6),
    save_dir: pathlib.Path | None = None,
) -> plt.Figure:
    """SECONDARY figure (§11, mandatory §14): 3×1 small-multiples by q_a.

    Shows the same 5-strategy cumulative-profit panel at each q_a in q_a_grid.
    The default grid (0.8, 1.2, 1.6) spans the range from "all_T is deeply
    negative" to "all_T dominates" — the key §11 "it depends" insight.

    Y-axis treatment (parent architecture negative-profit-handling risk, R-10):
    Each panel has its own y-axis auto-range (NO sharey). At q_a=0.8, all_T
    cumulative profit goes deeply negative (low AI productivity + full automation
    cost); at q_a=1.6 it is positive. sharey=True would compress one panel to
    unreadability. Per-panel auto-range is the standard small-multiples choice
    when absolute scale varies meaningfully across facets (D-02 from plan).
    Note: y-axis range varies per panel; cross-panel y-comparisons not meaningful.

    Shared-firm-instance policy: each panel uses a fresh firm built from
    params_qa = replace(params, q_a=q_a). All 5 strategies in a panel share
    that panel's firm instance (R-08).

    Saved as:
      save_dir/fig2_small_multiples_q_a.png  (dpi=300)
      save_dir/fig2_small_multiples_q_a.svg  (vector)

    Args:
        params: Base FirmParams (q_a overridden per panel). Default: FirmParams(seed=0).
        q_a_grid: Tuple of q_a values for the panels (top to bottom).
        save_dir: Directory for output files. Default: pathlib.Path("results/figures").

    Returns:
        matplotlib.figure.Figure (caller decides when to close).
    """
    if params is None:
        params = FirmParams(seed=0)
    if save_dir is None:
        save_dir = pathlib.Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    n_panels = len(q_a_grid)
    # NO sharey — per-panel y-axis auto-range (D-02, R-10)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, q_a in zip(axes, q_a_grid):
        params_qa = dataclasses.replace(params, q_a=q_a)
        firm = make_firm(params_qa)

        for strategy, color, marker in zip(_STRATEGIES, _COLORS, _MARKERS):
            df = run_simulation(firm, strategy)
            ax.plot(
                df["t"],
                df["pi"].cumsum(),
                label=strategy.__name__,
                color=color,
                marker=marker,
                markevery=10,
                linewidth=1.0,
            )

        ax.set_title(f"q_a = {q_a}")
        ax.axhline(0, color="lightgray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("cumulative profit")
        ax.grid(alpha=0.3)
        # Let each panel auto-range independently
        ax.relim()
        ax.autoscale_view()

    axes[-1].set_xlabel("period")

    # Single figure-level legend (avoids 3 redundant per-panel legends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Phase 1: Cumulative Profit by Strategy across q_a\n"
        "(y-axis range varies per panel; cross-panel comparisons not meaningful)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "fig2_small_multiples_q_a.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig2_small_multiples_q_a.svg", bbox_inches="tight")
    return fig


def fig3_mode_mix_greedy(
    params: FirmParams | None = None,
    save_dir: pathlib.Path | None = None,
) -> plt.Figure:
    """OPTIONAL figure (§11): stacked bar of mode mix per period under greedy_profit.

    Shows what fraction of tasks are in each mode {H, A, T} at each period for
    the greedy_profit strategy. Demonstrates how the firm shifts toward automation
    over time.

    Bar heights sum to exactly 1.0 per period (sanity: all N tasks accounted for).

    Colors: H=blue (#4C72B0), A=green (#55A868), T=orange (#C44E52).

    Saved as:
      save_dir/fig3_mode_mix_greedy.png  (dpi=300)
      save_dir/fig3_mode_mix_greedy.svg  (vector)

    Args:
        params: FirmParams to use. Default: FirmParams(seed=0).
        save_dir: Directory for output files. Default: pathlib.Path("results/figures").

    Returns:
        matplotlib.figure.Figure (caller decides when to close).
    """
    if params is None:
        params = FirmParams(seed=0)
    if save_dir is None:
        save_dir = pathlib.Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    firm = make_firm(params)
    df = run_simulation(firm, greedy_profit)

    N = params.N
    periods = df["t"].values
    modes_series = df["modes"].values

    frac_H = np.array([(m == int(Mode.H)).sum() / N for m in modes_series])
    frac_A = np.array([(m == int(Mode.A)).sum() / N for m in modes_series])
    frac_T = np.array([(m == int(Mode.T)).sum() / N for m in modes_series])

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(periods, frac_H, label="H (human)", color="#4C72B0")
    ax.bar(periods, frac_A, bottom=frac_H, label="A (augmented)", color="#55A868")
    ax.bar(periods, frac_T, bottom=frac_H + frac_A, label="T (automated)", color="#C44E52")

    ax.set_xlabel("period")
    ax.set_ylabel("fraction of tasks")
    ax.set_title("Phase 1: greedy_profit Mode Mix per Period")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_dir / "fig3_mode_mix_greedy.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig3_mode_mix_greedy.svg", bbox_inches="tight")
    return fig
