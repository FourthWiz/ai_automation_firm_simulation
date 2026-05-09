"""Streamlit-native plot helpers for the Phase 1.5 dashboard.

Each function accepts a run DataFrame (produced by run_simulation) and returns
a matplotlib.figure.Figure. NO savefig calls, NO disk I/O, NO mkdir. The
dashboard calls st.pyplot(fig) directly.

Color scheme for mode-mix plots matches viz.fig3_mode_mix_greedy:
  H (human):    #4C72B0 (blue)
  A (augmented): #55A868 (green)
  T (automated): #C44E52 (red/orange)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from firm_ai_abm.production import Mode

# Use non-interactive Agg backend (safe in Streamlit environment)
matplotlib.use("Agg")


def fig_pi_over_time(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Cumulative profit over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int, 1-indexed) and
            'pi' (float, per-period profit). Shape: (T, 13+).
    Output:
        Figure with one Axes: line plot of df['pi'].cumsum() vs df['t'],
        plus a dashed gray zero-reference line.
    Color: single line in #4C72B0 (blue).
    No disk I/O — returns Figure only; caller calls st.pyplot(fig).
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    cum_pi = df["pi"].cumsum()
    ax.plot(df["t"], cum_pi, color="#4C72B0", linewidth=1.5)
    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("period")
    ax.set_ylabel("cumulative profit")
    ax.set_title("Cumulative Profit")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_K_over_time(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Worker headcount over time: packed formula (K) vs actual active (K_active).

    K (packed formula) = ceil(n_H_or_A / tasks_per_worker) — underestimates
    actual paying headcount when H/A tasks are scattered across task slots.
    K_active = unique workers assigned to at least one H/A task — matches
    wage_bill exactly. When N is small relative to tasks_per_worker the two
    lines coincide; for large N they diverge and K_active is the correct count.

    Input:
        df: run_simulation DataFrame with columns 't', 'K', 'K_active'.
    Output:
        Figure with one Axes, two lines. No disk I/O.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["t"], df["K_active"], color="#55A868", linewidth=1.5, label="K_active (actual)")
    ax.plot(df["t"], df["K"], color="#4C72B0", linewidth=1.0, linestyle="--", label="K (packed formula)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("period")
    ax.set_ylabel("workers")
    ax.set_title("Active Workers Over Time")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_mode_mix_area(df: pd.DataFrame, N: int) -> matplotlib.figure.Figure:
    """Stacked area chart of H/A/T mode fractions over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'modes'
            (object column of numpy int arrays of length N). Shape: (T, 13+).
        N: number of tasks (used to normalize fractions to [0, 1]).
    Output:
        Figure with one Axes: stacked area chart. Three layers (H, A, T) sum
        to 1.0 at each period.
    Colors match viz.fig3_mode_mix_greedy:
        H=#4C72B0 (blue), A=#55A868 (green), T=#C44E52 (red).
    No disk I/O — returns Figure only.
    """
    periods = df["t"].values
    modes_series = df["modes"].values

    frac_H = np.array([(m == int(Mode.H)).sum() / N for m in modes_series])
    frac_A = np.array([(m == int(Mode.A)).sum() / N for m in modes_series])
    frac_T = np.array([(m == int(Mode.T)).sum() / N for m in modes_series])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(
        periods,
        frac_H, frac_A, frac_T,
        labels=["H (human)", "A (augmented)", "T (automated)"],
        colors=["#4C72B0", "#55A868", "#C44E52"],
        alpha=0.85,
    )
    ax.set_xlabel("period")
    ax.set_ylabel("fraction of tasks")
    ax.set_title("Mode Mix (stacked area)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_wage_bill_over_time(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Total wage bill over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'wage_bill'
            (float, sum of wages paid to all active workers). Shape: (T, 13+).
    Output:
        Figure with one Axes: line plot of df['wage_bill'] vs df['t'].
    Color: single line in #8172B3 (purple).
    No disk I/O — returns Figure only.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["t"], df["wage_bill"], color="#8172B3", linewidth=1.5)
    ax.set_xlabel("period")
    ax.set_ylabel("wage bill")
    ax.set_title("Wage Bill Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_theta_histogram(workforce_theta: np.ndarray) -> matplotlib.figure.Figure:
    """Histogram of the final-period worker skill (theta) distribution.

    Input:
        workforce_theta: 1-D numpy array of worker theta values from the
            final period of the simulation (shape: (K,) where K = active
            worker count). Values in [theta_min, theta_max].
    Output:
        Figure with one Axes: histogram with 20 bins (matplotlib defaults for
        color). Dashed vertical line at mean theta.
    Color: matplotlib default histogram color.
    No disk I/O — returns Figure only.
    """
    theta = np.asarray(workforce_theta)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(theta, bins=20, edgecolor="white", alpha=0.8)
    if len(theta) > 0:
        ax.axvline(float(np.mean(theta)), color="#C44E52", linewidth=1.2,
                   linestyle="--", label=f"mean={np.mean(theta):.3f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("theta (worker skill)")
    ax.set_ylabel("count")
    ax.set_title("Final-Period Theta Distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_mean_theta_over_time(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Mean worker skill (theta) over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'mean_theta'
            (float, mean of active worker theta values). Shape: (T, 13+).
    Output:
        Figure with one Axes: line plot of df['mean_theta'] vs df['t'].
    Color: single line in #CCB974 (gold/ochre).
    No disk I/O — returns Figure only.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    col = df.get("mean_theta", pd.Series(np.nan, index=df.index))
    ax.plot(df["t"], col, color="#CCB974", linewidth=1.5)
    ax.set_xlabel("period")
    ax.set_ylabel("mean theta")
    ax.set_title("Mean Worker Skill Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_firing_events(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Firing events at review periods.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and
            'n_review_fired' (int, workers fired at this period's review;
            0 for non-review periods). Shape: (T, 13+).
    Output:
        Figure with one Axes: scatter of t values where n_review_fired > 0,
        marker size proportional to count (markersize = 40 * count / max_count,
        minimum 40). Baseline at y=0 via axhline. Only firing periods have
        markers; non-review or zero-fired periods have none.
        If no firings occurred (all-T strategy, T_review=inf), the plot
        renders an empty scatter with the baseline line and a text note.
    Color: markers in #C44E52 (red).
    No disk I/O — returns Figure only.
    """
    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    mask = fired_col > 0
    t_fired = df["t"][mask].values
    n_fired = fired_col[mask].values

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")

    if len(t_fired) > 0:
        max_n = float(n_fired.max())
        sizes = [max(40, 40 * v / max_n * 5) for v in n_fired]
        ax.scatter(t_fired, n_fired, s=sizes, color="#C44E52", alpha=0.8,
                   zorder=3, label="workers fired")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no firing events\n(T_review=inf or no firings)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")

    ax.set_xlabel("period")
    ax.set_ylabel("workers fired")
    ax.set_title("Firing Events at Review Periods")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_trained_capital(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Cumulative trained workers (n_a_trained) over time.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and 'n_a_trained'
            (int, count of workers who have completed H→A training and retained
            it through the current period). Shape: (T, 13+).
    Output:
        Figure with one Axes: line plot of df['n_a_trained'] vs df['t'].
        Y-axis uses integer ticks (MaxNLocator with integer=True).
    Color: single line in #55A868 (green).
    No disk I/O — returns Figure only.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    col = df.get("n_a_trained", pd.Series(0, index=df.index))
    ax.plot(df["t"], col, color="#55A868", linewidth=1.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("period")
    ax.set_ylabel("trained workers (n_a_trained)")
    ax.set_title("Trained-Capital Workers Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
