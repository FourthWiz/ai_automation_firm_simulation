"""Streamlit-native plot helpers for the Phase 1.5 dashboard.

Each function accepts a run DataFrame (produced by run_simulation) and returns
a matplotlib.figure.Figure. NO savefig calls, NO disk I/O, NO mkdir. The
dashboard calls st.pyplot(fig) directly.

Color scheme for mode-mix plots matches viz.fig3_mode_mix_greedy:
  H (human):    #4C72B0 (blue)
  A (augmented): #55A868 (green)
  T (automated): #C44E52 (red/orange)
"""

import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from firm_ai_abm.production import Mode

# Use non-interactive Agg backend (safe in Streamlit environment)
matplotlib.use("Agg")


def fig_pi_per_period_over_time(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Per-period profit over time with mean line and optional first-firing annotation.

    Input:
        df: run_simulation DataFrame with columns 't', 'pi', and optionally
            'n_review_fired'. Shape: (T, 13+).
    Output:
        Figure with one Axes: line plot of df['pi'] vs df['t'], horizontal dashed
        line at df['pi'].mean(), optional vertical dashed line at the first period
        where n_review_fired > 0 (if any). No disk I/O.
    Color: line in #4C72B0 (blue), mean line in gray, firing marker in #C44E52 (red).
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["t"], df["pi"], color="#4C72B0", linewidth=1.5, label="per-period π")
    mean_pi = float(df["pi"].mean())
    ax.axhline(mean_pi, color="gray", linewidth=0.8, linestyle="--",
               label=f"mean={mean_pi:.2f}")

    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    first_fire_rows = df.loc[fired_col > 0, "t"]
    if len(first_fire_rows) > 0:
        first_fire_t = int(first_fire_rows.iloc[0])
        ax.axvline(first_fire_t, color="#C44E52", linewidth=0.8, linestyle=":",
                   label=f"first firing t={first_fire_t}")

    ax.set_xlabel("period")
    ax.set_ylabel("profit")
    ax.set_title("Per-Period Profit")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
    """Worker headcount over time with firing-event overlay.

    K (packed formula) = ceil(n_H_or_A / tasks_per_worker).
    K_active = unique workers assigned to at least one H/A task.
    When firings occur, red bars on the right y-axis show workers fired per
    review period. K stays constant because fired workers are immediately
    replaced; the bars reveal when turnover happens.

    Input:
        df: run_simulation DataFrame with columns 't', 'K', 'K_active',
            and optionally 'n_review_fired'.
    Output:
        Figure with one Axes (two lines) and, when firings > 0, a twin
        right-axis with firing bars. No disk I/O.
    """
    fired_col = df.get("n_review_fired", pd.Series(0, index=df.index))
    has_firings = bool((fired_col > 0).any())

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["t"], df["K_active"], color="#55A868", linewidth=1.5, label="K_active (actual)")
    ax.plot(df["t"], df["K"], color="#4C72B0", linewidth=1.0, linestyle="--", label="K (packed formula)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("period")
    ax.set_ylabel("workers")
    ax.set_title("Active Workers Over Time")
    ax.grid(alpha=0.3)

    if has_firings:
        ax2 = ax.twinx()
        mask = fired_col > 0
        ax2.bar(df["t"][mask], fired_col[mask], color="#C44E52", alpha=0.4,
                width=0.8, label="fired")
        ax2.set_ylabel("workers fired", color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False)
    else:
        ax.legend(fontsize=7, frameon=False)

    ax.text(0.01, -0.18, "K = workforce headcount = N / tasks_per_worker",
            transform=ax.transAxes, fontsize=7, color="gray")
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


def fig_firing_events(df: pd.DataFrame, T_review: float = math.inf) -> matplotlib.figure.Figure:
    """Firing events at review periods.

    Input:
        df: run_simulation DataFrame with columns 't' (int) and
            'n_review_fired' (int, workers fired at this period's review;
            0 for non-review periods). Shape: (T, 13+).
        T_review: the T_review parameter used for the run (default math.inf).
            Used only to distinguish "review disabled" from "review active
            but no workers met the firing threshold" in the fallback message.
    Output:
        Figure with one Axes: scatter of t values where n_review_fired > 0,
        marker size proportional to count (markersize = 40 * count / max_count,
        minimum 40). Baseline at y=0 via axhline. Only firing periods have
        markers; non-review or zero-fired periods have none.
        If no firings occurred, the plot renders an empty scatter with the
        baseline line and a context-aware text note.
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
        if math.isinf(T_review):
            msg = "review disabled\n(T_review=∞)"
        else:
            msg = f"no firings at this threshold\n(T_review={int(T_review)}, raise firing_threshold)"
        ax.text(0.5, 0.5, msg,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")

    # Always set x-axis to the full simulation range, not matplotlib's [0,1] default
    t_max = int(df["t"].max()) if len(df) > 0 else 1
    ax.set_xlim(-0.5, t_max + 0.5)
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

    # Degenerate annotation: flat nonzero series means all workers trained from early on
    col_vals = col.values
    if len(col_vals) > 0 and col_vals.min() == col_vals.max() and col_vals.max() > 0:
        value = int(col_vals.max())
        first_t = int(df.loc[col == value, "t"].iloc[0])
        ax.text(0.5, 0.5, f"all {value} workers trained by t={first_t}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#55A868", alpha=0.7)

    fig.tight_layout()
    return fig


def fig_wage_histogram(workforce_wage: np.ndarray) -> matplotlib.figure.Figure:
    """Histogram of the final-period worker wage distribution.

    Input:
        workforce_wage: 1-D numpy array of worker wage values from the
            final period (shape: (K,) where K = active worker count).
    Output:
        Figure with one Axes: histogram with 20 bins, dashed vertical mean line.
    Color: #8172B3 (purple, matches fig_wage_bill_over_time).
    No disk I/O — returns Figure only.
    """
    wage = np.asarray(workforce_wage)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(wage, bins=20, color="#8172B3", edgecolor="white", alpha=0.8)
    if len(wage) > 0:
        ax.axvline(float(np.mean(wage)), color="#C44E52", linewidth=1.2,
                   linestyle="--", label=f"mean={np.mean(wage):.3f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("wage")
    ax.set_ylabel("count")
    ax.set_title("Final-Period Wage Distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_wage_vs_mean_output(
    wages_final: np.ndarray,
    mean_output_per_worker: np.ndarray,
    a_trained_final: np.ndarray,
) -> matplotlib.figure.Figure:
    """Scatter of worker wage vs mean per-period output, colored by training status.

    Input:
        wages_final: 1-D array of final-period wages, shape (K,).
        mean_output_per_worker: 1-D array of NaN-averaged per-period output per worker,
            shape (K,). Workers where all output columns are NaN are excluded.
        a_trained_final: 1-D bool array of training status, shape (K,).
    Output:
        Figure with scatter: X=wage, Y=mean output. Two colors:
          trained (#55A868 green), untrained (#4C72B0 blue). Legend shows counts.
        Workers where all output is NaN are excluded (fired before producing; misleading as 0).
    No disk I/O — returns Figure only.
    """
    wages = np.asarray(wages_final)
    mean_out = np.asarray(mean_output_per_worker)
    trained = np.asarray(a_trained_final, dtype=bool)

    valid = ~np.isnan(mean_out)
    wages = wages[valid]
    mean_out = mean_out[valid]
    trained = trained[valid]

    fig, ax = plt.subplots(figsize=(5, 3))

    n_trained = int(trained.sum())
    n_untrained = int((~trained).sum())

    if n_untrained > 0:
        ax.scatter(wages[~trained], mean_out[~trained], color="#4C72B0", alpha=0.7,
                   s=20, label=f"untrained (n={n_untrained})")
    if n_trained > 0:
        ax.scatter(wages[trained], mean_out[trained], color="#55A868", alpha=0.7,
                   s=20, label=f"trained (n={n_trained})")

    if n_trained > 0 or n_untrained > 0:
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel("wage")
    ax.set_ylabel("mean output per period (NaN-averaged)")
    ax.set_title("Wage vs. Mean Output")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_hiring_events(df: pd.DataFrame, enable_hiring: bool = False) -> matplotlib.figure.Figure:
    """Hiring events timeline at review periods.

    Mirrors fig_firing_events: scatter of t values where n_hired > 0.
    Input:
        df: run_simulation DataFrame with columns 't' (int) and
            'n_hired' (int, workers hired this period; 0 when enable_hiring=False).
        enable_hiring: the enable_hiring flag used for the run (default False).
            Used to distinguish "hiring disabled" vs "no hires despite being enabled".
    Output:
        Figure with one Axes: scatter of t values where n_hired > 0.
        Baseline at y=0. Empty-state message when no hires occurred.
    Color: markers in #55A868 (green — opposite semantic of firing red).
    No disk I/O — returns Figure only.
    """
    hired_col = df.get("n_hired", pd.Series(0, index=df.index))
    mask = hired_col > 0
    t_hired = df["t"][mask].values
    n_hired = hired_col[mask].values

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")

    if len(t_hired) > 0:
        max_n = float(n_hired.max())
        sizes = [max(40, 40 * v / max_n * 5) for v in n_hired]
        ax.scatter(t_hired, n_hired, s=sizes, color="#55A868", alpha=0.8,
                   zorder=3, label="workers hired")
        ax.legend(frameon=False, fontsize=8)
    else:
        if not enable_hiring:
            msg = "hiring disabled\n(enable_hiring=False)"
        else:
            msg = "no hiring events\n(no firings to backfill)"
        ax.text(0.5, 0.5, msg,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")

    t_max = int(df["t"].max()) if len(df) > 0 else 1
    ax.set_xlim(-0.5, t_max + 0.5)
    ax.set_xlabel("period")
    ax.set_ylabel("workers hired")
    ax.set_title("Hiring Events Over Time")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
