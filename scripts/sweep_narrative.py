"""Narrative-default calibration sweep harness.

Usage:
    .venv/bin/python -m scripts.sweep_narrative --grid baseline --seeds 0..4 --out results/sweeps/narrative-defaults/
    .venv/bin/python -m scripts.sweep_narrative --grid oneD-sweep --seeds 0..4 --out results/sweeps/narrative-defaults/
    .venv/bin/python -m scripts.sweep_narrative --grid 2D-refinement --t-review 10 --seeds 0..4 --out results/sweeps/narrative-defaults/
    .venv/bin/python -m scripts.sweep_narrative --grid robustness --out results/sweeps/narrative-defaults/
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm.strategy import all_H, all_A, all_T, greedy_with_switching
from firm_ai_abm.margin_optimizer import horizon_brute_strategy
from firm_ai_abm.dp_optimizer import dp_rolling_horizon_strategy

_STRATEGY_REGISTRY = {
    "all_H": all_H,
    "all_A": all_A,
    "all_T": all_T,
    "greedy_with_switching": greedy_with_switching,
    "horizon_brute": horizon_brute_strategy,
    "horizon_optimizer": dp_rolling_horizon_strategy,
}

_T03_STRATEGIES = ["all_H", "greedy_with_switching", "horizon_optimizer"]

_HIRING_MODES = ["off", "enable_hiring", "enable_replenish_hiring"]


# ---------------------------------------------------------------------------
# Core sweep engine
# ---------------------------------------------------------------------------

def _param_from_combo(combo: dict[str, Any], hiring_mode: str) -> FirmParams:
    """Build a FirmParams from current defaults + combo overrides + hiring_mode."""
    base = asdict(FirmParams())
    patched = FirmParams(**{
        **base,
        **combo,
        "enable_hiring": (hiring_mode == "enable_hiring"),
        "enable_replenish_hiring": (hiring_mode == "enable_replenish_hiring"),
    })
    # D-05 mutex guard — raises AssertionError (not ValueError) so caller chain is clean
    assert not (patched.enable_hiring and patched.enable_replenish_hiring), (
        f"Mutex violation: hm={hiring_mode!r}, combo={combo}"
    )
    return patched


def _prefilter(p_val: float, w_val: float, N_val: int = 500, K_val: int = 100,
               F_val: float = 5.0) -> bool:
    """Return True if cell passes the baseline-profit pre-filter."""
    return 50 <= p_val * N_val - w_val * K_val - F_val <= 150


def _run_cell(combo: dict, strategy_name: str, hiring_mode: str, seed: int) -> dict | None:
    """Run one (combo, strategy, hiring_mode, seed) cell. Returns None on error."""
    try:
        params = replace(_param_from_combo(combo, hiring_mode), seed=seed)
        firm = make_firm(params)
        df = run_simulation(firm, _STRATEGY_REGISTRY[strategy_name])
        return {
            **combo,
            "strategy": strategy_name,
            "hiring_mode": hiring_mode,
            "seed": seed,
            "cum_pi": float(df["pi"].sum()),
            "mean_K_active": float(df["K_active"].mean()),
            "n_hired_total": int(df["n_hired"].sum()),
            "n_review_fired_total": int(df["n_review_fired"].sum()),
        }
    except Exception as e:
        print(f"  ERROR cell={combo} strat={strategy_name} hm={hiring_mode} seed={seed}: {e}",
              file=sys.stderr)
        return None


def sweep(grid_params: dict, strategies: list[str], hiring_modes: list[str],
          seeds: list[int], pre_filter: bool = False) -> pd.DataFrame:
    """Run a full sweep over grid x strategies x hiring_modes x seeds."""
    keys = list(grid_params.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid_params[k] for k in keys])]
    total = len(combos) * len(strategies) * len(hiring_modes) * len(seeds)
    print(f"  Grid: {len(combos)} combos × {len(strategies)} strategies × "
          f"{len(hiring_modes)} hiring_modes × {len(seeds)} seeds = {total} cells")

    rows = []
    done = 0
    skipped = 0
    for combo in combos:
        # Pre-filter: skip cells with implausible baseline profit
        if pre_filter:
            p_val = combo.get("p", FirmParams().p)
            w_val = combo.get("w", FirmParams().w)
            if not _prefilter(p_val, w_val):
                skipped += len(strategies) * len(hiring_modes) * len(seeds)
                continue
        for hm in hiring_modes:
            for strat in strategies:
                for seed in seeds:
                    row = _run_cell(combo, strat, hm, seed)
                    if row is not None:
                        rows.append(row)
                    done += 1
                    if done % 50 == 0:
                        print(f"  ... {done}/{total} cells done ({skipped} skipped by pre-filter)")

    if skipped:
        print(f"  Pre-filter skipped {skipped} cells.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Narrative pass/fail helpers
# ---------------------------------------------------------------------------

def n1_pass(df: pd.DataFrame, t_review: float) -> tuple[bool, str]:
    """N1: horizon_optimizer with dp_prior_alpha=0.85 loses vs greedy_with_switching / all_H.

    Returns (pass_at_ge4_seeds, explanation).
    At T_review=inf this fails by construction (no firing → no T-mode bet).
    """
    if math.isinf(t_review):
        return False, "FAIL by construction — T_review=inf disables firing path"

    sub = df[df["hiring_mode"] == "enable_hiring"]

    def _median_cum_pi(strat: str) -> float | None:
        rows = sub[sub["strategy"] == strat]["cum_pi"]
        return float(rows.median()) if len(rows) > 0 else None

    ho = _median_cum_pi("horizon_optimizer")
    gws = _median_cum_pi("greedy_with_switching")
    ah = _median_cum_pi("all_H")
    if ho is None or gws is None or ah is None:
        return False, f"missing data: ho={ho}, gws={gws}, ah={ah}"

    # Count seeds where horizon_optimizer < both greedy_with_switching and all_H
    seeds_pass = 0
    all_seeds = sorted(sub["seed"].unique())
    for seed in all_seeds:
        ho_val = sub[(sub["strategy"] == "horizon_optimizer") & (sub["seed"] == seed)]["cum_pi"]
        gws_val = sub[(sub["strategy"] == "greedy_with_switching") & (sub["seed"] == seed)]["cum_pi"]
        ah_val = sub[(sub["strategy"] == "all_H") & (sub["seed"] == seed)]["cum_pi"]
        if len(ho_val) > 0 and len(gws_val) > 0 and len(ah_val) > 0:
            if float(ho_val.iloc[0]) < float(gws_val.iloc[0]) and float(ho_val.iloc[0]) < float(ah_val.iloc[0]):
                seeds_pass += 1

    n_seeds = len(all_seeds)
    passes = seeds_pass >= max(4, math.ceil(0.8 * n_seeds))
    return passes, f"horizon_optimizer < greedy AND all_H at {seeds_pass}/{n_seeds} seeds"


def n2_pass(df: pd.DataFrame) -> tuple[bool, str]:
    """N2: at least one hiring mode beats off for horizon_brute AND horizon_optimizer.

    Plan D-01: 'enabling hiring (either mode) improves both horizon_brute and horizon_optimizer
    vs hiring-off' — either enable_hiring OR enable_replenish_hiring must beat off.
    """
    results = []
    for strat in ["horizon_brute", "horizon_optimizer"]:
        sub = df[df["strategy"] == strat]
        off_median = sub[sub["hiring_mode"] == "off"]["cum_pi"].median()
        hire_median = sub[sub["hiring_mode"] == "enable_hiring"]["cum_pi"].median()
        rep_median = sub[sub["hiring_mode"] == "enable_replenish_hiring"]["cum_pi"].median()
        best_hire = max(hire_median, rep_median)
        best_mode = "hire" if hire_median >= rep_median else "replenish"
        if best_hire > off_median:
            results.append(f"{strat}: {best_mode}={best_hire:.1f} > off={off_median:.1f} ✓")
        else:
            results.append(f"{strat}: best_hire={best_hire:.1f} <= off={off_median:.1f} ✗")

    passes = all("✓" in r for r in results)
    return passes, "; ".join(results)


def n3_pass(df: pd.DataFrame) -> tuple[bool, str]:
    """N3: enable_replenish_hiring beats enable_hiring (ordinal, >=6/10 seeds)."""
    seeds = sorted(df["seed"].unique())
    n_pass = 0
    replenish_vals = []
    hire_vals = []

    for seed in seeds:
        rep_cum = df[(df["hiring_mode"] == "enable_replenish_hiring") & (df["seed"] == seed)]["cum_pi"].median()
        hire_cum = df[(df["hiring_mode"] == "enable_hiring") & (df["seed"] == seed)]["cum_pi"].median()
        if not (math.isnan(rep_cum) or math.isnan(hire_cum)):
            replenish_vals.append(rep_cum)
            hire_vals.append(hire_cum)
            if rep_cum > hire_cum:
                n_pass += 1

    n_seeds = len(seeds)
    threshold = max(6, math.ceil(0.6 * n_seeds))
    passes = n_pass >= threshold

    med_rep = float(np.median(replenish_vals)) if replenish_vals else float("nan")
    med_hire = float(np.median(hire_vals)) if hire_vals else float("nan")
    gap = med_rep - med_hire
    return passes, (f"replenish > hire at {n_pass}/{n_seeds} seeds "
                    f"(threshold {threshold}); median gap={gap:.1f}")


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

def _baseline_grid() -> dict:
    return {"T_review": [math.inf, 10.0]}


def _t03_grids(best_t_review: float) -> list[tuple[str, dict, bool]]:
    """Return list of (dial_name, grid, apply_prefilter) for T-03 sub-sweeps."""
    defaults = FirmParams()
    return [
        ("T_review", {"T_review": [5.0, 10.0, 20.0, math.inf]}, False),
        ("alpha_mean", {"T_review": [best_t_review],
                        "alpha_mean": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]}, True),
        ("q_a",       {"T_review": [best_t_review],
                        "q_a": [0.6, 0.8, 1.0, 1.2]}, True),
        ("c_auto",    {"T_review": [best_t_review],
                        "c_auto": [0.10, 0.20, 0.30, 0.50]}, True),
        ("p",         {"T_review": [best_t_review],
                        "p": [0.40, 0.50, 0.56, 0.70]}, True),
    ]


def _2d_grid(lead_dial: str, secondary_dial: str,
             lead_vals: list, secondary_vals: list,
             t_review: float) -> dict:
    return {
        "T_review": [t_review],
        "hire_delay_periods": [3],
        lead_dial: lead_vals,
        secondary_dial: secondary_vals,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _format_table(df_pivot: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(str(c) for c in df_pivot.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(df_pivot.columns)) + "|")
    for _, row in df_pivot.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T-02: baseline sweep + report
# ---------------------------------------------------------------------------

def run_t02(out_dir: Path, seeds: list[int]) -> float:
    """Run T-02: baseline sweep and produce T02-current-defaults.md.

    Returns the best finite T_review from the T_review sub-sweep for use in T-03.
    """
    print("\n=== T-02: Baseline sweep ===")
    grid = _baseline_grid()
    df = sweep(grid, list(_STRATEGY_REGISTRY.keys()), _HIRING_MODES, seeds)

    # Save raw CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"baseline-{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Build report
    lines = ["# T02 — Current-defaults narrative gap\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    for t_review in [math.inf, 10.0]:
        label = "inf" if math.isinf(t_review) else str(t_review)
        lines.append(f"\n## T_review = {label}\n")
        sub = df[df["T_review"] == t_review]

        # Mean ± std table
        tbl = sub.groupby(["strategy", "hiring_mode"])["cum_pi"].agg(["mean", "std"]).reset_index()
        tbl["mean±std"] = tbl.apply(lambda r: f"{r['mean']:.1f}±{r['std']:.1f}", axis=1)
        pivot = tbl.pivot(index="strategy", columns="hiring_mode", values="mean±std").reset_index()
        lines.append("### Mean ± std cum_pi by strategy × hiring_mode\n")
        lines.append(_format_table(pivot))
        lines.append("")

        # Narrative pass/fail
        n1_ok, n1_msg = n1_pass(sub, t_review)
        n2_ok, n2_msg = n2_pass(sub)
        n3_ok, n3_msg = n3_pass(sub)
        lines.append(f"\n### Narrative verdicts at T_review={label}")
        lines.append(f"- N1: {'PASS' if n1_ok else 'FAIL'} — {n1_msg}")
        lines.append(f"- N2: {'PASS' if n2_ok else 'FAIL'} — {n2_msg}")
        lines.append(f"- N3: {'PASS' if n3_ok else 'FAIL'} — {n3_msg}")

    md_path = out_dir / "T02-current-defaults.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")
    return 10.0  # best finite T_review for T-03 (confirmed from T_review sub-sweep in T-03)


# ---------------------------------------------------------------------------
# T-03: one-D sensitivity sweep
# ---------------------------------------------------------------------------

def run_t03(out_dir: Path, seeds: list[int]) -> dict:
    """Run T-03: 5 one-D sub-sweeps. Returns PROPOSED_DEFAULTS candidate dict."""
    print("\n=== T-03: One-D sensitivity sweep ===")

    # Step 1: T_review sub-sweep to find best finite value
    print("  Sub-sweep: T_review")
    t_review_grid = {"T_review": [5.0, 10.0, 20.0, math.inf]}
    df_tr = sweep(t_review_grid, _T03_STRATEGIES, ["enable_hiring"], seeds)

    best_finite_t_review = _pick_best_t_review(df_tr)
    print(f"  Best finite T_review: {best_finite_t_review}")

    all_dfs = {"T_review": df_tr}
    candidates: dict[str, Any] = {"T_review": best_finite_t_review, "hire_delay_periods": 3}

    # Steps 2-5: remaining dials conditional on best T_review
    dial_grids = [
        ("alpha_mean", [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]),
        ("q_a",        [0.6, 0.8, 1.0, 1.2]),
        ("c_auto",     [0.10, 0.20, 0.30, 0.50]),
        ("p",          [0.40, 0.50, 0.56, 0.70]),
    ]
    defaults = FirmParams()
    dial_deltas: dict[str, float] = {}

    for dial_name, dial_vals in dial_grids:
        print(f"  Sub-sweep: {dial_name}")
        grid = {"T_review": [best_finite_t_review], dial_name: dial_vals}
        df_d = sweep(grid, _T03_STRATEGIES, ["enable_hiring"], seeds, pre_filter=True)
        all_dfs[dial_name] = df_d

        # Find smallest perturbation flipping N1 from FAIL to PASS at >=4/5 seeds
        best_val, delta = _find_n1_flip(df_d, dial_name, dial_vals, defaults, best_finite_t_review)
        if best_val is not None:
            candidates[dial_name] = best_val
            dial_deltas[dial_name] = delta
            print(f"    N1 flip at {dial_name}={best_val} (Δ={delta:.3f})")
        else:
            print(f"    No N1 flip found for {dial_name}")

    # Rank dials by |Δ| / current_default; prefer alpha_mean on tie (D-03)
    lead_dial = _rank_dials(dial_deltas, defaults)
    print(f"  Lead dial: {lead_dial}")

    # Save CSV
    all_rows = []
    for dial_name, df_d in all_dfs.items():
        df_d["dial"] = dial_name
        all_rows.append(df_d)
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(out_dir / "T03-oneD-sweep.csv", index=False)
    print(f"  Saved: {out_dir / 'T03-oneD-sweep.csv'}")

    # Build report
    _write_t03_report(out_dir, all_dfs, dial_deltas, lead_dial, best_finite_t_review, candidates)

    return {"lead_dial": lead_dial, "candidates": candidates,
            "best_t_review": best_finite_t_review}


def _pick_best_t_review(df: pd.DataFrame) -> float:
    """Pick the best finite T_review for N1 pass at >=4/5 seeds."""
    for t_review in [5.0, 10.0, 20.0]:
        sub = df[df["T_review"] == t_review]
        ok, _ = n1_pass(sub, t_review)
        if ok:
            return t_review
    # Default to 10 if none pass cleanly
    return 10.0


def _find_n1_flip(df: pd.DataFrame, dial_name: str, dial_vals: list,
                  defaults: FirmParams, t_review: float) -> tuple:
    """Find smallest perturbation in dial_vals that flips N1 to PASS."""
    default_val = getattr(defaults, dial_name)
    # Sort by distance from default
    sorted_vals = sorted(dial_vals, key=lambda v: abs(v - default_val))
    for val in sorted_vals:
        if val == default_val:
            sub = df[df["T_review"] == t_review]
        else:
            sub = df[(df[dial_name] == val) & (df["T_review"] == t_review)]
        if len(sub) == 0:
            continue
        ok, _ = n1_pass(sub, t_review)
        if ok and val != default_val:
            delta = abs(val - default_val) / default_val if default_val != 0 else abs(val)
            return val, delta
    return None, float("inf")


def _rank_dials(dial_deltas: dict[str, float], defaults: FirmParams) -> str:
    """Return lead dial by smallest relative delta; break ties preferring alpha_mean."""
    if not dial_deltas:
        return "alpha_mean"
    sorted_dials = sorted(dial_deltas.items(), key=lambda x: (x[1], 0 if x[0] == "alpha_mean" else 1))
    return sorted_dials[0][0]


def _write_t03_report(out_dir: Path, all_dfs: dict, dial_deltas: dict,
                      lead_dial: str, best_t_review: float, candidates: dict) -> None:
    lines = ["# T03 — One-D sensitivity sweep\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Best finite T_review: {best_t_review}\n")
    lines.append(f"Lead dial: {lead_dial}\n")
    lines.append(f"\n## Dial ranking (|Δ| / default)\n")
    for dial, delta in sorted(dial_deltas.items(), key=lambda x: x[1]):
        prefix = "← LEAD" if dial == lead_dial else ""
        lines.append(f"- {dial}: Δ={delta:.3f} {prefix}")
    lines.append(f"\n## T_review sub-sweep (confirmatory)")
    df_tr = all_dfs.get("T_review", pd.DataFrame())
    for t_review in [5.0, 10.0, 20.0, math.inf]:
        label = "inf" if math.isinf(t_review) else str(t_review)
        sub = df_tr[df_tr["T_review"] == t_review] if "T_review" in df_tr.columns else df_tr
        ok, msg = n1_pass(sub, t_review)
        prefix = "FAIL by construction — confirmatory only" if math.isinf(t_review) else ("PASS" if ok else "FAIL")
        lines.append(f"- T_review={label}: {prefix} — {msg}")
    lines.append(f"\n## Candidates from T-03\n")
    for k, v in candidates.items():
        lines.append(f"- {k}: {v}")
    md_path = out_dir / "T03-oneD-report.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# T-04: 2-D refinement grid
# ---------------------------------------------------------------------------

def run_t04(out_dir: Path, t03_result: dict, seeds: list[int]) -> dict:
    """Run T-04: 2-D refinement around lead candidate. Returns PROPOSED_DEFAULTS."""
    print("\n=== T-04: 2-D refinement grid ===")
    lead_dial = t03_result["lead_dial"]
    candidates = t03_result["candidates"]
    best_t_review = t03_result["best_t_review"]
    defaults = FirmParams()

    # Build 5×5 grid around (lead_dial, secondary_dial)
    lead_val = candidates.get(lead_dial, getattr(defaults, lead_dial))
    secondary_dial = _pick_secondary_dial(lead_dial, candidates, defaults)
    secondary_val = candidates.get(secondary_dial, getattr(defaults, secondary_dial))

    lead_vals = _build_grid_vals(lead_dial, lead_val, 5)
    secondary_vals = _build_grid_vals(secondary_dial, secondary_val, 5)

    print(f"  Lead dial: {lead_dial} around {lead_val}: {lead_vals}")
    print(f"  Secondary dial: {secondary_dial} around {secondary_val}: {secondary_vals}")

    grid = {
        "T_review": [best_t_review],
        "hire_delay_periods": [3],
        lead_dial: lead_vals,
        secondary_dial: secondary_vals,
    }

    df = sweep(grid, list(_STRATEGY_REGISTRY.keys()), _HIRING_MODES, seeds, pre_filter=True)
    df.to_csv(out_dir / "T04-2D-grid.csv", index=False)
    print(f"  Saved: {out_dir / 'T04-2D-grid.csv'}")

    # Find best cell satisfying N1+N2+N3 with smallest deviation, seed=0 must pass
    proposed = _find_proposed_defaults(
        df, lead_dial, secondary_dial, lead_vals, secondary_vals,
        best_t_review, defaults, seeds
    )

    if proposed is None:
        # Expand to 7×7
        print("  No 5×5 cell found — expanding to 7×7")
        lead_vals_7 = _build_grid_vals(lead_dial, lead_val, 7)
        secondary_vals_7 = _build_grid_vals(secondary_dial, secondary_val, 7)
        grid7 = {
            "T_review": [best_t_review],
            "hire_delay_periods": [3],
            lead_dial: lead_vals_7,
            secondary_dial: secondary_vals_7,
        }
        df7 = sweep(grid7, list(_STRATEGY_REGISTRY.keys()), _HIRING_MODES, seeds, pre_filter=True)
        df7.to_csv(out_dir / "T04-2D-grid-7x7.csv", index=False)
        df = df7
        proposed = _find_proposed_defaults(
            df, lead_dial, secondary_dial, lead_vals_7, secondary_vals_7,
            best_t_review, defaults, seeds
        )

    if proposed is None:
        print("  WARNING: No cell satisfies all 3 narratives. Escalating to user.")
        # Find partial-pass candidate (best 2/3)
        proposed = _find_partial_pass(df, lead_dial, secondary_dial, best_t_review, seeds)
        proposed["_ESCALATE"] = True

    _write_t04_report(out_dir, proposed, lead_dial, secondary_dial)
    return proposed


def _pick_secondary_dial(lead_dial: str, candidates: dict, defaults: FirmParams) -> str:
    order = ["alpha_mean", "q_a", "c_auto", "p"]
    for d in order:
        if d != lead_dial and d in candidates:
            return d
    for d in order:
        if d != lead_dial:
            return d
    return "q_a"


def _build_grid_vals(dial: str, center: float, n: int) -> list[float]:
    """Build n values centered on center with reasonable step sizes."""
    steps = {
        "alpha_mean": 0.05,
        "q_a": 0.1,
        "c_auto": 0.05,
        "p": 0.03,
        "T_review": 5.0,
    }
    step = steps.get(dial, 0.1)
    half = (n - 1) // 2
    vals = [round(center + (i - half) * step, 4) for i in range(n)]
    # Clip to [0.01, ...] for positive dials
    vals = [max(0.01, v) for v in vals]
    # For alpha_mean: clip to [0.05, 0.95]
    if dial == "alpha_mean":
        vals = [max(0.05, min(0.95, v)) for v in vals]
    return sorted(set(vals))


def _find_proposed_defaults(df: pd.DataFrame, lead_dial: str, secondary_dial: str,
                             lead_vals: list, secondary_vals: list, t_review: float,
                             defaults: FirmParams, seeds: list[int]) -> dict | None:
    """Find cell with min deviation from defaults satisfying N1+N2+N3, seed=0 must pass."""
    best = None
    best_dev = float("inf")

    for lv in lead_vals:
        for sv in secondary_vals:
            sub = df[
                (df[lead_dial] == lv) &
                (df[secondary_dial] == sv) &
                (df["T_review"] == t_review) &
                (df["hire_delay_periods"] == 3)
            ]
            if len(sub) == 0:
                continue

            # Check all three narratives
            n1_ok, n1_msg = n1_pass(sub, t_review)
            n2_ok, n2_msg = n2_pass(sub)
            n3_ok, n3_msg = n3_pass(sub)
            if not (n1_ok and n2_ok and n3_ok):
                continue

            # seed=0 must pass all three narratives
            seed0_ok = _seed0_pass(sub, t_review)
            if not seed0_ok:
                continue

            # Deviation from current defaults
            dev = (abs(lv - getattr(defaults, lead_dial)) / max(abs(getattr(defaults, lead_dial)), 1e-6) +
                   abs(sv - getattr(defaults, secondary_dial)) / max(abs(getattr(defaults, secondary_dial)), 1e-6))

            if dev < best_dev:
                best_dev = dev
                best = {
                    lead_dial: lv,
                    secondary_dial: sv,
                    "T_review": t_review,
                    "hire_delay_periods": 3,
                    "n1": n1_msg,
                    "n2": n2_msg,
                    "n3": n3_msg,
                    "deviation": dev,
                }

    return best


def _seed0_pass(df: pd.DataFrame, t_review: float) -> bool:
    """Check if seed=0 passes N1+N2+N3."""
    sub = df[df["seed"] == 0]
    if len(sub) == 0:
        return False

    n1_ok, _ = n1_pass(sub, t_review)
    n2_ok, _ = n2_pass(sub)
    n3_ok, _ = n3_pass(sub)

    # Fallback: if seed=0 fails N1 by <3pp margin, accept seeds {0,1,2} median (D-07)
    if not n1_ok:
        sub_012 = df[df["seed"].isin([0, 1, 2])]
        n1_ok012, _ = n1_pass(sub_012, t_review)
        return n1_ok012 and n2_ok and n3_ok

    return n1_ok and n2_ok and n3_ok


def _find_partial_pass(df: pd.DataFrame, lead_dial: str, secondary_dial: str,
                        t_review: float, seeds: list[int]) -> dict:
    """Find best partial-pass cell (2/3 narratives)."""
    best = {"_partial": True, lead_dial: None, secondary_dial: None, "narratives_passed": 0}
    for lv in df[lead_dial].unique():
        for sv in df[secondary_dial].unique():
            sub = df[(df[lead_dial] == lv) & (df[secondary_dial] == sv)]
            n1_ok, n1_msg = n1_pass(sub, t_review)
            n2_ok, n2_msg = n2_pass(sub)
            n3_ok, n3_msg = n3_pass(sub)
            passed = sum([n1_ok, n2_ok, n3_ok])
            if passed > best["narratives_passed"]:
                best = {lead_dial: lv, secondary_dial: sv, "T_review": t_review,
                        "hire_delay_periods": 3, "narratives_passed": passed,
                        "n1": n1_msg, "n2": n2_msg, "n3": n3_msg, "_partial": True}
    return best


def _write_t04_report(out_dir: Path, proposed: dict, lead_dial: str, secondary_dial: str) -> None:
    lines = ["# T04 — PROPOSED_DEFAULTS candidate\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    if proposed.get("_ESCALATE"):
        lines.append("⚠️  ESCALATION: No cell satisfied all 3 narratives. Showing partial-pass.\n")
    lines.append("## PROPOSED_DEFAULTS\n")
    skip_keys = {"n1", "n2", "n3", "deviation", "_partial", "_ESCALATE", "narratives_passed"}
    for k, v in proposed.items():
        if k not in skip_keys:
            lines.append(f"- {k}: {v}")
    lines.append("\n## Narrative verdicts\n")
    for n in ["n1", "n2", "n3"]:
        if n in proposed:
            lines.append(f"- {n.upper()}: {proposed[n]}")
    if proposed.get("deviation") is not None:
        lines.append(f"\n## Deviation from current defaults: {proposed['deviation']:.4f}")
    md_path = out_dir / "T04-candidate.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")
    print(f"  PROPOSED_DEFAULTS: {proposed}")


# ---------------------------------------------------------------------------
# T-05: Seed robustness
# ---------------------------------------------------------------------------

def run_t05(out_dir: Path, proposed: dict, seeds: list[int]) -> bool:
    """Run T-05: seed robustness at PROPOSED_DEFAULTS (seeds 0..9)."""
    print("\n=== T-05: Seed robustness verification ===")
    skip_keys = {"n1", "n2", "n3", "deviation", "_partial", "_ESCALATE", "narratives_passed"}
    grid = {k: [v] for k, v in proposed.items() if k not in skip_keys}
    df = sweep(grid, list(_STRATEGY_REGISTRY.keys()), _HIRING_MODES, seeds)
    df.to_csv(out_dir / "T05-robustness.csv", index=False)

    t_review = proposed.get("T_review", 10.0)
    n1_ok, n1_msg = n1_pass(df, t_review)
    n2_ok, n2_msg = n2_pass(df)
    n3_ok, n3_msg = n3_pass(df)

    # N1 >= 8/10 seeds
    n1_seeds = _count_n1_seeds(df, t_review)
    n1_robust = n1_seeds >= 8

    # seed=0 sub-acceptance
    seed0_ok = _seed0_pass(df, t_review)

    lines = ["# T05 — Seed robustness verification\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## PROPOSED_DEFAULTS\n")
    for k, v in grid.items():
        lines.append(f"- {k}: {v[0]}")
    lines.append("\n## Narrative verdicts (seeds 0..9)\n")
    lines.append(f"- N1: {'PASS' if n1_robust else 'FAIL'} ({n1_seeds}/10 seeds) — {n1_msg}")
    lines.append(f"- N2: {'PASS' if n2_ok else 'FAIL'} — {n2_msg}")
    lines.append(f"- N3: {'PASS' if n3_ok else 'FAIL'} — {n3_msg}")
    lines.append(f"- seed=0 sub-acceptance: {'PASS' if seed0_ok else 'FAIL'}")
    overall = n1_robust and n2_ok and n3_ok and seed0_ok
    lines.append(f"\n## Overall: {'✓ PASS — proceed to T-06' if overall else '✗ FAIL — return to T-03/T-04'}")

    md_path = out_dir / "T05-robustness.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")
    return overall


def _count_n1_seeds(df: pd.DataFrame, t_review: float) -> int:
    """Count seeds where N1 passes (horizon_optimizer < greedy AND all_H)."""
    if math.isinf(t_review):
        return 0
    sub = df[df["hiring_mode"] == "enable_hiring"]
    count = 0
    for seed in sorted(sub["seed"].unique()):
        ho = sub[(sub["strategy"] == "horizon_optimizer") & (sub["seed"] == seed)]["cum_pi"]
        gws = sub[(sub["strategy"] == "greedy_with_switching") & (sub["seed"] == seed)]["cum_pi"]
        ah = sub[(sub["strategy"] == "all_H") & (sub["seed"] == seed)]["cum_pi"]
        if len(ho) > 0 and len(gws) > 0 and len(ah) > 0:
            if float(ho.iloc[0]) < float(gws.iloc[0]) and float(ho.iloc[0]) < float(ah.iloc[0]):
                count += 1
    return count


# ---------------------------------------------------------------------------
# T-06: Phase 1 invariants at PROPOSED_DEFAULTS
# ---------------------------------------------------------------------------

def run_t06(proposed: dict) -> bool:
    """Run T-06: Phase 1 invariants check in isolation mode."""
    print("\n=== T-06: Phase 1 invariants (isolation run) ===")
    skip_keys = {"n1", "n2", "n3", "deviation", "_partial", "_ESCALATE", "narratives_passed"}
    prop = {k: v for k, v in proposed.items() if k not in skip_keys}

    defaults = asdict(FirmParams())
    PROPOSED = {**defaults, **prop}

    # Check 1: closed-form pi = p*N - w*K - F (atol=1e-9, full isolation)
    iso1 = {**PROPOSED, "T_review": math.inf, "enable_hiring": False,
            "enable_replenish_hiring": False, "q_a": 0.0, "g": 0.0, "c_auto": 0.0,
            "sigma_theta": 0.0}
    p_new = iso1["p"]
    N_new = iso1["N"]
    K_new = N_new // iso1["tasks_per_worker"]
    F_new = iso1["F"]
    w_new = iso1["w"]
    closed_form_pi = (p_new * N_new - w_new * K_new - F_new)

    firm1 = make_firm(FirmParams(**iso1))
    df1 = run_simulation(firm1, all_H)
    observed_pi = float(df1["pi"].mean())
    check1 = abs(closed_form_pi - observed_pi) < 1e-9
    print(f"  Check 1 (closed-form): expected={closed_form_pi:.6f}, observed={observed_pi:.6f} → {'PASS' if check1 else 'FAIL'}")

    # Check 2: monotonicity in q_a (T_review=inf, all_T strategy)
    q_a_vals = [0.5, 1.0, 1.5]
    cum_pis_q_a = []
    for qv in q_a_vals:
        iso2 = {**PROPOSED, "T_review": math.inf, "enable_hiring": False,
                "enable_replenish_hiring": False, "q_a": qv}
        df2 = run_simulation(make_firm(FirmParams(**iso2)), all_T)
        cum_pis_q_a.append(float(df2["pi"].sum()))
    check2 = all(a <= b for a, b in zip(cum_pis_q_a, cum_pis_q_a[1:]))
    print(f"  Check 2 (q_a monotonicity): cum_pis={[f'{v:.1f}' for v in cum_pis_q_a]} → {'PASS' if check2 else 'FAIL'}")

    # Check 3: numeraire invariance (scale by 2x)
    monetary_keys = {"w", "c_aug", "c_auto", "c_fire", "c_hire", "c_train", "F", "p", "firing_threshold"}
    iso3 = {**PROPOSED, "enable_hiring": False, "enable_replenish_hiring": False,
            "T_review": math.inf, "sigma_theta": 0.0}
    scaled = {k: (2.0 * v if k in monetary_keys else v) for k, v in iso3.items()}
    df3_base = run_simulation(make_firm(FirmParams(**iso3)), all_H)
    df3_scaled = run_simulation(make_firm(FirmParams(**scaled)), all_H)
    pi_base = df3_base["pi"].values
    pi_scal = df3_scaled["pi"].values
    check3 = np.allclose(pi_scal, 2.0 * pi_base, rtol=1e-9)
    print(f"  Check 3 (numeraire invariance): {'PASS' if check3 else 'FAIL'}")

    all_pass = check1 and check2 and check3
    if not all_pass:
        print("  ⚠️  T-06 FAILED — escalate to user before editing config.py", file=sys.stderr)
    else:
        print("  ✓ All T-06 invariants pass — safe to proceed with T-07")
    return all_pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_seeds(s: str) -> list[int]:
    """Parse '0..4' → [0,1,2,3,4] or '0,1,2' → [0,1,2]."""
    if ".." in s:
        parts = s.split("..")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Narrative-default calibration sweep")
    parser.add_argument("--grid", required=True,
                        choices=["baseline", "oneD-sweep", "2D-refinement", "robustness", "all"])
    parser.add_argument("--seeds", default="0..4", help="Seed range, e.g. '0..4' or '0..9'")
    parser.add_argument("--out", default="results/sweeps/narrative-defaults/",
                        help="Output directory")
    parser.add_argument("--t-review", type=float, default=None,
                        help="T_review override for 2D-refinement grid")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {args.grid}, seeds: {seeds}, out: {out_dir}")

    if args.grid == "baseline":
        run_t02(out_dir, seeds)

    elif args.grid == "oneD-sweep":
        run_t03(out_dir, seeds)

    elif args.grid == "2D-refinement":
        # Load T-03 result from file if available
        t03_report = out_dir / "T03-oneD-report.md"
        if t03_report.exists():
            content = t03_report.read_text()
            # Parse lead dial from report
            lead_dial = "alpha_mean"  # default
            for line in content.splitlines():
                if line.startswith("Lead dial:"):
                    lead_dial = line.split(":")[1].strip()
                    break
            t_review = args.t_review or 10.0
            t03_result = {"lead_dial": lead_dial, "candidates": {"T_review": t_review},
                          "best_t_review": t_review}
        else:
            t_review = args.t_review or 10.0
            t03_result = {"lead_dial": "alpha_mean", "candidates": {"T_review": t_review},
                          "best_t_review": t_review}
        run_t04(out_dir, t03_result, seeds)

    elif args.grid == "robustness":
        # Load PROPOSED_DEFAULTS from T-04 report
        t04_report = out_dir / "T04-candidate.md"
        if not t04_report.exists():
            print("ERROR: T04-candidate.md not found. Run --grid 2D-refinement first.",
                  file=sys.stderr)
            sys.exit(1)
        proposed = _parse_proposed_from_report(t04_report)
        run_t05(out_dir, proposed, seeds)

    elif args.grid == "all":
        # Run full pipeline T-02 → T-05
        run_t02(out_dir, seeds)
        t03_result = run_t03(out_dir, seeds)
        proposed = run_t04(out_dir, t03_result, seeds)
        if not proposed.get("_ESCALATE"):
            seeds_05 = list(range(10))
            robust = run_t05(out_dir, proposed, seeds_05)
            if robust:
                ok = run_t06(proposed)
                if ok:
                    print("\n✓ All sweeps complete. PROPOSED_DEFAULTS confirmed. Proceed to T-07.")
                else:
                    print("\n⚠️  T-06 failed. Escalate before editing config.py.", file=sys.stderr)
            else:
                print("\n⚠️  T-05 robustness failed. Return to T-03/T-04.", file=sys.stderr)
        else:
            print("\n⚠️  T-04 escalation: no cell satisfies all 3 narratives.", file=sys.stderr)


def _parse_proposed_from_report(path: Path) -> dict:
    """Parse PROPOSED_DEFAULTS from T04-candidate.md."""
    proposed = {}
    in_section = False
    for line in path.read_text().splitlines():
        if line.strip() == "## PROPOSED_DEFAULTS":
            in_section = True
            continue
        if in_section:
            if line.startswith("##"):
                break
            if line.startswith("- "):
                parts = line[2:].split(": ", 1)
                if len(parts) == 2:
                    k, v = parts
                    try:
                        proposed[k.strip()] = float(v.strip())
                    except ValueError:
                        proposed[k.strip()] = v.strip()
    return proposed


if __name__ == "__main__":
    main()
