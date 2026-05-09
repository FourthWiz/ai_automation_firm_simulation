"""Phase 1 deliverables runner.

Single command to produce all Phase 1 outputs (§14 DoD):
  - Tier A + Tier B validation (all 6 checks)
  - results/figures/fig1_primary_lines.{png,svg}  — MANDATORY
  - results/figures/fig2_small_multiples_q_a.{png,svg}  — MANDATORY
  - results/figures/fig3_mode_mix_greedy.{png,svg}  — OPTIONAL

Note: results/observation.md is NOT generated here — it is a hand-written
personal observation (§14 item 5, plan D-03). Write it after inspecting the
figures produced by this script.

Usage:
    .venv/bin/python notebooks/run_phase1_deliverables.py
"""
import json
import pathlib
import sys

# Ensure project root is on sys.path when run directly
_root = pathlib.Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from firm_ai_abm.validate import run_all_checks
from firm_ai_abm.viz import fig1_primary_lines, fig2_small_multiples_q_a, fig3_mode_mix_greedy
import matplotlib.pyplot as plt

SAVE_DIR = _root / "results" / "figures"


def main():
    print("=" * 60)
    print("Phase 1 deliverables runner")
    print("=" * 60)

    # 1. Validation: all six checks
    print("\n[1/4] Running all validation checks (Tier A + Tier B)...")
    results = run_all_checks()
    print(json.dumps(
        {
            "all_passed": results["all_passed"],
            "tier_a": {
                k: {"passed": v["passed"]}
                for k, v in results["tier_a"].items()
                if k != "all_passed"
            },
            "tier_a_all_passed": results["tier_a"]["all_passed"],
            "tier_b": {
                k: {"passed": v["passed"]}
                for k, v in results["tier_b"].items()
                if k != "all_passed"
            },
            "tier_b_all_passed": results["tier_b"]["all_passed"],
        },
        indent=2,
    ))

    if not results["all_passed"]:
        print("\nWARNING: Some validation checks FAILED. See details above.")
    else:
        print("\nAll 6 validation checks PASSED.")

    # 2. fig1: primary lines (MANDATORY)
    print(f"\n[2/4] Generating fig1_primary_lines -> {SAVE_DIR} ...")
    fig1 = fig1_primary_lines(save_dir=SAVE_DIR)
    plt.close(fig1)
    print(f"  Saved: {SAVE_DIR}/fig1_primary_lines.png")
    print(f"  Saved: {SAVE_DIR}/fig1_primary_lines.svg")

    # 3. fig2: small multiples by q_a (MANDATORY)
    print(f"\n[3/4] Generating fig2_small_multiples_q_a -> {SAVE_DIR} ...")
    fig2 = fig2_small_multiples_q_a(save_dir=SAVE_DIR)
    plt.close(fig2)
    print(f"  Saved: {SAVE_DIR}/fig2_small_multiples_q_a.png")
    print(f"  Saved: {SAVE_DIR}/fig2_small_multiples_q_a.svg")

    # 4. fig3: mode mix (OPTIONAL)
    print(f"\n[4/4] Generating fig3_mode_mix_greedy -> {SAVE_DIR} (optional)...")
    fig3 = fig3_mode_mix_greedy(save_dir=SAVE_DIR)
    plt.close(fig3)
    print(f"  Saved: {SAVE_DIR}/fig3_mode_mix_greedy.png")
    print(f"  Saved: {SAVE_DIR}/fig3_mode_mix_greedy.svg")

    # Summary
    print("\n" + "=" * 60)
    obs_path = _root / "results" / "observation.md"
    if obs_path.exists() and obs_path.stat().st_size > 50:
        print("results/observation.md: EXISTS and non-empty. Phase 1 DoD complete.")
    else:
        print(
            "REMINDER: results/observation.md is missing or empty.\n"
            "Write your personal observation about strategy ranking after\n"
            "inspecting the figures above (§14 item 5, plan D-03)."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
