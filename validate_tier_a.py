"""Tier A gate driver — run from project root.

Usage:
    python validate_tier_a.py

Exits 0 if all four Tier A checks pass, non-zero otherwise.
This is the kernel gate: if any check fails, do not proceed to Stage 4.
"""
import sys

from firm_ai_abm.validate import run_tier_a

results = run_tier_a()

for name in ("check1", "check3", "check4", "check5"):
    status = "PASS" if results[name]["passed"] else "FAIL"
    print(f"{name}: {status}")
    if not results[name]["passed"]:
        print(f"  details: {results[name]['details']}")

print(f"\nTier A overall: {'PASS' if results['all_passed'] else 'FAIL'}")
sys.exit(0 if results["all_passed"] else 1)
