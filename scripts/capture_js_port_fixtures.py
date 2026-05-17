"""Capture JSON reference fixtures for the JS port byte-parity tests.

Run from FirmBehavior root:
    .venv/bin/python scripts/capture_js_port_fixtures.py

Outputs to .workflow_artifacts/js-embed-simulator/fixtures/:
  fixture_baseline_seed0.json      (all_H)
  fixture_greedy_seed0.json        (greedy_profit)
  fixture_switching_seed0.json     (greedy_with_switching)
  fixture_all_T_seed0.json         (all_T, finite T_review=10, exercises firing+hire-back)
  _provenance.json                 (commit hash, capture time, config hash)

JS port imports alpha[] and beta[] as taskAttrsOverride so byte-parity is
achievable despite mulberry32 != numpy PCG64 (CRIT-4 fix).
"""
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone

from firm_ai_abm.config import FirmParams
from firm_ai_abm.firm import make_firm
from firm_ai_abm.simulate import run_simulation
from firm_ai_abm import all_H, all_A, all_T, greedy_profit, greedy_with_switching

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "tests", "fixtures", "js_port",
)
os.makedirs(OUT_DIR, exist_ok=True)

# Shared params for all fixtures: N=100 (smaller than 500 default for fixture size),
# sigma_theta=0 (no worker heterogeneity; D-03), all optional features dormant.
_BASE_KWARGS = dict(
    N=100,
    T=60,
    seed=0,
    sigma_theta=0.0,
    T_review=math.inf,
    enable_hiring=True,
    enable_replenish_hiring=False,
    enable_training_delay=False,
    scenario_mode="price",
    belief_alpha=None,
)

STRATEGY_FNS = {
    "all_H": all_H,
    "all_A": all_A,
    "all_T": all_T,
    "greedy_profit": greedy_profit,
    "greedy_with_switching": greedy_with_switching,
}


def _config_hash() -> str:
    """sha1 of sorted-key JSON of no-arg FirmParams() defaults."""
    defaults = FirmParams()
    d = {}
    for k, v in defaults.__dict__.items():
        if v == math.inf:
            d[k] = "Infinity"
        elif isinstance(v, float) and math.isnan(v):
            d[k] = "NaN"
        else:
            d[k] = v
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


def _params_to_json(p: FirmParams) -> dict:
    d = {}
    for k, v in p.__dict__.items():
        if v == math.inf:
            d[k] = "Infinity"
        elif isinstance(v, float) and math.isnan(v):
            d[k] = "NaN"
        else:
            d[k] = v
    return d


def capture(strategy_name: str, params: FirmParams, path: str) -> None:
    fn = STRATEGY_FNS[strategy_name]
    firm = make_firm(params)
    df = run_simulation(firm, fn)

    alpha = firm.alpha.tolist()
    beta = firm.beta.tolist()

    assert len(df) == params.T, f"Expected {params.T} rows, got {len(df)}"
    assert len(alpha) == params.N
    assert len(beta) == params.N

    history = []
    for _, row in df.iterrows():
        rec = {
            "t": int(row["t"]),
            "Y": float(row["Y"]),
            "C": float(row["C"]),
            "pi": float(row["pi"]),
            "K": int(row["K"]),
            "K_active": int(row["K_active"]),
            "wage_bill": float(row["wage_bill"]),
            "adj_cost": float(row["adj_cost"]),
            "n_review_fired": int(row["n_review_fired"]),
            "n_hired": int(row["n_hired"]),
            "modes": row["modes"].tolist() if hasattr(row["modes"], "tolist") else list(row["modes"]),
        }
        history.append(rec)

    # MAJ-3: fixture non-degeneracy guard.
    # With sigma_theta=0.0, greedy strategies lock into a single dominant mode
    # for the entire run (task scores are constant). That is valid model behavior,
    # not a bug. Instead, verify each fixture exercises the intended code path:
    #   greedy_*: at least 1 non-H mode at t=0 (greedy score calculation ran)
    #   all_T: at least one n_review_fired > 0 (firing path exercised)
    if strategy_name in ("greedy_profit", "greedy_with_switching"):
        modes0 = history[0]["modes"]
        non_h = sum(m != 0 for m in modes0)
        assert non_h > 0, (
            f"Fixture '{strategy_name}' has all-H at t=0 — greedy score "
            "calculation produced no A/T tasks, which is unexpected."
        )
    elif strategy_name == "all_T":
        fired = sum(r["n_review_fired"] for r in history)
        assert fired > 0, (
            "all_T fixture: expected at least one firing event (finite T_review=10) "
            "but n_review_fired is 0 for all periods."
        )

    fixture = {
        "strategy": strategy_name,
        "seed": params.seed,
        "params": _params_to_json(params),
        "alpha": alpha,
        "beta": beta,
        "history": history,
    }
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"    wrote {os.path.basename(path)}  ({len(history)} periods, {len(alpha)} tasks)")


def main() -> None:
    print("Capturing JS port reference fixtures...")

    fixtures = [
        ("all_H",                  FirmParams(**_BASE_KWARGS),
         "fixture_baseline_seed0.json"),
        ("greedy_profit",          FirmParams(**_BASE_KWARGS),
         "fixture_greedy_seed0.json"),
        ("greedy_with_switching",  FirmParams(**_BASE_KWARGS),
         "fixture_switching_seed0.json"),
        # all_T with finite T_review=10 so firing+hire-back path is exercised
        ("all_T",                  FirmParams(**{**_BASE_KWARGS, "T_review": 10.0}),
         "fixture_all_T_seed0.json"),
    ]

    for strategy, params, fname in fixtures:
        print(f"  [{strategy}]")
        capture(strategy, params, os.path.join(OUT_DIR, fname))

    # _provenance.json
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    provenance = {
        "firmBehaviorCommit": commit,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "configHash": _config_hash(),
    }
    prov_path = os.path.join(OUT_DIR, "_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  wrote _provenance.json  (configHash={provenance['configHash'][:12]}...)")
    print("Done.")


if __name__ == "__main__":
    main()
