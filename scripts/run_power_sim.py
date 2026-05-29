#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 power simulation script.

Usage:
    python scripts/run_power_sim.py               # default grid, n_sims=300
    python scripts/run_power_sim.py --fast         # n_sims=100 for quick check
    python scripts/run_power_sim.py --full         # n_sims=500, n_boot=500 (paper quality)
    python scripts/run_power_sim.py --write-yaml   # also update experiment.yaml with target_n

Outputs (to results/power_sim/):
    power_results.csv       raw cell-level results
    power_table.md          formatted table for paper appendix
    power_summary.json      recommendation + scenario breakdown
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# make the package importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from judge_blindspot.power import (
    run_power_grid, recommend_n, results_to_csv, results_to_table_md,
    DEFAULT_PHIS, DEFAULT_MARGINALS, DEFAULT_N_VALUES,
    DEFAULT_CI_HW_TARGET, DEFAULT_POWER_THRESHOLD,
)


def _update_yaml(yaml_path: Path, target_n: int) -> None:
    """Update target_n in experiment.yaml in-place (simple line replacement)."""
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated = []
    found = False
    for line in lines:
        if line.strip().startswith("target_n:"):
            updated.append(f"target_n: {target_n}  # set by Phase 6 power simulation")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"target_n: {target_n}  # set by Phase 6 power simulation")
    yaml_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"Updated target_n={target_n} in {yaml_path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 6 power simulation")
    ap.add_argument("--fast",       action="store_true",
                    help="n_sims=100, n_boot=100 for quick sanity check")
    ap.add_argument("--full",       action="store_true",
                    help="n_sims=500, n_boot=500 for paper-quality results")
    ap.add_argument("--write-yaml", action="store_true",
                    help="update target_n in experiment.yaml")
    ap.add_argument("--out",        default="results/power_sim",
                    help="output directory (default: results/power_sim)")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--power-threshold", type=float, default=DEFAULT_POWER_THRESHOLD,
                    help=f"minimum power to declare adequate n (default {DEFAULT_POWER_THRESHOLD})")
    a = ap.parse_args()

    if a.fast:
        n_sims, n_boot = 100, 100
        label = "fast"
    elif a.full:
        n_sims, n_boot = 500, 500
        label = "full"
    else:
        n_sims, n_boot = 300, 200
        label = "default"

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PHASE 6 POWER SIMULATION  (mode={label}, n_sims={n_sims}, n_boot={n_boot})")
    print(f"  criterion  : CI half-width <= {DEFAULT_CI_HW_TARGET} "
          f"in >= {a.power_threshold*100:.0f}% of simulations")
    print(f"  phi grid   : {DEFAULT_PHIS}")
    print(f"  marginals  : {DEFAULT_MARGINALS}")
    print(f"  n sweep    : {DEFAULT_N_VALUES}")
    print(f"  seed       : {a.seed}")
    print("=" * 70)

    t0 = time.time()
    results = run_power_grid(
        n_sims=n_sims,
        n_boot=n_boot,
        seed=a.seed,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # --- recommendation ---
    rec = recommend_n(results, power_threshold=a.power_threshold)

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"  Target n per defect class : {rec['target_n']}")
    print(f"  (raw worst-case min n = {rec['raw_max_n']}, "
          f"+ {(rec['safety_margin']-1)*100:.0f}% margin, rounded to 10)")
    print(f"  Infeasible scenarios      : {rec['n_infeasible']}")
    print()
    print("  Scenario breakdown (minimum n to reach power threshold):")
    for scenario, min_n in sorted(rec["scenario_min_n"].items()):
        flag = "⚠ never reached" if min_n is None else ""
        print(f"    {scenario:<35} min_n = {min_n!s:<6} {flag}")
    print("=" * 70)

    # --- save outputs ---
    csv_path  = out_dir / "power_results.csv"
    md_path   = out_dir / "power_table.md"
    json_path = out_dir / "power_summary.json"

    csv_path.write_text(results_to_csv(results), encoding="utf-8")
    md_path.write_text(
        results_to_table_md(results, a.power_threshold, DEFAULT_CI_HW_TARGET),
        encoding="utf-8",
    )
    rec_serializable = dict(rec)
    rec_serializable["elapsed_s"]   = round(elapsed, 1)
    rec_serializable["n_sims"]      = n_sims
    rec_serializable["n_boot"]      = n_boot
    rec_serializable["seed"]        = a.seed
    rec_serializable["mode"]        = label
    json_path.write_text(json.dumps(rec_serializable, indent=2), encoding="utf-8")

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  {csv_path.name}  {md_path.name}  {json_path.name}")

    # --- update experiment.yaml ---
    if a.write_yaml:
        repo_root = Path(__file__).parent.parent
        yaml_path = repo_root / "experiment.yaml"
        if yaml_path.exists():
            _update_yaml(yaml_path, rec["target_n"])
        else:
            print(f"WARNING: {yaml_path} not found — skipping yaml update")

    return rec["target_n"]


if __name__ == "__main__":
    main()
