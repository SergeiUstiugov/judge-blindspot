# -*- coding: utf-8 -*-
"""CLI: judge-blindspot {selftest | doctor | run | calibrate | report}."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .stats import independence_report, bootstrap_ci, _phi, _ratio
from .verdict import verdict_for_pair, VerdictLabel


# ─────────────────────────── selftest ─────────────────────────────────────────

def _run_selftest(n_boot: int = 2000, seed: int = 0) -> int:
    """Synthetic calibration gate: tool must both pass and fail by design.

    Positive control  (known-independent): ratio ≈ 1, phi-CI covers 0.
    Negative control  (known-dependent):   ratio >> 1, phi-CI clearly > 0.
    """
    rng = np.random.default_rng(seed)
    n = 200
    print("=" * 65)
    print("JUDGE-BLINDSPOT SELFTEST (synthetic; must both pass and fail)")
    print("=" * 65)

    # --- A: known-independent (judges miss at independent fixed rates) ---
    Ma = np.column_stack([
        (rng.random(n) < 0.30).astype(int),
        (rng.random(n) < 0.40).astype(int),
    ])
    phi_a, ratio_a, null_a = independence_report(Ma, n_boot=n_boot, seed=seed)
    phi_a_ci   = bootstrap_ci(Ma, lambda m: _phi  (m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed)
    ratio_a_ci = bootstrap_ci(Ma, lambda m: _ratio(m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed+1)
    v_a = verdict_for_pair(phi_a, phi_a_ci, ratio_a_ci, n)
    ok_a = v_a in (VerdictLabel.INDEPENDENT, VerdictLabel.INCONCLUSIVE)

    print(f"\n--- A: deliberately INDEPENDENT (expected: INDEPENDENT or INCONCLUSIVE) ---")
    print(f"  phi   = {phi_a:+.3f}  CI [{phi_a_ci[0]:+.3f}, {phi_a_ci[1]:+.3f}]  shuffle≈{null_a:+.3f}")
    print(f"  ratio = {ratio_a:.3f}  CI [{ratio_a_ci[0]:.3f}, {ratio_a_ci[1]:.3f}]")
    print(f"  marginals: A={Ma[:,0].mean():.2f}  B={Ma[:,1].mean():.2f}")
    print(f"  → verdict: {v_a.value}  ({'OK' if ok_a else 'FAIL — expected INDEPENDENT/INCONCLUSIVE'})")

    # --- B: known-dependent (shared latent 'hard defect' causes joint misses) ---
    hard = rng.random(n) < 0.40
    Mb = np.column_stack([
        (rng.random(n) < 0.15).astype(int),
        (rng.random(n) < 0.15).astype(int),
    ])
    Mb[hard, :] = 1
    phi_b, ratio_b, null_b = independence_report(Mb, n_boot=n_boot, seed=seed)
    phi_b_ci   = bootstrap_ci(Mb, lambda m: _phi  (m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed)
    ratio_b_ci = bootstrap_ci(Mb, lambda m: _ratio(m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed+1)
    v_b = verdict_for_pair(phi_b, phi_b_ci, ratio_b_ci, n)
    ok_b = v_b in (VerdictLabel.OVERLAP, VerdictLabel.DUPLICATE)

    print(f"\n--- B: deliberately DEPENDENT (expected: OVERLAP or DUPLICATE) ---")
    print(f"  phi   = {phi_b:+.3f}  CI [{phi_b_ci[0]:+.3f}, {phi_b_ci[1]:+.3f}]  shuffle≈{null_b:+.3f}")
    print(f"  ratio = {ratio_b:.3f}  CI [{ratio_b_ci[0]:.3f}, {ratio_b_ci[1]:.3f}]")
    print(f"  marginals: A={Mb[:,0].mean():.2f}  B={Mb[:,1].mean():.2f}")
    print(f"  → verdict: {v_b.value}  ({'OK' if ok_b else 'FAIL — expected OVERLAP/DUPLICATE'})")

    ok = ok_a and ok_b
    print(f"\n{'=' * 65}")
    if ok:
        print("SELFTEST PASSED: tool distinguishes independent from dependent.")
        print("  (A→INDEPENDENT/INCONCLUSIVE, B→OVERLAP/DUPLICATE — as required)")
    else:
        print("SELFTEST FAILED: tool does NOT distinguish cases → do NOT run on data.")
    print("=" * 65)
    return 0 if ok else 1


# ─────────────────────────── doctor ───────────────────────────────────────────

def _run_doctor() -> int:
    print("=" * 55)
    print("JUDGE-BLINDSPOT DOCTOR — environment check")
    print("=" * 55)
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    enc = (sys.stdout.encoding or "").lower()
    print(f"Encoding: stdout={sys.stdout.encoding}")
    if enc and "utf" not in enc:
        print("  ! WARNING: stdout not UTF-8 — set PYTHONUTF8=1")

    all_ok = True
    for pkg in ("numpy", "pandas", "matplotlib", "yaml"):
        try:
            mod = __import__(pkg)
            print(f"  [OK ] {pkg}=={getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  [    ] {pkg} — NOT installed (required)")
            all_ok = False
    for pkg in ("httpx", "pytest"):
        try:
            mod = __import__(pkg)
            print(f"  [OK ] {pkg}=={getattr(mod, '__version__', '?')} (optional)")
        except ImportError:
            print(f"  [    ] {pkg} — not installed (optional)")

    print("")
    if all_ok:
        print("Environment OK — run `judge-blindspot selftest` to verify the meter.")
        return 0
    print("Install missing packages: pip install -e .[dev]")
    return 1


# ─────────────────────────── main ─────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="judge-blindspot",
        description="Measures whether LLM judges share blind spots (miss-correlation per defect class).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="synthetic calibration — must both pass and fail")
    st.add_argument("--n-boot", type=int, default=2000)
    st.add_argument("--seed",   type=int, default=0)

    sub.add_parser("doctor", help="check environment: packages, Python, encoding")

    run_p = sub.add_parser("run", help="run judges on corpus (requires API credentials)")
    run_p.add_argument("--config",       default="experiment.yaml")
    run_p.add_argument("--class",        dest="defect_class", default=None)
    run_p.add_argument("--dry-run",      action="store_true")
    run_p.add_argument("--force",        action="store_true", help="override cost-guard")
    run_p.add_argument("--out",          default="results/")

    cal_p = sub.add_parser("calibrate", help="Phase 5 calibration gate (run before Table 3)")
    cal_p.add_argument("--config", default="experiment.yaml")
    cal_p.add_argument("--out",    default="results/calibration/")

    rep_p = sub.add_parser("report", help="emit tables.md and forest plot from results.json")
    rep_p.add_argument("results_json")
    rep_p.add_argument("--out",   default="results/")
    rep_p.add_argument("--class", dest="class_label", default="")

    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        sys.exit(_run_selftest(a.n_boot, a.seed))

    if a.cmd == "doctor":
        sys.exit(_run_doctor())

    if a.cmd == "run":
        print("'run' requires a live corpus and API credentials (Phase 2+).")
        print("Implement judges.py call_fn for your provider, then re-run.")
        sys.exit(1)

    if a.cmd == "calibrate":
        print("'calibrate' is the Phase 5 hard gate.")
        print("Build the Phase 1 corpus first, then run this command.")
        sys.exit(1)

    if a.cmd == "report":
        from .report import save_tables_md, save_forest_plot, save_results
        with open(a.results_json, encoding="utf-8") as f:
            pairwise = json.load(f)
        out = Path(a.out)
        t = save_tables_md(pairwise, out, a.class_label)
        print(f"Tables : {t}")
        fp = save_forest_plot(pairwise, out, a.class_label)
        if fp:
            print(f"Forest plot: {fp}")
        sys.exit(0)


if __name__ == "__main__":
    main()
