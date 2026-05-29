# -*- coding: utf-8 -*-
"""Phase 6 power simulation: find minimum n where CI half-width <= 0.20 with
high probability across the full hypothesis grid.

Pre-registered grid (lock before data collection):
  phi values : 0.0 (H3 null), 0.3 (H2 low), 0.5 (H2 high), 0.8 (H1 duplicate)
  marginals  : 0.2/0.2, 0.3/0.3, 0.4/0.4, 0.5/0.5, 0.3/0.5 (asymmetric)
  n sweep    : 30, 50, 70, 90, 110, 130, 150, 200
  criterion  : CI half-width <= 0.20 in >= 80% of simulations

Generation model: exact 2x2 joint distribution parameterised by
  P(A=1,B=1) = p1*p2 + phi * sqrt(p1*(1-p1)*p2*(1-p2))
so the sample has the target phi by construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stats import _phi, bootstrap_ci


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized bootstrap for phi — avoids Python loop, ~20-50x faster
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_phi_ci_fast(
    M: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Vectorized percentile CI for phi. All n_boot resamples in one numpy call.

    Replaces the generic bootstrap_ci (which has a Python for-loop) with a
    batched matrix approach: sample indices shape (n_boot, n), then compute
    phi across all rows simultaneously.
    """
    n = len(M)
    # (n_boot, n) integer indices — one numpy call, no Python loop
    idx = rng.integers(0, n, size=(n_boot, n))
    # (n_boot, n, 2) — fancy indexing resamples all bootstraps at once
    S = M[idx].astype(float)
    A, B = S[:, :, 0], S[:, :, 1]              # each (n_boot, n)
    A_m = A.mean(axis=1, keepdims=True)
    B_m = B.mean(axis=1, keepdims=True)
    A_s = A.std(axis=1)
    B_s = B.std(axis=1)
    valid = (A_s > 0) & (B_s > 0)
    cov   = ((A - A_m) * (B - B_m)).mean(axis=1)
    phi_v = np.where(valid, cov / np.where(valid, A_s * B_s, 1.0), np.nan)
    clean = phi_v[~np.isnan(phi_v)]
    if len(clean) < 10:
        return (float("nan"), float("nan"))
    return float(np.percentile(clean, 100 * alpha / 2)), \
           float(np.percentile(clean, 100 * (1 - alpha / 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Default pre-registered grid (written to experiment.yaml after run)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PHIS: List[float] = [0.0, 0.3, 0.5, 0.8]
DEFAULT_MARGINALS: List[Tuple[float, float]] = [
    (0.2, 0.2),
    (0.3, 0.3),
    (0.4, 0.4),
    (0.5, 0.5),
    (0.3, 0.5),   # asymmetric — one stricter, one laxer judge
]
DEFAULT_N_VALUES: List[int] = [30, 50, 70, 90, 110, 130, 150, 200]
DEFAULT_CI_HW_TARGET: float = 0.20
DEFAULT_POWER_THRESHOLD: float = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _joint_probs(
    true_phi: float, p1: float, p2: float
) -> Optional[Tuple[float, float, float, float]]:
    """Return (p00, p01, p10, p11) for given marginals and target phi.

    Returns None if the combination is geometrically infeasible (phi out of
    the achievable range for these marginals).
    """
    denom = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    if denom < 1e-12:
        return None
    p11 = p1 * p2 + true_phi * denom
    p10 = p1 - p11
    p01 = p2 - p11
    p00 = 1.0 - p1 - p2 + p11
    if any(v < -1e-9 for v in (p00, p01, p10, p11)):
        return None
    # clamp tiny floating-point negatives and renormalise
    p00, p01, p10, p11 = (max(0.0, v) for v in (p00, p01, p10, p11))
    total = p00 + p01 + p10 + p11
    return p00 / total, p01 / total, p10 / total, p11 / total


def _sample_matrix(
    n: int,
    probs: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n i.i.d. rows from the 2x2 joint distribution."""
    counts = rng.multinomial(n, list(probs))
    rows = (
        [[0, 0]] * int(counts[0]) +
        [[0, 1]] * int(counts[1]) +
        [[1, 0]] * int(counts[2]) +
        [[1, 1]] * int(counts[3])
    )
    if not rows:
        return np.zeros((n, 2), dtype=int)
    M = np.array(rows, dtype=int)
    return M[rng.permutation(len(M))]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    true_phi:    float
    p1:          float
    p2:          float
    n:           int
    power:       float         # P(CI half-width <= ci_hw_target)
    median_hw:   float         # median CI half-width across simulations
    p90_hw:      float         # 90th-percentile CI half-width
    feasible:    bool          # False if (phi, marginals) combination is impossible
    n_sims_used: int


def simulate_power(
    true_phi: float,
    marginals: Tuple[float, float],
    n: int,
    n_sims: int = 300,
    n_boot: int = 200,
    seed: int = 0,
    ci_hw_target: float = DEFAULT_CI_HW_TARGET,
) -> SimResult:
    """Estimate P(CI half-width <= ci_hw_target) for one (phi, marginals, n) cell."""
    p1, p2 = marginals
    probs = _joint_probs(true_phi, p1, p2)
    if probs is None:
        return SimResult(true_phi, p1, p2, n, 0.0, float("nan"), float("nan"),
                         feasible=False, n_sims_used=0)

    rng = np.random.default_rng(seed)
    hw_vals: List[float] = []

    for _ in range(n_sims):
        M = _sample_matrix(n, probs, rng)
        lo, hi = _bootstrap_phi_ci_fast(M, n_boot, rng)
        if lo == lo and hi == hi:   # not nan
            hw_vals.append((hi - lo) / 2.0)

    if not hw_vals:
        return SimResult(true_phi, p1, p2, n, 0.0, float("nan"), float("nan"),
                         feasible=True, n_sims_used=0)

    power = float(np.mean([hw <= ci_hw_target for hw in hw_vals]))
    return SimResult(
        true_phi=true_phi, p1=p1, p2=p2, n=n,
        power=power,
        median_hw=float(np.median(hw_vals)),
        p90_hw=float(np.percentile(hw_vals, 90)),
        feasible=True,
        n_sims_used=len(hw_vals),
    )


def run_power_grid(
    phis: List[float] = None,
    marginal_pairs: List[Tuple[float, float]] = None,
    n_values: List[int] = None,
    n_sims: int = 300,
    n_boot: int = 200,
    seed: int = 0,
    ci_hw_target: float = DEFAULT_CI_HW_TARGET,
    verbose: bool = True,
) -> List[SimResult]:
    """Run the full power grid. Returns all SimResult objects."""
    phis          = phis          or DEFAULT_PHIS
    marginal_pairs = marginal_pairs or DEFAULT_MARGINALS
    n_values      = n_values      or DEFAULT_N_VALUES

    results: List[SimResult] = []
    total = len(phis) * len(marginal_pairs) * len(n_values)
    done = 0

    for phi in phis:
        for margs in marginal_pairs:
            for n in n_values:
                r = simulate_power(phi, margs, n, n_sims, n_boot, seed, ci_hw_target)
                results.append(r)
                done += 1
                if verbose:
                    status = f"power={r.power:.2f} med_hw={r.median_hw:.3f}" if r.feasible else "INFEASIBLE"
                    print(f"  [{done:3d}/{total}] phi={phi:.1f} "
                          f"marg=({margs[0]:.1f},{margs[1]:.1f}) n={n:3d}  {status}",
                          flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_n(
    results: List[SimResult],
    power_threshold: float = DEFAULT_POWER_THRESHOLD,
    safety_margin: float = 1.10,
) -> dict:
    """Find minimum n meeting power_threshold per scenario; recommend max + margin.

    The target n is the worst-case minimum n across all (phi, marginals)
    combinations, rounded up to the nearest 10 with a 10% safety margin.
    """
    from collections import defaultdict
    scenarios: Dict[tuple, List[SimResult]] = defaultdict(list)
    for r in results:
        scenarios[(r.true_phi, r.p1, r.p2)].append(r)

    scenario_min_n: Dict[str, Optional[int]] = {}
    for key, rlist in scenarios.items():
        sorted_r = sorted([r for r in rlist if r.feasible], key=lambda x: x.n)
        min_n = next((r.n for r in sorted_r if r.power >= power_threshold), None)
        scenario_min_n[f"phi={key[0]:.1f} p=({key[1]:.1f},{key[2]:.1f})"] = min_n

    valid_ns = [n for n in scenario_min_n.values() if n is not None]
    raw_max = max(valid_ns) if valid_ns else 200
    # round before ceil to avoid floating-point artefacts (e.g. 100*1.1 → 110.000...01)
    raw_with_margin = round(raw_max * safety_margin)
    target_n = int(np.ceil(raw_with_margin / 10) * 10)

    return {
        "target_n":           target_n,
        "raw_max_n":          raw_max,
        "power_threshold":    power_threshold,
        "safety_margin":      safety_margin,
        "ci_hw_target":       DEFAULT_CI_HW_TARGET,
        "scenario_min_n":     scenario_min_n,
        "n_infeasible":       sum(1 for n in scenario_min_n.values() if n is None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def results_to_table_md(
    results: List[SimResult],
    power_threshold: float = DEFAULT_POWER_THRESHOLD,
    ci_hw_target: float = DEFAULT_CI_HW_TARGET,
) -> str:
    """Render results as a Markdown table (grouped by phi)."""
    from itertools import groupby
    lines = [
        f"## Power simulation — CI half-width ≤ {ci_hw_target} in ≥ "
        f"{power_threshold*100:.0f}% of simulations\n",
        "| phi | marginals | n | power | med hw | p90 hw | meets criterion |",
        "|-----|-----------|---|-------|--------|--------|-----------------|",
    ]
    for r in results:
        if not r.feasible:
            meets = "INFEASIBLE"
            med, p90 = "—", "—"
        else:
            meets = "✅" if r.power >= power_threshold else "—"
            med  = f"{r.median_hw:.3f}"
            p90  = f"{r.p90_hw:.3f}"
        lines.append(
            f"| {r.true_phi:.1f} | ({r.p1:.1f},{r.p2:.1f}) | {r.n} "
            f"| {r.power:.2f} | {med} | {p90} | {meets} |"
        )
    return "\n".join(lines) + "\n"


def results_to_csv(results: List[SimResult]) -> str:
    """Render results as CSV."""
    header = "true_phi,p1,p2,n,power,median_hw,p90_hw,feasible,n_sims_used"
    rows = [header]
    for r in results:
        rows.append(
            f"{r.true_phi},{r.p1},{r.p2},{r.n},"
            f"{r.power:.4f},{r.median_hw:.4f},{r.p90_hw:.4f},"
            f"{r.feasible},{r.n_sims_used}"
        )
    return "\n".join(rows) + "\n"
