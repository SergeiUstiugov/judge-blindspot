# -*- coding: utf-8 -*-
"""Core statistics: phi correlation, joint/product ratio, bootstrap CI, shuffle null."""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi (Pearson) correlation of two binary vectors. Returns nan if zero variance."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    """joint / product of marginals. Returns nan if product is zero."""
    joint = float(np.mean(a & b))
    prod = float(a.mean() * b.mean())
    return joint / prod if prod > 0 else float("nan")


def independence_report(
    miss: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    """Core metric for a (n, 2) binary miss matrix of two judges in ONE defect class.

    Returns (phi, ratio, shuffle_null):
      phi   -- phi miss-correlation (approx 0 under independence);
      ratio -- joint/product: how many times more often joint miss occurs than under
               independence (approx 1 under independence, >1 with shared blind spot);
      null  -- shuffle control: phi level with relationship destroyed (floor estimate).
    """
    rng = np.random.default_rng(seed)
    a, b = miss[:, 0].astype(int), miss[:, 1].astype(int)
    phi = _phi(a, b)
    ratio = _ratio(a, b)
    if a.std() == 0 or b.std() == 0:
        null = 0.0
    else:
        null = float(np.mean([_phi(a, rng.permutation(b)) for _ in range(n_boot)]))
    return phi, ratio, null


def bootstrap_ci(
    miss: np.ndarray,
    stat_fn: Callable,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile CI of stat_fn(miss) via row resampling (2000x by default)."""
    rng = np.random.default_rng(seed)
    n = len(miss)
    vals = []
    for _ in range(n_boot):
        v = stat_fn(miss[rng.integers(0, n, n)])
        if v == v:  # not nan
            vals.append(v)
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def pairwise_report(
    M: np.ndarray,
    judge_names: List[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Full pairwise stats for a (n, k) binary miss matrix.

    Returns nested dict keyed by 'A|B' with phi, phi_ci, ratio, ratio_ci,
    shuffle_null, marginals, n, and 2x2 miss table (N00/N01/N10/N11).
    """
    k = M.shape[1]
    seed_mat = np.random.default_rng(seed).integers(0, 2**31, (k, k))
    results = {}
    for i in range(k):
        for j in range(i + 1, k):
            key = f"{judge_names[i]}|{judge_names[j]}"
            sub = M[:, [i, j]]
            ai, bi = sub[:, 0].astype(int), sub[:, 1].astype(int)
            phi_val, ratio_val, null_val = independence_report(
                sub, n_boot=n_boot, seed=int(seed_mat[i, j])
            )
            phi_ci = bootstrap_ci(
                sub,
                lambda m: _phi(m[:, 0].astype(int), m[:, 1].astype(int)),
                n_boot=n_boot, seed=int(seed_mat[i, j]), alpha=alpha,
            )
            ratio_ci = bootstrap_ci(
                sub,
                lambda m: _ratio(m[:, 0].astype(int), m[:, 1].astype(int)),
                n_boot=n_boot, seed=int(seed_mat[j, i]), alpha=alpha,
            )
            results[key] = {
                "phi":         phi_val,
                "phi_ci":      phi_ci,
                "ratio":       ratio_val,
                "ratio_ci":    ratio_ci,
                "shuffle_null": null_val,
                "marginal_i":  float(ai.mean()),
                "marginal_j":  float(bi.mean()),
                "n":           int(M.shape[0]),
                "N00": int(((ai == 0) & (bi == 0)).sum()),
                "N01": int(((ai == 0) & (bi == 1)).sum()),
                "N10": int(((ai == 1) & (bi == 0)).sum()),
                "N11": int(((ai == 1) & (bi == 1)).sum()),
            }
    return results
