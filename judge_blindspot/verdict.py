# -*- coding: utf-8 -*-
"""Pre-registered verdict rule — locked before any data collection."""
from __future__ import annotations
from enum import Enum
from typing import Tuple


class VerdictLabel(str, Enum):
    INDEPENDENT   = "INDEPENDENT"
    OVERLAP       = "OVERLAP"
    DUPLICATE     = "DUPLICATE"
    INCONCLUSIVE  = "INCONCLUSIVE"


# ── Pre-registered thresholds (do not change after data collection begins) ──
_DUPLICATE_PHI_MIN        = 0.7    # phi >= this → duplicate candidate
_DUPLICATE_CI_HW_MAX      = 0.15   # AND CI half-width <= this → DUPLICATE
_INCONCLUSIVE_CI_HW       = 0.2    # CI half-width > this → INCONCLUSIVE
_MIN_N                    = 30     # n below this → INCONCLUSIVE


def verdict_for_pair(
    phi: float,
    phi_ci: Tuple[float, float],
    ratio_ci: Tuple[float, float],
    n: int,
) -> VerdictLabel:
    """Apply the pre-registered decision rule to a single judge pair.

    Priority order:
      1. INCONCLUSIVE  -- nan, CI half-width > 0.2, or n < MIN_N
      2. DUPLICATE     -- phi >= 0.7 AND CI half-width <= 0.15
      3. INDEPENDENT   -- phi CI covers 0 AND ratio CI covers 1
      4. OVERLAP       -- phi CI clearly > 0 AND phi <= 0.7
      5. INCONCLUSIVE  -- fallback
    """
    lo_phi, hi_phi = phi_ci
    lo_ratio, hi_ratio = ratio_ci

    # nan guard
    if phi != phi or lo_phi != lo_phi or lo_ratio != lo_ratio:
        return VerdictLabel.INCONCLUSIVE

    phi_hw = (hi_phi - lo_phi) / 2
    if phi_hw > _INCONCLUSIVE_CI_HW or n < _MIN_N:
        return VerdictLabel.INCONCLUSIVE

    if phi >= _DUPLICATE_PHI_MIN and phi_hw <= _DUPLICATE_CI_HW_MAX:
        return VerdictLabel.DUPLICATE

    phi_covers_zero  = lo_phi <= 0 <= hi_phi
    ratio_covers_one = lo_ratio <= 1 <= hi_ratio
    if phi_covers_zero and ratio_covers_one:
        return VerdictLabel.INDEPENDENT

    if lo_phi > 0 and phi < _DUPLICATE_PHI_MIN:
        return VerdictLabel.OVERLAP

    return VerdictLabel.INCONCLUSIVE


def apply_verdicts(pairwise: dict) -> dict:
    """Apply verdict_for_pair to all entries in a pairwise_report dict."""
    result = {}
    for key, p in pairwise.items():
        label = verdict_for_pair(
            phi=p["phi"],
            phi_ci=p["phi_ci"],
            ratio_ci=p["ratio_ci"],
            n=p["n"],
        )
        result[key] = {**p, "verdict": label.value}
    return result
