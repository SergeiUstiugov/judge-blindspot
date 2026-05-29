# -*- coding: utf-8 -*-
"""Tests for verdict.py: pre-registered decision rule."""
import pytest
from judge_blindspot.verdict import verdict_for_pair, apply_verdicts, VerdictLabel


# helper: n=150 so it never triggers the n<MIN_N guard
N = 150


def test_independent():
    v = verdict_for_pair(0.05, (-0.15, 0.25), (0.7, 1.4), n=N)
    assert v == VerdictLabel.INDEPENDENT


def test_duplicate():
    v = verdict_for_pair(0.85, (0.78, 0.92), (3.0, 5.0), n=N)
    assert v == VerdictLabel.DUPLICATE


def test_duplicate_boundary_phi():
    # phi exactly at threshold — should be DUPLICATE
    v = verdict_for_pair(0.70, (0.63, 0.77), (2.0, 4.0), n=N)
    assert v == VerdictLabel.DUPLICATE


def test_overlap():
    # phi CI half-width = (0.56 - 0.20) / 2 = 0.18 <= 0.20, lo_phi=0.20 > 0
    v = verdict_for_pair(0.40, (0.20, 0.56), (1.5, 3.0), n=N)
    assert v == VerdictLabel.OVERLAP


def test_inconclusive_wide_ci():
    # half-width = (0.80 - (-0.20)) / 2 = 0.50 > 0.20
    v = verdict_for_pair(0.30, (-0.20, 0.80), (0.8, 2.5), n=N)
    assert v == VerdictLabel.INCONCLUSIVE


def test_inconclusive_small_n():
    v = verdict_for_pair(0.50, (0.20, 0.80), (1.5, 3.0), n=20)
    assert v == VerdictLabel.INCONCLUSIVE


def test_inconclusive_nan_phi():
    v = verdict_for_pair(float("nan"), (float("nan"), float("nan")), (1.0, 2.0), n=N)
    assert v == VerdictLabel.INCONCLUSIVE


def test_inconclusive_nan_ratio_ci():
    v = verdict_for_pair(0.10, (-0.10, 0.30), (float("nan"), float("nan")), n=N)
    assert v == VerdictLabel.INCONCLUSIVE


def test_apply_verdicts_adds_verdict_field():
    pairwise = {
        "A|B": {
            "phi": 0.05, "phi_ci": (-0.10, 0.20),
            "ratio": 1.1, "ratio_ci": (0.7, 1.5), "n": N,
        }
    }
    result = apply_verdicts(pairwise)
    assert "verdict" in result["A|B"]
    assert result["A|B"]["verdict"] == VerdictLabel.INDEPENDENT.value


def test_apply_verdicts_preserves_original_fields():
    pairwise = {
        "X|Y": {
            "phi": 0.85, "phi_ci": (0.78, 0.92),
            "ratio": 4.0, "ratio_ci": (2.5, 6.0), "n": N,
            "N00": 10, "N01": 5, "N10": 3, "N11": 132,
        }
    }
    result = apply_verdicts(pairwise)
    assert result["X|Y"]["N00"] == 10
    assert result["X|Y"]["verdict"] == VerdictLabel.DUPLICATE.value
