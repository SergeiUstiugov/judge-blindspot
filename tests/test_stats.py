# -*- coding: utf-8 -*-
"""Tests for stats.py: phi, ratio, bootstrap CI, pairwise_report."""
import numpy as np
import pytest
from judge_blindspot.stats import (
    _phi, _ratio, independence_report, bootstrap_ci, pairwise_report,
)


def _make_independent(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        (rng.random(n) < 0.30).astype(int),
        (rng.random(n) < 0.40).astype(int),
    ])


def _make_dependent(n=200, seed=0):
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.40
    M = np.column_stack([
        (rng.random(n) < 0.15).astype(int),
        (rng.random(n) < 0.15).astype(int),
    ])
    M[hard, :] = 1
    return M


# ── _phi ──────────────────────────────────────────────────────────────────────

def test_phi_independent_near_zero():
    M = _make_independent()
    assert abs(_phi(M[:, 0], M[:, 1])) < 0.3


def test_phi_dependent_positive():
    M = _make_dependent()
    assert _phi(M[:, 0], M[:, 1]) > 0.4


def test_phi_zero_variance_returns_nan():
    a = np.zeros(50, dtype=int)
    b = (np.random.default_rng(0).random(50) < 0.5).astype(int)
    assert np.isnan(_phi(a, b))


def test_phi_perfect_correlation():
    v = np.array([0, 1, 1, 0, 1], dtype=int)
    assert abs(_phi(v, v) - 1.0) < 1e-9


# ── _ratio ────────────────────────────────────────────────────────────────────

def test_ratio_independent_near_one():
    M = _make_independent()
    r = _ratio(M[:, 0], M[:, 1])
    assert 0.4 < r < 2.5


def test_ratio_dependent_high():
    M = _make_dependent()
    assert _ratio(M[:, 0], M[:, 1]) > 1.5


def test_ratio_zero_product_returns_nan():
    a = np.zeros(50, dtype=int)
    b = np.ones(50, dtype=int)
    assert np.isnan(_ratio(a, b))


# ── independence_report ───────────────────────────────────────────────────────

def test_independence_report_returns_three_values():
    phi, ratio, null = independence_report(_make_independent(), n_boot=100)
    assert phi == phi      # not nan
    assert ratio == ratio


def test_independence_report_null_near_zero():
    _, _, null = independence_report(_make_independent(), n_boot=200)
    assert abs(null) < 0.15


# ── bootstrap_ci ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_independent_covers_zero():
    M = _make_independent()
    lo, hi = bootstrap_ci(
        M, lambda m: _phi(m[:, 0].astype(int), m[:, 1].astype(int)), n_boot=500
    )
    assert lo < 0 < hi


def test_bootstrap_ci_dependent_clearly_positive():
    M = _make_dependent()
    lo, hi = bootstrap_ci(
        M, lambda m: _phi(m[:, 0].astype(int), m[:, 1].astype(int)), n_boot=500
    )
    assert lo > 0


def test_bootstrap_ci_ratio_independent_covers_one():
    M = _make_independent()
    lo, hi = bootstrap_ci(
        M, lambda m: _ratio(m[:, 0].astype(int), m[:, 1].astype(int)), n_boot=500
    )
    assert lo < 1 < hi


# ── pairwise_report ───────────────────────────────────────────────────────────

def test_pairwise_report_keys():
    rng = np.random.default_rng(0)
    M = (rng.random((100, 3)) < 0.4).astype(int)
    r = pairwise_report(M, ["A", "B", "C"], n_boot=100)
    assert set(r.keys()) == {"A|B", "A|C", "B|C"}


def test_pairwise_report_fields():
    rng = np.random.default_rng(0)
    M = (rng.random((100, 2)) < 0.4).astype(int)
    r = pairwise_report(M, ["X", "Y"], n_boot=100)
    p = r["X|Y"]
    for field in ("phi", "phi_ci", "ratio", "ratio_ci", "shuffle_null",
                  "marginal_i", "marginal_j", "n", "N00", "N01", "N10", "N11"):
        assert field in p, f"missing field: {field}"


def test_pairwise_report_n_equals_matrix_rows():
    rng = np.random.default_rng(0)
    M = (rng.random((77, 2)) < 0.4).astype(int)
    r = pairwise_report(M, ["A", "B"], n_boot=50)
    assert r["A|B"]["n"] == 77


def test_pairwise_2x2_sums_to_n():
    rng = np.random.default_rng(0)
    M = (rng.random((80, 2)) < 0.4).astype(int)
    r = pairwise_report(M, ["A", "B"], n_boot=50)
    p = r["A|B"]
    assert p["N00"] + p["N01"] + p["N10"] + p["N11"] == 80
