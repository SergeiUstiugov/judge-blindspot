# -*- coding: utf-8 -*-
"""Tests for power.py: joint distribution, simulation, recommendation."""
import numpy as np
import pytest
from judge_blindspot.power import (
    _joint_probs, _sample_matrix, simulate_power, recommend_n,
    results_to_csv, results_to_table_md, SimResult,
    DEFAULT_CI_HW_TARGET, DEFAULT_POWER_THRESHOLD,
)


# ── _joint_probs ──────────────────────────────────────────────────────────────

def test_joint_probs_zero_phi_equals_product():
    # phi=0 → joint = p1*p2 (independence)
    p00, p01, p10, p11 = _joint_probs(0.0, 0.3, 0.4)
    assert abs(p11 - 0.3 * 0.4) < 1e-9
    assert abs(p00 - (1 - 0.3) * (1 - 0.4)) < 1e-9


def test_joint_probs_sums_to_one():
    for phi in [0.0, 0.3, 0.5, 0.8]:
        for p1, p2 in [(0.2, 0.2), (0.3, 0.5), (0.5, 0.5)]:
            probs = _joint_probs(phi, p1, p2)
            if probs is not None:
                assert abs(sum(probs) - 1.0) < 1e-9, f"phi={phi} p=({p1},{p2})"


def test_joint_probs_all_nonnegative():
    probs = _joint_probs(0.8, 0.3, 0.3)
    assert all(p >= 0 for p in probs)


def test_joint_probs_infeasible_returns_none():
    # phi=+1.0 is only achievable when p1==p2
    # phi=1.0 with p1=0.2, p2=0.8: p11 = 0.16 + 1.0*sqrt(0.16*0.16) = 0.16+0.16=0.32
    # but min(p1,p2)=0.2 → p11 > 0.2 → infeasible
    result = _joint_probs(1.0, 0.2, 0.8)
    assert result is None


def test_joint_probs_marginals_recovered():
    # p(A=1) = p10 + p11 should equal p1
    for phi in [0.0, 0.5]:
        p1, p2 = 0.3, 0.4
        p00, p01, p10, p11 = _joint_probs(phi, p1, p2)
        assert abs(p10 + p11 - p1) < 1e-9
        assert abs(p01 + p11 - p2) < 1e-9


# ── _sample_matrix ────────────────────────────────────────────────────────────

def test_sample_matrix_shape():
    probs = _joint_probs(0.3, 0.4, 0.4)
    rng = np.random.default_rng(0)
    M = _sample_matrix(100, probs, rng)
    assert M.shape == (100, 2)
    assert M.dtype == int


def test_sample_matrix_values_binary():
    probs = _joint_probs(0.5, 0.3, 0.3)
    rng = np.random.default_rng(0)
    M = _sample_matrix(200, probs, rng)
    assert set(np.unique(M)).issubset({0, 1})


def test_sample_matrix_marginals_approx():
    # with large n, empirical marginals should be close to true marginals
    p1, p2 = 0.4, 0.3
    probs = _joint_probs(0.0, p1, p2)
    rng = np.random.default_rng(42)
    M = _sample_matrix(5000, probs, rng)
    assert abs(M[:, 0].mean() - p1) < 0.03
    assert abs(M[:, 1].mean() - p2) < 0.03


# ── simulate_power ────────────────────────────────────────────────────────────

def test_simulate_power_returns_simresult():
    r = simulate_power(0.0, (0.3, 0.3), n=60, n_sims=30, n_boot=50, seed=0)
    assert isinstance(r, SimResult)
    assert r.feasible is True
    assert 0.0 <= r.power <= 1.0
    assert r.n_sims_used > 0


def test_simulate_power_infeasible():
    r = simulate_power(1.0, (0.2, 0.8), n=100, n_sims=10, n_boot=50, seed=0)
    assert r.feasible is False
    assert r.power == 0.0


def test_simulate_power_larger_n_higher_power():
    # larger n should give narrower CI → higher power
    r_small = simulate_power(0.3, (0.4, 0.4), n=30,  n_sims=100, n_boot=100, seed=0)
    r_large = simulate_power(0.3, (0.4, 0.4), n=200, n_sims=100, n_boot=100, seed=0)
    assert r_large.power >= r_small.power - 0.1   # large should not be much worse


def test_simulate_power_larger_n_smaller_median_hw():
    r_small = simulate_power(0.0, (0.4, 0.4), n=30,  n_sims=100, n_boot=100, seed=0)
    r_large = simulate_power(0.0, (0.4, 0.4), n=200, n_sims=100, n_boot=100, seed=0)
    assert r_large.median_hw < r_small.median_hw


def test_simulate_power_phi_zero_large_n_power_high():
    # phi=0, n=200, symmetric marginals: should achieve high power easily
    r = simulate_power(0.0, (0.4, 0.4), n=200, n_sims=150, n_boot=150, seed=0)
    assert r.power >= 0.7, f"Expected high power, got {r.power:.2f}"


# ── recommend_n ───────────────────────────────────────────────────────────────

def _make_results():
    """Small synthetic result set for recommendation tests."""
    results = []
    for phi in [0.0, 0.5]:
        for n, power in [(50, 0.5), (100, 0.85), (150, 0.95)]:
            results.append(SimResult(
                true_phi=phi, p1=0.3, p2=0.3, n=n,
                power=power, median_hw=0.25 - n*0.001,
                p90_hw=0.30 - n*0.001,
                feasible=True, n_sims_used=200,
            ))
    return results


def test_recommend_n_returns_dict():
    rec = recommend_n(_make_results())
    assert "target_n" in rec
    assert "raw_max_n" in rec
    assert "scenario_min_n" in rec


def test_recommend_n_uses_max_across_scenarios():
    # Both scenarios first reach threshold at n=100; max=100, +10% → 110
    rec = recommend_n(_make_results(), power_threshold=0.80, safety_margin=1.1)
    assert rec["raw_max_n"] == 100
    assert rec["target_n"] == 110


def test_recommend_n_scenario_none_when_never_reached():
    # power never reaches threshold
    results = [
        SimResult(0.8, 0.2, 0.2, n, 0.3, 0.25, 0.30, True, 200)
        for n in [30, 50, 100]
    ]
    rec = recommend_n(results, power_threshold=0.80)
    assert rec["n_infeasible"] == 1   # one scenario never reached threshold


def test_recommend_n_safety_margin_applied():
    results = [SimResult(0.0, 0.4, 0.4, 100, 0.90, 0.18, 0.22, True, 200)]
    rec = recommend_n(results, power_threshold=0.80, safety_margin=1.2)
    # raw=100, *1.2=120, round to 10 → 120
    assert rec["target_n"] == 120


# ── reporting helpers ─────────────────────────────────────────────────────────

def test_results_to_csv_header():
    results = [SimResult(0.0, 0.3, 0.3, 100, 0.85, 0.15, 0.18, True, 200)]
    csv = results_to_csv(results)
    assert csv.startswith("true_phi,p1,p2,n,power")
    assert "0.0,0.3,0.3,100" in csv


def test_results_to_table_md_contains_phi():
    results = [SimResult(0.5, 0.4, 0.4, 120, 0.88, 0.17, 0.21, True, 200)]
    md = results_to_table_md(results)
    assert "0.5" in md
    assert "| phi |" in md
