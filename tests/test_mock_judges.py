# -*- coding: utf-8 -*-
"""Tests for mock_judges.py: determinism, positive/orthogonal control properties."""
import numpy as np
import pytest
from judge_blindspot.corpus import CorpusItem
from judge_blindspot.mock_judges import MockJudge, make_calibration_judges
from judge_blindspot.judges import VERDICT_PASS, VERDICT_FAIL, VERDICT_INVALID
from judge_blindspot.stats import _phi


def _item(item_id: str, gt_label: str = "defective") -> CorpusItem:
    return CorpusItem(
        item_id=item_id, spec="x", candidate_code="def solution(): pass",
        gt_label=gt_label, defect_type="off_by_one",
        source="test", contamination_flag=False,
    )


def _corpus(n: int = 40) -> list:
    items = []
    for i in range(n // 2):
        items.append(_item(f"def_{i}", "defective"))
    for i in range(n // 2):
        items.append(_item(f"cor_{i}", "correct"))
    return items


# ── determinism ──────────────────────────────────────────────────────────────

def test_same_judge_same_item_deterministic():
    j = MockJudge(seed=0)
    item = _item("x1")
    assert j.judge(item).verdict == j.judge(item).verdict


def test_same_seed_two_instances_identical():
    j1 = MockJudge(seed=7)
    j2 = MockJudge(seed=7, judge_id="clone")
    items = _corpus(20)
    for it in items:
        assert j1.judge(it).verdict == j2.judge(it).verdict


def test_different_seeds_produce_different_verdicts():
    j0 = MockJudge(seed=0)
    j1 = MockJudge(seed=999)
    items = _corpus(40)
    diffs = sum(1 for it in items if j0.judge(it).verdict != j1.judge(it).verdict)
    # With error_rate=0.25 and different seeds, expect some differences
    assert diffs > 0


# ── miss() method ────────────────────────────────────────────────────────────

def test_miss_correct_item_not_missed():
    j = MockJudge(error_rate=0.0)  # never makes errors
    item = _item("c1", "correct")
    result = j.judge(item)
    assert result.verdict == VERDICT_PASS
    assert result.miss("correct") == 0


def test_miss_defective_item_caught():
    j = MockJudge(error_rate=0.0)  # never makes errors
    item = _item("d1", "defective")
    result = j.judge(item)
    assert result.verdict == VERDICT_FAIL
    assert result.miss("defective") == 0


def test_error_rate_zero_no_misses():
    j = MockJudge(error_rate=0.0, seed=0)
    items = _corpus(20)
    misses = [j.judge(it).miss(it.gt_label) for it in items]
    assert all(m == 0 for m in misses)


def test_error_rate_one_all_miss():
    j = MockJudge(error_rate=1.0, seed=0)
    items = _corpus(20)
    misses = [j.judge(it).miss(it.gt_label) for it in items]
    assert all(m == 1 for m in misses)


# ── calibration controls ─────────────────────────────────────────────────────

def test_positive_control_phi_equals_one():
    # Same seed → identical miss vectors → phi must be exactly +1
    pos_a, pos_b, _, _ = make_calibration_judges(error_rate=0.3, seed=0)
    items = _corpus(60)
    col_a = np.array([pos_a.judge(it).miss(it.gt_label) for it in items])
    col_b = np.array([pos_b.judge(it).miss(it.gt_label) for it in items])
    assert (col_a == col_b).all(), "Positive control: miss columns must be identical"
    phi = _phi(col_a, col_b)
    assert abs(phi - 1.0) < 1e-9 or np.isnan(phi)  # nan only if all-same (zero var)


def test_orthogonal_control_phi_near_zero():
    # Different seeds → independent misses → phi should be near 0
    _, _, orth_a, orth_b = make_calibration_judges(error_rate=0.3, seed=0)
    items = _corpus(200)
    col_a = np.array([orth_a.judge(it).miss(it.gt_label) for it in items])
    col_b = np.array([orth_b.judge(it).miss(it.gt_label) for it in items])
    phi = _phi(col_a, col_b)
    assert abs(phi) < 0.3, f"Orthogonal control: phi={phi:.3f} should be near 0"
