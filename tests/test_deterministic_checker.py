# -*- coding: utf-8 -*-
"""Tests for DeterministicChecker — verify_gt wrapper.

All tests mock verify_gt to avoid launching a real subprocess.
We are testing the wrapper logic, not verify_gt itself
(verify_gt has its own coverage in test_corpus.py).
"""
from __future__ import annotations
from unittest.mock import patch

from judge_blindspot.corpus import CorpusItem
from judge_blindspot.deterministic_checker import DeterministicChecker
from judge_blindspot.judges import VERDICT_FAIL, VERDICT_INVALID, VERDICT_PASS


def _item(item_id: str = "t01", gt_label: str = "correct") -> CorpusItem:
    return CorpusItem(
        item_id=item_id,
        spec="Return the sum of two numbers.",
        candidate_code="def f(a, b): return a + b",
        gt_label=gt_label,
        defect_type=None if gt_label == "correct" else "wrong_operator",
        source="test_fixture",
        contamination_flag=False,
        hidden_tests="def test_f(): assert f(1, 2) == 3",
    )


def test_correct_item_returns_pass():
    """verify_gt returning True → PASS; never INVALID."""
    checker = DeterministicChecker()
    item = _item(item_id="t01", gt_label="correct")
    with patch("judge_blindspot.deterministic_checker.verify_gt", return_value=True):
        result = checker.judge(item)
    assert result.verdict == VERDICT_PASS
    assert result.verdict != VERDICT_INVALID
    assert result.item_id == "t01"
    assert result.judge_id == "deterministic_gt"


def test_defective_item_returns_fail():
    """verify_gt returning False → FAIL; never INVALID."""
    checker = DeterministicChecker()
    item = _item(item_id="t02", gt_label="defective")
    with patch("judge_blindspot.deterministic_checker.verify_gt", return_value=False):
        result = checker.judge(item)
    assert result.verdict == VERDICT_FAIL
    assert result.verdict != VERDICT_INVALID
    assert result.item_id == "t02"
    assert result.judge_id == "deterministic_gt"


def test_same_item_same_verdict_on_repeat():
    """DeterministicChecker is stateless — repeated calls on the same item agree."""
    checker = DeterministicChecker()
    item = _item(item_id="t03", gt_label="correct")
    with patch("judge_blindspot.deterministic_checker.verify_gt", return_value=True) as mock_vgt:
        r1 = checker.judge(item)
        r2 = checker.judge(item)
    assert r1.verdict == r2.verdict == VERDICT_PASS
    # verify_gt was called each time (checker is not caching — determinism comes from the oracle)
    assert mock_vgt.call_count == 2
