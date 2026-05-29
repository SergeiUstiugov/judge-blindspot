# -*- coding: utf-8 -*-
"""Tests for mutate.py.

Rule: when a test expectation changes, comment WHY the expectation was wrong,
not the code.
"""
import pytest
from judge_blindspot.mutate import MutationSpec, verify_mutation, verify_task_mutations

CORRECT = "def solution(lst):\n    return sum(lst)\n"
TESTS = (
    "def test_sum():\n"
    "    assert solution([]) == 0\n"
    "    assert solution([1, 2, 3]) == 6\n"
    "    assert solution([-1, 1]) == 0\n"
)


# ── verify_mutation ───────────────────────────────────────────────────────────

def test_defective_mutation_detected():
    mut = MutationSpec(
        source="def solution(lst):\n    return -sum(lst)\n",
        defect_type="wrong_operator",
        description="negates sum",
    )
    result = verify_mutation("t1", CORRECT, mut, TESTS)
    assert result.is_defective is True
    assert result.is_equivalent is False
    assert result.error == ""


def test_equivalent_mutation_detected():
    # Adding 'and True' is a dead-code change — the function behaves identically.
    # Expectation: is_equivalent=True because all tests still pass.
    mut = MutationSpec(
        source=(
            "def solution(lst):\n"
            "    if True and True:\n"
            "        return sum(lst)\n"
        ),
        defect_type="wrong_operator",
        description="dead 'and True' — equivalent",
    )
    result = verify_mutation("t1", CORRECT, mut, TESTS)
    assert result.is_equivalent is True
    assert result.is_defective is False


def test_correct_source_sanity_check():
    # A mutation with the exact same source as correct → also equivalent
    mut = MutationSpec(source=CORRECT, defect_type="off_by_one", description="no change")
    result = verify_mutation("t1", CORRECT, mut, TESTS)
    assert result.is_equivalent is True


# ── verify_task_mutations ─────────────────────────────────────────────────────

def test_task_mutations_counts_equivalents():
    mutations = [
        MutationSpec(
            source="def solution(lst):\n    return -sum(lst)\n",
            defect_type="wrong_operator",
            description="defective",
        ),
        MutationSpec(
            source=CORRECT,
            defect_type="off_by_one",
            description="equivalent copy",
        ),
    ]
    results, n_equiv = verify_task_mutations("t1", CORRECT, TESTS, mutations)
    assert n_equiv == 1
    defective = [r for r in results if r.is_defective]
    assert len(defective) == 1


def test_task_mutations_all_defective():
    mutations = [
        MutationSpec(
            source="def solution(lst):\n    return sum(lst) + 99\n",
            defect_type="off_by_one",
            description="add 99",
        ),
        MutationSpec(
            source="def solution(lst):\n    return 0\n",
            defect_type="wrong_return",
            description="always 0",
        ),
    ]
    results, n_equiv = verify_task_mutations("t1", CORRECT, TESTS, mutations)
    assert n_equiv == 0
    assert all(r.is_defective for r in results)
