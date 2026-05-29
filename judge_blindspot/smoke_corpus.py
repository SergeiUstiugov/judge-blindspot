# -*- coding: utf-8 -*-
"""Smoke corpus: 10 hand-authored tasks with hidden tests and controlled mutations.

Source: "synthetic_smoke" — NOT HumanEval/MBPP.
Run build_smoke_corpus() to generate, verify, and save the JSONL.
"""
from __future__ import annotations
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .corpus import CorpusItem, save_corpus, verify_corpus
from .mutate import MutationSpec, verify_task_mutations


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# Each entry: task_id, spec, correct source, hidden tests, mutations list.
# Mutations include at least one EQUIVALENT mutant per task where noted.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TaskDef:
    task_id: str
    spec: str
    correct: str
    tests: str
    mutations: List[MutationSpec]
    equivalent_mutations: List[MutationSpec] = field(default_factory=list)


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


_TASKS: List[_TaskDef] = [

    _TaskDef(
        task_id="smoke_sum_list",
        spec="Given a list of numbers, return their sum. Return 0 for an empty list.",
        correct=_dedent("""
            def solution(lst):
                total = 0
                for x in lst:
                    total += x
                return total
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([]) == 0
                assert solution([1, 2, 3]) == 6
                assert solution([-1, 1]) == 0
                assert solution([5]) == 5
                assert solution([1, -2, 3]) == 2
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        total = 0
                        for x in lst:
                            total -= x
                        return total
                """),
                defect_type="wrong_operator",
                description="subtract instead of add — obvious",
            ),
        ],
        equivalent_mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        total = 0
                        for x in lst:
                            if True:
                                total += x
                        return total
                """),
                defect_type="wrong_operator",
                description="dead 'if True' branch — equivalent mutant",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_find_max",
        spec="Return the maximum element of a non-empty list of numbers.",
        correct=_dedent("""
            def solution(lst):
                m = lst[0]
                for x in lst[1:]:
                    if x > m:
                        m = x
                return m
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([3, 1, 4, 1, 5]) == 5
                assert solution([1]) == 1
                assert solution([-3, -1, -4]) == -1
                assert solution([7, 7, 7]) == 7
                assert solution([0, -1, 1]) == 1
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        m = lst[0]
                        for x in lst[1:]:
                            if x < m:
                                m = x
                        return m
                """),
                defect_type="wrong_operator",
                description="< instead of > — finds min instead of max, obvious",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_count_even",
        spec="Return the count of even numbers in the list.",
        correct=_dedent("""
            def solution(lst):
                count = 0
                for x in lst:
                    if x % 2 == 0:
                        count += 1
                return count
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([1, 2, 3]) == 1
                assert solution([2, 4, 6]) == 3
                assert solution([1, 3, 5]) == 0
                assert solution([]) == 0
                assert solution([0, 1]) == 1
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        count = 0
                        for x in lst:
                            if x % 2 != 0:
                                count += 1
                        return count
                """),
                defect_type="wrong_operator",
                description="!= instead of == — counts odd numbers, obvious",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_linear_search",
        spec="Return the index of the first occurrence of target in lst, or -1 if not found.",
        correct=_dedent("""
            def solution(lst, target):
                for i, x in enumerate(lst):
                    if x == target:
                        return i
                return -1
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([1, 2, 3], 2) == 1
                assert solution([1, 2, 3], 4) == -1
                assert solution([5, 1, 5], 5) == 0
                assert solution([], 1) == -1
                assert solution([0], 0) == 0
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst, target):
                        for i, x in enumerate(lst):
                            if x == target:
                                return i + 1
                        return -1
                """),
                defect_type="off_by_one",
                description="return i+1 instead of i — subtle (only fails when target is found)",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_factorial",
        spec="Return n! (factorial of n). Return 1 for n <= 0.",
        correct=_dedent("""
            def solution(n):
                if n <= 0:
                    return 1
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
        """),
        tests=_dedent("""
            def test_basic():
                assert solution(0) == 1
                assert solution(1) == 1
                assert solution(5) == 120
                assert solution(3) == 6
                assert solution(4) == 24
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(n):
                        if n <= 0:
                            return 1
                        result = 1
                        for i in range(1, n):
                            result *= i
                        return result
                """),
                defect_type="off_by_one",
                description="range(1, n) instead of range(1, n+1) — misses last factor, subtle",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_is_palindrome",
        spec="Return True if the string is a palindrome, False otherwise.",
        correct=_dedent("""
            def solution(s):
                return s == s[::-1]
        """),
        tests=_dedent("""
            def test_basic():
                assert solution("racecar") is True
                assert solution("hello") is False
                assert solution("") is True
                assert solution("a") is True
                assert solution("ab") is False
                assert solution("aba") is True
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(s):
                        return s != s[::-1]
                """),
                defect_type="wrong_operator",
                description="!= instead of == — inverts result, obvious",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_clamp",
        spec="Clamp val to [lo, hi]: return lo if val < lo, hi if val > hi, else val.",
        correct=_dedent("""
            def solution(val, lo, hi):
                if val < lo:
                    return lo
                if val > hi:
                    return hi
                return val
        """),
        tests=_dedent("""
            def test_basic():
                assert solution(5, 0, 10) == 5
                assert solution(-1, 0, 10) == 0
                assert solution(15, 0, 10) == 10
                assert solution(0, 0, 10) == 0
                assert solution(10, 0, 10) == 10
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(val, lo, hi):
                        if val < lo:
                            return hi
                        if val > hi:
                            return lo
                        return val
                """),
                defect_type="swapped_args",
                description="returns wrong bound — returns hi when below lo, subtle",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_first_duplicate",
        spec="Return the first element that appears more than once in lst, or None if no duplicates.",
        correct=_dedent("""
            def solution(lst):
                seen = set()
                for x in lst:
                    if x in seen:
                        return x
                    seen.add(x)
                return None
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([1, 2, 3, 2]) == 2
                assert solution([1, 2, 3]) is None
                assert solution([1, 1, 2]) == 1
                assert solution([]) is None
                assert solution([3, 3]) == 3
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        seen = set()
                        for x in lst:
                            seen.add(x)
                            if x in seen:
                                return x
                        return None
                """),
                defect_type="dropped_guard",
                description="add before check — always returns first element, subtle",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_running_sum",
        spec="Return a list where element i is the sum of lst[0..i] (running/prefix sum).",
        correct=_dedent("""
            def solution(lst):
                result = []
                total = 0
                for x in lst:
                    total += x
                    result.append(total)
                return result
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([1, 2, 3]) == [1, 3, 6]
                assert solution([]) == []
                assert solution([5]) == [5]
                assert solution([1, -1, 1]) == [1, 0, 1]
                assert solution([3, 3, 3]) == [3, 6, 9]
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst):
                        result = []
                        total = 0
                        for x in lst:
                            result.append(total)
                            total += x
                        return result
                """),
                defect_type="off_by_one",
                description="append before update — each element is one step behind, subtle",
            ),
        ],
    ),

    _TaskDef(
        task_id="smoke_count_above",
        spec="Return the count of elements in lst that are strictly greater than threshold.",
        correct=_dedent("""
            def solution(lst, threshold):
                count = 0
                for x in lst:
                    if x > threshold:
                        count += 1
                return count
        """),
        tests=_dedent("""
            def test_basic():
                assert solution([1, 2, 3, 4], 2) == 2
                assert solution([5, 5, 5], 5) == 0
                assert solution([1, 2, 3], 0) == 3
                assert solution([], 0) == 0
                assert solution([5, 6], 5) == 1
        """),
        mutations=[
            MutationSpec(
                source=_dedent("""
                    def solution(lst, threshold):
                        count = 0
                        for x in lst:
                            if x >= threshold:
                                count += 1
                        return count
                """),
                defect_type="wrong_operator",
                description=">= instead of > — also counts equal elements, subtle",
            ),
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder
# ─────────────────────────────────────────────────────────────────────────────

def build_smoke_corpus(
    output_path: str | Path = "data/synthetic_smoke.jsonl",
    abort_on_mismatch: bool = True,
    verbose: bool = True,
) -> dict:
    """Build, verify, and save the smoke corpus. Returns stats dict.

    For every task:
      - Verifies correct solution passes all hidden tests.
      - Classifies each mutation (defective / equivalent-discarded).
      - Aborts if any gt_label does not match test results.

    Stats returned: n_correct, n_defective, n_equivalent_discarded,
    defect_type_counts, items_by_task.
    """
    items: List[CorpusItem] = []
    n_equiv_total = 0
    defect_type_counts: dict = {}

    for task in _TASKS:
        # -- correct item --
        items.append(CorpusItem(
            item_id=f"{task.task_id}_correct",
            spec=task.spec,
            candidate_code=task.correct,
            gt_label="correct",
            defect_type=None,
            source="synthetic_smoke",
            contamination_flag=False,
            hidden_tests=task.tests,
        ))

        # -- verify mutations (including equivalent ones for logging) --
        all_mutations = task.mutations + task.equivalent_mutations
        results, n_equiv = verify_task_mutations(
            task.task_id, task.correct, task.tests, all_mutations
        )
        n_equiv_total += n_equiv

        for res in results:
            if res.is_equivalent:
                if verbose:
                    print(f"  [DISCARD equiv] {task.task_id}: {res.spec.description}")
                continue
            if not res.is_defective:
                raise RuntimeError(
                    f"Mutation verification error for {task.task_id}: {res.error}"
                )
            items.append(CorpusItem(
                item_id=f"{task.task_id}_{res.spec.defect_type}",
                spec=task.spec,
                candidate_code=res.spec.source,
                gt_label="defective",
                defect_type=res.spec.defect_type,
                source="synthetic_smoke",
                contamination_flag=False,
                hidden_tests=task.tests,
            ))
            defect_type_counts[res.spec.defect_type] = (
                defect_type_counts.get(res.spec.defect_type, 0) + 1
            )

    # -- 100% GT verification (abort on mismatch) --
    if verbose:
        print(f"\nRunning 100% GT verification on {len(items)} items...")
    mismatches = verify_corpus(items, abort_on_mismatch=abort_on_mismatch)
    if mismatches:
        raise ValueError(f"GT mismatches found: {mismatches}")

    # -- save --
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_corpus(items, output_path)

    n_correct   = sum(1 for it in items if it.gt_label == "correct")
    n_defective = sum(1 for it in items if it.gt_label == "defective")

    stats = {
        "n":                     len(items),
        "n_correct":             n_correct,
        "n_defective":           n_defective,
        "balance_pct_correct":   round(100 * n_correct / len(items), 1),
        "n_equivalent_discarded": n_equiv_total,
        "defect_type_counts":    defect_type_counts,
        "output_path":           str(output_path),
    }

    if verbose:
        print(f"\nSmoke corpus built: {len(items)} items "
              f"({n_correct} correct / {n_defective} defective)")
        print(f"Equivalent mutants discarded: {n_equiv_total}")
        print(f"Defect types: {defect_type_counts}")
        print(f"Saved to: {output_path}")

    return stats
