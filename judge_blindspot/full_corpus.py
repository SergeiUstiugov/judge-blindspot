# -*- coding: utf-8 -*-
"""Full corpus: 40 hand-authored tasks, patch-based mutations, ≥170 items/class target.

Source tag: "synthetic_full" — NOT derived from HumanEval/MBPP.
Build:  judge-blindspot build-corpus --full [--output PATH]

Each task stores patches (old_text, new_text, defect_type, description).
The builder applies each patch once to produce a MutationSpec, verifies it
against hidden tests, and keeps only defective (non-equivalent) items.
"""
from __future__ import annotations
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from .corpus import CorpusItem, save_corpus, verify_corpus
from .mutate import MutationSpec, verify_task_mutations


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TaskDef:
    task_id:  str
    spec:     str
    correct:  str   # complete function source after _d()
    tests:    str   # pytest source (no imports needed; preamble added by verify_gt)
    patches:  List[Tuple[str, str, str, str]]  # (old, new, defect_type, description)

    # ------------------------------------------------------------------
    # Patch resolution: patches are written with stripped (or 16-space)
    # indentation for readability inside the file.  At build time we
    # normalise them to the actual indentation used in self.correct so
    # simple .replace() works correctly.
    # ------------------------------------------------------------------

    def _normalise(self, old: str, new: str):
        """Return (actual_old, actual_new) adjusted to match self.correct."""
        if old in self.correct:
            return old, new
        old_lines = old.split("\n")
        new_lines = new.split("\n") if new else []
        for n_sp in (4, 8, 12, 0, 16):
            sp = " " * n_sp
            norm_old = "\n".join(
                sp + ln.lstrip() if ln.strip() else "" for ln in old_lines
            )
            if norm_old in self.correct:
                norm_new = (
                    "\n".join(sp + ln.lstrip() if ln.strip() else "" for ln in new_lines)
                    if new and new_lines
                    else new
                )
                return norm_old, norm_new
        return old, new  # not found — raise below

    def build_mutations(self) -> List[MutationSpec]:
        seen: set = set()
        result: List[MutationSpec] = []
        for old, new, dtype, desc in self.patches:
            actual_old, actual_new = self._normalise(old, new)
            if actual_old not in self.correct:
                raise ValueError(
                    f"[{self.task_id}] patch old_text not found: {old!r}"
                )
            mutated = self.correct.replace(actual_old, actual_new, 1)
            if mutated == self.correct:
                raise ValueError(
                    f"[{self.task_id}] patch produced no change: {old!r} -> {new!r}"
                )
            key = mutated.strip()
            if key not in seen:
                seen.add(key)
                result.append(MutationSpec(mutated, dtype, desc))
        return result


def _d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions  (40 tasks)
# ─────────────────────────────────────────────────────────────────────────────

_TASKS: List[_TaskDef] = [

    # ── 1. clamp ──────────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_clamp",
        spec="Clamp val to [lo, hi]: return lo if val < lo, hi if val > hi, else val. Raise ValueError if lo > hi.",
        correct=_d("""
            def solution(val, lo, hi):
                if lo > hi: raise ValueError("lo must be <= hi")
                if val < lo: return lo
                if val > hi: return hi
                return val
        """),
        tests=_d("""
            import pytest
            def test_below():  assert solution(1, 5, 10) == 5
            def test_above():  assert solution(15, 5, 10) == 10
            def test_inside(): assert solution(7, 5, 10) == 7
            def test_at_lo():  assert solution(5, 5, 10) == 5
            def test_at_hi():  assert solution(10, 5, 10) == 10
            def test_invalid():
                with pytest.raises(ValueError): solution(5, 10, 5)
            def test_neg():    assert solution(-5, -3, 3) == -3
            def test_above_neg(): assert solution(5, -3, 3) == 3
        """),
        patches=[
            ("val < lo", "val <= lo",   "wrong_operator", "<= makes at-lo return lo incorrectly (double)"),
            ("val > hi", "val >= hi",   "wrong_operator", ">= makes at-hi return hi incorrectly (double)"),
            ("lo > hi",  "lo >= hi",    "wrong_operator", "guard fires on equal lo==hi"),
            ("val < lo", "val > lo",    "wrong_operator", "inverted lo check"),
            ("val > hi", "val < hi",    "wrong_operator", "inverted hi check"),
            ("if lo > hi: raise ValueError(\"lo must be <= hi\")\n                ", "",
             "dropped_guard", "remove lo<=hi guard"),
            ("if val < lo: return lo\n                if val > hi: return hi\n                return val",
             "if val > lo: return lo\n                if val < hi: return hi\n                return val",
             "swapped_args", "inverted both clamp directions"),
            ("return lo", "return hi",  "swapped_args", "return hi when below lo"),
            ("return hi", "return lo",  "swapped_args", "return lo when above hi"),
            ("val < lo",  "val < hi",   "swapped_args", "compare val to hi instead of lo for lower bound"),
            ("val > hi",  "val > lo",   "swapped_args", "compare val to lo instead of hi for upper bound"),
        ],
    ),

    # ── 2. factorial ─────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_factorial",
        spec="Return n! (n factorial). Raise ValueError for negative n. factorial(0) = 1.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be >= 0")
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
        """),
        tests=_d("""
            import pytest
            def test_zero():    assert solution(0) == 1
            def test_one():     assert solution(1) == 1
            def test_five():    assert solution(5) == 120
            def test_ten():     assert solution(10) == 3628800
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("result *= i",        "result += i",   "wrong_operator", "add instead of multiply"),
            ("result *= i",        "result -= i",   "wrong_operator", "subtract instead of multiply"),
            ("range(1, n + 1)",    "range(1, n)",   "off_by_one",    "stop before n — miss last factor"),
            ("range(1, n + 1)",    "range(2, n + 1)", "off_by_one",  "start at 2 — skip factor 1 for n=1 edge"),
            ("range(1, n + 1)",    "range(0, n + 1)", "off_by_one",  "start at 0 — multiply by zero"),
            ("if n < 0: raise ValueError(\"n must be >= 0\")\n                ", "",
             "dropped_guard", "remove negative-n guard"),
            ("if n < 0:",           "if n <= 0:",   "wrong_operator", "guard rejects n=0 as well"),
        ],
    ),

    # ── 3. power ─────────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_power",
        spec="Return base ** exp using a loop. exp must be a non-negative integer. Raise ValueError if exp < 0.",
        correct=_d("""
            def solution(base, exp):
                if exp < 0: raise ValueError("exp must be >= 0")
                result = 1
                for _ in range(exp):
                    result *= base
                return result
        """),
        tests=_d("""
            import pytest
            def test_zero_exp():  assert solution(5, 0) == 1
            def test_one_exp():   assert solution(5, 1) == 5
            def test_square():    assert solution(3, 2) == 9
            def test_cube():      assert solution(2, 3) == 8
            def test_base_zero(): assert solution(0, 5) == 0
            def test_neg_exp():
                with pytest.raises(ValueError): solution(2, -1)
        """),
        patches=[
            ("result *= base",     "result += base",  "wrong_operator", "add instead of multiply"),
            ("result *= base",     "result -= base",  "wrong_operator", "subtract instead of multiply"),
            ("range(exp)",         "range(exp - 1)",  "off_by_one",    "one too few iterations"),
            ("range(exp)",         "range(exp + 1)",  "off_by_one",    "one extra multiplication"),
            ("if exp < 0: raise ValueError(\"exp must be >= 0\")\n                ", "",
             "dropped_guard", "remove negative-exp guard"),
            ("result = 1",         "result = 0",      "wrong_operator", "wrong identity element"),
            ("result *= base",     "result *= exp",   "swapped_args",  "multiply by exp instead of base"),
            ("range(exp)",         "range(base)",     "swapped_args",  "iterate base times instead of exp"),
        ],
    ),

    # ── 4. nth_fibonacci ─────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_nth_fibonacci",
        spec="Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...). Raise ValueError if n < 0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be >= 0")
                if n == 0: return 0
                a, b = 0, 1
                for _ in range(1, n):
                    a, b = b, a + b
                return b
        """),
        tests=_d("""
            import pytest
            def test_fib0():  assert solution(0) == 0
            def test_fib1():  assert solution(1) == 1
            def test_fib2():  assert solution(2) == 1
            def test_fib5():  assert solution(5) == 5
            def test_fib10(): assert solution(10) == 55
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("a, b = b, a + b",   "a, b = b, a - b",  "wrong_operator", "subtract instead of add"),
            ("a, b = b, a + b",   "a, b = a, a + b",  "wrong_operator", "don't advance a"),
            ("range(1, n)",       "range(1, n - 1)",   "off_by_one",    "stop one short"),
            ("range(1, n)",       "range(1, n + 1)",   "off_by_one",    "one extra iteration"),
            ("range(1, n)",       "range(0, n)",       "off_by_one",    "start from 0 — extra iteration"),
            ("if n < 0: raise ValueError(\"n must be >= 0\")\n                ", "",
             "dropped_guard", "remove negative-n guard"),
            ("if n == 0: return 0\n                ", "",
             "dropped_guard", "remove base case n=0"),
            ("a, b = b, a + b",   "a, b = a + b, b",  "swapped_args",  "swap which value advances"),
            ("a, b = 0, 1",       "a, b = 1, 0",      "swapped_args",  "swap initial a and b"),
        ],
    ),

    # ── 5. count_multiples ───────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_multiples",
        spec="Count integers in [lo, hi] (inclusive) that are divisible by k. Raise ValueError if k == 0 or lo > hi.",
        correct=_d("""
            def solution(lo, hi, k):
                if k == 0: raise ValueError("k must not be zero")
                if lo > hi: raise ValueError("lo must be <= hi")
                return sum(1 for x in range(lo, hi + 1) if x % k == 0)
        """),
        tests=_d("""
            import pytest
            def test_basic():  assert solution(1, 10, 2) == 5
            def test_three():  assert solution(1, 9, 3) == 3
            def test_same():   assert solution(6, 6, 3) == 1
            def test_none():   assert solution(1, 5, 7) == 0
            def test_neg_lo():  assert solution(-6, 6, 3) == 5
            def test_k_zero():
                with pytest.raises(ValueError): solution(1, 10, 0)
            def test_lo_gt_hi():
                with pytest.raises(ValueError): solution(10, 1, 2)
        """),
        patches=[
            ("x % k == 0",         "x % k != 0",      "wrong_operator", "count non-multiples"),
            ("range(lo, hi + 1)",   "range(lo, hi)",   "off_by_one",    "exclude hi"),
            ("range(lo, hi + 1)",   "range(lo + 1, hi + 1)", "off_by_one", "exclude lo"),
            ("if k == 0: raise ValueError(\"k must not be zero\")\n                ", "",
             "dropped_guard", "remove k=0 guard"),
            ("if lo > hi: raise ValueError(\"lo must be <= hi\")\n                ", "",
             "dropped_guard", "remove lo>hi guard"),
            ("lo, hi, k",          "hi, lo, k",        "swapped_args",  "swap lo and hi"),
            ("range(lo, hi + 1)",  "range(hi, lo + 1)", "swapped_args", "range from hi to lo"),
        ],
    ),

    # ── 6. sum_list ──────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_sum_list",
        spec="Return the sum of a list of numbers. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                total = 0
                for x in lst:
                    total += x
                return total
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1, 2, 3]) == 6
            def test_single():  assert solution([5]) == 5
            def test_neg():     assert solution([-1, 1]) == 0
            def test_float():   assert abs(solution([1.5, 2.5]) - 4.0) < 1e-9
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("total += x",     "total -= x",    "wrong_operator", "subtract instead of add"),
            ("total += x",     "total *= x",    "wrong_operator", "multiply instead of add"),
            ("total = 0",      "total = 1",     "wrong_operator", "wrong initial value"),
            ("for x in lst:", "for x in lst[:-1]:", "off_by_one", "skip last element"),
            ("for x in lst:", "for x in lst[1:]:",  "off_by_one", "skip first element"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty-list guard"),
        ],
    ),

    # ── 7. find_max ──────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_find_max",
        spec="Return the maximum element of lst. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                result = lst[0]
                for x in lst[1:]:
                    if x > result:
                        result = x
                return result
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution([3, 1, 4, 1, 5]) == 5
            def test_single():   assert solution([7]) == 7
            def test_neg():      assert solution([-3, -1, -2]) == -1
            def test_all_same(): assert solution([4, 4, 4]) == 4
            def test_first_max():assert solution([9, 1, 2]) == 9
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("x > result",        "x < result",        "wrong_operator", "find min instead of max"),
            ("x > result",        "x >= result",       "wrong_operator", "use >= (ok for distinct but changes semantics)"),
            ("for x in lst[1:]:", "for x in lst:",     "off_by_one",    "start from lst[0] again"),
            ("for x in lst[1:]:", "for x in lst[2:]:", "off_by_one",    "skip first two elements"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty-list guard"),
            ("result = lst[0]",   "result = lst[-1]",  "off_by_one",    "seed with last element — fails if last is min"),
        ],
    ),

    # ── 8. count_in_range ────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_in_range",
        spec="Count elements of lst in [lo, hi] inclusive. Raise ValueError if lo > hi.",
        correct=_d("""
            def solution(lst, lo, hi):
                if lo > hi: raise ValueError("lo must be <= hi")
                return sum(1 for x in lst if lo <= x <= hi)
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,2,3,4,5], 2, 4) == 3
            def test_all():     assert solution([1,2,3], 1, 3) == 3
            def test_none():    assert solution([1,2,3], 5, 9) == 0
            def test_one():     assert solution([5], 5, 5) == 1
            def test_exact_lo():assert solution([1,2,3], 1, 1) == 1
            def test_exact_hi():assert solution([1,2,3], 3, 3) == 1
            def test_invalid():
                with pytest.raises(ValueError): solution([1,2,3], 5, 3)
        """),
        patches=[
            ("lo <= x <= hi",    "lo < x < hi",      "wrong_operator", "strict — excludes lo and hi"),
            ("lo <= x <= hi",    "lo < x <= hi",     "wrong_operator", "strict on lo only"),
            ("lo <= x <= hi",    "lo <= x < hi",     "wrong_operator", "strict on hi only"),
            ("lo <= x <= hi",    "hi <= x <= lo",    "swapped_args",  "swap lo and hi in check"),
            ("if lo > hi: raise ValueError(\"lo must be <= hi\")\n                ", "",
             "dropped_guard", "remove lo<=hi guard"),
            ("lo <= x <= hi",    "lo <= x",          "dropped_guard", "drop upper bound check"),
            ("lo <= x <= hi",    "x <= hi",          "dropped_guard", "drop lower bound check"),
            ("sum(1 for x in lst if lo <= x <= hi)",
             "sum(1 for x in lst if hi <= x <= lo)", "swapped_args",  "swap lo/hi in condition"),
        ],
    ),

    # ── 9. running_sum ───────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_running_sum",
        spec="Return a new list where result[i] is the sum of lst[0..i] inclusive. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                result = []
                total = 0
                for x in lst:
                    total += x
                    result.append(total)
                return result
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1, 2, 3]) == [1, 3, 6]
            def test_single():  assert solution([5]) == [5]
            def test_neg():     assert solution([-1, -2, -3]) == [-1, -3, -6]
            def test_mixed():   assert solution([1, -1, 2]) == [1, 0, 2]
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("total += x",           "total -= x",          "wrong_operator", "subtract instead of add"),
            ("result.append(total)", "result.append(x)",    "wrong_operator", "append original value not running sum"),
            ("total += x\n                    result.append(total)",
             "result.append(total)\n                    total += x",
             "off_by_one", "append before adding — shifts output by one"),
            ("total = 0",            "total = lst[0]",      "off_by_one",    "initialise with first element"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("for x in lst:",       "for x in lst[:-1]:",   "off_by_one",   "skip last element"),
        ],
    ),

    # ── 10. rotate_left ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_rotate_left",
        spec="Return lst rotated left by k positions. If lst is empty raise ValueError. k is taken modulo len(lst).",
        correct=_d("""
            def solution(lst, k):
                if not lst: raise ValueError("list must not be empty")
                k = k % len(lst)
                return lst[k:] + lst[:k]
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,2,3,4,5], 2) == [3,4,5,1,2]
            def test_zero():    assert solution([1,2,3], 0) == [1,2,3]
            def test_full():    assert solution([1,2,3], 3) == [1,2,3]
            def test_one():     assert solution([1,2,3], 1) == [2,3,1]
            def test_large_k(): assert solution([1,2,3], 7) == [2,3,1]
            def test_empty():
                with pytest.raises(ValueError): solution([], 2)
        """),
        patches=[
            ("lst[k:] + lst[:k]", "lst[:k] + lst[k:]",  "swapped_args",  "rotate right instead of left"),
            ("k = k % len(lst)",  "k = k % (len(lst) - 1)", "off_by_one","wrong modulus — fails on k=len-1"),
            ("k = k % len(lst)",  "k = k % (len(lst) + 1)", "off_by_one","larger modulus — k can exceed len"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("lst[k:] + lst[:k]", "lst[k:] + lst[k:]",  "wrong_operator","duplicate tail — lose prefix"),
            ("lst[k:] + lst[:k]", "lst[:k] + lst[:k]",  "wrong_operator","duplicate head — lose tail"),
        ],
    ),

    # ── 11. first_above ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_first_above",
        spec="Return the index of the first element strictly greater than threshold. Return -1 if none. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst, threshold):
                if not lst: raise ValueError("list must not be empty")
                for i in range(len(lst)):
                    if lst[i] > threshold:
                        return i
                return -1
        """),
        tests=_d("""
            import pytest
            def test_first():   assert solution([1,5,3], 2) == 1
            def test_last():    assert solution([1,2,5], 3) == 2
            def test_none():    assert solution([1,2,3], 5) == -1
            def test_first_elem(): assert solution([9,1,2], 5) == 0
            def test_threshold_eq(): assert solution([3,3,3], 3) == -1
            def test_empty():
                with pytest.raises(ValueError): solution([], 5)
        """),
        patches=[
            ("lst[i] > threshold",  "lst[i] >= threshold",  "wrong_operator", ">= includes threshold"),
            ("lst[i] > threshold",  "lst[i] < threshold",   "wrong_operator", "find first below"),
            ("lst[i] > threshold",  "threshold > lst[i]",   "swapped_args",   "swap operand order"),
            ("range(len(lst))",     "range(len(lst) - 1)",  "off_by_one",    "skip last element"),
            ("range(len(lst))",     "range(1, len(lst))",   "off_by_one",    "skip index 0"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return -1",           "return len(lst)",       "wrong_operator","wrong sentinel value"),
        ],
    ),

    # ── 12. sliding_window_sum ───────────────────────────────────────────────
    _TaskDef(
        task_id="full_sliding_window_sum",
        spec="Return a list of sums of consecutive windows of size k. len(result) == len(lst) - k + 1. Raise ValueError if k <= 0 or k > len(lst).",
        correct=_d("""
            def solution(lst, k):
                if k <= 0: raise ValueError("k must be positive")
                if k > len(lst): raise ValueError("k must be <= len(lst)")
                return [sum(lst[i:i + k]) for i in range(len(lst) - k + 1)]
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution([1,2,3,4,5], 3) == [6,9,12]
            def test_k1():       assert solution([1,2,3], 1) == [1,2,3]
            def test_full():     assert solution([1,2,3], 3) == [6]
            def test_two():      assert solution([1,2,3,4], 2) == [3,5,7]
            def test_k_zero():
                with pytest.raises(ValueError): solution([1,2,3], 0)
            def test_k_too_big():
                with pytest.raises(ValueError): solution([1,2], 5)
        """),
        patches=[
            ("lst[i:i + k]",         "lst[i:i + k - 1]",   "off_by_one",   "window one short"),
            ("lst[i:i + k]",         "lst[i:i + k + 1]",   "off_by_one",   "window one long"),
            ("range(len(lst) - k + 1)", "range(len(lst) - k)", "off_by_one","one fewer window — skip last"),
            ("range(len(lst) - k + 1)", "range(len(lst) - k + 2)", "off_by_one","one extra window — index error"),
            ("if k <= 0: raise ValueError(\"k must be positive\")\n                ", "",
             "dropped_guard", "remove k<=0 guard"),
            ("if k > len(lst): raise ValueError(\"k must be <= len(lst)\")\n                ", "",
             "dropped_guard", "remove k>len guard"),
            ("sum(lst[i:i + k])",    "sum(lst[i:i + k]) - lst[i]", "wrong_operator", "subtract first of window"),
            ("len(lst) - k + 1",     "len(lst) - k - 1",   "off_by_one",   "two fewer windows"),
        ],
    ),

    # ── 13. merge_sorted ─────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_merge_sorted",
        spec="Merge two sorted lists a and b into a single sorted list. Both may be empty.",
        correct=_d("""
            def solution(a, b):
                result = []
                i = j = 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result
        """),
        tests=_d("""
            def test_basic():     assert solution([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
            def test_empty_a():   assert solution([],[1,2,3]) == [1,2,3]
            def test_empty_b():   assert solution([1,2,3],[]) == [1,2,3]
            def test_both_empty():assert solution([],[]) == []
            def test_dup():       assert solution([1,2],[1,3]) == [1,1,2,3]
            def test_single():    assert solution([2],[1]) == [1,2]
        """),
        patches=[
            ("a[i] <= b[j]",       "a[i] < b[j]",         "wrong_operator", "strict < — equal elements wrong order"),
            ("a[i] <= b[j]",       "a[i] >= b[j]",        "wrong_operator", "inverted comparison"),
            ("result.extend(a[i:])\n                result.extend(b[j:])",
             "result.extend(b[j:])\n                result.extend(a[i:])",
             "swapped_args",  "append b leftovers before a leftovers"),
            ("i += 1",             "j += 1",               "swapped_args",  "advance j when a was smaller"),
            ("j += 1",             "i += 1",               "swapped_args",  "advance i when b was smaller"),
            ("result.append(a[i])\n                        i += 1",
             "result.append(b[j])\n                        i += 1",
             "swapped_args",  "append b[j] when a[i] is smaller"),
            ("i < len(a) and j < len(b)", "i < len(a) or j < len(b)", "wrong_operator","or loop runs past one list end"),
            ("i = j = 0",          "i = 1\n                j = 0",     "off_by_one",  "start i at 1"),
        ],
    ),

    # ── 14. second_largest ───────────────────────────────────────────────────
    _TaskDef(
        task_id="full_second_largest",
        spec="Return the second-largest distinct value in lst. Raise ValueError if fewer than 2 distinct values.",
        correct=_d("""
            def solution(lst):
                unique = sorted(set(lst), reverse=True)
                if len(unique) < 2: raise ValueError("need at least 2 distinct values")
                return unique[1]
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([3,1,4,1,5]) == 4
            def test_sorted():  assert solution([1,2,3,4,5]) == 4
            def test_two():     assert solution([7,3]) == 3
            def test_dupes():   assert solution([5,5,5,3]) == 3
            def test_one():
                with pytest.raises(ValueError): solution([1,1,1])
        """),
        patches=[
            ("unique[1]",          "unique[0]",            "off_by_one",   "return largest, not second"),
            ("unique[1]",          "unique[2]",            "off_by_one",   "return third (fails if only 2 distinct)"),
            ("reverse=True",       "reverse=False",        "wrong_operator","ascending sort — index 1 is second-smallest"),
            ("len(unique) < 2",    "len(unique) < 1",      "off_by_one",   "guard too lenient"),
            ("len(unique) < 2",    "len(unique) <= 2",     "wrong_operator","guard too strict — rejects exactly 2 distinct"),
            ("if len(unique) < 2: raise ValueError(\"need at least 2 distinct values\")\n                ", "",
             "dropped_guard", "remove minimum-distinct guard"),
            ("sorted(set(lst), reverse=True)", "sorted(lst, reverse=True)", "wrong_operator","don't deduplicate"),
        ],
    ),

    # ── 15. longest_run ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_longest_run",
        spec="Return the length of the longest consecutive run of equal elements. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                best = 1
                curr = 1
                for i in range(1, len(lst)):
                    if lst[i] == lst[i - 1]:
                        curr += 1
                        if curr > best:
                            best = curr
                    else:
                        curr = 1
                return best
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,1,2,2,2,3]) == 3
            def test_all():     assert solution([4,4,4,4]) == 4
            def test_none():    assert solution([1,2,3,4]) == 1
            def test_single():  assert solution([7]) == 1
            def test_tail():    assert solution([1,2,2,2]) == 3
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("lst[i] == lst[i - 1]",  "lst[i] != lst[i - 1]",  "wrong_operator","increment when different"),
            ("curr > best",            "curr >= best",           "wrong_operator","ok but consider ties"),
            ("range(1, len(lst))",     "range(len(lst))",        "off_by_one",   "start at 0 — compare with index -1"),
            ("range(1, len(lst))",     "range(1, len(lst) - 1)", "off_by_one",   "skip last element"),
            ("lst[i - 1]",             "lst[i + 1]",            "off_by_one",   "compare with next instead of prev"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("best = 1\n                curr = 1",  "best = 0\n                curr = 0",
             "off_by_one", "initialise with 0 — wrong for single-element list"),
        ],
    ),

    # ── 16. linear_search ────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_linear_search",
        spec="Return the index of the first occurrence of target in arr, or -1 if not found. Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr, target):
                if not arr: raise ValueError("arr must not be empty")
                for i in range(len(arr)):
                    if arr[i] == target:
                        return i
                return -1
        """),
        tests=_d("""
            import pytest
            def test_found_first():  assert solution([1,2,3], 1) == 0
            def test_found_mid():    assert solution([1,2,3], 2) == 1
            def test_found_last():   assert solution([1,2,3], 3) == 2
            def test_not_found():    assert solution([1,2,3], 9) == -1
            def test_first_of_dup(): assert solution([2,2,2], 2) == 0
            def test_empty():
                with pytest.raises(ValueError): solution([], 1)
        """),
        patches=[
            ("arr[i] == target",   "arr[i] != target",    "wrong_operator", "return index when not equal"),
            ("return i",           "return i + 1",        "off_by_one",    "return i+1 instead of i"),
            ("return i",           "return i - 1",        "off_by_one",    "return i-1 instead of i"),
            ("range(len(arr))",    "range(1, len(arr))",  "off_by_one",    "skip index 0"),
            ("range(len(arr))",    "range(len(arr) - 1)", "off_by_one",    "skip last index"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("arr[i] == target",   "target == arr[i + 1]", "swapped_args","compare with next element"),
        ],
    ),

    # ── 17. binary_search ────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_binary_search",
        spec="Return the index of target in sorted arr, or -1 if not found. Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr, target):
                if not arr: raise ValueError("arr must not be empty")
                lo, hi = 0, len(arr) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return -1
        """),
        tests=_d("""
            import pytest
            def test_found():    assert solution([1,3,5,7,9], 5) == 2
            def test_first():    assert solution([1,3,5,7,9], 1) == 0
            def test_last():     assert solution([1,3,5,7,9], 9) == 4
            def test_missing():  assert solution([1,3,5,7,9], 4) == -1
            def test_one_elem(): assert solution([42], 42) == 0
            def test_empty():
                with pytest.raises(ValueError): solution([], 5)
        """),
        patches=[
            ("lo <= hi",          "lo < hi",            "wrong_operator", "strict < — misses single-element"),
            ("arr[mid] < target", "arr[mid] > target",  "wrong_operator", "inverted direction"),
            ("arr[mid] == target","arr[mid] != target",  "wrong_operator","inverted equality"),
            ("lo = mid + 1",      "lo = mid",            "off_by_one",    "lo doesn't advance past mid — infinite loop"),
            ("hi = mid - 1",      "hi = mid",            "off_by_one",    "hi doesn't retreat past mid"),
            ("lo, hi = 0, len(arr) - 1", "lo, hi = 0, len(arr)", "off_by_one","hi one past end — index error"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("lo = mid + 1",      "hi = mid + 1",        "swapped_args",  "advance hi instead of lo"),
            ("hi = mid - 1",      "lo = mid - 1",        "swapped_args",  "retreat lo instead of hi"),
            ("mid = (lo + hi) // 2", "mid = (lo + hi + 1) // 2", "off_by_one","ceiling mid — may not converge"),
        ],
    ),

    # ── 18. lower_bound ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_lower_bound",
        spec="Return the first index i in sorted arr where arr[i] >= target. Return len(arr) if all < target. Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr, target):
                if not arr: raise ValueError("arr must not be empty")
                lo, hi = 0, len(arr)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if arr[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        """),
        tests=_d("""
            import pytest
            def test_found():    assert solution([1,2,3,4,5], 3) == 2
            def test_before():   assert solution([1,3,5,7], 4) == 2
            def test_first():    assert solution([2,3,4], 1) == 0
            def test_past_end(): assert solution([1,2,3], 9) == 3
            def test_dup():      assert solution([1,2,2,2,3], 2) == 1
            def test_empty():
                with pytest.raises(ValueError): solution([], 1)
        """),
        patches=[
            ("arr[mid] < target", "arr[mid] <= target",  "wrong_operator","<= gives upper_bound not lower_bound"),
            ("arr[mid] < target", "arr[mid] > target",   "wrong_operator","inverted condition"),
            ("lo < hi",           "lo <= hi",            "wrong_operator","<= — doesn't converge to correct bound"),
            ("lo, hi = 0, len(arr)", "lo, hi = 0, len(arr) - 1", "off_by_one","hi one short — miss last element"),
            ("lo = mid + 1",      "lo = mid",            "off_by_one",   "lo doesn't advance — infinite loop"),
            ("hi = mid",          "hi = mid - 1",        "off_by_one",   "hi retreats past valid boundary"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return lo",         "return hi",           "swapped_args", "return hi instead of lo"),
            ("lo = mid + 1",      "hi = mid + 1",        "swapped_args", "advance hi instead of lo"),
        ],
    ),

    # ── 19. upper_bound ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_upper_bound",
        spec="Return first index i in sorted arr where arr[i] > target. Return len(arr) if all <= target. Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr, target):
                if not arr: raise ValueError("arr must not be empty")
                lo, hi = 0, len(arr)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if arr[mid] <= target:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        """),
        tests=_d("""
            import pytest
            def test_found():    assert solution([1,2,3,4,5], 3) == 3
            def test_before():   assert solution([1,3,5,7], 4) == 2
            def test_first():    assert solution([2,3,4], 1) == 0
            def test_past_end(): assert solution([1,2,3], 9) == 3
            def test_dup():      assert solution([1,2,2,2,3], 2) == 4
            def test_empty():
                with pytest.raises(ValueError): solution([], 1)
        """),
        patches=[
            ("arr[mid] <= target","arr[mid] < target",   "wrong_operator","< gives lower_bound not upper_bound"),
            ("arr[mid] <= target","arr[mid] >= target",  "wrong_operator","inverted condition"),
            ("lo < hi",          "lo <= hi",             "wrong_operator","<= — doesn't converge"),
            ("lo, hi = 0, len(arr)", "lo, hi = 0, len(arr) - 1","off_by_one","hi one short"),
            ("lo = mid + 1",     "lo = mid",             "off_by_one",   "lo doesn't advance"),
            ("hi = mid",         "hi = mid - 1",         "off_by_one",   "hi retreats past boundary"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return lo",        "return hi",            "swapped_args", "return hi instead of lo"),
            ("lo = mid + 1",     "hi = mid + 1",         "swapped_args", "advance hi instead of lo"),
        ],
    ),

    # ── 20. max_in_slice ─────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_max_in_slice",
        spec="Return the maximum of arr[lo:hi] (exclusive hi). Raise ValueError if lo >= hi or indices out of range.",
        correct=_d("""
            def solution(arr, lo, hi):
                if lo >= hi: raise ValueError("lo must be < hi")
                if lo < 0 or hi > len(arr): raise ValueError("indices out of range")
                return max(arr[lo:hi])
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([3,1,4,1,5,9,2], 1, 5) == 5
            def test_full():    assert solution([1,2,3], 0, 3) == 3
            def test_single():  assert solution([7,2,5], 2, 3) == 5
            def test_lo_ge_hi():
                with pytest.raises(ValueError): solution([1,2,3], 2, 2)
            def test_neg_lo():
                with pytest.raises(ValueError): solution([1,2,3], -1, 2)
            def test_hi_oob():
                with pytest.raises(ValueError): solution([1,2,3], 0, 5)
        """),
        patches=[
            ("lo >= hi",          "lo > hi",             "wrong_operator","allows lo==hi — empty slice"),
            ("lo < 0 or hi > len(arr)", "lo < 0 or hi >= len(arr)", "off_by_one","rejects hi==len which is valid"),
            ("max(arr[lo:hi])",   "min(arr[lo:hi])",     "wrong_operator","return min instead of max"),
            ("if lo >= hi: raise ValueError(\"lo must be < hi\")\n                ", "",
             "dropped_guard", "remove lo<hi guard"),
            ("if lo < 0 or hi > len(arr): raise ValueError(\"indices out of range\")\n                ", "",
             "dropped_guard", "remove bounds guard"),
            ("arr[lo:hi]",        "arr[hi:lo]",          "swapped_args", "swap lo and hi in slice"),
            ("arr[lo:hi]",        "arr[lo:hi - 1]",      "off_by_one",  "hi exclusive then -1"),
        ],
    ),

    # ── 21. sum_slice ────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_sum_slice",
        spec="Return sum(arr[lo:hi]) (exclusive hi). Raise ValueError if lo >= hi or indices out of range.",
        correct=_d("""
            def solution(arr, lo, hi):
                if lo >= hi: raise ValueError("lo must be < hi")
                if lo < 0 or hi > len(arr): raise ValueError("indices out of range")
                return sum(arr[lo:hi])
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution([1,2,3,4,5], 1, 4) == 9
            def test_full():     assert solution([1,2,3], 0, 3) == 6
            def test_single_el():assert solution([7,2,5], 0, 1) == 7
            def test_lo_ge_hi():
                with pytest.raises(ValueError): solution([1,2,3], 2, 2)
            def test_oob():
                with pytest.raises(ValueError): solution([1,2,3], 0, 5)
        """),
        patches=[
            ("lo >= hi",         "lo > hi",              "wrong_operator","allows lo==hi — empty slice"),
            ("sum(arr[lo:hi])",  "sum(arr[hi:lo])",      "swapped_args", "reversed slice — empty"),
            ("sum(arr[lo:hi])",  "sum(arr[lo:hi - 1])",  "off_by_one",  "short by one element"),
            ("sum(arr[lo:hi])",  "sum(arr[lo + 1:hi])",  "off_by_one",  "skip first element of slice"),
            ("if lo >= hi: raise ValueError(\"lo must be < hi\")\n                ", "",
             "dropped_guard", "remove lo<hi guard"),
            ("if lo < 0 or hi > len(arr): raise ValueError(\"indices out of range\")\n                ", "",
             "dropped_guard", "remove bounds guard"),
            ("hi > len(arr)",    "hi >= len(arr)",       "off_by_one",  "rejects valid hi==len"),
        ],
    ),

    # ── 22. ranges_overlap ───────────────────────────────────────────────────
    _TaskDef(
        task_id="full_ranges_overlap",
        spec="Return True if intervals [a0,a1] and [b0,b1] overlap (inclusive). Raise ValueError if a0>a1 or b0>b1.",
        correct=_d("""
            def solution(a0, a1, b0, b1):
                if a0 > a1: raise ValueError("a0 must be <= a1")
                if b0 > b1: raise ValueError("b0 must be <= b1")
                return a0 <= b1 and b0 <= a1
        """),
        tests=_d("""
            import pytest
            def test_overlap():    assert solution(1, 5, 3, 7) == True
            def test_touch():      assert solution(1, 3, 3, 5) == True
            def test_no_overlap(): assert solution(1, 3, 4, 6) == False
            def test_contained():  assert solution(1, 10, 3, 7) == True
            def test_single_pt():  assert solution(5, 5, 5, 5) == True
            def test_invalid_a():
                with pytest.raises(ValueError): solution(5, 1, 2, 4)
            def test_invalid_b():
                with pytest.raises(ValueError): solution(1, 4, 7, 3)
        """),
        patches=[
            ("a0 <= b1 and b0 <= a1", "a0 < b1 and b0 < a1",   "wrong_operator","strict — touching intervals return False"),
            ("a0 <= b1 and b0 <= a1", "a0 >= b1 and b0 >= a1",  "wrong_operator","inverted — returns True for non-overlap"),
            ("a0 <= b1 and b0 <= a1", "a0 <= b1 or b0 <= a1",   "wrong_operator","or — too permissive"),
            ("a0 <= b1 and b0 <= a1", "a1 <= b1 and b0 <= a0",  "swapped_args",  "swap a0/a1 in condition"),
            ("a0 <= b1 and b0 <= a1", "a0 <= b0 and b1 <= a1",  "swapped_args",  "containment check instead of overlap"),
            ("if a0 > a1: raise ValueError(\"a0 must be <= a1\")\n                ", "",
             "dropped_guard", "remove a-interval guard"),
            ("if b0 > b1: raise ValueError(\"b0 must be <= b1\")\n                ", "",
             "dropped_guard", "remove b-interval guard"),
            ("a0 <= b1 and b0 <= a1", "b0 <= a1 and a0 <= b1",  "swapped_args",  "swap the two conditions (same result — likely equivalent)"),
        ],
    ),

    # ── 23. interval_contains ────────────────────────────────────────────────
    _TaskDef(
        task_id="full_interval_contains",
        spec="Return True if [b0,b1] is entirely contained in [a0,a1]. Raise ValueError if a0>a1 or b0>b1.",
        correct=_d("""
            def solution(a0, a1, b0, b1):
                if a0 > a1: raise ValueError("a0 must be <= a1")
                if b0 > b1: raise ValueError("b0 must be <= b1")
                return a0 <= b0 and b1 <= a1
        """),
        tests=_d("""
            import pytest
            def test_contained():    assert solution(1, 10, 3, 7) == True
            def test_equal():        assert solution(2, 5, 2, 5) == True
            def test_not_contain():  assert solution(3, 7, 1, 10) == False
            def test_partial():      assert solution(1, 5, 3, 8) == False
            def test_touch_lo():     assert solution(2, 8, 2, 6) == True
            def test_touch_hi():     assert solution(2, 8, 4, 8) == True
            def test_invalid():
                with pytest.raises(ValueError): solution(5, 1, 2, 4)
        """),
        patches=[
            ("a0 <= b0 and b1 <= a1","a0 < b0 and b1 < a1",    "wrong_operator","strict — touching bounds not allowed"),
            ("a0 <= b0 and b1 <= a1","a0 <= b1 and b0 <= a1",  "wrong_operator","overlap check instead of containment"),
            ("a0 <= b0 and b1 <= a1","b0 <= a0 and a1 <= b1",  "swapped_args",  "containment reversed — b contains a"),
            ("a0 <= b0 and b1 <= a1","a0 <= b0 and a1 <= b1",  "swapped_args",  "a1<=b1 instead of b1<=a1"),
            ("if a0 > a1: raise ValueError(\"a0 must be <= a1\")\n                ", "",
             "dropped_guard", "remove a guard"),
            ("if b0 > b1: raise ValueError(\"b0 must be <= b1\")\n                ", "",
             "dropped_guard", "remove b guard"),
            ("a0 <= b0 and b1 <= a1","a0 <= b0 or b1 <= a1",   "wrong_operator","or — too permissive"),
        ],
    ),

    # ── 24. insert_position ──────────────────────────────────────────────────
    _TaskDef(
        task_id="full_insert_position",
        spec="Return the index where val should be inserted in sorted arr to keep it sorted (leftmost position). arr may be empty.",
        correct=_d("""
            def solution(arr, val):
                lo, hi = 0, len(arr)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if arr[mid] < val:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        """),
        tests=_d("""
            def test_empty():     assert solution([], 5) == 0
            def test_before():    assert solution([2,4,6], 1) == 0
            def test_after():     assert solution([2,4,6], 9) == 3
            def test_between():   assert solution([2,4,6], 5) == 2
            def test_exact():     assert solution([1,3,3,5], 3) == 1
            def test_after_dup(): assert solution([1,2,2,4], 2) == 1
        """),
        patches=[
            ("arr[mid] < val",   "arr[mid] <= val",    "wrong_operator","<= gives rightmost not leftmost"),
            ("arr[mid] < val",   "arr[mid] > val",     "wrong_operator","inverted comparison"),
            ("lo < hi",          "lo <= hi",           "wrong_operator","<= — infinite loop on equal"),
            ("lo, hi = 0, len(arr)", "lo, hi = 0, len(arr) - 1", "off_by_one","hi one short — can't insert at end"),
            ("lo = mid + 1",     "lo = mid",           "off_by_one",   "lo stuck — infinite loop"),
            ("hi = mid",         "hi = mid - 1",       "off_by_one",   "hi retreats too far"),
            ("return lo",        "return hi",          "swapped_args", "return hi instead of lo"),
            ("lo = mid + 1",     "hi = mid + 1",       "swapped_args", "update hi when arr[mid]<val"),
        ],
    ),

    # ── 25. count_less ───────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_less",
        spec="Return the number of elements in sorted arr that are strictly less than target. Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr, target):
                if not arr: raise ValueError("arr must not be empty")
                lo, hi = 0, len(arr)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if arr[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        """),
        tests=_d("""
            import pytest
            def test_none():    assert solution([1,2,3,4,5], 1) == 0
            def test_all():     assert solution([1,2,3,4,5], 9) == 5
            def test_three():   assert solution([1,2,3,4,5], 4) == 3
            def test_dup():     assert solution([1,2,2,2,3], 2) == 1
            def test_empty():
                with pytest.raises(ValueError): solution([], 5)
        """),
        patches=[
            ("arr[mid] < target","arr[mid] <= target",  "wrong_operator","counts elements <= target"),
            ("arr[mid] < target","arr[mid] > target",   "wrong_operator","inverted — counts elements > target"),
            ("lo < hi",         "lo <= hi",             "wrong_operator","<= — doesn't converge"),
            ("lo = mid + 1",    "lo = mid",             "off_by_one",   "lo doesn't advance"),
            ("hi = mid",        "hi = mid - 1",         "off_by_one",   "hi overshoots"),
            ("lo, hi = 0, len(arr)", "lo, hi = 0, len(arr) - 1","off_by_one","hi one short"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return lo",       "return hi",            "swapped_args", "return hi instead of lo"),
        ],
    ),

    # ── 26. count_char ───────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_char",
        spec="Return the number of times character c appears in string s. Raise ValueError if c is not a single character.",
        correct=_d("""
            def solution(s, c):
                if len(c) != 1: raise ValueError("c must be a single character")
                return sum(1 for ch in s if ch == c)
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution("hello", "l") == 2
            def test_none():    assert solution("hello", "z") == 0
            def test_all():     assert solution("aaa", "a") == 3
            def test_empty_s(): assert solution("", "a") == 0
            def test_bad_c():
                with pytest.raises(ValueError): solution("hello", "ll")
        """),
        patches=[
            ("ch == c",          "ch != c",             "wrong_operator","count non-matching"),
            ("len(c) != 1",      "len(c) == 1",         "wrong_operator","guard inverted — only errors if c is 1 char"),
            ("if len(c) != 1: raise ValueError(\"c must be a single character\")\n                ", "",
             "dropped_guard", "remove single-char guard"),
            ("for ch in s",      "for ch in s[1:]",     "off_by_one",   "skip first char"),
            ("for ch in s",      "for ch in s[:-1]",    "off_by_one",   "skip last char"),
            ("ch == c",          "c == ch",             "swapped_args", "same logic — likely equivalent"),
        ],
    ),

    # ── 27. is_palindrome ────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_is_palindrome",
        spec="Return True if s is a palindrome (case-insensitive, ignoring spaces). Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                cleaned = s.replace(" ", "").lower()
                n = len(cleaned)
                for i in range(n // 2):
                    if cleaned[i] != cleaned[n - 1 - i]:
                        return False
                return True
        """),
        tests=_d("""
            import pytest
            def test_yes():       assert solution("racecar") == True
            def test_no():        assert solution("hello") == False
            def test_case():      assert solution("Racecar") == True
            def test_space():     assert solution("race bar") == False
            def test_palindrome_space(): assert solution("a man a plan a canal panama".replace(" ","")) == True
            def test_single():    assert solution("a") == True
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("cleaned[i] != cleaned[n - 1 - i]", "cleaned[i] == cleaned[n - 1 - i]",
             "wrong_operator", "return False when chars match"),
            ("n - 1 - i",         "n - i",               "off_by_one",   "off-by-one in mirror index"),
            ("range(n // 2)",     "range(n // 2 + 1)",   "off_by_one",   "compare one past the middle"),
            ("range(n // 2)",     "range(n // 2 - 1)",   "off_by_one",   "one fewer comparison"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("s.replace(\" \", \"\").lower()", "s.lower()", "dropped_guard","spaces not removed"),
            ("s.replace(\" \", \"\").lower()", "s.replace(\" \", \"\")", "dropped_guard","not lowercased"),
        ],
    ),

    # ── 28. count_words ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_words",
        spec="Return the number of words in s, where words are separated by whitespace. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                return len(s.split())
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution("hello world") == 2
            def test_single():   assert solution("hi") == 1
            def test_spaces():   assert solution("  hello   world  ") == 2
            def test_three():    assert solution("one two three") == 3
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("len(s.split())",   "len(s.split()) - 1",  "off_by_one",  "one fewer word"),
            ("len(s.split())",   "len(s.split()) + 1",  "off_by_one",  "one extra word"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("s.split()",        "s.split(' ')",        "wrong_operator","split on single space — multi-spaces break it"),
            ("len(s.split())",   "len(s)",              "wrong_operator","count chars not words"),
        ],
    ),

    # ── 29. longest_common_prefix ─────────────────────────────────────────────
    _TaskDef(
        task_id="full_longest_common_prefix",
        spec="Return the longest common prefix of all strings in words. Return '' if words is empty or no common prefix. Raise ValueError if any word is empty.",
        correct=_d("""
            def solution(words):
                if not words: return ""
                if any(not w for w in words): raise ValueError("words must not contain empty strings")
                prefix = words[0]
                for word in words[1:]:
                    while not word.startswith(prefix):
                        prefix = prefix[:-1]
                        if not prefix: return ""
                return prefix
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution(["flower","flow","flight"]) == "fl"
            def test_all():      assert solution(["abc","abc","abc"]) == "abc"
            def test_none():     assert solution(["dog","car","racecar"]) == ""
            def test_empty_list():assert solution([]) == ""
            def test_single():   assert solution(["hello"]) == "hello"
            def test_empty_word():
                with pytest.raises(ValueError): solution(["abc",""])
        """),
        patches=[
            ("prefix = prefix[:-1]", "prefix = prefix[1:]", "off_by_one","trim from front not back"),
            ("words[1:]",          "words[2:]",             "off_by_one","skip second word"),
            ("words[1:]",          "words[:-1]",            "off_by_one","skip last word"),
            ("if any(not w for w in words): raise ValueError(\"words must not contain empty strings\")",
             "",
             "dropped_guard", "remove empty-word guard"),
            ("if not words: return \"\"", "",
             "dropped_guard", "remove empty-list guard"),
            ("word.startswith(prefix)", "prefix.startswith(word)","swapped_args","check if prefix starts with word"),
            ("not word.startswith(prefix)", "word.startswith(prefix)", "wrong_operator","inverted — trim when prefix DOES match"),
        ],
    ),

    # ── 30. encode_rle ───────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_encode_rle",
        spec="Return run-length encoding of s as list of (count, char) pairs. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                result = []
                count = 1
                for i in range(1, len(s)):
                    if s[i] == s[i - 1]:
                        count += 1
                    else:
                        result.append((count, s[i - 1]))
                        count = 1
                result.append((count, s[-1]))
                return result
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution("aaabbc") == [(3,'a'),(2,'b'),(1,'c')]
            def test_single():  assert solution("z") == [(1,'z')]
            def test_all_diff():assert solution("abc") == [(1,'a'),(1,'b'),(1,'c')]
            def test_all_same():assert solution("aaaa") == [(4,'a')]
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("s[i] == s[i - 1]", "s[i] != s[i - 1]",   "wrong_operator","increment on change"),
            ("count += 1",       "count -= 1",           "wrong_operator","decrement count"),
            ("range(1, len(s))", "range(len(s))",        "off_by_one",   "compare s[0] with s[-1]"),
            ("range(1, len(s))", "range(1, len(s) - 1)", "off_by_one",   "skip last char"),
            ("s[i - 1]",         "s[i]",                "off_by_one",   "append current char not previous"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("result.append((count, s[-1]))", "pass",   "dropped_guard", "omit final group"),
            ("count = 1\n                for i in range",
             "count = 0\n                for i in range","off_by_one",   "initialise count at 0"),
        ],
    ),

    # ── 31. first_duplicate_char ──────────────────────────────────────────────
    _TaskDef(
        task_id="full_first_duplicate_char",
        spec="Return the first character that appears more than once in s, or None if none. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                seen = set()
                for ch in s:
                    if ch in seen:
                        return ch
                    seen.add(ch)
                return None
        """),
        tests=_d("""
            import pytest
            def test_found():   assert solution("abca") == "a"
            def test_first():   assert solution("abbc") == "b"
            def test_none():    assert solution("abcd") is None
            def test_all_dup(): assert solution("aabb") == "a"
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("if ch in seen",    "if ch not in seen",   "wrong_operator","return first non-duplicate"),
            ("seen.add(ch)",     "seen.remove(ch) if ch in seen else seen.add(ch)",
             "wrong_operator",  "toggle — wrong logic"),
            ("for ch in s:",    "for ch in s[1:]:",     "off_by_one",   "skip first char"),
            ("for ch in s:",    "for ch in s[:-1]:",    "off_by_one",   "skip last char"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return ch",       "return s[0]",         "wrong_operator","always return first char"),
        ],
    ),

    # ── 32. count_uppercase ──────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_uppercase",
        spec="Return the number of uppercase letters in s. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                return sum(1 for ch in s if ch.isupper())
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution("Hello World") == 2
            def test_none():    assert solution("hello") == 0
            def test_all():     assert solution("ABC") == 3
            def test_digit():   assert solution("A1B2") == 2
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("ch.isupper()",     "ch.islower()",        "wrong_operator","count lowercase"),
            ("ch.isupper()",     "not ch.isupper()",    "wrong_operator","count non-uppercase"),
            ("for ch in s",     "for ch in s[1:]",     "off_by_one",   "skip first char"),
            ("for ch in s",     "for ch in s[:-1]",    "off_by_one",   "skip last char"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
        ],
    ),

    # ── 33. reverse_words ────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_reverse_words",
        spec="Reverse the order of words in s (words separated by single spaces). Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                words = s.split(" ")
                return " ".join(reversed(words))
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution("hello world") == "world hello"
            def test_three():   assert solution("one two three") == "three two one"
            def test_single():  assert solution("word") == "word"
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ('" ".join(reversed(words))', '" ".join(words)',  "wrong_operator","no reversal"),
            ('" ".join(reversed(words))', '" ".join(words[::-1])', "wrong_operator","same as correct — equivalent"),
            ("s.split(\" \")",  "s.split()",           "wrong_operator","split on whitespace — different for multi-spaces"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ('" ".join(reversed(words))', '"-".join(reversed(words))', "wrong_operator","wrong separator"),
        ],
    ),

    # ── 34. count_vowels ─────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_vowels",
        spec="Count vowels (a,e,i,o,u, case-insensitive) in s. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                return sum(1 for ch in s if ch.lower() in "aeiou")
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution("hello") == 2
            def test_upper():   assert solution("HELLO") == 2
            def test_none():    assert solution("bcdf") == 0
            def test_all():     assert solution("aeiou") == 5
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ('ch.lower() in "aeiou"', 'ch.lower() not in "aeiou"',"wrong_operator","count consonants"),
            ('ch.lower() in "aeiou"', 'ch in "aeiou"',           "wrong_operator","no lowercase — misses uppercase"),
            ('ch.lower() in "aeiou"', 'ch.lower() in "aeio"',    "off_by_one",   "missing 'u'"),
            ("for ch in s",    "for ch in s[1:]",               "off_by_one",   "skip first char"),
            ("for ch in s",    "for ch in s[:-1]",              "off_by_one",   "skip last char"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
        ],
    ),

    # ── 35. max_subarray_sum ─────────────────────────────────────────────────
    _TaskDef(
        task_id="full_max_subarray_sum",
        spec="Return the maximum subarray sum (Kadane's algorithm). Raise ValueError if arr is empty.",
        correct=_d("""
            def solution(arr):
                if not arr: raise ValueError("arr must not be empty")
                best = arr[0]
                curr = arr[0]
                for x in arr[1:]:
                    curr = max(x, curr + x)
                    if curr > best:
                        best = curr
                return best
        """),
        tests=_d("""
            import pytest
            def test_all_pos():  assert solution([1,2,3,4]) == 10
            def test_mixed():    assert solution([-2,1,-3,4,-1,2,1,-5,4]) == 6
            def test_all_neg():  assert solution([-3,-1,-4]) == -1
            def test_single():   assert solution([5]) == 5
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("curr = max(x, curr + x)", "curr = min(x, curr + x)",  "wrong_operator","minimum subarray"),
            ("if curr > best:",         "if curr < best:",           "wrong_operator","update best when curr is smaller"),
            ("for x in arr[1:]:",       "for x in arr:",             "off_by_one",   "re-process arr[0] — double-counts"),
            ("for x in arr[1:]:",       "for x in arr[2:]:",         "off_by_one",   "skip arr[1]"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("best = arr[0]\n                curr = arr[0]",
             "best = 0\n                curr = 0",               "off_by_one",   "initialise with 0 — wrong for all-negative"),
        ],
    ),

    # ── 36. count_pairs ──────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_pairs",
        spec="Count pairs (i,j) with i < j such that lst[i] + lst[j] == target. Raise ValueError if lst has fewer than 2 elements.",
        correct=_d("""
            def solution(lst, target):
                if len(lst) < 2: raise ValueError("need at least 2 elements")
                count = 0
                for i in range(len(lst) - 1):
                    for j in range(i + 1, len(lst)):
                        if lst[i] + lst[j] == target:
                            count += 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution([1,2,3,4,5], 6) == 2
            def test_none():     assert solution([1,2,3], 9) == 0
            def test_one():      assert solution([1,5], 6) == 1
            def test_dup():      assert solution([3,3,3], 6) == 3
            def test_too_short():
                with pytest.raises(ValueError): solution([1], 2)
        """),
        patches=[
            ("lst[i] + lst[j] == target", "lst[i] + lst[j] != target",  "wrong_operator","count non-matching pairs"),
            ("lst[i] + lst[j] == target", "lst[i] + lst[j] < target",   "wrong_operator","wrong comparison"),
            ("range(i + 1, len(lst))",    "range(i, len(lst))",          "off_by_one",   "include i==j pairs"),
            ("range(len(lst) - 1)",       "range(len(lst))",             "off_by_one",   "outer loop includes last index"),
            ("range(i + 1, len(lst))",    "range(i + 2, len(lst))",      "off_by_one",   "skip j==i+1"),
            ("if len(lst) < 2: raise ValueError(\"need at least 2 elements\")\n                ", "",
             "dropped_guard", "remove size guard"),
            ("lst[i] + lst[j]",           "lst[j] + lst[i]",             "swapped_args", "swap — commutative, likely equivalent"),
            ("count += 1",                "count += 2",                  "wrong_operator","count each pair twice"),
        ],
    ),

    # ── 37. cumulative_product ───────────────────────────────────────────────
    _TaskDef(
        task_id="full_cumulative_product",
        spec="Return a new list where result[i] = product of lst[0..i]. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                result = []
                product = 1
                for x in lst:
                    product *= x
                    result.append(product)
                return result
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,2,3,4]) == [1,2,6,24]
            def test_single():  assert solution([5]) == [5]
            def test_zero():    assert solution([1,2,0,3]) == [1,2,0,0]
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("product *= x",          "product += x",          "wrong_operator","sum instead of product"),
            ("product = 1",           "product = 0",           "wrong_operator","wrong identity element"),
            ("product *= x\n                    result.append(product)",
             "result.append(product)\n                    product *= x",
             "off_by_one", "append before multiplying — shifts by one"),
            ("for x in lst:",        "for x in lst[:-1]:",    "off_by_one",   "skip last element"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("result.append(product)","result.append(x)",     "wrong_operator","append element not product"),
        ],
    ),

    # ── 38. find_peak ────────────────────────────────────────────────────────
    _TaskDef(
        task_id="full_find_peak",
        spec="Return the index of any local maximum (element >= both neighbours). Raise ValueError if arr has fewer than 1 element. For single-element arr return 0.",
        correct=_d("""
            def solution(arr):
                if not arr: raise ValueError("arr must not be empty")
                n = len(arr)
                lo, hi = 0, n - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if arr[mid] < arr[mid + 1]:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo
        """),
        tests=_d("""
            import pytest
            def test_basic():
                idx = solution([1,3,5,3,1]); assert idx == 2
            def test_left():
                idx = solution([5,3,1]); assert idx == 0
            def test_right():
                idx = solution([1,3,5]); assert idx == 2
            def test_single():  assert solution([7]) == 0
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("arr[mid] < arr[mid + 1]", "arr[mid] > arr[mid + 1]",  "wrong_operator","find trough instead of peak"),
            ("arr[mid] < arr[mid + 1]", "arr[mid] <= arr[mid + 1]", "wrong_operator","<= shifts which half to keep"),
            ("lo < hi",                 "lo <= hi",                  "wrong_operator","<= — out-of-bounds on mid+1"),
            ("lo = mid + 1",            "lo = mid",                  "off_by_one",   "lo doesn't advance"),
            ("hi = mid",                "hi = mid - 1",              "off_by_one",   "hi overshoots"),
            ("if not arr: raise ValueError(\"arr must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("lo = mid + 1",            "hi = mid + 1",              "swapped_args", "update hi when going right"),
            ("hi = mid",                "lo = mid",                  "swapped_args", "update lo when going left"),
        ],
    ),

    # ── 39. count_less_than ──────────────────────────────────────────────────
    _TaskDef(
        task_id="full_count_less_than",
        spec="Return the count of elements in lst strictly less than threshold. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst, threshold):
                if not lst: raise ValueError("lst must not be empty")
                return sum(1 for x in lst if x < threshold)
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,2,3,4,5], 3) == 2
            def test_none():    assert solution([5,6,7], 5) == 0
            def test_all():     assert solution([1,2,3], 10) == 3
            def test_at_thresh():assert solution([3,3,3], 3) == 0
            def test_empty():
                with pytest.raises(ValueError): solution([], 5)
        """),
        patches=[
            ("x < threshold",   "x <= threshold",     "wrong_operator","count elements <= threshold"),
            ("x < threshold",   "x > threshold",      "wrong_operator","count elements above"),
            ("x < threshold",   "threshold < x",      "swapped_args", "swap operands — same as count above"),
            ("for x in lst",   "for x in lst[1:]",    "off_by_one",   "skip first element"),
            ("for x in lst",   "for x in lst[:-1]",   "off_by_one",   "skip last element"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
        ],
    ),

    # ── 40. partition_count ──────────────────────────────────────────────────
    _TaskDef(
        task_id="full_partition_count",
        spec="Return (count_less, count_equal, count_greater) for elements of lst relative to pivot. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst, pivot):
                if not lst: raise ValueError("lst must not be empty")
                less = sum(1 for x in lst if x < pivot)
                equal = sum(1 for x in lst if x == pivot)
                greater = sum(1 for x in lst if x > pivot)
                return (less, equal, greater)
        """),
        tests=_d("""
            import pytest
            def test_basic():   assert solution([1,2,3,4,5], 3) == (2,1,2)
            def test_none_less():assert solution([3,3,3], 3) == (0,3,0)
            def test_all_less(): assert solution([1,2,3], 5) == (3,0,0)
            def test_all_gt():   assert solution([5,6,7], 3) == (0,0,3)
            def test_empty():
                with pytest.raises(ValueError): solution([], 3)
        """),
        patches=[
            ("x < pivot",        "x <= pivot",        "wrong_operator","count <=pivot as less"),
            ("x > pivot",        "x >= pivot",        "wrong_operator","count >=pivot as greater"),
            ("x < pivot",        "x > pivot",         "swapped_args",  "swap less and greater counts"),
            ("x > pivot",        "x < pivot",         "swapped_args",  "count less in 'greater' position"),
            ("less = sum(1 for x in lst if x < pivot)\n                equal",
             "less = sum(1 for x in lst if x <= pivot)\n                equal",
             "off_by_one", "include equal in less — double-counts"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n                ", "",
             "dropped_guard", "remove empty guard"),
            ("return (less, equal, greater)", "return (greater, equal, less)",
             "swapped_args", "return in reversed order"),
        ],
    ),

]  # end _TASKS


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_full_corpus(
    output_path: str | Path = "data/full_corpus.jsonl",
    abort_on_mismatch: bool = True,
    verbose: bool = True,
) -> dict:
    """Build, verify, and save the full corpus.

    For each task:
    1. Materialise mutations from patches.
    2. Verify each mutation with hidden tests (discard equivalent mutants).
    3. Verify the correct item.
    4. Append all passing items to the corpus.

    Returns a stats dict.
    """
    from .mutate import verify_mutation, MutationResult

    all_items: list = []
    n_equiv = 0
    defect_counts: dict = {}
    task_errors: list = []

    for task in _TASKS:
        if verbose:
            print(f"  [{task.task_id}] building mutations...", end=" ", flush=True)

        # 1. Materialise mutations
        try:
            mutations = task.build_mutations()
        except ValueError as exc:
            task_errors.append(str(exc))
            print(f"PATCH ERROR: {exc}")
            continue

        # 2. Verify each mutation
        kept: list = []
        n_task_equiv = 0
        for mut in mutations:
            res = verify_mutation(task.task_id, task.correct, mut, task.tests)
            if res.error:
                task_errors.append(f"[{task.task_id}/{mut.defect_type}] {res.error}")
                if verbose:
                    print(f"\n    ERROR in mutation: {res.error[:80]}")
                continue
            if res.is_defective:
                item = CorpusItem(
                    item_id=f"{task.task_id}_{mut.defect_type}_{len(kept):02d}",
                    spec=task.spec,
                    candidate_code=mut.source,
                    gt_label="defective",
                    defect_type=mut.defect_type,
                    source="synthetic_full",
                    contamination_flag=False,
                    hidden_tests=task.tests,
                    metadata={"description": mut.description},
                )
                kept.append(item)
                defect_counts[mut.defect_type] = defect_counts.get(mut.defect_type, 0) + 1
            else:
                n_task_equiv += 1
                n_equiv += 1

        if verbose:
            dtype_summary = {}
            for it in kept:
                dtype_summary[it.defect_type] = dtype_summary.get(it.defect_type, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(dtype_summary.items()))
            print(f"{len(kept)} kept ({n_task_equiv} equiv). [{summary}]")

        # 3. Correct item
        correct_item = CorpusItem(
            item_id=f"{task.task_id}_correct",
            spec=task.spec,
            candidate_code=task.correct,
            gt_label="correct",
            defect_type=None,
            source="synthetic_full",
            contamination_flag=False,
            hidden_tests=task.tests,
        )
        all_items.append(correct_item)
        all_items.extend(kept)

    # 4. Final GT verification
    if verbose:
        print(f"\nVerifying {len(all_items)} items...", end=" ", flush=True)
    mismatches = verify_corpus(all_items, abort_on_mismatch=abort_on_mismatch)
    if verbose:
        print("done." if not mismatches else f"MISMATCHES: {mismatches}")

    # 5. Save
    save_corpus(all_items, output_path)

    n_correct  = sum(1 for it in all_items if it.gt_label == "correct")
    n_defective= sum(1 for it in all_items if it.gt_label == "defective")

    stats = {
        "n_total":            len(all_items),
        "n_correct":          n_correct,
        "n_defective":        n_defective,
        "n_equivalent_discarded": n_equiv,
        "defect_type_counts": defect_counts,
        "task_errors":        task_errors,
        "output_path":        str(output_path),
    }
    if verbose:
        print("\n-- Full corpus stats -----------------------------------------")
        print(f"  total items   : {stats['n_total']}")
        print(f"  correct       : {n_correct}")
        print(f"  defective     : {n_defective}")
        print(f"  equiv discard : {n_equiv}")
        for dt, cnt in sorted(defect_counts.items()):
            mark = "[OK]" if cnt >= 170 else ("[~] " if cnt >= 100 else "[LO]")
            print(f"  {dt:<22}: {cnt:3d}  {mark}")
        if task_errors:
            print(f"  errors        : {len(task_errors)}")
        print(f"  output        : {output_path}")
    return stats
