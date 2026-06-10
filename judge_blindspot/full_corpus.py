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

_TaskDef(
        task_id="full_gcd",
        spec="GCD of non-negative integers a, b via Euclidean algorithm (while b != 0: a,b = b, a%b). Raise ValueError if either is negative.",
        correct=_d("""
            def solution(a, b):
                if a < 0 or b < 0: raise ValueError("arguments must be non-negative")
                while b != 0:
                    a, b = b, a % b
                return a
        """),
        tests=_d("""
            import pytest
            def test_basic():      assert solution(12, 8) == 4
            def test_coprime():    assert solution(7, 13) == 1
            def test_same():       assert solution(9, 9) == 9
            def test_zero_b():     assert solution(5, 0) == 5
            def test_zero_a():     assert solution(0, 5) == 5
            def test_both_zero():  assert solution(0, 0) == 0
            def test_neg_a():
                with pytest.raises(ValueError): solution(-1, 5)
            def test_neg_b():
                with pytest.raises(ValueError): solution(5, -1)
        """),
        patches=[
            ("a % b", "a // b", "wrong_operator", "% replaced with // breaks Euclidean algorithm"),
            ("b != 0", "b == 0", "wrong_operator", "loop condition inverted, never iterates"),
            ("a, b = b, a % b", "a, b = a % b, b", "swapped_args", "swap order wrong, algorithm does not converge correctly"),
            ("if a < 0 or b < 0: raise ValueError(\"arguments must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative check guard"),
        ],
    ),

    _TaskDef(
        task_id="full_lcm",
        spec="LCM of non-negative integers a, b via formula (a*b)//gcd(a,b). lcm(0,x)=0. Raise ValueError if either is negative.",
        correct=_d("""
            def solution(a, b):
                if a < 0 or b < 0: raise ValueError("arguments must be non-negative")
                if a == 0 or b == 0: return 0
                def gcd(x, y):
                    while y != 0:
                        x, y = y, x % y
                    return x
                return (a * b) // gcd(a, b)
        """),
        tests=_d("""
            import pytest
            def test_basic():     assert solution(4, 6) == 12
            def test_coprime():   assert solution(3, 7) == 21
            def test_same():      assert solution(5, 5) == 5
            def test_zero_a():    assert solution(0, 7) == 0
            def test_zero_b():    assert solution(7, 0) == 0
            def test_one():       assert solution(1, 8) == 8
            def test_neg_a():
                with pytest.raises(ValueError): solution(-1, 5)
            def test_neg_b():
                with pytest.raises(ValueError): solution(5, -1)
        """),
        patches=[
            ("(a * b) // gcd(a, b)", "(a + b) // gcd(a, b)", "wrong_operator", "+ instead of * gives wrong LCM"),
            ("(a * b) // gcd(a, b)", "(a * b) % gcd(a, b)", "wrong_operator", "% instead of // gives wrong result"),
            ("if a == 0 or b == 0: return 0\n    ", "",
             "dropped_guard", "remove zero guard causes division by zero"),
            ("if a < 0 or b < 0: raise ValueError(\"arguments must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative check guard"),
        ],
    ),

    _TaskDef(
        task_id="full_is_prime",
        spec="Return True if n>=2 is prime (no divisor i with 2<=i and i*i<=n divides n). Return False for n<2. Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                if n < 2: return False
                i = 2
                while i * i <= n:
                    if n % i == 0:
                        return False
                    i += 1
                return True
        """),
        tests=_d("""
            import pytest
            def test_two():      assert solution(2) == True
            def test_three():    assert solution(3) == True
            def test_four():     assert solution(4) == False
            def test_prime_17(): assert solution(17) == True
            def test_one():      assert solution(1) == False
            def test_zero():     assert solution(0) == False
            def test_composite(): assert solution(9) == False
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("i * i <= n", "i * i < n", "wrong_operator", "< instead of <= misses perfect square composites like 4,9"),
            ("n % i == 0", "n % i != 0", "wrong_operator", "!= returns False for primes instead of composites"),
            ("if n < 2: return False\n    ", "",
             "dropped_guard", "remove n<2 guard makes 0 and 1 return True"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
        ],
    ),

    _TaskDef(
        task_id="full_digit_sum",
        spec="Return sum of decimal digits of non-negative integer n. Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                return sum(int(c) for c in str(n))
        """),
        tests=_d("""
            import pytest
            def test_basic():    assert solution(123) == 6
            def test_zero():     assert solution(0) == 0
            def test_hundred():  assert solution(100) == 1
            def test_nines():    assert solution(999) == 27
            def test_single():   assert solution(7) == 7
            def test_large():    assert solution(12345) == 15
            def test_neg():
                with pytest.raises(ValueError): solution(-5)
        """),
        patches=[
            ("sum(int(c) for c in str(n))", "sum(int(c) for c in str(n)) - 1", "wrong_operator", "subtracting 1 gives wrong sum"),
            ("sum(int(c) for c in str(n))", "len(str(n))", "wrong_operator", "len counts digits not their sum"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
            ("sum(int(c) for c in str(n))", "sum(int(c) for c in str(n + 1))", "wrong_operator", "n+1 shifts all results by at least 1"),
        ],
    ),

    _TaskDef(
        task_id="full_count_divisors",
        spec="Count positive divisors of n (including 1 and n). Raise ValueError if n<=0.",
        correct=_d("""
            def solution(n):
                if n <= 0: raise ValueError("n must be positive")
                count = 0
                i = 1
                while i * i <= n:
                    if n % i == 0:
                        if i * i == n:
                            count += 1
                        else:
                            count += 2
                    i += 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_one():      assert solution(1) == 1
            def test_six():      assert solution(6) == 4
            def test_twelve():   assert solution(12) == 6
            def test_prime():    assert solution(7) == 2
            def test_square():   assert solution(9) == 3
            def test_sixteen():  assert solution(16) == 5
            def test_zero():
                with pytest.raises(ValueError): solution(0)
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("i * i <= n", "i * i < n", "wrong_operator", "< instead of <= misses perfect square root divisor"),
            ("i * i == n", "i * i != n", "wrong_operator", "inverted check swaps single/double counting"),
            ("count += 2", "count += 1", "wrong_operator", "counts only one divisor instead of two for non-square pairs"),
            ("if n <= 0: raise ValueError(\"n must be positive\")\n    ", "",
             "dropped_guard", "remove guard allows n<=0"),
        ],
    ),

    _TaskDef(
        task_id="full_collatz_steps",
        spec="Count steps to reach 1 from n>0 via Collatz (even: n//2, odd: 3*n+1). collatz_steps(1)=0. Raise ValueError if n<=0.",
        correct=_d("""
            def solution(n):
                if n <= 0: raise ValueError("n must be positive")
                steps = 0
                while n != 1:
                    if n % 2 == 0:
                        n = n // 2
                    else:
                        n = 3 * n + 1
                    steps += 1
                return steps
        """),
        tests=_d("""
            import pytest
            def test_one():   assert solution(1) == 0
            def test_two():   assert solution(2) == 1
            def test_four():  assert solution(4) == 2
            def test_six():   assert solution(6) == 8
            def test_three(): assert solution(3) == 7
            def test_zero():
                with pytest.raises(ValueError): solution(0)
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("n % 2 == 0", "n % 2 != 0", "wrong_operator", "inverts even/odd check, applies wrong rule"),
            ("n // 2", "n // 3", "wrong_operator", "wrong divisor in even step"),
            ("3 * n + 1", "3 * n - 1", "wrong_operator", "wrong constant in odd step"),
            ("if n <= 0: raise ValueError(\"n must be positive\")\n    ", "",
             "dropped_guard", "remove guard allows n<=0"),
        ],
    ),

    _TaskDef(
        task_id="full_count_set_bits",
        spec="Count 1-bits (popcount) of non-negative integer n. Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                count = 0
                while n > 0:
                    count += n & 1
                    n >>= 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_zero():      assert solution(0) == 0
            def test_one():       assert solution(1) == 1
            def test_seven():     assert solution(7) == 3
            def test_eight():     assert solution(8) == 1
            def test_255():       assert solution(255) == 8
            def test_two():       assert solution(2) == 1
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("count += n & 1", "count += n | 1", "wrong_operator", "| instead of & always adds 1 per iteration"),
            ("n >>= 1", "n <<= 1", "wrong_operator", "left shift instead of right shift causes infinite loop"),
            ("while n > 0", "while n >= 0", "wrong_operator", ">= causes infinite loop on n==0"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
        ],
    ),

    _TaskDef(
        task_id="full_is_power_of_two",
        spec="Return True if n is a positive power of two (1,2,4,8,...). Return False for n=0. Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                return n > 0 and (n & (n - 1)) == 0
        """),
        tests=_d("""
            import pytest
            def test_one():    assert solution(1) == True
            def test_two():    assert solution(2) == True
            def test_four():   assert solution(4) == True
            def test_eight():  assert solution(8) == True
            def test_three():  assert solution(3) == False
            def test_six():    assert solution(6) == False
            def test_zero():   assert solution(0) == False
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("(n & (n - 1)) == 0", "(n & (n - 1)) != 0", "wrong_operator", "!= inverts the power-of-two bit trick"),
            ("n > 0 and", "n >= 0 and", "wrong_operator", ">= includes 0 making it return True for n=0"),
            ("n & (n - 1)", "n | (n - 1)", "wrong_operator", "| instead of & breaks the bit trick"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
        ],
    ),

    _TaskDef(
        task_id="full_bit_length",
        spec="Return minimum bits to represent n: bit_length(0)=1, bit_length(1)=1, bit_length(2)=2. Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                if n == 0: return 1
                count = 0
                while n > 0:
                    count += 1
                    n >>= 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_zero():   assert solution(0) == 1
            def test_one():    assert solution(1) == 1
            def test_two():    assert solution(2) == 2
            def test_seven():  assert solution(7) == 3
            def test_eight():  assert solution(8) == 4
            def test_large():  assert solution(255) == 8
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("n >>= 1", "n <<= 1", "wrong_operator", "left shift causes infinite loop"),
            ("while n > 0", "while n >= 0", "wrong_operator", ">= causes infinite loop"),
            ("if n == 0: return 1\n    ", "",
             "dropped_guard", "remove zero special case, returns 0 instead of 1 for n=0"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
        ],
    ),

    _TaskDef(
        task_id="full_xor_range",
        spec="Return XOR of all integers from 1 to n inclusive. xor_range(0)=0 (empty). Raise ValueError if n<0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                result = 0
                for i in range(1, n + 1):
                    result ^= i
                return result
        """),
        tests=_d("""
            import pytest
            def test_zero():  assert solution(0) == 0
            def test_one():   assert solution(1) == 1
            def test_two():   assert solution(2) == 3
            def test_three(): assert solution(3) == 0
            def test_four():  assert solution(4) == 4
            def test_six():   assert solution(6) == 7
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("result ^= i", "result |= i", "wrong_operator", "| instead of ^ gives wrong cumulative result"),
            ("range(1, n + 1)", "range(0, n + 1)", "off_by_one", "includes 0 in range, XOR with 0 is identity so no change for n>=1 but semantically wrong"),
            ("result ^= i", "result &= i", "wrong_operator", "& instead of ^ gives wrong result"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative guard"),
        ],
    ),

    _TaskDef(
        task_id="full_parity",
        spec="Return 1 if n has odd number of set bits, 0 if even. Raise ValueError if n < 0.",
        correct=_d("""
            def solution(n):
                if n < 0: raise ValueError("n must be non-negative")
                count = 0
                while n:
                    count += n & 1
                    n >>= 1
                return count % 2
        """),
        tests=_d("""
            import pytest
            def test_zero():    assert solution(0) == 0
            def test_one():     assert solution(1) == 1
            def test_two():     assert solution(2) == 1
            def test_three():   assert solution(3) == 0
            def test_seven():   assert solution(7) == 1
            def test_eight():   assert solution(8) == 1
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("count += n & 1", "count += n & 0", "wrong_operator", "& 0 always adds 0, count stays 0"),
            ("count % 2", "count % 3", "wrong_operator", "% 3 gives wrong parity for count==2"),
            ("n < 0", "n <= 0", "wrong_operator", "guard fires on n==0 which is valid"),
            ("if n < 0: raise ValueError(\"n must be non-negative\")\n    ", "",
             "dropped_guard", "remove negative input guard"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_transpose",
        spec="Transpose an N×M matrix (list of lists). Raise ValueError if matrix is empty or rows have different lengths.",
        correct=_d("""
            def solution(matrix):
                if not matrix: raise ValueError("matrix must not be empty")
                row_len = len(matrix[0])
                for row in matrix:
                    if len(row) != row_len: raise ValueError("all rows must have the same length")
                return [[matrix[r][c] for r in range(len(matrix))] for c in range(row_len)]
        """),
        tests=_d("""
            import pytest
            def test_2x3():
                m = [[1,2,3],[4,5,6]]
                assert solution(m) == [[1,4],[2,5],[3,6]]
            def test_1x1():
                assert solution([[7]]) == [[7]]
            def test_single_row():
                assert solution([[1,2,3]]) == [[1],[2],[3]]
            def test_empty():
                with pytest.raises(ValueError): solution([])
            def test_ragged():
                with pytest.raises(ValueError): solution([[1,2],[3]])
        """),
        patches=[
            ("for r in range(len(matrix))", "for r in range(len(matrix) - 1)", "off_by_one", "misses last row in transpose"),
            ("for c in range(row_len)", "for c in range(row_len - 1)", "off_by_one", "misses last column in transpose"),
            ("matrix[r][c]", "matrix[c][r]", "swapped_args", "swaps row and column indices"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_diagonal_sum",
        spec="Sum of main diagonal elements of an N×N square matrix. Raise ValueError if matrix is empty or not square.",
        correct=_d("""
            def solution(matrix):
                if not matrix: raise ValueError("matrix must not be empty")
                n = len(matrix)
                for row in matrix:
                    if len(row) != n: raise ValueError("matrix must be square")
                return sum(matrix[i][i] for i in range(n))
        """),
        tests=_d("""
            import pytest
            def test_2x2():
                assert solution([[1,2],[3,4]]) == 5
            def test_3x3():
                assert solution([[1,0,0],[0,2,0],[0,0,3]]) == 6
            def test_1x1():
                assert solution([[9]]) == 9
            def test_non_square():
                with pytest.raises(ValueError): solution([[1,2,3],[4,5,6]])
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("matrix[i][i]", "matrix[i][n-1-i]", "swapped_args", "sums anti-diagonal instead of main diagonal"),
            ("range(n)", "range(n - 1)", "off_by_one", "skips last diagonal element"),
            ("len(row) != n", "len(row) == n", "wrong_operator", "guard inverted, rejects square matrices"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_antidiagonal_sum",
        spec="Sum of anti-diagonal elements (top-right to bottom-left) of N×N matrix. Raise ValueError if not square or empty.",
        correct=_d("""
            def solution(matrix):
                if not matrix: raise ValueError("matrix must not be empty")
                n = len(matrix)
                for row in matrix:
                    if len(row) != n: raise ValueError("matrix must be square")
                return sum(matrix[i][n - 1 - i] for i in range(n))
        """),
        tests=_d("""
            import pytest
            def test_2x2():
                assert solution([[1,2],[3,4]]) == 5
            def test_3x3():
                assert solution([[1,2,3],[4,5,6],[7,8,9]]) == 15
            def test_1x1():
                assert solution([[5]]) == 5
            def test_non_square():
                with pytest.raises(ValueError): solution([[1,2],[3,4],[5,6]])
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("matrix[i][n - 1 - i]", "matrix[i][i]", "swapped_args", "sums main diagonal instead of anti-diagonal"),
            ("n - 1 - i", "n - i", "off_by_one", "index out of bounds or wrong column"),
            ("len(row) != n", "len(row) != n + 1", "wrong_operator", "wrong square check threshold"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_row_max_sum",
        spec="Return index of row with maximum sum. Raise ValueError if matrix is empty.",
        correct=_d("""
            def solution(matrix):
                if not matrix: raise ValueError("matrix must not be empty")
                return max(range(len(matrix)), key=lambda r: sum(matrix[r]))
        """),
        tests=_d("""
            import pytest
            def test_clear_winner():
                assert solution([[1,2],[10,20],[3,4]]) == 1
            def test_single_row():
                assert solution([[5,6,7]]) == 0
            def test_negative_rows():
                assert solution([[-1,-2],[-3,-4]]) == 0
            def test_first_wins():
                assert solution([[9,9],[1,2],[3,4]]) == 0
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("max(range(len(matrix))", "min(range(len(matrix))", "wrong_operator", "returns row with minimum sum"),
            ("sum(matrix[r])", "sum(matrix[r]) * -1", "wrong_operator", "negates sums, inverting order"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
            ("key=lambda r: sum(matrix[r])", "key=lambda r: len(matrix[r])", "swapped_args", "compares row length instead of row sum"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_scalar_multiply",
        spec="Return new matrix with all elements multiplied by scalar k. Raise ValueError if matrix is empty.",
        correct=_d("""
            def solution(matrix, k):
                if not matrix: raise ValueError("matrix must not be empty")
                return [[k * matrix[r][c] for c in range(len(matrix[r]))] for r in range(len(matrix))]
        """),
        tests=_d("""
            import pytest
            def test_multiply_by_2():
                assert solution([[1,2],[3,4]], 2) == [[2,4],[6,8]]
            def test_multiply_by_0():
                assert solution([[1,2],[3,4]], 0) == [[0,0],[0,0]]
            def test_multiply_by_neg1():
                assert solution([[1,-2],[3,-4]], -1) == [[-1,2],[-3,4]]
            def test_single_element():
                assert solution([[5]], 3) == [[15]]
            def test_empty():
                with pytest.raises(ValueError): solution([], 2)
        """),
        patches=[
            ("k * matrix[r][c]", "k + matrix[r][c]", "wrong_operator", "adds instead of multiplies"),
            ("for c in range(len(matrix[r]))", "for c in range(len(matrix[r]) - 1)", "off_by_one", "skips last column"),
            ("for r in range(len(matrix))", "for r in range(len(matrix) - 1)", "off_by_one", "skips last row"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
        ],
    ),

    _TaskDef(
        task_id="full_matrix_all_positive",
        spec="Return True if ALL elements > 0. Raise ValueError if matrix is empty.",
        correct=_d("""
            def solution(matrix):
                if not matrix: raise ValueError("matrix must not be empty")
                return all(matrix[r][c] > 0 for r in range(len(matrix)) for c in range(len(matrix[r])))
        """),
        tests=_d("""
            import pytest
            def test_all_positive():
                assert solution([[1,2],[3,4]]) == True
            def test_one_zero():
                assert solution([[1,2],[0,4]]) == False
            def test_one_negative():
                assert solution([[1,-2],[3,4]]) == False
            def test_single_pos():
                assert solution([[7]]) == True
            def test_single_zero():
                assert solution([[0]]) == False
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("matrix[r][c] > 0", "matrix[r][c] >= 0", "wrong_operator", ">= 0 allows zero elements"),
            ("all(", "any(", "wrong_operator", "any() returns True if at least one positive"),
            ("matrix[r][c] > 0", "matrix[r][c] < 0", "wrong_operator", "checks for negative instead of positive"),
            ("if not matrix: raise ValueError(\"matrix must not be empty\")\n    ", "",
             "dropped_guard", "remove empty matrix guard"),
        ],
    ),

    _TaskDef(
        task_id="full_is_balanced_brackets",
        spec="Return True if s has balanced brackets using only ()[]{}. Raise ValueError if s is empty.",
        correct=_d("""
            def solution(s):
                if not s: raise ValueError("s must not be empty")
                matching = {')': '(', ']': '[', '}': '{'}
                stack = []
                for ch in s:
                    if ch in '([{':
                        stack.append(ch)
                    elif ch in ')]}':
                        if not stack or stack[-1] != matching[ch]:
                            return False
                        stack.pop()
                return len(stack) == 0
        """),
        tests=_d("""
            import pytest
            def test_nested():       assert solution("(())") == True
            def test_mixed():        assert solution("()[]{}") == True
            def test_wrong_close():  assert solution("(]") == False
            def test_interleaved():  assert solution("{[}]") == False
            def test_unclosed():     assert solution("((") == False
            def test_empty():
                with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("stack[-1] != matching[ch]", "stack[-1] == matching[ch]", "wrong_operator", "inverted mismatch check"),
            ("len(stack) == 0", "len(stack) != 0", "wrong_operator", "returns True when stack non-empty"),
            ("if not stack or stack[-1] != matching[ch]:", "if stack and stack[-1] != matching[ch]:", "wrong_operator", "removes empty-stack check"),
            ("if not s: raise ValueError(\"s must not be empty\")\n    ", "",
             "dropped_guard", "remove empty string guard"),
        ],
    ),

    _TaskDef(
        task_id="full_evaluate_postfix",
        spec="Evaluate postfix expression. Input: list of tokens (strings), operators: + - * //. Raise ValueError if stack has != 1 element at end, or division by zero.",
        correct=_d("""
            def solution(tokens):
                stack = []
                for token in tokens:
                    if token in ('+', '-', '*', '//'):
                        if len(stack) < 2: raise ValueError("not enough operands")
                        b = stack.pop()
                        a = stack.pop()
                        if token == '+':
                            stack.append(a + b)
                        elif token == '-':
                            stack.append(a - b)
                        elif token == '*':
                            stack.append(a * b)
                        elif token == '//':
                            if b == 0: raise ValueError("division by zero")
                            stack.append(a // b)
                    else:
                        stack.append(int(token))
                if len(stack) != 1: raise ValueError("invalid expression")
                return stack[0]
        """),
        tests=_d("""
            import pytest
            def test_add():
                assert solution(["3","4","+"]) == 7
            def test_complex():
                assert solution(["5","1","2","+","4","*","+","3","-"]) == 14
            def test_floordiv():
                assert solution(["6","2","//"]) == 3
            def test_div_zero():
                with pytest.raises(ValueError): solution(["5","0","//"])
            def test_invalid_expr():
                with pytest.raises(ValueError): solution(["1","2"])
        """),
        patches=[
            ("a + b", "b + a", "wrong_operator", "addition is commutative so swap a and b with subtraction"),
            ("a - b", "b - a", "wrong_operator", "subtraction order reversed gives wrong result"),
            ("a // b", "b // a", "swapped_args", "division operands swapped"),
            ("b == 0", "b != 0", "wrong_operator", "division by zero guard inverted"),
        ],
    ),

    _TaskDef(
        task_id="full_next_greater_element",
        spec="For each element in lst, return the next greater element to its right, or -1 if none. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("lst must not be empty")
                n = len(lst)
                result = [-1] * n
                stack = []
                for i in range(n):
                    while stack and lst[stack[-1]] < lst[i]:
                        result[stack.pop()] = lst[i]
                    stack.append(i)
                return result
        """),
        tests=_d("""
            import pytest
            def test_typical():
                assert solution([4,5,2,25]) == [5,25,25,-1]
            def test_decreasing():
                assert solution([13,7,6,12]) == [-1,12,12,-1]
            def test_single():
                assert solution([1]) == [-1]
            def test_empty():
                with pytest.raises(ValueError): solution([])
            def test_all_same():
                assert solution([3,3,3]) == [-1,-1,-1]
        """),
        patches=[
            ("lst[stack[-1]] < lst[i]", "lst[stack[-1]] > lst[i]", "wrong_operator", "finds next smaller instead of next greater"),
            ("lst[stack[-1]] < lst[i]", "lst[stack[-1]] <= lst[i]", "wrong_operator", "includes equal elements as next greater"),
            ("result = [-1] * n", "result = [0] * n", "wrong_operator", "default value 0 instead of -1"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n    ", "",
             "dropped_guard", "remove empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_stack_minimum",
        spec="Return the minimum element of lst. Raise ValueError if empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                result = lst[0]
                for x in lst[1:]:
                    if x < result:
                        result = x
                return result
        """),
        tests=_d("""
            import pytest
            def test_mixed():    assert solution([3,1,4,1,5]) == 1
            def test_single():   assert solution([5]) == 5
            def test_negative():  assert solution([-3,-1,-2]) == -3
            def test_empty():
                with pytest.raises(ValueError): solution([])
            def test_all_same(): assert solution([2,2,2]) == 2
        """),
        patches=[
            ("if x < result:", "if x > result:",
             "wrong_operator", "returns max instead of min"),
            ("result = lst[0]", "result = lst[-1]",
             "off_by_one", "seed with last element instead of first"),
            ("for x in lst[1:]:", "for x in lst:",
             "off_by_one", "re-processes lst[0] including seed element"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty-list guard"),
        ],
    ),
    _TaskDef(
        task_id="full_decode_rle",
        spec="Decode a run-length encoded list of (count, value) pairs into a flat list. Raise ValueError if list is empty or any count <= 0.",
        correct=_d("""
            def solution(pairs):
                if not pairs: raise ValueError("pairs must not be empty")
                result = []
                for count, val in pairs:
                    if count <= 0: raise ValueError("count must be positive")
                    result.extend([val] * count)
                return result
        """),
        tests=_d("""
            import pytest
            def test_basic():     assert solution([(3,'a'),(2,'b')]) == ['a','a','a','b','b']
            def test_single():    assert solution([(1,5)]) == [5]
            def test_empty():
                with pytest.raises(ValueError): solution([])
            def test_zero_count():
                with pytest.raises(ValueError): solution([(0,'x')])
            def test_negative_count():
                with pytest.raises(ValueError): solution([(-1,'x')])
            def test_multi_count(): assert solution([(4,7)]) == [7,7,7,7]
        """),
        patches=[
            ("if count <= 0:", "if count == 0:",
             "wrong_operator", "allows negative counts through"),
            ("result.extend([val] * count)", "result.append(val)",
             "wrong_operator", "ignores count, appends only once"),
            ("result.extend([val] * count)", "result.extend([val] * (count - 1))",
             "off_by_one", "extends one fewer element than required"),
            ("if not pairs: raise ValueError(\"pairs must not be empty\")\n                ", "",
             "dropped_guard", "remove empty-pairs guard"),
        ],
    ),
    _TaskDef(
        task_id="full_tree_height",
        spec="Return height of binary tree stored as list (None=absent). Index i has children 2i+1 and 2i+2. Height = max depth from root. Empty tree or None root has height 0. Single node has height 1. Raise ValueError if tree is None.",
        correct=_d("""
            def solution(tree):
                if tree is None: raise ValueError("tree must not be None")
                if not tree or tree[0] is None: return 0
                n = len(tree)
                def h(i):
                    if i >= n or tree[i] is None: return 0
                    return 1 + max(h(2*i+1), h(2*i+2))
                return h(0)
        """),
        tests=_d("""
            import pytest
            def test_full():     assert solution([1,2,3,4,5]) == 3
            def test_single():   assert solution([1]) == 1
            def test_right():    assert solution([1,None,3]) == 2
            def test_empty():    assert solution([]) == 0
            def test_none_root(): assert solution([None]) == 0
            def test_none_tree():
                with pytest.raises(ValueError): solution(None)
        """),
        patches=[
            ("1 + max(h(2*i+1), h(2*i+2))", "1 + min(h(2*i+1), h(2*i+2))",
             "wrong_operator", "returns minimum-height path instead of maximum"),
            ("h(2*i+1)", "h(2*i)",
             "off_by_one", "left child index is wrong, skips a level"),
            ("h(2*i+2)", "h(2*i+1)",
             "off_by_one", "both children point to left child"),
            ("if tree is None: raise ValueError(\"tree must not be None\")\n                ", "",
             "dropped_guard", "remove None-tree guard"),
        ],
    ),
    _TaskDef(
        task_id="full_count_leaves",
        spec="Count leaf nodes in binary tree (array representation, index i has children 2i+1 and 2i+2). Raise ValueError if tree is None. Empty tree has 0 leaves.",
        correct=_d("""
            def solution(tree):
                if tree is None: raise ValueError("tree must not be None")
                if not tree or tree[0] is None: return 0
                n = len(tree)
                count = 0
                def visit(i):
                    nonlocal count
                    if i >= n or tree[i] is None: return
                    left = 2*i+1
                    right = 2*i+2
                    if (left >= n or tree[left] is None) and (right >= n or tree[right] is None):
                        count += 1
                    else:
                        visit(left)
                        visit(right)
                visit(0)
                return count
        """),
        tests=_d("""
            import pytest
            def test_two_leaves():   assert solution([1,2,3]) == 2
            def test_single():       assert solution([1]) == 1
            def test_three_leaves(): assert solution([1,2,3,4,5]) == 3
            def test_empty():        assert solution([]) == 0
            def test_none_tree():
                with pytest.raises(ValueError): solution(None)
        """),
        patches=[
            ("count += 1", "count += 2",
             "wrong_operator", "double-counts each leaf"),
            ("left = 2*i+1", "left = 2*i",
             "off_by_one", "left child index is off by one"),
            ("if tree is None: raise ValueError(\"tree must not be None\")\n                ", "",
             "dropped_guard", "remove None-tree guard"),
            ("(left >= n or tree[left] is None) and (right >= n or tree[right] is None)",
             "(left >= n or tree[left] is None) or (right >= n or tree[right] is None)",
             "wrong_operator", "counts nodes with any absent child as leaf"),
        ],
    ),
    _TaskDef(
        task_id="full_bfs_level_count",
        spec="Count nodes at level k (0-indexed, root=level 0) in a binary tree (array representation). Raise ValueError if tree is None or k<0.",
        correct=_d("""
            def solution(tree, k):
                if tree is None: raise ValueError("tree must not be None")
                if k < 0: raise ValueError("k must be non-negative")
                if not tree or tree[0] is None: return 0
                n = len(tree)
                from collections import deque
                queue = deque([(0, 0)])
                count = 0
                while queue:
                    idx, level = queue.popleft()
                    if idx >= n or tree[idx] is None: continue
                    if level == k:
                        count += 1
                    elif level < k:
                        queue.append((2*idx+1, level+1))
                        queue.append((2*idx+2, level+1))
                return count
        """),
        tests=_d("""
            import pytest
            def test_level0():  assert solution([1,2,3,4,5,6,7], 0) == 1
            def test_level1():  assert solution([1,2,3,4,5,6,7], 1) == 2
            def test_level2():  assert solution([1,2,3,4,5,6,7], 2) == 4
            def test_level3():  assert solution([1,2,3,4,5,6,7], 3) == 0
            def test_empty():   assert solution([], 0) == 0
            def test_neg_k():
                with pytest.raises(ValueError): solution([1], -1)
            def test_none_tree():
                with pytest.raises(ValueError): solution(None, 0)
        """),
        patches=[
            ("if level == k:", "if level != k:",
             "wrong_operator", "counts nodes not at level k"),
            ("elif level < k:", "elif level <= k:",
             "wrong_operator", "also pushes children for nodes at level k"),
            ("if k < 0: raise ValueError(\"k must be non-negative\")\n                ", "",
             "dropped_guard", "remove negative-k guard"),
            ("queue.append((2*idx+1, level+1))", "queue.append((2*idx, level+1))",
             "off_by_one", "left child index is off by one"),
        ],
    ),
    _TaskDef(
        task_id="full_dfs_reachable_count",
        spec="Count nodes reachable from source in undirected graph (adjacency list as dict). Raise ValueError if source not in graph.",
        correct=_d("""
            def solution(graph, source):
                if source not in graph: raise ValueError("source not in graph")
                visited = set()
                stack = [source]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        for nb in graph[node]:
                            if nb not in visited:
                                stack.append(nb)
                return len(visited)
        """),
        tests=_d("""
            import pytest
            def test_connected():
                g = {0:[1,2],1:[0],2:[0,3],3:[2]}
                assert solution(g, 0) == 4
            def test_isolated():
                g = {0:[],1:[2],2:[1]}
                assert solution(g, 0) == 1
            def test_source_missing():
                with pytest.raises(ValueError): solution({0:[1],1:[0]}, 99)
            def test_single_node():
                assert solution({5:[]}, 5) == 1
        """),
        patches=[
            ("if node not in visited:", "if node in visited:",
             "wrong_operator", "skips unvisited nodes, never explores"),
            ("if nb not in visited:", "if nb in visited:",
             "wrong_operator", "only appends already-visited neighbours"),
            ("return len(visited)", "return len(visited) - 1",
             "off_by_one", "excludes source from count"),
            ("if source not in graph: raise ValueError(\"source not in graph\")\n                ", "",
             "dropped_guard", "remove missing-source guard"),
        ],
    ),
    _TaskDef(
        task_id="full_has_cycle_undirected",
        spec="Return True if undirected graph (adjacency list as dict, nodes are ints) contains a cycle. Return False if empty graph.",
        correct=_d("""
            def solution(graph):
                if not graph: return False
                visited = set()
                def dfs(node, parent):
                    visited.add(node)
                    for nb in graph.get(node, []):
                        if nb not in visited:
                            if dfs(nb, node): return True
                        elif nb != parent:
                            return True
                    return False
                for node in graph:
                    if node not in visited:
                        if dfs(node, -1): return True
                return False
        """),
        tests=_d("""
            def test_triangle():
                g = {0:[1,2],1:[0,2],2:[0,1]}
                assert solution(g) == True
            def test_tree():
                g = {0:[1,2],1:[0],2:[0]}
                assert solution(g) == False
            def test_single():
                assert solution({0:[]}) == False
            def test_empty():
                assert solution({}) == False
            def test_chain_no_cycle():
                g = {0:[1],1:[0,2],2:[1]}
                assert solution(g) == False
        """),
        patches=[
            ("if nb not in visited:", "if nb in visited:",
             "wrong_operator", "inverts visit check, logic broken"),
            ("elif nb != parent:", "elif nb == parent:",
             "wrong_operator", "wrong cycle detection condition"),
            ("if node not in visited:", "if node in visited:",
             "wrong_operator", "skips unvisited components"),
            ("            if nb not in visited:\n                if dfs(nb, node): return True\n            elif nb != parent:\n                return True\n",
             "            if nb not in visited:\n                if dfs(nb, node): return True\n",
             "dropped_guard", "remove parent-check cycle detection"),
        ],
    ),
    _TaskDef(
        task_id="full_insertion_sort",
        spec="Sort list in ascending order using insertion sort, return new list. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("list must not be empty")
                arr = list(lst)
                for i in range(1, len(arr)):
                    key = arr[i]
                    j = i - 1
                    while j >= 0 and arr[j] > key:
                        arr[j + 1] = arr[j]
                        j -= 1
                    arr[j + 1] = key
                return arr
        """),
        tests=_d("""
            import pytest
            def test_mixed():    assert solution([3,1,4,1,5]) == [1,1,3,4,5]
            def test_single():   assert solution([1]) == [1]
            def test_reverse():  assert solution([5,4,3,2,1]) == [1,2,3,4,5]
            def test_sorted():   assert solution([1,2,3]) == [1,2,3]
            def test_empty():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("arr[j] > key", "arr[j] < key",
             "wrong_operator", "sorts in descending order"),
            ("while j >= 0 and arr[j] > key:", "while j > 0 and arr[j] > key:",
             "wrong_operator", "skips comparison at index 0"),
            ("arr[j + 1] = arr[j]", "arr[j] = arr[j + 1]",
             "swapped_args", "shifts in wrong direction, corrupts array"),
            ("for i in range(1, len(arr)):", "for i in range(len(arr)):",
             "off_by_one", "starts outer loop at 0, compares element with itself"),
            ("if not lst: raise ValueError(\"list must not be empty\")\n                ", "",
             "dropped_guard", "remove empty-list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_count_inversions",
        spec="Count inversions in lst: pairs (i,j) with i<j and lst[i]>lst[j]. Raise ValueError if fewer than 2 elements.",
        correct=_d("""
            def solution(lst):
                if len(lst) < 2: raise ValueError("need at least 2 elements")
                count = 0
                for i in range(len(lst) - 1):
                    for j in range(i + 1, len(lst)):
                        if lst[i] > lst[j]:
                            count += 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_two_inversions(): assert solution([3,1,2]) == 2
            def test_no_inversions(): assert solution([1,2,3]) == 0
            def test_all_inversions(): assert solution([3,2,1]) == 3
            def test_mixed(): assert solution([1,3,2,3,1]) == 4
            def test_too_short_raises():
                with pytest.raises(ValueError): solution([1])
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("lst[i] > lst[j]", "lst[i] >= lst[j]",
             "wrong_operator", "counts equal pairs as inversions too"),
            ("lst[i] > lst[j]", "lst[i] < lst[j]",
             "wrong_operator", "counts non-inversions instead of inversions"),
            ("range(len(lst) - 1)", "range(len(lst))",
             "off_by_one", "i can equal len(lst)-1, j range becomes empty but wastes iteration; more critically i==j-1 edge accessed"),
            ("range(i + 1, len(lst))", "range(i, len(lst))",
             "off_by_one", "includes i==j pairs, comparing element to itself"),
            ("if len(lst) < 2: raise ValueError(\"need at least 2 elements\")\n            ", "",
             "dropped_guard", "removes the too-short guard"),
        ],
    ),

    _TaskDef(
        task_id="full_dutch_flag_partition",
        spec="Partition lst into three lists: elements < pivot, == pivot, > pivot. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst, pivot):
                if not lst: raise ValueError("lst must not be empty")
                less = [x for x in lst if x < pivot]
                equal = [x for x in lst if x == pivot]
                greater = [x for x in lst if x > pivot]
                return (less, equal, greater)
        """),
        tests=_d("""
            import pytest
            def test_mixed():
                l, e, g = solution([1,4,2,3,4,5], 3)
                assert l == [1,2] and e == [3] and g == [4,4,5]
            def test_all_equal():
                l, e, g = solution([2,2,2], 2)
                assert l == [] and e == [2,2,2] and g == []
            def test_all_less():
                l, e, g = solution([1,2,3], 5)
                assert l == [1,2,3] and e == [] and g == []
            def test_all_greater():
                l, e, g = solution([4,5,6], 3)
                assert l == [] and e == [] and g == [4,5,6]
            def test_empty_raises():
                with pytest.raises(ValueError): solution([], 1)
        """),
        patches=[
            ("x < pivot", "x <= pivot",
             "wrong_operator", "equal elements go to less instead of equal"),
            ("x > pivot", "x >= pivot",
             "wrong_operator", "equal elements go to greater instead of equal"),
            ("x == pivot", "x != pivot",
             "wrong_operator", "equal partition gets non-equal elements"),
            ("[x for x in lst if x < pivot]", "[x for x in lst if x > pivot]",
             "wrong_operator", "less and greater conditions swapped"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_kth_smallest",
        spec="Return k-th smallest element (1-indexed) in lst. Raise ValueError if lst is empty or k out of range.",
        correct=_d("""
            def solution(lst, k):
                if not lst: raise ValueError("lst must not be empty")
                if k < 1 or k > len(lst): raise ValueError("k out of range")
                return sorted(lst)[k - 1]
        """),
        tests=_d("""
            import pytest
            def test_first(): assert solution([3,1,4,1,5], 1) == 1
            def test_middle(): assert solution([3,1,4,1,5], 3) == 3
            def test_last(): assert solution([3,1,4,1,5], 5) == 5
            def test_k_zero_raises():
                with pytest.raises(ValueError): solution([1,2,3], 0)
            def test_k_too_large_raises():
                with pytest.raises(ValueError): solution([1,2,3], 4)
            def test_empty_raises():
                with pytest.raises(ValueError): solution([], 1)
            def test_single(): assert solution([7], 1) == 7
        """),
        patches=[
            ("sorted(lst)[k - 1]", "sorted(lst)[k]",
             "off_by_one", "returns k+1-th element instead of k-th (0-indexed offset)"),
            ("sorted(lst)[k - 1]", "sorted(lst, reverse=True)[k - 1]",
             "wrong_operator", "returns k-th largest instead of k-th smallest"),
            ("k < 1", "k < 0",
             "wrong_operator", "allows k=0 which would be invalid 0-indexed access"),
            ("k > len(lst)", "k >= len(lst)",
             "wrong_operator", "rejects k=len(lst) which is a valid last element"),
        ],
    ),

    _TaskDef(
        task_id="full_is_sorted_asc",
        spec="Return True if lst is non-strictly ascending (each element >= previous). Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("lst must not be empty")
                for i in range(1, len(lst)):
                    if lst[i] < lst[i - 1]:
                        return False
                return True
        """),
        tests=_d("""
            import pytest
            def test_strictly_asc(): assert solution([1,2,3]) == True
            def test_non_strict_asc(): assert solution([1,1,2]) == True
            def test_desc(): assert solution([3,2,1]) == False
            def test_single(): assert solution([1]) == True
            def test_dip(): assert solution([2,1,3]) == False
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("lst[i] < lst[i - 1]", "lst[i] <= lst[i - 1]",
             "wrong_operator", "rejects equal adjacent elements, so [1,1,2] returns False incorrectly"),
            ("lst[i] < lst[i - 1]", "lst[i] > lst[i - 1]",
             "wrong_operator", "checks descending condition instead of ascending violation"),
            ("range(1, len(lst))", "range(len(lst))",
             "off_by_one", "starts at i=0, accesses lst[-1] which wraps around"),
            ("lst[i] < lst[i - 1]", "lst[i - 1] < lst[i]",
             "swapped_args", "returns False when ascending (correct) instead of when descending"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_mode",
        spec="Return the most frequent element in lst (any one if tie). Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("lst must not be empty")
                counts = {}
                for x in lst:
                    counts[x] = counts.get(x, 0) + 1
                best = None
                best_count = 0
                for x, c in counts.items():
                    if c > best_count:
                        best_count = c
                        best = x
                return best
        """),
        tests=_d("""
            import pytest
            def test_clear_mode(): assert solution([1,2,2,3]) == 2
            def test_single(): assert solution([1]) == 1
            def test_first_encountered_wins():
                result = solution([3,3,2,2,1])
                assert result == 3
            def test_all_same(): assert solution([5,5,5]) == 5
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("c > best_count", "c < best_count",
             "wrong_operator", "returns least frequent element instead of most frequent"),
            ("counts.get(x, 0) + 1", "counts.get(x, 0) - 1",
             "wrong_operator", "decrements count instead of incrementing"),
            ("c > best_count", "c >= best_count",
             "wrong_operator", "tie-breaking changes: returns last max element instead of first, so [3,3,2,2,1] returns 2 not 3"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_all_unique",
        spec="Return True if all elements in lst are distinct. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("lst must not be empty")
                return len(lst) == len(set(lst))
        """),
        tests=_d("""
            import pytest
            def test_all_unique(): assert solution([1,2,3]) == True
            def test_has_duplicate(): assert solution([1,2,1]) == False
            def test_single(): assert solution([1]) == True
            def test_all_same(): assert solution([2,2,2]) == False
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("len(lst) == len(set(lst))", "len(lst) != len(set(lst))",
             "wrong_operator", "returns True when duplicates exist and False when all unique"),
            ("len(lst) == len(set(lst))", "len(lst) - 1 == len(set(lst))",
             "wrong_operator", "off by one: single-element list returns False"),
            ("len(lst) == len(set(lst))", "len(lst) == len(set(lst)) + 1",
             "wrong_operator", "always returns False since len(lst) can never equal len(set)+1 when all unique"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_intersection_count",
        spec="Count elements common to both lists a and b (set intersection, no multiplicity). Raise ValueError if either is empty.",
        correct=_d("""
            def solution(a, b):
                if not a or not b: raise ValueError("lists must not be empty")
                return len(set(a) & set(b))
        """),
        tests=_d("""
            import pytest
            def test_two_common(): assert solution([1,2,3], [2,3,4]) == 2
            def test_no_common(): assert solution([1,2], [3,4]) == 0
            def test_with_duplicates(): assert solution([1,1,2], [1,2,2]) == 2
            def test_all_common(): assert solution([1,2], [1,2]) == 2
            def test_empty_a_raises():
                with pytest.raises(ValueError): solution([], [1,2])
            def test_empty_b_raises():
                with pytest.raises(ValueError): solution([1,2], [])
        """),
        patches=[
            ("set(a) & set(b)", "set(a) | set(b)",
             "wrong_operator", "returns union count instead of intersection count"),
            ("set(a) & set(b)", "set(a) - set(b)",
             "wrong_operator", "returns set difference count instead of intersection count"),
            ("not a or not b", "not a and not b",
             "wrong_operator", "only raises if BOTH lists are empty, not just one"),
            ("if not a or not b: raise ValueError(\"lists must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_majority_element",
        spec="Return element appearing > n//2 times, or None if none. Raise ValueError if lst is empty.",
        correct=_d("""
            def solution(lst):
                if not lst: raise ValueError("lst must not be empty")
                n = len(lst)
                counts = {}
                for x in lst:
                    counts[x] = counts.get(x, 0) + 1
                for x, c in counts.items():
                    if c > n // 2:
                        return x
                return None
        """),
        tests=_d("""
            import pytest
            def test_has_majority(): assert solution([3,3,4,2,3]) == 3
            def test_no_majority(): assert solution([1,2,3,4]) is None
            def test_single(): assert solution([1]) == 1
            def test_even_no_majority(): assert solution([1,2,1,2]) is None
            def test_three_same(): assert solution([2,2,2]) == 2
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("c > n // 2", "c >= n // 2",
             "wrong_operator", "[2,2] has n=2, c=2>=1 returns 2 incorrectly instead of None"),
            ("c > n // 2", "c > n",
             "wrong_operator", "c can never exceed n so always returns None"),
            ("counts.get(x, 0) + 1", "counts.get(x, 0) + 2",
             "wrong_operator", "double-increments count, inflating all frequencies"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_count_anagram_pairs",
        spec="Count pairs (i,j) i<j in words list where words[i] and words[j] are anagrams. Raise ValueError if fewer than 2 words.",
        correct=_d("""
            def solution(words):
                if len(words) < 2: raise ValueError("need at least 2 words")
                count = 0
                for i in range(len(words) - 1):
                    for j in range(i + 1, len(words)):
                        if sorted(words[i]) == sorted(words[j]):
                            count += 1
                return count
        """),
        tests=_d("""
            import pytest
            def test_multiple_pairs():
                assert solution(["eat","tea","tan","ate","nat","bat"]) == 4
            def test_one_pair(): assert solution(["ab","ba"]) == 1
            def test_no_pairs(): assert solution(["abc","def"]) == 0
            def test_too_few_raises():
                with pytest.raises(ValueError): solution(["only"])
            def test_empty_raises():
                with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("sorted(words[i]) == sorted(words[j])", "sorted(words[i]) != sorted(words[j])",
             "wrong_operator", "counts non-anagram pairs instead of anagram pairs"),
            ("range(len(words) - 1)", "range(len(words))",
             "off_by_one", "outer loop goes to last index, inner range(i+1,len) is empty but allows i==len-1"),
            ("range(i + 1, len(words))", "range(i, len(words))",
             "off_by_one", "includes i==j case, comparing word to itself"),
            ("count += 1", "count += 2",
             "wrong_operator", "double-counts each anagram pair"),
            ("if len(words) < 2: raise ValueError(\"need at least 2 words\")\n            ", "",
             "dropped_guard", "removes the too-few words guard"),
        ],
    ),

    _TaskDef(
        task_id="full_count_substring_occurrences",
        spec="Count non-overlapping occurrences of pattern in s. Raise ValueError if pattern is empty.",
        correct=_d("""
            def solution(s, pattern):
                if not pattern: raise ValueError("pattern must not be empty")
                if not s: return 0
                count = 0
                start = 0
                while start <= len(s) - len(pattern):
                    idx = s.find(pattern, start)
                    if idx == -1: break
                    count += 1
                    start = idx + len(pattern)
                return count
        """),
        tests=_d("""
            import pytest
            def test_multiple(): assert solution("aababab", "ab") == 3
            def test_non_overlapping(): assert solution("aaaa", "aa") == 2
            def test_not_found(): assert solution("hello", "xyz") == 0
            def test_empty_string(): assert solution("", "x") == 0
            def test_empty_pattern_raises():
                with pytest.raises(ValueError): solution("hello", "")
            def test_single_char(): assert solution("aaaa", "a") == 4
            def test_full_match(): assert solution("ab", "ab") == 1
        """),
        patches=[
            ("start = idx + len(pattern)", "start = idx + 1",
             "wrong_operator", "allows overlapping matches, so 'aaaa'/'aa' returns 3 instead of 2"),
            ("start <= len(s) - len(pattern)", "start < len(s) - len(pattern)",
             "wrong_operator", "misses the last valid starting position"),
            ("count += 1", "count += 2",
             "wrong_operator", "double-counts every occurrence"),
            ("if not pattern: raise ValueError(\"pattern must not be empty\")\n            ", "",
             "dropped_guard", "removes the empty pattern guard"),
            ("idx == -1", "idx != -1",
             "wrong_operator", "breaks when pattern is found instead of when not found"),
        ],
    ),

    _TaskDef(
        task_id="full_is_subsequence",
        spec="Return True if s is a subsequence of t. Raise ValueError if either s or t is empty.",
        correct=_d("""
                def solution(s, t):
                    if not s or not t: raise ValueError("s and t must not be empty")
                    i = 0
                    for c in t:
                        if i < len(s) and c == s[i]:
                            i += 1
                    return i == len(s)
        """),
        tests=_d("""
                import pytest
                def test_true():       assert solution("ace", "abcde") == True
                def test_false():      assert solution("aec", "abcde") == False
                def test_exact():      assert solution("abc", "abc") == True
                def test_single():     assert solution("b", "abc") == True
                def test_missing():    assert solution("z", "abc") == False
                def test_empty_s():
                    with pytest.raises(ValueError): solution("", "abc")
                def test_empty_t():
                    with pytest.raises(ValueError): solution("abc", "")
        """),
        patches=[
            ("c == s[i]", "c != s[i]", "wrong_operator", "!= advances index when chars differ instead of match"),
            ("i == len(s)", "i != len(s)", "wrong_operator", "inverts final check, returns True when not all matched"),
            ("i += 1", "i += 2", "off_by_one", "skips every other character in s"),
            ("c == s[i]", "c == s[i - 1]", "off_by_one", "compares against previous index in s"),
            ("if not s or not t: raise ValueError(\"s and t must not be empty\")\n                ", "",
             "dropped_guard", "remove empty string guard"),
        ],
    ),

    _TaskDef(
        task_id="full_caesar_cipher",
        spec="Shift each letter in s by k positions (mod 26), preserving case and non-letters. Raise ValueError if k is negative.",
        correct=_d("""
                def solution(s, k):
                    if k < 0: raise ValueError("k must be non-negative")
                    k = k % 26
                    result = []
                    for c in s:
                        if c.isalpha():
                            base = ord('A') if c.isupper() else ord('a')
                            result.append(chr((ord(c) - base + k) % 26 + base))
                        else:
                            result.append(c)
                    return ''.join(result)
        """),
        tests=_d("""
                import pytest
                def test_shift_1():     assert solution("abc", 1) == "bcd"
                def test_wrap():        assert solution("xyz", 3) == "abc"
                def test_mixed():       assert solution("Hello!", 2) == "Jgnnq!"
                def test_full_cycle():  assert solution("ABC", 26) == "ABC"
                def test_zero():        assert solution("abc", 0) == "abc"
                def test_negative():
                    with pytest.raises(ValueError): solution("abc", -1)
        """),
        patches=[
            ("ord(c) - base + k", "ord(c) - base - k", "wrong_operator", "shifts backwards instead of forwards"),
            ("% 26 + base", "% 26 - base", "wrong_operator", "subtracts base instead of adding, giving wrong ordinals"),
            ("ord(c) - base + k", "ord(c) + base + k", "wrong_operator", "adds base instead of subtracting, wrong offset"),
            ("if k < 0: raise ValueError(\"k must be non-negative\")\n                ", "",
             "dropped_guard", "remove negative k guard"),
        ],
    ),

    _TaskDef(
        task_id="full_remove_consecutive_duplicates",
        spec="Remove consecutive duplicate characters from string s. Raise ValueError if s is empty.",
        correct=_d("""
                def solution(s):
                    if not s: raise ValueError("s must not be empty")
                    result = [s[0]]
                    for c in s[1:]:
                        if c != result[-1]:
                            result.append(c)
                    return ''.join(result)
        """),
        tests=_d("""
                import pytest
                def test_basic():     assert solution("aabbcc") == "abc"
                def test_no_dup():    assert solution("abcd") == "abcd"
                def test_all_same():  assert solution("aaa") == "a"
                def test_alternating():assert solution("aba") == "aba"
                def test_single():    assert solution("x") == "x"
                def test_empty():
                    with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("c != result[-1]", "c == result[-1]", "wrong_operator", "keeps duplicates instead of removing them"),
            ("result[-1]", "result[0]", "wrong_operator", "compares to first character always, not previous"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty string guard"),
            ("s[1:]", "s", "off_by_one", "starts loop from s[0], comparing s[0] to result[-1] which is s[0]"),
        ],
    ),

    _TaskDef(
        task_id="full_longest_palindromic_length",
        spec="Return the length of the longest palindromic substring of s. Raise ValueError if s is empty.",
        correct=_d("""
                def solution(s):
                    if not s: raise ValueError("s must not be empty")
                    n = len(s)
                    best = 1
                    for center in range(n):
                        lo, hi = center, center
                        while lo >= 0 and hi < n and s[lo] == s[hi]:
                            best = max(best, hi - lo + 1)
                            lo -= 1
                            hi += 1
                        lo, hi = center, center + 1
                        while lo >= 0 and hi < n and s[lo] == s[hi]:
                            best = max(best, hi - lo + 1)
                            lo -= 1
                            hi += 1
                    return best
        """),
        tests=_d("""
                import pytest
                def test_babad():    assert solution("babad") == 3
                def test_cbbd():     assert solution("cbbd") == 2
                def test_single():   assert solution("a") == 1
                def test_racecar():  assert solution("racecar") == 7
                def test_no_pal():   assert solution("abcd") == 1
                def test_empty():
                    with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("hi - lo + 1", "hi - lo", "off_by_one", "undercounts palindrome length by 1"),
            ("s[lo] == s[hi]", "s[lo] != s[hi]", "wrong_operator", "expands when characters differ instead of match"),
            ("lo >= 0", "lo > 0", "wrong_operator", "stops expansion one step too early"),
            ("hi < n", "hi <= n", "wrong_operator", "allows out-of-bounds index access"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty string guard"),
        ],
    ),

    _TaskDef(
        task_id="full_capitalize_words",
        spec="Return s with the first letter of each whitespace-separated word capitalized and the rest lowercase. Raise ValueError if s is empty.",
        correct=_d("""
                def solution(s):
                    if not s: raise ValueError("s must not be empty")
                    return ' '.join(word[0].upper() + word[1:].lower() if word else word for word in s.split(' '))
        """),
        tests=_d("""
                import pytest
                def test_basic():     assert solution("hello world") == "Hello World"
                def test_upper():     assert solution("HELLO") == "Hello"
                def test_mixed():     assert solution("already Capital") == "Already Capital"
                def test_single():    assert solution("x") == "X"
                def test_empty():
                    with pytest.raises(ValueError): solution("")
        """),
        patches=[
            ("word[0].upper()", "word[0].lower()", "wrong_operator", "lowercases first letter instead of uppercasing"),
            ("word[1:].lower()", "word[1:].upper()", "wrong_operator", "uppercases rest instead of lowercasing"),
            ("if not s: raise ValueError(\"s must not be empty\")\n                ", "",
             "dropped_guard", "remove empty string guard"),
            ("s.split(' ')", "s.split()", "wrong_operator", "split() removes multiple spaces differently than split(' ')"),
        ],
    ),

    _TaskDef(
        task_id="full_manhattan_distance",
        spec="Return the Manhattan distance between points (x1, y1) and (x2, y2): abs(x1-x2) + abs(y1-y2).",
        correct=_d("""
                def solution(x1, y1, x2, y2):
                    return abs(x1 - x2) + abs(y1 - y2)
        """),
        tests=_d("""
                def test_basic():      assert solution(0, 0, 3, 4) == 7
                def test_same():       assert solution(1, 1, 1, 1) == 0
                def test_negative():   assert solution(-1, -1, 1, 1) == 4
                def test_neg_target(): assert solution(0, 0, -3, -4) == 7
                def test_axis():       assert solution(0, 0, 5, 0) == 5
        """),
        patches=[
            ("abs(x1 - x2) + abs(y1 - y2)", "abs(x1 + x2) + abs(y1 + y2)", "wrong_operator", "adds coordinates instead of subtracting"),
            ("abs(x1 - x2) + abs(y1 - y2)", "abs(x1 - x2) * abs(y1 - y2)", "wrong_operator", "multiplies axis distances instead of summing"),
            ("abs(x1 - x2) + abs(y1 - y2)", "(x1 - x2) ** 2 + (y1 - y2) ** 2", "wrong_operator", "computes squared Euclidean distance instead of Manhattan"),
        ],
    ),

    _TaskDef(
        task_id="full_rectangle_area",
        spec="Return the area of an axis-aligned rectangle given bottom-left (x1, y1) and top-right (x2, y2). Raise ValueError if x1>=x2 or y1>=y2.",
        correct=_d("""
                def solution(x1, y1, x2, y2):
                    if x1 >= x2: raise ValueError("x1 must be less than x2")
                    if y1 >= y2: raise ValueError("y1 must be less than y2")
                    return (x2 - x1) * (y2 - y1)
        """),
        tests=_d("""
                import pytest
                def test_basic():   assert solution(0, 0, 3, 4) == 12
                def test_small():   assert solution(1, 1, 4, 3) == 6
                def test_unit():    assert solution(0, 0, 1, 1) == 1
                def test_bad_x():
                    with pytest.raises(ValueError): solution(3, 0, 1, 4)
                def test_bad_y():
                    with pytest.raises(ValueError): solution(0, 4, 3, 1)
                def test_equal_x():
                    with pytest.raises(ValueError): solution(1, 0, 1, 4)
                def test_equal_y():
                    with pytest.raises(ValueError): solution(0, 1, 3, 1)
        """),
        patches=[
            ("(x2 - x1) * (y2 - y1)", "(x2 + x1) * (y2 + y1)", "wrong_operator", "adds coordinates instead of computing dimensions"),
            ("(x2 - x1) * (y2 - y1)", "(x2 - x1) + (y2 - y1)", "wrong_operator", "computes half-perimeter instead of area"),
            ("x1 >= x2", "x1 > x2", "wrong_operator", "allows equal x1==x2, producing zero-area rectangle"),
            ("y1 >= y2", "y1 > y2", "wrong_operator", "allows equal y1==y2, producing zero-area rectangle"),
            ("if x1 >= x2: raise ValueError(\"x1 must be less than x2\")\n                ", "",
             "dropped_guard", "remove x1>=x2 guard"),
        ],
    ),

    _TaskDef(
        task_id="full_are_collinear",
        spec="Return True if three points (x1,y1), (x2,y2), (x3,y3) are collinear using the cross product test.",
        correct=_d("""
                def solution(x1, y1, x2, y2, x3, y3):
                    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)
        """),
        tests=_d("""
                def test_collinear_diag():    assert solution(0, 0, 1, 1, 2, 2) == True
                def test_not_collinear():     assert solution(0, 0, 1, 0, 2, 1) == False
                def test_vertical():          assert solution(0, 0, 0, 1, 0, 2) == True
                def test_same_point():        assert solution(1, 1, 1, 1, 1, 1) == True
                def test_not_col2():          assert solution(0, 0, 1, 2, 3, 1) == False
        """),
        patches=[
            ("(x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)",
             "(x2 - x1) * (y3 - y1) != (x3 - x1) * (y2 - y1)",
             "wrong_operator", "inverts result, returns True when not collinear"),
            ("(x2 - x1) * (y3 - y1)", "(x2 + x1) * (y3 + y1)", "wrong_operator", "uses addition instead of subtraction in cross product"),
            ("(x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)",
             "(x2 - x1) * (y3 - y1) < (x3 - x1) * (y2 - y1)",
             "wrong_operator", "uses < instead of == for cross product comparison"),
            ("(y3 - y1)", "(y3 + y1)", "wrong_operator", "adds y1 instead of subtracting in cross product term"),
        ],
    ),

    _TaskDef(
        task_id="full_staircase_ways",
        spec="Count the number of ways to climb n stairs taking 1 or 2 steps at a time. Raise ValueError if n <= 0.",
        correct=_d("""
                def solution(n):
                    if n <= 0: raise ValueError("n must be positive")
                    if n == 1: return 1
                    a, b = 1, 1
                    for _ in range(2, n + 1):
                        a, b = b, a + b
                    return b
        """),
        tests=_d("""
                import pytest
                def test_n1():  assert solution(1) == 1
                def test_n2():  assert solution(2) == 2
                def test_n3():  assert solution(3) == 3
                def test_n4():  assert solution(4) == 5
                def test_n5():  assert solution(5) == 8
                def test_zero():
                    with pytest.raises(ValueError): solution(0)
                def test_neg():
                    with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("a + b", "a * b", "wrong_operator", "multiplies instead of adds fibonacci recurrence"),
            ("a + b", "a - b", "wrong_operator", "subtracts instead of adds fibonacci recurrence"),
            ("a, b = b, a + b", "a, b = a + b, b", "swapped_args", "assigns sum to a instead of b, leaving b unchanged"),
            ("range(2, n + 1)", "range(2, n)", "off_by_one", "loop stops one step short, missing last fibonacci step"),
            ("if n <= 0: raise ValueError(\"n must be positive\")\n                ", "",
             "dropped_guard", "remove non-positive n guard"),
        ],
    ),

    _TaskDef(
        task_id="full_lis_length",
        spec="Return the length of the longest strictly increasing subsequence in lst. Raise ValueError if lst is empty.",
        correct=_d("""
                def solution(lst):
                    if not lst: raise ValueError("lst must not be empty")
                    n = len(lst)
                    dp = [1] * n
                    for i in range(1, n):
                        for j in range(i):
                            if lst[j] < lst[i]:
                                if dp[j] + 1 > dp[i]:
                                    dp[i] = dp[j] + 1
                    return max(dp)
        """),
        tests=_d("""
                import pytest
                def test_basic():     assert solution([3, 1, 2]) == 2
                def test_sorted():    assert solution([1, 2, 3, 4]) == 4
                def test_reverse():   assert solution([4, 3, 2, 1]) == 1
                def test_equal():     assert solution([2, 2, 2]) == 1
                def test_mixed():     assert solution([1, 3, 2, 3, 1, 4]) == 4
                def test_empty():
                    with pytest.raises(ValueError): solution([])
        """),
        patches=[
            ("lst[j] < lst[i]", "lst[j] <= lst[i]", "wrong_operator", "allows equal elements, finds non-strictly increasing subsequence"),
            ("lst[j] < lst[i]", "lst[j] > lst[i]", "wrong_operator", "finds longest decreasing subsequence instead"),
            ("dp[j] + 1 > dp[i]", "dp[j] + 1 < dp[i]", "wrong_operator", "updates only when new length is smaller, keeps minimum"),
            ("if not lst: raise ValueError(\"lst must not be empty\")\n                ", "",
             "dropped_guard", "remove empty list guard"),
        ],
    ),

    _TaskDef(
        task_id="full_closest_pair_distance",
        spec="Return the minimum Euclidean distance between any two points in the list. Raise ValueError if fewer than 2 points.",
        correct=_d("""
            def solution(points):
                import math
                if len(points) < 2: raise ValueError("need at least 2 points")\n                min_dist = float("inf")
                n = len(points)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        dx = points[i][0] - points[j][0]
                        dy = points[i][1] - points[j][1]
                        d = math.sqrt(dx * dx + dy * dy)
                        if d < min_dist:
                            min_dist = d
                return min_dist
        """),
        tests=_d("""
            import pytest, math
            def test_basic():    assert solution([(0,0),(3,4)]) == pytest.approx(5.0)
            def test_unit():     assert solution([(0,0),(1,0),(0,1)]) == pytest.approx(1.0)
            def test_two():      assert solution([(1,1),(4,5)]) == pytest.approx(5.0)
            def test_three():    assert solution([(0,0),(1,1),(10,10)]) == pytest.approx(math.sqrt(2))
            def test_neg():      assert solution([(-1,-1),(1,1)]) == pytest.approx(math.sqrt(8))
            def test_short():
                with pytest.raises(ValueError): solution([(0,0)])
        """),
        patches=[
            ("dx * dx + dy * dy",   "dx * dx - dy * dy",  "wrong_operator", "subtraction breaks distance calculation"),
            ("if d < min_dist:",     "if d > min_dist:",    "wrong_operator", "finds maximum distance instead of minimum"),
            ("dx = points[i][0] - points[j][0]", "dx = points[i][0] + points[j][0]", "wrong_operator", "addition instead of subtraction for dx"),
            ("range(n - 1)",         "range(n)",            "off_by_one",    "outer loop includes last index — j loops out of useful range"),
            ("if len(points) < 2: raise ValueError(\"need at least 2 points\")\n                ", "",
             "dropped_guard", "remove minimum-points guard"),
        ],
    ),

    _TaskDef(
        task_id="full_min_coins",
        spec="Return minimum coins to make amount using denominations [1,5,10,25] (greedy, canonical set). Raise ValueError if amount < 0.",
        correct=_d("""
            def solution(amount):
                if amount < 0: raise ValueError("amount must be non-negative")
                coins = [25, 10, 5, 1]
                count = 0
                for coin in coins:
                    count += amount // coin
                    amount %= coin
                return count
        """),
        tests=_d("""
            import pytest
            def test_zero():     assert solution(0) == 0
            def test_one():      assert solution(1) == 1
            def test_five():     assert solution(5) == 1
            def test_thirty():   assert solution(30) == 2
            def test_eleven():   assert solution(11) == 2
            def test_fortyone(): assert solution(41) == 4
            def test_neg():
                with pytest.raises(ValueError): solution(-1)
        """),
        patches=[
            ("amount // coin",  "amount % coin",    "wrong_operator", "modulo instead of floor division gives wrong coin count"),
            ("amount %= coin",  "amount //= coin",  "wrong_operator", "floor division instead of modulo gives wrong remainder"),
            ("count += amount // coin", "count -= amount // coin", "wrong_operator", "subtract instead of add gives negative count"),
            ("if amount < 0: raise ValueError(\"amount must be non-negative\")\n                ", "",
             "dropped_guard", "remove negative guard"),
            ("coins = [25, 10, 5, 1]", "coins = [1, 5, 10, 25]", "swapped_args", "ascending order causes greedy to pick smallest coins first"),
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
