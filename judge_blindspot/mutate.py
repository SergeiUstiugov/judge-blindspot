# -*- coding: utf-8 -*-
"""Mutation verification engine.

Takes hand-authored MutationSpec objects (source already written), runs hidden
tests to classify each as defective (fails ≥1 test) or equivalent (passes all
tests). Equivalent mutants are discarded and counted.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class MutationSpec:
    source: str        # complete module-level Python source of the mutant
    defect_type: str   # off_by_one | wrong_operator | dropped_guard | swapped_args | wrong_return
    description: str   # human-readable explanation of what was changed


@dataclass
class MutationResult:
    spec: MutationSpec
    is_defective: bool    # True  = fails >=1 test  → keep as defective corpus item
    is_equivalent: bool   # True  = passes all tests → discard (equivalent mutant)
    error: str            # non-empty if verification itself crashed


def _run_tests(source: str, hidden_tests: str) -> Tuple[bool, str]:
    """Run hidden_tests against source. Returns (all_pass: bool, stderr: str)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "candidate.py"
        test_path = Path(tmpdir) / "test_candidate.py"
        code_path.write_text(source, encoding="utf-8")
        # Register in sys.modules so 'from candidate import *' is deterministic
        # regardless of whether pytest adds tmpdir to sys.path.
        preamble = (
            "import sys, importlib.util as _ilu\n"
            f"_spec = _ilu.spec_from_file_location('candidate', r'{code_path}')\n"
            "_mod = _ilu.module_from_spec(_spec)\n"
            "sys.modules['candidate'] = _mod\n"
            "_spec.loader.exec_module(_mod)\n"
            "from candidate import *\n"
        )
        test_path.write_text(preamble + hidden_tests, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=30,
        )
    return result.returncode == 0, result.stderr


def verify_mutation(
    task_id: str,
    correct_source: str,
    mutation: MutationSpec,
    hidden_tests: str,
) -> MutationResult:
    """Verify that correct source passes all tests, then classify the mutation."""
    try:
        correct_ok, err = _run_tests(correct_source, hidden_tests)
        if not correct_ok:
            return MutationResult(mutation, False, False,
                                  f"task {task_id!r}: correct source failed tests")
        mutant_ok, err = _run_tests(mutation.source, hidden_tests)
    except Exception as e:
        return MutationResult(mutation, False, False, str(e))

    if mutant_ok:
        # passes all tests → equivalent mutant
        return MutationResult(mutation, False, True, "")
    else:
        # fails ≥1 test → genuine defect
        return MutationResult(mutation, True, False, "")


def verify_task_mutations(
    task_id: str,
    correct_source: str,
    hidden_tests: str,
    mutations: List[MutationSpec],
) -> Tuple[List[MutationResult], int]:
    """Verify all mutations for one task.

    Returns (results, n_equivalent_discarded).
    """
    results = []
    n_equiv = 0
    for mut in mutations:
        r = verify_mutation(task_id, correct_source, mut, hidden_tests)
        if r.error:
            raise RuntimeError(f"[{task_id}] mutation verification error: {r.error}")
        if r.is_equivalent:
            n_equiv += 1
        results.append(r)
    return results, n_equiv
