# -*- coding: utf-8 -*-
"""Corpus management: JSONL schema, loading, saving, GT verification via hidden tests."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CorpusItem:
    """One judging item. GT is fixed and verified before any judge runs."""
    item_id:           str
    spec:              str            # problem statement shown to the judge
    candidate_code:    str            # code to be judged
    gt_label:          str            # "correct" | "defective"
    defect_type:       Optional[str]  # None for correct items; e.g. "off_by_one"
    source:            str            # e.g. "synthesized", "humaneval-debug-only"
    contamination_flag: bool          # True = possibly in model training data
    hidden_tests:      Optional[str] = None   # test source; NOT shown to judges
    metadata:          dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "CorpusItem":
        d = dict(d)
        d.setdefault("hidden_tests", None)
        d.setdefault("metadata", {})
        return CorpusItem(**d)


def save_corpus(items: List[CorpusItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_jsonl() + "\n")


def load_corpus(path: str | Path) -> List[CorpusItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(CorpusItem.from_dict(json.loads(line)))
    return items


def verify_gt(item: CorpusItem) -> bool:
    """Run hidden tests to confirm gt_label. Returns True if GT is consistent.

    Correct items must pass all hidden tests.
    Defective items must fail at least one hidden test.
    Items without hidden_tests are skipped (returns True).
    """
    if item.hidden_tests is None:
        return True
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "candidate.py"
        test_path = Path(tmpdir) / "test_candidate.py"
        code_path.write_text(item.candidate_code, encoding="utf-8")
        # inject candidate module into the test file
        preamble = (
            "import sys, importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('candidate', r'{code_path}')\n"
            "_mod = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_mod)\n"
            "from candidate import *\n"
        )
        test_path.write_text(preamble + item.hidden_tests, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=30,
        )
    tests_pass = result.returncode == 0
    return tests_pass if item.gt_label == "correct" else not tests_pass


def verify_corpus(
    items: List[CorpusItem], abort_on_mismatch: bool = True
) -> List[str]:
    """Verify GT for all items that have hidden_tests. Returns list of mismatch item_ids.

    Aborts (raises ValueError) on first mismatch if abort_on_mismatch=True.
    """
    mismatches = []
    for item in items:
        if item.hidden_tests is not None:
            if not verify_gt(item):
                mismatches.append(item.item_id)
                if abort_on_mismatch:
                    raise ValueError(
                        f"GT mismatch for {item.item_id!r}: "
                        f"gt_label={item.gt_label!r} contradicted by hidden tests."
                    )
    return mismatches
