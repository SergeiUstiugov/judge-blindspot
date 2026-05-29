# -*- coding: utf-8 -*-
"""Tests for corpus.py: CorpusItem, save/load, GT verification."""
import json
import tempfile
from pathlib import Path
import pytest
from judge_blindspot.corpus import CorpusItem, save_corpus, load_corpus, verify_gt


def _item(item_id="t01", gt_label="defective", defect_type="off_by_one",
          hidden_tests=None):
    return CorpusItem(
        item_id=item_id,
        spec="Return the sum of a list.",
        candidate_code="def f(lst):\n    return sum(lst) + 1\n",
        gt_label=gt_label,
        defect_type=defect_type,
        source="synthesized",
        contamination_flag=False,
        hidden_tests=hidden_tests,
    )


# ── serialisation ─────────────────────────────────────────────────────────────

def test_to_jsonl_roundtrip():
    item = _item()
    restored = CorpusItem.from_dict(json.loads(item.to_jsonl()))
    assert restored.item_id  == item.item_id
    assert restored.gt_label == item.gt_label


def test_save_load_roundtrip():
    items = [_item("a"), _item("b", gt_label="correct", defect_type=None)]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    save_corpus(items, path)
    loaded = load_corpus(path)
    assert len(loaded) == 2
    assert loaded[0].item_id == "a"
    assert loaded[1].gt_label == "correct"


def test_load_skips_blank_lines():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w",
                                     encoding="utf-8") as f:
        f.write(_item("x").to_jsonl() + "\n\n")
        path = f.name
    loaded = load_corpus(path)
    assert len(loaded) == 1


# ── verify_gt ────────────────────────────────────────────────────────────────

def test_verify_gt_no_hidden_tests_returns_true():
    assert verify_gt(_item()) is True


CORRECT_CODE = "def add(a, b):\n    return a + b\n"
DEFECTIVE_CODE = "def add(a, b):\n    return a - b\n"
HIDDEN_TESTS = (
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "    assert add(0, 0) == 0\n"
)


def test_verify_gt_correct_item_passes():
    item = CorpusItem(
        item_id="c1", spec="Add two numbers.", candidate_code=CORRECT_CODE,
        gt_label="correct", defect_type=None, source="synthesized",
        contamination_flag=False, hidden_tests=HIDDEN_TESTS,
    )
    assert verify_gt(item) is True


def test_verify_gt_defective_item_passes():
    item = CorpusItem(
        item_id="d1", spec="Add two numbers.", candidate_code=DEFECTIVE_CODE,
        gt_label="defective", defect_type="wrong_operator", source="synthesized",
        contamination_flag=False, hidden_tests=HIDDEN_TESTS,
    )
    assert verify_gt(item) is True


def test_verify_gt_detects_mislabeled_correct():
    # defective code but labelled correct → mismatch
    item = CorpusItem(
        item_id="bad", spec="Add two numbers.", candidate_code=DEFECTIVE_CODE,
        gt_label="correct", defect_type=None, source="synthesized",
        contamination_flag=False, hidden_tests=HIDDEN_TESTS,
    )
    assert verify_gt(item) is False
