# -*- coding: utf-8 -*-
"""Tests for judging prompts and supporting functions in judges.py."""
from pathlib import Path
import pytest
from judge_blindspot.judges import (
    _apply_template, _parse_verdict, _prompt_hash, load_prompt,
    VERDICT_PASS, VERDICT_FAIL, VERDICT_INVALID,
)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_FILES = ["strict_passfail.txt", "rubric.txt", "confidence.txt"]


# ── prompt file basics ────────────────────────────────────────────────────────

def test_all_prompt_files_exist():
    for fname in PROMPT_FILES:
        assert (PROMPTS_DIR / fname).exists(), f"Missing: {fname}"


def test_all_prompts_contain_spec_placeholder():
    for fname in PROMPT_FILES:
        text = (PROMPTS_DIR / fname).read_text(encoding="utf-8")
        assert "{spec}" in text, f"{fname} missing {{spec}}"


def test_all_prompts_contain_code_placeholder():
    for fname in PROMPT_FILES:
        text = (PROMPTS_DIR / fname).read_text(encoding="utf-8")
        assert "{candidate_code}" in text, f"{fname} missing {{candidate_code}}"


def test_all_prompts_mention_verdict_pass_fail():
    for fname in PROMPT_FILES:
        text = (PROMPTS_DIR / fname).read_text(encoding="utf-8")
        assert "VERDICT: PASS" in text and "VERDICT: FAIL" in text, \
            f"{fname} does not show both VERDICT options"


def test_prompt_hashes_are_unique():
    texts = [(PROMPTS_DIR / f).read_text(encoding="utf-8") for f in PROMPT_FILES]
    hashes = [_prompt_hash(t) for t in texts]
    assert len(set(hashes)) == len(hashes), "Two prompts have the same hash"


def test_load_prompt_returns_string():
    text = load_prompt(PROMPTS_DIR / "strict_passfail.txt")
    assert isinstance(text, str) and len(text) > 50


# ── _apply_template ───────────────────────────────────────────────────────────

def test_apply_template_basic():
    tmpl = "Spec:\n{spec}\nCode:\n{candidate_code}"
    result = _apply_template(tmpl, "add two numbers", "def f(a, b): return a + b")
    assert "add two numbers" in result
    assert "def f(a, b)" in result


def test_apply_template_safe_with_dict_literal():
    # candidate_code with { } must not raise or corrupt output
    tmpl = "Spec:\n{spec}\nCode:\n{candidate_code}"
    code = "def f(x):\n    return {k: v for k, v in x.items()}\n"
    result = _apply_template(tmpl, "build dict", code)
    assert "k: v for k, v in x.items()" in result


def test_apply_template_safe_with_set_literal():
    tmpl = "{spec}\n{candidate_code}"
    code = "def f():\n    return {1, 2, 3}\n"
    result = _apply_template(tmpl, "return set", code)
    assert "{1, 2, 3}" in result


def test_apply_template_safe_with_fstring():
    tmpl = "{spec}\n{candidate_code}"
    code = 'def f(x):\n    return f"value={x}"\n'
    result = _apply_template(tmpl, "format", code)
    assert 'f"value={x}"' in result


def test_apply_template_replaces_both_placeholders():
    tmpl = "{spec} :: {candidate_code}"
    result = _apply_template(tmpl, "SPEC", "CODE")
    assert result == "SPEC :: CODE"


# ── _parse_verdict ────────────────────────────────────────────────────────────

def test_parse_verdict_pass():
    assert _parse_verdict("some reasoning\nVERDICT: PASS") == VERDICT_PASS


def test_parse_verdict_fail():
    assert _parse_verdict("analysis\nVERDICT: FAIL") == VERDICT_FAIL


def test_parse_verdict_case_insensitive():
    assert _parse_verdict("verdict: pass") == VERDICT_PASS
    assert _parse_verdict("Verdict: Fail") == VERDICT_FAIL


def test_parse_verdict_confidence_format():
    # confidence.txt puts Confidence: N before VERDICT
    raw = "Issues: none\nConfidence: 4\nVERDICT: PASS"
    assert _parse_verdict(raw) == VERDICT_PASS


def test_parse_verdict_trailing_whitespace():
    assert _parse_verdict("VERDICT: PASS   \n") == VERDICT_PASS


def test_parse_verdict_missing_returns_invalid():
    assert _parse_verdict("No verdict in this output") == VERDICT_INVALID
    assert _parse_verdict("") == VERDICT_INVALID
    assert _parse_verdict("VERDICT: MAYBE") == VERDICT_INVALID


def test_parse_verdict_takes_last_occurrence():
    # If VERDICT appears multiple times (e.g. in reasoning), last one wins
    raw = "The code might VERDICT: FAIL but actually\nVERDICT: PASS"
    assert _parse_verdict(raw) == VERDICT_PASS


# ── markdown-decorated verdict lines (regression for pilot bug) ───────────────

def test_parse_verdict_markdown_bold():
    assert _parse_verdict("some reasoning\n**VERDICT: PASS**") == VERDICT_PASS


def test_parse_verdict_plain_fail():
    assert _parse_verdict("VERDICT: FAIL") == VERDICT_FAIL


def test_parse_verdict_leading_spaces_lowercase():
    assert _parse_verdict("  verdict: pass") == VERDICT_PASS


def test_parse_verdict_markdown_heading():
    assert _parse_verdict("analysis\n### VERDICT: PASS") == VERDICT_PASS


def test_parse_verdict_no_verdict_line():
    assert _parse_verdict("The solution looks correct.") == VERDICT_INVALID


# ── round-trip: load prompt → apply template → parse verdict ──────────────────

def test_round_trip_strict_passfail():
    tmpl = load_prompt(PROMPTS_DIR / "strict_passfail.txt")
    prompt = _apply_template(tmpl, "return sum", "def f(x): return sum(x)")
    assert "return sum" in prompt
    assert "def f(x)" in prompt
    # simulate model appending verdict
    response = prompt + "\nVERDICT: PASS"
    assert _parse_verdict(response) == VERDICT_PASS
