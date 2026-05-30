# -*- coding: utf-8 -*-
"""Unit tests for _parse_judge_specs in cli.py.

No HTTP calls — make_ollama_judge / make_anthropic_judge only allocate
Judge objects; actual network I/O happens on judge.judge(), not here.
"""
from __future__ import annotations
import pytest

from judge_blindspot.cli import _parse_judge_specs
from judge_blindspot.judges import Judge
from judge_blindspot.mock_judges import MockJudge

DUMMY_PROMPT = "Spec:\n{spec}\nCode:\n{candidate_code}\nVERDICT: PASS or FAIL"


# ── mock ──────────────────────────────────────────────────────────────────────

def test_mock_returns_two_mock_judges():
    judges = _parse_judge_specs("mock", "", 0, None)
    assert len(judges) == 2
    assert all(isinstance(j, MockJudge) for j in judges)


def test_mock_judges_have_different_ids():
    judges = _parse_judge_specs("mock", "", 0, None)
    assert judges[0].judge_id != judges[1].judge_id


def test_mock_uses_judge_seed():
    j_s0 = _parse_judge_specs("mock", "", 0, None)
    j_s5 = _parse_judge_specs("mock", "", 5, None)
    # Different seeds → different judge_ids (MockJudge embeds seed in model_id string)
    assert j_s0[0].seed == 0
    assert j_s5[0].seed == 5
    assert j_s0[1].seed == 1
    assert j_s5[1].seed == 6


# ── ollama single spec (intra-model axis) ────────────────────────────────────

def test_ollama_single_creates_two_judges():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 42, None)
    assert len(judges) == 2


def test_ollama_single_both_have_same_model_id():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 42, None)
    assert judges[0].model_id == "llama3.1:8b"
    assert judges[1].model_id == "llama3.1:8b"


def test_ollama_single_clone_has_different_judge_id():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 42, None)
    assert judges[0].judge_id != judges[1].judge_id


def test_ollama_single_clone_suffix_in_id():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 42, None)
    assert "_clone" in judges[1].judge_id


def test_ollama_single_seed_propagated():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 99, None)
    assert judges[0].decoding["seed"] == 99
    assert judges[1].decoding["seed"] == 99


# ── ollama two-spec pair ──────────────────────────────────────────────────────

def test_ollama_pair_different_models():
    judges = _parse_judge_specs(
        "ollama:llama3.1:8b,ollama:mistral:7b", DUMMY_PROMPT, 42, None
    )
    assert len(judges) == 2
    assert judges[0].model_id == "llama3.1:8b"
    assert judges[1].model_id == "mistral:7b"


def test_ollama_pair_different_models_have_distinct_ids():
    judges = _parse_judge_specs(
        "ollama:llama3.1:8b,ollama:mistral:7b", DUMMY_PROMPT, 42, None
    )
    assert judges[0].judge_id != judges[1].judge_id


def test_ollama_pair_same_model_gets_suffix():
    # Explicitly listing the same model twice (same judge_id collision → _0/_1 suffixes)
    judges = _parse_judge_specs(
        "ollama:llama3.1:8b,ollama:llama3.1:8b", DUMMY_PROMPT, 42, None
    )
    assert judges[0].judge_id != judges[1].judge_id
    assert "_0" in judges[0].judge_id
    assert "_1" in judges[1].judge_id


# ── anthropic pair (inter-capability axis) ────────────────────────────────────

def test_anthropic_pair_haiku_sonnet():
    judges = _parse_judge_specs(
        "anthropic:claude-haiku-4-5-20251001,anthropic:claude-sonnet-4-6",
        DUMMY_PROMPT, 42, "sk-test",
    )
    assert len(judges) == 2
    assert judges[0].model_id == "claude-haiku-4-5-20251001"
    assert judges[1].model_id == "claude-sonnet-4-6"


def test_anthropic_pair_distinct_judge_ids():
    judges = _parse_judge_specs(
        "anthropic:claude-haiku-4-5-20251001,anthropic:claude-sonnet-4-6",
        DUMMY_PROMPT, 42, "sk-test",
    )
    assert judges[0].judge_id != judges[1].judge_id


def test_anthropic_single_creates_clone(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    judges = _parse_judge_specs(
        "anthropic:claude-haiku-4-5-20251001", DUMMY_PROMPT, 42, None
    )
    assert len(judges) == 2
    assert judges[0].model_id == judges[1].model_id
    assert "_clone" in judges[1].judge_id


# ── cross-type pair (ollama + anthropic) ─────────────────────────────────────

def test_cross_type_ollama_anthropic():
    judges = _parse_judge_specs(
        "ollama:llama3.1:8b,anthropic:claude-haiku-4-5-20251001",
        DUMMY_PROMPT, 42, "sk-test",
    )
    assert len(judges) == 2
    assert judges[0].model_id == "llama3.1:8b"
    assert judges[1].model_id == "claude-haiku-4-5-20251001"


def test_cross_type_distinct_judge_ids():
    judges = _parse_judge_specs(
        "ollama:llama3.1:8b,anthropic:claude-haiku-4-5-20251001",
        DUMMY_PROMPT, 42, "sk-test",
    )
    assert judges[0].judge_id != judges[1].judge_id


# ── result type checks ────────────────────────────────────────────────────────

def test_ollama_result_is_judge_instance():
    judges = _parse_judge_specs("ollama:llama3.1:8b", DUMMY_PROMPT, 42, None)
    assert all(isinstance(j, Judge) for j in judges)


def test_anthropic_result_is_judge_instance():
    judges = _parse_judge_specs(
        "anthropic:claude-haiku-4-5-20251001,anthropic:claude-sonnet-4-6",
        DUMMY_PROMPT, 42, "sk-test",
    )
    assert all(isinstance(j, Judge) for j in judges)


# ── error cases ───────────────────────────────────────────────────────────────

def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        _parse_judge_specs("openai:gpt-4", DUMMY_PROMPT, 42, None)


def test_bad_format_no_colon_raises():
    with pytest.raises(ValueError, match="spec"):
        _parse_judge_specs("nocoLon", DUMMY_PROMPT, 42, None)


def test_three_specs_raises():
    with pytest.raises(ValueError):
        _parse_judge_specs("ollama:a,ollama:b,ollama:c", DUMMY_PROMPT, 42, None)


def test_empty_string_raises():
    with pytest.raises(ValueError):
        _parse_judge_specs("", DUMMY_PROMPT, 42, None)


def test_anthropic_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        _parse_judge_specs("anthropic:claude-haiku-4-5-20251001", DUMMY_PROMPT, 42, None)


def test_anthropic_explicit_key_overrides_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Should NOT raise because api_key is provided explicitly
    judges = _parse_judge_specs(
        "anthropic:claude-haiku-4-5-20251001", DUMMY_PROMPT, 42, "sk-explicit"
    )
    assert len(judges) == 2
