# -*- coding: utf-8 -*-
"""Unit tests for the Ollama adapter in judges.py.

All HTTP calls are mocked — no running Ollama server required.
"""
from __future__ import annotations
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from judge_blindspot.judges import (
    OLLAMA_DEFAULT_HOST,
    VERDICT_FAIL,
    VERDICT_INVALID,
    VERDICT_PASS,
    Judge,
    _ollama_call,
    is_ollama_available,
    list_ollama_models,
    make_ollama_judge,
)

DUMMY_PROMPT = "Spec:\n{spec}\nCode:\n{candidate_code}\nReply VERDICT: PASS or VERDICT: FAIL"


def _mock_urlopen(body: dict) -> MagicMock:
    """Return a MagicMock that behaves like urllib.request.urlopen(...) used as a context manager.

    The code does: `with urlopen(req) as resp: resp.read()`.
    MagicMock.__enter__ returns a *new* MagicMock by default, not self — so
    we must wire it explicitly to return the same object that has read set up.
    """
    data = json.dumps(body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class _FakeItem:
    item_id = "item_001"
    spec = "return sum of list"
    candidate_code = "def f(x): return sum(x)"
    gt_label = "correct"
    defect_type = None


# ── _ollama_call ──────────────────────────────────────────────────────────────

def test_ollama_call_returns_response_text():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen({"response": "VERDICT: PASS"})):
        raw, logprob = _ollama_call("prompt", {"temperature": 0.0, "max_tokens": 64}, 42, "llama3.1:8b")
    assert raw == "VERDICT: PASS"
    assert logprob is None


def test_ollama_call_sends_correct_payload():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _mock_urlopen({"response": ""})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _ollama_call("my prompt", {"temperature": 0.5, "max_tokens": 128}, 7, "mistral:7b")

    assert captured["model"] == "mistral:7b"
    assert captured["prompt"] == "my prompt"
    assert captured["stream"] is False
    assert captured["options"]["temperature"] == 0.5
    assert captured["options"]["seed"] == 7
    assert captured["options"]["num_predict"] == 128


def test_ollama_call_uses_custom_host():
    captured_url: list = []

    def fake_urlopen(req, timeout=None):
        captured_url.append(req.full_url)
        return _mock_urlopen({"response": ""})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _ollama_call("p", {}, 0, "llama3:8b", host="http://myhost:11434")

    assert captured_url[0] == "http://myhost:11434/api/generate"


def test_ollama_call_defaults_to_standard_host():
    captured_url: list = []

    def fake_urlopen(req, timeout=None):
        captured_url.append(req.full_url)
        return _mock_urlopen({"response": ""})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _ollama_call("p", {}, 0, "llama3:8b")

    assert captured_url[0] == f"{OLLAMA_DEFAULT_HOST}/api/generate"


def test_ollama_call_missing_response_key_returns_empty():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen({})):
        raw, _ = _ollama_call("p", {}, 0, "model")
    assert raw == ""


# ── is_ollama_available ───────────────────────────────────────────────────────

def test_is_ollama_available_true():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen({"models": []})):
        assert is_ollama_available() is True


def test_is_ollama_available_false_on_url_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert is_ollama_available() is False


def test_is_ollama_available_false_on_os_error():
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        assert is_ollama_available() is False


def test_is_ollama_available_uses_api_tags_endpoint():
    captured: list = []

    def fake_urlopen(url, timeout=None):
        captured.append(url)
        return _mock_urlopen({})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        is_ollama_available(host="http://remotehost:11434")

    assert "remotehost:11434/api/tags" in captured[0]


# ── list_ollama_models ────────────────────────────────────────────────────────

def test_list_ollama_models_returns_names():
    body = {"models": [{"name": "llama3.1:8b"}, {"name": "mistral:7b"}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
        names = list_ollama_models()
    assert names == ["llama3.1:8b", "mistral:7b"]


def test_list_ollama_models_empty_list():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen({"models": []})):
        assert list_ollama_models() == []


def test_list_ollama_models_empty_on_url_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert list_ollama_models() == []


def test_list_ollama_models_empty_when_no_models_key():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen({})):
        assert list_ollama_models() == []


# ── make_ollama_judge: factory ────────────────────────────────────────────────

def test_make_ollama_judge_returns_judge_instance():
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
    assert isinstance(judge, Judge)


def test_make_ollama_judge_model_id():
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
    assert judge.model_id == "llama3.1:8b"


def test_make_ollama_judge_id_contains_model():
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
    assert "llama3.1:8b" in judge.judge_id


def test_make_ollama_judge_with_suffix():
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, judge_id_suffix="_clone")
    assert "_clone" in judge.judge_id
    assert "llama3.1:8b" in judge.judge_id


def test_make_ollama_judge_default_decoding():
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
    assert judge.decoding["temperature"] == 0.0
    assert judge.decoding["seed"] == 42
    assert judge.decoding["max_tokens"] == 256


def test_make_ollama_judge_custom_decoding():
    dec = {"temperature": 0.1, "max_tokens": 128, "seed": 99}
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, decoding=dec)
    assert judge.decoding["seed"] == 99
    assert judge.decoding["temperature"] == 0.1
    assert judge.decoding["max_tokens"] == 128


def test_make_ollama_judge_decoding_is_copy():
    dec = {"temperature": 0.0, "max_tokens": 256, "seed": 42}
    judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, decoding=dec)
    dec["seed"] = 999
    assert judge.decoding["seed"] == 42  # mutation of original dict must not affect judge


# ── judge() round-trip ────────────────────────────────────────────────────────

def test_ollama_judge_returns_pass():
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "Looks correct.\nVERDICT: PASS"})):
        result = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT).judge(_FakeItem())
    assert result.verdict == VERDICT_PASS
    assert result.item_id == "item_001"
    assert result.model_id == "llama3.1:8b"
    assert result.logprob is None


def test_ollama_judge_returns_fail():
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "Found a bug.\nVERDICT: FAIL"})):
        result = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT).judge(_FakeItem())
    assert result.verdict == VERDICT_FAIL


def test_ollama_judge_records_judge_id():
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "VERDICT: PASS"})):
        judge = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
        result = judge.judge(_FakeItem())
    assert result.judge_id == judge.judge_id


def test_ollama_judge_retries_on_invalid_then_pass():
    """First call returns no verdict (INVALID); second call returns PASS."""
    responses = [
        _mock_urlopen({"response": "I cannot determine."}),
        _mock_urlopen({"response": "VERDICT: PASS"}),
    ]
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        resp = responses[call_count[0]]
        call_count[0] += 1
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, max_retries=1).judge(_FakeItem())

    assert result.verdict == VERDICT_PASS
    assert call_count[0] == 2


def test_ollama_judge_records_invalid_when_max_retries_exhausted():
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "Not sure."})):
        result = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, max_retries=0).judge(_FakeItem())
    assert result.verdict == VERDICT_INVALID


def test_ollama_judge_latency_ms_positive():
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "VERDICT: PASS"})):
        result = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT).judge(_FakeItem())
    assert result.latency_ms >= 0.0


# ── intra-model axis: same seed → identical verdicts ─────────────────────────

def test_intra_model_same_seed_same_verdict():
    """Core guarantee for H1: two clones with identical seed → identical verdict."""
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen({"response": "VERDICT: PASS"})):
        ja = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, decoding={"seed": 42})
        jb = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, decoding={"seed": 42},
                               judge_id_suffix="_clone")
        ra = ja.judge(_FakeItem())
        rb = jb.judge(_FakeItem())

    assert ra.verdict == rb.verdict


def test_two_judges_same_model_same_seed_have_different_judge_ids():
    """Clone suffix must produce a distinct judge_id (needed for pairwise analysis)."""
    ja = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT)
    jb = make_ollama_judge("llama3.1:8b", DUMMY_PROMPT, judge_id_suffix="_clone")
    assert ja.judge_id != jb.judge_id
