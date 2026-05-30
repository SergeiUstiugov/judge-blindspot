# -*- coding: utf-8 -*-
"""Unit tests for the Anthropic adapter in judges.py.

All HTTP calls are mocked — no live Anthropic API key required.
"""
from __future__ import annotations
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from judge_blindspot.judges import (
    ANTHROPIC_API_BASE,
    ANTHROPIC_API_VERSION,
    VERDICT_FAIL,
    VERDICT_INVALID,
    VERDICT_PASS,
    Judge,
    _anthropic_call,
    make_anthropic_judge,
)

DUMMY_PROMPT = "Spec:\n{spec}\nCode:\n{candidate_code}\nReply VERDICT: PASS or VERDICT: FAIL"

# Minimal well-formed Anthropic Messages API response
_RESPONSE_PASS = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "VERDICT: PASS"}],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 5},
}

_RESPONSE_FAIL = {
    **_RESPONSE_PASS,
    "content": [{"type": "text", "text": "Found a bug.\nVERDICT: FAIL"}],
}

_RESPONSE_NO_VERDICT = {
    **_RESPONSE_PASS,
    "content": [{"type": "text", "text": "I cannot determine."}],
}


def _mock_urlopen(body: dict) -> MagicMock:
    """Return a MagicMock that behaves like urllib.request.urlopen(...) as context manager."""
    data = json.dumps(body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class _FakeItem:
    item_id = "item_002"
    spec = "return product of two numbers"
    candidate_code = "def f(a, b): return a + b"
    gt_label = "defective"
    defect_type = "wrong_operator"


# ── _anthropic_call ───────────────────────────────────────────────────────────

def test_anthropic_call_returns_text():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_PASS)):
        raw, logprob = _anthropic_call(
            "prompt", {"temperature": 0.0, "max_tokens": 64}, 0,
            "claude-haiku-4-5-20251001", "sk-test",
        )
    assert raw == "VERDICT: PASS"
    assert logprob is None


def test_anthropic_call_sends_model_and_max_tokens():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("hello", {"temperature": 0.0, "max_tokens": 128}, 0,
                        "claude-haiku-4-5-20251001", "sk-ant-test")

    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["max_tokens"] == 128
    assert captured["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_call_omits_temperature_when_zero():
    """temperature=0.0 must NOT be sent (some providers reject explicit 0)."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("p", {"temperature": 0.0, "max_tokens": 64}, 0,
                        "claude-haiku-4-5-20251001", "key")

    assert "temperature" not in captured


def test_anthropic_call_includes_temperature_when_nonzero():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured.update(json.loads(req.data.decode("utf-8")))
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("p", {"temperature": 0.5, "max_tokens": 64}, 0,
                        "claude-haiku-4-5-20251001", "key")

    assert captured["temperature"] == 0.5


def test_anthropic_call_sends_api_key_header():
    captured_req: list = []

    def fake_urlopen(req, timeout=None):
        captured_req.append(req)
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("p", {}, 0, "claude-haiku-4-5-20251001", "sk-secret")

    req = captured_req[0]
    # urllib normalises header names: first char upper, rest lower
    assert req.get_header("X-api-key") == "sk-secret"


def test_anthropic_call_sends_anthropic_version_header():
    captured_req: list = []

    def fake_urlopen(req, timeout=None):
        captured_req.append(req)
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("p", {}, 0, "claude-haiku-4-5-20251001", "key")

    assert captured_req[0].get_header("Anthropic-version") == ANTHROPIC_API_VERSION


def test_anthropic_call_posts_to_messages_endpoint():
    captured_url: list = []

    def fake_urlopen(req, timeout=None):
        captured_url.append(req.full_url)
        return _mock_urlopen(_RESPONSE_PASS)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _anthropic_call("p", {}, 0, "claude-haiku-4-5-20251001", "key")

    assert captured_url[0] == f"{ANTHROPIC_API_BASE}/v1/messages"


# ── make_anthropic_judge: factory ─────────────────────────────────────────────

def test_make_anthropic_judge_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT)


def test_make_anthropic_judge_reads_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT)
    assert isinstance(judge, Judge)


def test_make_anthropic_judge_explicit_key():
    judge = make_anthropic_judge("claude-sonnet-4-6", DUMMY_PROMPT, api_key="sk-test")
    assert isinstance(judge, Judge)
    assert judge.model_id == "claude-sonnet-4-6"


def test_make_anthropic_judge_id_contains_model():
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k")
    assert "claude-haiku-4-5-20251001" in judge.judge_id


def test_make_anthropic_judge_with_suffix():
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT,
                                 api_key="k", judge_id_suffix="_b")
    assert "_b" in judge.judge_id
    assert "claude-haiku-4-5-20251001" in judge.judge_id


def test_make_anthropic_judge_default_decoding():
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k")
    assert judge.decoding["temperature"] == 0.0
    assert judge.decoding["max_tokens"] == 256


def test_make_anthropic_judge_custom_decoding():
    dec = {"temperature": 0.2, "max_tokens": 512}
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT,
                                 api_key="k", decoding=dec)
    assert judge.decoding["temperature"] == 0.2
    assert judge.decoding["max_tokens"] == 512


def test_make_anthropic_judge_decoding_is_copy():
    dec = {"temperature": 0.0, "max_tokens": 256}
    judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT,
                                 api_key="k", decoding=dec)
    dec["max_tokens"] = 9999
    assert judge.decoding["max_tokens"] == 256


# ── judge() round-trip ────────────────────────────────────────────────────────

def test_anthropic_judge_returns_pass():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_PASS)):
        judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k")
        result = judge.judge(_FakeItem())
    assert result.verdict == VERDICT_PASS
    assert result.item_id == "item_002"
    assert result.model_id == "claude-haiku-4-5-20251001"
    assert result.logprob is None


def test_anthropic_judge_returns_fail():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_FAIL)):
        result = make_anthropic_judge(
            "claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k"
        ).judge(_FakeItem())
    assert result.verdict == VERDICT_FAIL


def test_anthropic_judge_records_judge_id():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_PASS)):
        judge = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k")
        result = judge.judge(_FakeItem())
    assert result.judge_id == judge.judge_id


def test_anthropic_judge_retries_on_invalid_then_pass():
    responses = [
        _mock_urlopen(_RESPONSE_NO_VERDICT),
        _mock_urlopen(_RESPONSE_PASS),
    ]
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        resp = responses[call_count[0]]
        call_count[0] += 1
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = make_anthropic_judge(
            "claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k", max_retries=1
        ).judge(_FakeItem())

    assert result.verdict == VERDICT_PASS
    assert call_count[0] == 2


def test_anthropic_judge_records_invalid_when_max_retries_exhausted():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_NO_VERDICT)):
        result = make_anthropic_judge(
            "claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k", max_retries=0
        ).judge(_FakeItem())
    assert result.verdict == VERDICT_INVALID


def test_anthropic_judge_latency_ms_non_negative():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_RESPONSE_PASS)):
        result = make_anthropic_judge(
            "claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k"
        ).judge(_FakeItem())
    assert result.latency_ms >= 0.0


# ── inter-capability axis: haiku vs sonnet have distinct judge_ids ─────────────

def test_inter_capability_judges_have_different_ids():
    """Haiku and Sonnet must have distinct judge_ids for pairwise analysis."""
    haiku  = make_anthropic_judge("claude-haiku-4-5-20251001", DUMMY_PROMPT, api_key="k")
    sonnet = make_anthropic_judge("claude-sonnet-4-6",          DUMMY_PROMPT, api_key="k")
    assert haiku.judge_id != sonnet.judge_id
