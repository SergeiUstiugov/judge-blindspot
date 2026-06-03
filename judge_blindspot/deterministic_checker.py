# -*- coding: utf-8 -*-
"""Deterministic GT checker — wraps verify_gt as a Judge-compatible oracle.

Used as the deterministic arm of the H3 cross-type axis:
  (LLM judge) × (DeterministicChecker)  → expected φ ≈ 0
"""
from __future__ import annotations
import datetime
import time

from .corpus import verify_gt
from .judges import JudgeResult, VERDICT_PASS, VERDICT_FAIL

_JUDGE_ID    = "deterministic_gt"
_PROMPT_HASH = "deterministic:hidden_tests"


class DeterministicChecker:
    """Judge-compatible wrapper around verify_gt (hidden test suite).

    Runs item.hidden_tests via subprocess and maps the result to
    VERDICT_PASS (tests pass) or VERDICT_FAIL (at least one test fails).

    - Never produces VERDICT_INVALID: no LLM output to misparse.
    - Fully deterministic: same item → same verdict on every call.
    - No seed, no network calls, no prompt template.
    - seed=0 in JudgeResult is a placeholder (required field, no meaning here).

    Items without hidden_tests fall through to PASS because verify_gt returns
    True for them. In practice the full corpus always has hidden_tests on every
    item; this is a defensive fallback only.
    """

    judge_id:    str = _JUDGE_ID
    prompt_hash: str = _PROMPT_HASH

    def judge(self, item) -> JudgeResult:
        t0 = time.monotonic()
        passed = verify_gt(item)
        latency_ms = (time.monotonic() - t0) * 1000
        verdict = VERDICT_PASS if passed else VERDICT_FAIL
        return JudgeResult(
            item_id=item.item_id,
            judge_id=self.judge_id,
            verdict=verdict,
            raw_output=f"hidden_tests={'passed' if passed else 'failed'}",
            logprob=None,
            seed=0,
            latency_ms=round(latency_ms, 1),
            ts=datetime.datetime.utcnow().isoformat(),
            prompt_template_hash=self.prompt_hash,
            model_id=_JUDGE_ID,
            decoding_params={},
        )
