# -*- coding: utf-8 -*-
"""Judge abstraction: wraps an LLM (or deterministic checker) as a binary judge."""
from __future__ import annotations
import dataclasses
import datetime
import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


VERDICT_PASS    = "PASS"
VERDICT_FAIL    = "FAIL"
VERDICT_INVALID = "INVALID"


@dataclass
class JudgeResult:
    item_id:               str
    judge_id:              str
    verdict:               str           # PASS | FAIL | INVALID
    raw_output:            str
    logprob:               Optional[float]
    seed:                  int
    latency_ms:            float
    ts:                    str           # ISO UTC timestamp
    prompt_template_hash:  str
    model_id:              str
    decoding_params:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def miss(self, gt_label: str) -> Optional[int]:
        """Returns 1 (missed), 0 (caught), None (INVALID — exclude from analysis)."""
        if self.verdict == VERDICT_INVALID:
            return None
        judge_says_pass = self.verdict == VERDICT_PASS
        item_is_correct = gt_label == "correct"
        # miss = judge disagrees with ground truth
        return 0 if judge_says_pass == item_is_correct else 1

    @property
    def pass_defective(self) -> Optional[int]:
        """1 if judge passed a defective item, else 0/None."""
        if self.verdict == VERDICT_INVALID:
            return None
        # only meaningful for defective items; callers filter
        return 1 if self.verdict == VERDICT_PASS else 0

    @property
    def fail_correct(self) -> Optional[int]:
        """1 if judge failed a correct item, else 0/None."""
        if self.verdict == VERDICT_INVALID:
            return None
        return 1 if self.verdict == VERDICT_FAIL else 0


def _prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode()).hexdigest()[:12]


def _parse_verdict(raw: str) -> str:
    """Extract PASS/FAIL from 'VERDICT: PASS' or 'VERDICT: FAIL' in the last lines."""
    for line in reversed(raw.strip().splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            token = stripped.split(":", 1)[1].strip().upper()
            if token in (VERDICT_PASS, VERDICT_FAIL):
                return token
    return VERDICT_INVALID


class Judge:
    """Wraps a model + prompt template as a binary judge.

    call_fn signature: (prompt: str, decoding: dict, seed: int) -> (raw: str, logprob: float|None)
    For stubs/dry-runs pass a mock call_fn.
    """

    def __init__(
        self,
        model_id: str,
        decoding: dict,
        prompt_template: str,
        call_fn: Callable[[str, dict, int], Tuple[str, Optional[float]]],
        max_retries: int = 1,
    ):
        self.model_id        = model_id
        self.decoding        = decoding
        self.prompt_template = prompt_template
        self.prompt_hash     = _prompt_hash(prompt_template)
        self.call_fn         = call_fn
        self.max_retries     = max_retries
        self.judge_id        = f"{model_id}:{self.prompt_hash}"

    def judge(self, item) -> JudgeResult:
        """Judge one CorpusItem. Retries once on INVALID, then records as INVALID."""
        prompt = self.prompt_template.format(
            spec=item.spec, candidate_code=item.candidate_code
        )
        seed = self.decoding.get("seed", 0)
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            raw, logprob = self.call_fn(prompt, self.decoding, seed)
            latency_ms = (time.monotonic() - t0) * 1000
            verdict = _parse_verdict(raw)
            if verdict != VERDICT_INVALID or attempt == self.max_retries:
                break
        return JudgeResult(
            item_id=item.item_id,
            judge_id=self.judge_id,
            verdict=verdict,
            raw_output=raw,
            logprob=logprob,
            seed=seed,
            latency_ms=round(latency_ms, 1),
            ts=datetime.datetime.utcnow().isoformat(),
            prompt_template_hash=self.prompt_hash,
            model_id=self.model_id,
            decoding_params=dict(self.decoding),
        )
