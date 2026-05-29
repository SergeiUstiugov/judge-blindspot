# -*- coding: utf-8 -*-
"""Deterministic mock judge for pipeline testing (no real LLM calls).

Determinism guarantee: same (judge_seed, item_id) always produces the same
verdict, regardless of call order. Two MockJudges with identical seeds on the
same corpus produce identical miss columns (phi = +1). Two MockJudges with
different seeds produce independent miss columns (phi ≈ 0).
"""
from __future__ import annotations
import datetime
import hashlib
from typing import List, Tuple

import numpy as np

from .judges import JudgeResult, VERDICT_PASS, VERDICT_FAIL


def _item_rng(judge_seed: int, item_id: str) -> np.random.Generator:
    """Deterministic RNG seeded from (judge_seed, item_id). Order-independent."""
    key = f"{judge_seed}:{item_id}".encode()
    h = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(h)


class MockJudge:
    """Deterministic mock judge with configurable per-item error rate.

    Parameters
    ----------
    error_rate : float
        Probability of returning the WRONG verdict on any item.
    seed : int
        Controls which items get errors. Same seed → identical miss column.
    judge_id : str
        Identifier used in JudgeResult and pairwise keys.
    """

    def __init__(self, error_rate: float = 0.25, seed: int = 0,
                 judge_id: str = "mock"):
        self.error_rate = error_rate
        self.seed = seed
        self.judge_id = judge_id
        self.prompt_hash = f"mock:{seed}"

    def judge(self, item) -> JudgeResult:
        """Return a deterministic verdict for item based on (seed, item_id)."""
        rng = _item_rng(self.seed, item.item_id)
        correct_verdict = VERDICT_PASS if item.gt_label == "correct" else VERDICT_FAIL
        if rng.random() < self.error_rate:
            verdict = VERDICT_FAIL if correct_verdict == VERDICT_PASS else VERDICT_PASS
        else:
            verdict = correct_verdict
        return JudgeResult(
            item_id=item.item_id,
            judge_id=self.judge_id,
            verdict=verdict,
            raw_output=f"VERDICT: {verdict}",
            logprob=None,
            seed=self.seed,
            latency_ms=0.0,
            ts=datetime.datetime.utcnow().isoformat(),
            prompt_template_hash=self.prompt_hash,
            model_id=f"mock(seed={self.seed})",
            decoding_params={"error_rate": self.error_rate},
        )


def make_calibration_judges(
    error_rate: float = 0.25,
    seed: int = 0,
) -> Tuple[MockJudge, MockJudge, MockJudge, MockJudge]:
    """Return judge pairs for positive and orthogonal calibration controls.

    Returns (pos_a, pos_b, orth_a, orth_b):
      pos_a, pos_b  -- same seed → identical miss column → phi = +1
      orth_a, orth_b -- different seeds → independent miss columns → phi ≈ 0
    """
    pos_a  = MockJudge(error_rate=error_rate, seed=seed,   judge_id="cal_pos_a")
    pos_b  = MockJudge(error_rate=error_rate, seed=seed,   judge_id="cal_pos_b")
    orth_a = MockJudge(error_rate=error_rate, seed=seed,   judge_id="cal_orth_a")
    orth_b = MockJudge(error_rate=error_rate, seed=seed+1, judge_id="cal_orth_b")
    return pos_a, pos_b, orth_a, orth_b
