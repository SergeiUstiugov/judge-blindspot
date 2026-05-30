# -*- coding: utf-8 -*-
"""Judge abstraction: wraps an LLM (or deterministic checker) as a binary judge."""
from __future__ import annotations
import dataclasses
import datetime
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple


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


def load_prompt(path: str | Path) -> str:
    """Load a prompt template from a file. Placeholders: {spec}, {candidate_code}."""
    return Path(path).read_text(encoding="utf-8")


def _prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode()).hexdigest()[:12]


def _apply_template(template: str, spec: str, candidate_code: str) -> str:
    """Substitute {spec} and {candidate_code} without calling .format().

    .format() would fail if candidate_code contains { } (dict literals, f-strings,
    set comprehensions). Plain .replace() is safe and unambiguous.
    """
    return template.replace("{spec}", spec).replace("{candidate_code}", candidate_code)


def _parse_verdict(raw: str) -> str:
    """Extract PASS/FAIL scanning lines from the end; first VERDICT: line wins."""
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
        prompt = _apply_template(self.prompt_template, item.spec, item.candidate_code)
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


# ─────────────────────────────────────────────────────────────────────────────
# Ollama adapter — local LLM, zero API cost, deterministic seed
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def _ollama_call(
    prompt: str,
    decoding: dict,
    seed: int,
    model_id: str,
    host: str = OLLAMA_DEFAULT_HOST,
    timeout: int = 120,
) -> Tuple[str, None]:
    """POST to ollama /api/generate. Returns (raw_text, None).

    Uses stdlib urllib — no extra dependencies.
    Raises urllib.error.URLError if ollama is not running.
    """
    payload = {
        "model":  model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(decoding.get("temperature", 0.0)),
            "seed":        int(seed),
            "num_predict": int(decoding.get("max_tokens", 256)),
        },
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", ""), None


def is_ollama_available(host: str = OLLAMA_DEFAULT_HOST, timeout: int = 3) -> bool:
    """Return True if ollama server is reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def list_ollama_models(host: str = OLLAMA_DEFAULT_HOST) -> List[str]:
    """Return list of model names available in local ollama."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def make_ollama_judge(
    model_id: str,
    prompt_template: str,
    decoding: Optional[dict] = None,
    host: str = OLLAMA_DEFAULT_HOST,
    max_retries: int = 1,
    judge_id_suffix: str = "",
) -> Judge:
    """Factory: create a Judge backed by a local ollama model.

    Two judges with the same model_id and seed will produce identical verdicts
    (deterministic seed support in ollama) — use this for the intra-model axis.
    Different models give the inter-family axis.

    Example — intra-model positive control:
        ja = make_ollama_judge("llama3.1:8b", prompt, decoding={"seed": 42})
        jb = make_ollama_judge("llama3.1:8b", prompt, decoding={"seed": 42},
                               judge_id_suffix="_clone")
        # ja and jb should produce phi ≈ +1

    Example — inter-family:
        ja = make_ollama_judge("llama3.1:8b",  prompt)
        jb = make_ollama_judge("mistral:7b",   prompt)
    """
    if decoding is None:
        decoding = {"temperature": 0.0, "max_tokens": 256, "seed": 42}

    def call_fn(prompt_text: str, dec: dict, seed: int) -> Tuple[str, None]:
        return _ollama_call(prompt_text, dec, seed, model_id=model_id, host=host)

    judge = Judge(
        model_id=model_id,
        decoding=dict(decoding),
        prompt_template=prompt_template,
        call_fn=call_fn,
        max_retries=max_retries,
    )
    if judge_id_suffix:
        judge.judge_id = f"{model_id}{judge_id_suffix}:{judge.prompt_hash}"
    return judge


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic adapter — Messages API, zero extra dependencies (stdlib urllib)
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_BASE    = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"


def _anthropic_call(
    prompt: str,
    decoding: dict,
    seed: int,
    model_id: str,
    api_key: str,
    timeout: int = 60,
) -> Tuple[str, None]:
    """POST to Anthropic Messages API. Returns (raw_text, None).

    seed is recorded in JudgeResult but not forwarded — Anthropic API does not
    support deterministic seed. Use temperature=0.0 to minimise variance.
    Raises urllib.error.HTTPError on API errors (4xx/5xx).
    """
    payload: dict = {
        "model":      model_id,
        "max_tokens": int(decoding.get("max_tokens", 256)),
        "messages":   [{"role": "user", "content": prompt}],
    }
    temp = float(decoding.get("temperature", 0.0))
    if temp > 0.0:
        payload["temperature"] = temp
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
    }
    req = urllib.request.Request(
        f"{ANTHROPIC_API_BASE}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body["content"][0]["text"]
    return raw, None


def make_anthropic_judge(
    model_id: str,
    prompt_template: str,
    decoding: Optional[dict] = None,
    api_key: Optional[str] = None,
    max_retries: int = 1,
    judge_id_suffix: str = "",
) -> Judge:
    """Factory: create a Judge backed by the Anthropic Messages API.

    Uses stdlib urllib — no anthropic-sdk dependency required.
    Reads ANTHROPIC_API_KEY from the environment when api_key is not given.

    Note on determinism: Anthropic API does not guarantee identical outputs
    across calls even at temperature=0. Do NOT use this for the intra-model
    positive control (which needs bit-identical output via fixed seed).
    Use it for the inter-capability and cross-type axes only.

    Example — inter-capability axis:
        haiku  = make_anthropic_judge("claude-haiku-4-5-20251001", prompt)
        sonnet = make_anthropic_judge("claude-sonnet-4-6",          prompt)
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. "
            "Set ANTHROPIC_API_KEY env var or pass api_key=."
        )
    if decoding is None:
        decoding = {"temperature": 0.0, "max_tokens": 256}

    def call_fn(prompt_text: str, dec: dict, seed: int) -> Tuple[str, None]:
        return _anthropic_call(prompt_text, dec, seed, model_id=model_id, api_key=api_key)

    judge = Judge(
        model_id=model_id,
        decoding=dict(decoding),
        prompt_template=prompt_template,
        call_fn=call_fn,
        max_retries=max_retries,
    )
    if judge_id_suffix:
        judge.judge_id = f"{model_id}{judge_id_suffix}:{judge.prompt_hash}"
    return judge
