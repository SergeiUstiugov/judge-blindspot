# Finding: 6-7B Code-LLMs as Binary Judges — Distinct Failure Modes

**Date:** 2026-06-11  
**Status:** POST-HOC — hypothesis-generating only. NOT a registered metric.  
**Frame:** C-strict. This finding does not modify pre-registered hypotheses H1–H3.  
**Reference from:** PREREG_AMENDMENT_2026_05_31.md, Amendment 5

---

## Context

During instrument validation for H2 (inter-family judge pair, Amendment 4),
two models from different training lineages were evaluated as binary judge
candidates on the wrong_operator class (prompt: `prompts/strict_passfail.txt`).
Both failed instrument validation before any H2 measurement was taken.

---

## Evidence

### deepseek-coder:6.7b — deferred-judgment (instruction-following failure)

Smoke run: 5 wrong_operator items (`data/smoke5_wrong_operator.jsonl`).  
Result: **2/5 INVALID** verdicts.

Pattern: instead of committing to PASS or FAIL, the model described what a
judge *should* do:

> "If any of these tests fail, then VERDICT: FAIL should be returned.
>  If all tests pass, then VERDICT: PASS should be returned."

`_parse_verdict` correctly rejects this — the token after `"VERDICT:"` is
`"FAIL SHOULD BE RETURNED"`, which does not match `"PASS"` or `"FAIL"`.
Parser behavior is correct per prompt specification.

`qwen2.5-coder:7b` on the same 5 items: **5/5 clean verdicts**.  
This is a **differential** finding — not a general property of 7B code models.
deepseek-coder:6.7b specifically fails to close judgment; qwen does not.

---

### qwen2.5-coder:7b — two failures, two distinct mechanisms

Smoke run: 20 wrong_operator items (`data/smoke20_wrong_operator.jsonl`,
`--judge-seed 42`).  
Result: **2/20 failures**. Both produce syntactically valid verdict tokens;
invisible to `_parse_verdict`; detectable only by manual reasoning inspection.

**Failure 1 — code-content hallucination (`count_in_range_00`):**  
The model asserts a boundary condition the actual code does not have.
The reasoning misrepresents what the submitted code does — a hallucination
of code content.

**Failure 2 — evaluated-own-fix (`fibonacci_00`):**  
The model correctly identifies `a-b` as the defect (accurate code reading),
then reasons about a hypothetical corrected version and judges *that* instead
of the actual submitted code. This is a distinct failure mode: the model
substitutes what the code *should be* for what it *is*. The verdict is
rendered on a code that was never submitted.

These two mechanisms are different and should not be collapsed:

| Item | Mechanism | Code read correctly? | Verdict rendered on |
|---|---|---|---|
| count_in_range_00 | code-content hallucination | No | Misrepresented code |
| fibonacci_00 | evaluated-own-fix | Yes | Hypothetical fixed version |

**qwen narrow hallucination rate (code-content only): 1/20 (~5%)**  
**qwen broad failure rate (any reasoning error): 2/20 (~10%)**

---

### codegen:latest — two failures, code-content hallucination

Smoke run: same 20 items (`data/smoke20_wrong_operator.jsonl`, `--judge-seed 42`).  
Result: **2/20 failures**. Both are code-content hallucinations.

**Failure 1 — `factorial_00`:**  
The model asserts the code "multiplies" when the actual implementation uses
`result += i` (addition). Direct misreading of the operator.

**Failure 2 — `count_in_range_01`:**  
The model asserts boundary inclusion the code does not implement.
Same category as qwen's count_in_range_00.

**codegen code-content hallucination rate: 2/20 (~10%)**  
Both codegen failures are hallucination in the narrow sense. No evaluated-own-fix
observed in codegen.

---

## Failure mode taxonomy (this corpus)

Three distinct failure modes identified across the three models:

| Mode | Description | Models observed | Parser-visible? |
|---|---|---|---|
| deferred-judgment | Describes conditions instead of committing verdict | deepseek-coder:6.7b | Yes (INVALID) |
| code-content hallucination | Misrepresents what the code actually does | qwen (×1), codegen (×2) | No |
| evaluated-own-fix | Correctly reads code, then judges hypothetical fix instead | qwen (×1) | No |

The two parser-invisible modes are mechanistically different:
- Hallucination: wrong input to the judgment (misread code)
- Evaluated-own-fix: correct input, wrong object judged (substituted code)

---

## Cross-family character

The three models span three distinct training lineages:

| Model | Organization | Family |
|---|---|---|
| qwen2.5-coder:7b | Alibaba (Qwen team) | Qwen |
| codegen:latest | Salesforce | CodeGen |
| deepseek-coder:6.7b | DeepSeek | DeepSeek |

Parser-invisible failures appear in both qwen (Alibaba) and codegen (Salesforce) —
different pre-training corpora, different fine-tuning pipelines. The pattern
recurs across families, suggesting a capability-level property of ~7B
code-specialized models rather than a family-specific artifact.

The core finding is NOT "both models hallucinate at 10%". It is that both
models are unreliable, through different mechanisms. qwen fails via
evaluated-own-fix plus narrow hallucination; codegen fails via code-content
hallucination. Multiple distinct failure modes across families is the stronger
and more honest characterization.

---

## Rates summary

| Model | Code-content hallucination | Evaluated-own-fix | Total parser-invisible failures |
|---|---|---|---|
| qwen2.5-coder:7b | 1/20 (~5%) | 1/20 (~5%) | 2/20 (~10%) |
| codegen:latest | 2/20 (~10%) | 0/20 (0%) | 2/20 (~10%) |

Total broad failure rates are accidentally equal. Narrow rates differ. Mechanisms differ.
Do not report as "both 10% hallucination" — that collapses distinct phenomena.

---

## Implication for φ measurement

If double-faults (both judges fail simultaneously) are partially driven by
shared hallucination patterns (both misread the same operator boundary), then
a measured φ > 0 would reflect **independence of hallucination patterns**,
not independence of judgment errors on genuine code defects. The two quantities
are confounded and cannot be separated post-hoc without ground-truth reasoning labels.

The evaluated-own-fix mode adds a further confound: if qwen systematically
substitutes its own fix on cases where codegen also hallucinates, they may
correlate or anti-correlate on those items independently of the underlying defect.

This does not invalidate φ as a measurement instrument in general — it identifies
specific conditions under which φ conflates distinct phenomena.

---

## Hypothesis generated (NOT registered)

> **H-reliability:** 6-7B code-specialized LLMs exhibit at least three distinct
> judge-failure modes (deferred-judgment; code-content hallucination;
> evaluated-own-fix) that are (a) partially model-family differential,
> (b) invisible to verdict-string parsing for the two reasoning-failure modes,
> and (c) present at non-negligible rates (~5–10%) on wrong_operator semantic
> mutations.
>
> A robust automated judge instrument for semantic-mutation binary evaluation
> requires either larger models (>13B), chain-of-thought verification steps
> with explicit code-quotation requirements, or adversarial prompt calibration
> that forces the model to quote the evaluated code before rendering a verdict —
> none of which are available in this experiment's scope.

This is a **hypothesis-generating observation** from instrument validation.
It is not tested against a pre-registered decision rule and does not modify
any registered hypothesis. It belongs in the paper's Limitations section and
may serve as motivation for future work on judge calibration.

---

## Paper placement (Limitations)

> During instrument validation for H2, we evaluated three 6-7B
> coding-specialized models as binary judge candidates. deepseek-coder:6.7b
> produced deferred-judgment verdicts (2/5 INVALID; model describes testing
> methodology rather than committing to PASS/FAIL). qwen2.5-coder:7b and
> codegen:latest each failed in 2/20 smoke items, but through distinct
> mechanisms: qwen produced 1 code-content hallucination (count_in_range_00:
> boundary condition misread, ~5% narrow rate) and 1 evaluated-own-fix error
> (fibonacci_00: correctly identified the defect `a-b`, then judged a
> hypothetical corrected version rather than the submitted code); codegen
> produced 2 code-content hallucinations (factorial_00, count_in_range_01:
> operator semantics and boundary inclusion misread, ~10% narrow rate). Total
> parser-invisible failure rates are 2/20 for both models, but the underlying
> mechanisms differ. All parser-invisible failures produced syntactically valid
> verdict tokens invisible to `_parse_verdict`. We document H2 as NOT COMPLETED
> due to instrument unavailability, not design failure. The finding is
> hypothesis-generating: reliable automated binary judgment on semantic
> mutations may require models above 7B, explicit code-quotation in the
> reasoning chain, or adversarial calibration — and the failure modes to guard
> against are not uniform across model families.
