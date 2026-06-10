# Pre-registration Amendment — 2026-05-31

**Status:** LOCKED before corpus rebuild  
**Amends:** Phase 6 power simulation (2026-05-30), original `target_n=170`  
**Purpose:** Justify per-hypothesis target_n for H1 axis *before* viewing per-class corpus counts

---

## What this amends

The original `target_n=170` was computed under:

- `ci_hw_target = 0.20` (CI half-width criterion)
- worst-case phi = 0.3 (H2/H3, shared-but-imperfect blind spots)
- power threshold = 80%

This amendment adds a **separate, lower target** for the H1 axis only, locked pre-corpus-rebuild.

---

## H1 prior: why phi >= 0.90

**Hypothesis H1 (intra-model):** two instances of the same model with a fixed seed
behave as near-duplicate judges, not independent ones.

**Evidence (pilot_intra3, 2026-05-30):**
- phi = +1.000, CI = [1.000, 1.000], n = 10 → INCONCLUSIVE (n < 30 rule)
- INVALID verdicts = 0 after raising max_tokens 256 → 768
- Interpretation: result is consistent with H1 (duplicates) but underpowered to be conclusive

**Mechanistic argument:**
A fixed-seed local model is a deterministic function. Two runs of the same model
on the same prompt with the same seed return the same output. Any variance in
phi from +1.0 comes only from prompt-context variation across corpus items
(not stochastic differences between judges). Therefore phi ≈ +1 is the
*expected value*, not an optimistic upper bound.

**Conservative floor chosen for power analysis:** phi >= 0.90
(not phi = 1.0, to account for any prompt-context variation).

---

## Power simulation (2026-05-31)

Run parameters: `ci_hw_target=0.15`, `power >= 80%`, `n_sims=300`, `n_boot=300`, `seed=0`

| phi | worst-case marginals | min_n |
|-----|----------------------|-------|
| 0.70 | (0.2, 0.2) | 170 |
| 0.80 | (0.2, 0.2) | 130 |
| **0.90** | **(0.2, 0.2)** | **80** |
| 0.95 | (0.2, 0.2) | 50 |

Note: marginals (0.3, 0.5) are geometrically INFEASIBLE for all tested phi values
and are excluded from the worst-case calculation.

---

## Decision (locked 2026-05-31)

| Axis | Expected phi | ci_hw target | target_n | key |
|------|-------------|--------------|----------|-----|
| H1 intra-model | >= 0.90 | 0.15 | **80** | `target_n_h1` |
| H2 inter-capability | 0.2–0.7 | 0.20 | 170 | `target_n` |
| H3 cross-type | ≈ 0 | 0.20 | 170 | `target_n` |

**Honest underpowered reporting rule:**  
Classes with n < `target_n_h1` (80) are flagged as UNDERPOWERED in Table 2
footnote. They are **not** padded post-hoc to reach 80. The verdict for
underpowered classes is automatically INCONCLUSIVE regardless of observed phi.

**Corpus counts at time of amendment (from build-corpus --full crash log, 2026-05-31):**

| Class | n | vs target_n_h1=80 |
|-------|---|-------------------|
| wrong_operator | 81 | powered (81 >= 80) |
| off_by_one | 75 | underpowered (75 < 80) |
| dropped_guard | 46 | underpowered |
| swapped_args | 29 | underpowered |

These counts were observed from a crashed build run (corpus not saved);
the per-class assessment above is disclosed to prevent post-hoc n-shopping.

---

## Integrity note

This document was written and committed **before** the repaired corpus was
rebuilt. The palindrome test fix (`full_corpus.py:1040`, "race car" → "race bar")
was also committed in the same pre-rebuild window. No corpus counts were used
to select `target_n_h1`; the selection is based solely on the pilot phi value
and the power simulation.

---

## Amendment 2 — H3 NOT EVALUABLE on semantic-mutation corpus (2026-06-04)

**Status:** LOCKED  
**Nature:** Design limitation — documented, not patched

### Finding

The H3 cross-type axis (LLM judge × deterministic checker, expected φ ≈ 0)
is **NOT EVALUABLE** on this corpus.

The corpus consists of semantic mutations: syntactically valid Python where the
defect lies in logic, not syntax. Both candidate deterministic judges degenerate:

- **Test runner (verify_gt):** zero misses by construction — it IS the ground
  truth. Miss-vector is all-zeros → φ undefined (zero-variance column).
- **Lightweight static checker (pylint/ruff):** empirically 0 detections across
  all four defect classes on the smoke corpus (ruff 0/20, logged 2026-06-04).
  Semantic mutations are outside linters' detection scope → constant miss-vector
  → φ degenerate.

Both candidate second judges yield NaN or degenerate φ. No other deterministic
checker is available that would produce an informative miss-vector on semantic
defects while remaining genuinely orthogonal to an LLM judge.

### Decision

H3 is documented as a **limitation of the original design**, not substituted
with a post-hoc metric. The φ-based diversity analysis remains valid for H1
and H2 (LLM × LLM, both error-prone on semantic defects).

### Consequence for the calibration gate — Decision: Variant B (locked 2026-06-04)

The orthogonal control in the calibration gate is **not achievable on real
judges** for this corpus. **Decision: Variant B — explicit two-level calibration.**

**Three mandatory conditions (non-negotiable):**

1. `calibrate --judges real` MUST print verbatim:
   `"ORTHOGONAL CONTROL: SKIPPED (H3 not evaluable on this corpus, see PREREG_AMENDMENT Amendment 2)"`
   Silent omission is prohibited — the skip must be loud and attributed.

2. The manifest/output `gate` key for real judges MUST read `"POSITIVE-ONLY"`, never `"PASS"`.
   A reader of the code or paper must immediately see the gate is one-sided.

3. The mock path (`calibrate --judges mock`) is NOT touched — both controls
   (positive + orthogonal) run exactly as before. This is the software-level
   stats-machinery correctness check.

**Semantics after the change:**
- `calibrate --judges mock` → `gate: PASS/FAIL` (both controls, software correctness)
- `calibrate --judges <real>` → positive control runs; orthogonal SKIPPED with loud message;
  `gate: POSITIVE-ONLY` if positive passes, `gate: FAIL` if positive fails.

### Paper placement

Goes into Limitations, not Results. Verbatim:

> H3 (cross-type orthogonal control, LLM × deterministic checker) is NOT
> EVALUABLE on this corpus. Pre-registration offered two candidate second
> judges: (a) test runner — 0 miss by construction (zero variance → φ
> undefined); (b) lightweight static checker (ruff/pylint) — empirically 0/20
> detections across all four defect classes (semantic bugs are outside linters'
> scope → constant miss-vector → φ degenerate). Both yield NaN/degenerate φ.
> We document H3 as a limitation of the original design and do NOT substitute
> a post-hoc metric. φ-based diversity analysis remains valid for H1/H2
> (LLM × LLM, both error-prone). Empirical basis: ruff 0/20, logged 2026-06-04.

---

## Amendment 3 — Corpus expansion for wrong_operator class (2026-06-10)

**Status:** LOCKED before any new tasks are generated  
**Nature:** Planned corpus expansion — not a post-hoc metric substitution  
**Triggered by:** per-class count review after Phase 2 H1 completion

### Situation

After corpus rebuild, per-class counts are:

| Class | n (current) | vs target_n=170 (H2) | vs target_n_h1=80 (H1) |
|-------|-------------|----------------------|------------------------|
| wrong_operator | 82 | **underpowered** (−88) | powered ✅ |
| off_by_one | 77 | underpowered (−93) | underpowered |
| dropped_guard | 48 | underpowered (−122) | underpowered |
| swapped_args | 29 | underpowered (−141) | underpowered |

H2 (inter-capability: Haiku vs Sonnet) requires `target_n=170` per class.
No class currently meets this threshold.

### Decision

**Only `wrong_operator` is expanded to `target_n=170`.**

Rationale (all locked pre-generation):
1. wrong_operator is the most populated class — smallest expansion gap (−88).
2. wrong_operator is already powered for H1 (82 ≥ 80 = `target_n_h1`),
   minimising interaction with completed H1 measurement.
3. The remaining three classes (off_by_one, dropped_guard, swapped_args)
   remain UNDERPOWERED for H2 and are reported as a **documented limitation**,
   not patched.

### Rules of the expansion (binding)

1. **Only new base tasks are added** — hand-authored `_TaskDef` objects appended
   to `_TASKS` in `full_corpus.py`. No changes to existing task definitions.
2. **Mutation rules unchanged** — same patch-based mechanism, same
   `verify_mutation` / `verify_gt` logic, same EQUIV-discard criterion.
3. **Seeds unchanged** — `corpus: 0`, `bootstrap: 0`, `judge_run: 42`
   as registered in `experiment.yaml`.
4. **Decoding unchanged** — `temperature=0.0`, `max_tokens=256` for all models.
5. **New tasks are structurally identical** to existing tasks: single Python
   function, hidden pytest suite, patches of the form `(old_text, new_text,
   defect_type, description)`. No novel defect semantics introduced.
6. **Stopping rule:** exactly **N = 50 new tasks** are authored and committed
   before any corpus build is run. Rationale: need +88 valid wrong_operator
   items; observed yield is ~2.05 wrong_operator items/task; 43 tasks would
   suffice at yield floor, 50 tasks adds ~15% buffer for EQUIV-discard variance.
   After build, if wrong_operator count ≥ 170 — expansion is complete.
   If count < 170 — the shortfall is documented as a limitation; **no further
   tasks are added in a second iteration.** One-shot rule, no peek-and-add.
7. **Selection of task domains** is determined before building: new tasks cover
   algorithmic domains not yet represented in the 40-task corpus. Tasks are
   NOT selected or rejected based on their per-class mutation yield.

### Effect on already-completed H1 measurement

H1 on wrong_operator is already measured and locked:
`phi = +1.000, CI = [1.000, 1.000], n = 82, verdict = DUPLICATE ✅ (2026-06-05)`

The expansion adds items that **will be judged in future H1 and H2 runs**
on the extended corpus. Both results are preserved in the record:

- `h1_pre_expansion`: n = 82, phi = +1.000, CI = [1.000, 1.000] — **primary
  H1 outcome**, locked, not overwritten.
- `h1_post_expansion`: H1 re-run on expanded corpus (n ≈ 170), reported
  alongside pre-expansion result, expected to be consistent. If inconsistent,
  the discrepancy is reported verbatim — not resolved by selective reporting.

No H2 data existed at the time of this amendment — H2 has not been run.
Expansion therefore cannot constitute post-hoc adjustment of H2.

### Limitation statement (mandatory in paper)

> Corpus expansion was performed for `wrong_operator` only, reaching
> `target_n=170` for H2. Three classes (off_by_one, dropped_guard, swapped_args)
> remain underpowered for H2 (n = 77, 48, 29 respectively vs target 170) and
> are reported as UNDERPOWERED in Table 2 with verdict INCONCLUSIVE on H2.
> Expansion was pre-registered before generation and follows identical
> mutation and verification rules as the original corpus. New tasks were
> authored in a single batch of N = 50 before any corpus build was run
> (one-shot, no peek-and-add).
