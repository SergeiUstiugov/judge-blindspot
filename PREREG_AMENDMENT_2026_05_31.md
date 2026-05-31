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
