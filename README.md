# judge-blindspot

Measures whether LLM judges share blind spots — miss-correlation per defect
class across judge pairs.

Companion code for the paper *"Do LLM Judges Share Blind Spots?"*

> **Status: pre-registered, real-model runs pending.**
> The measurement infrastructure (Phases 0–6) is complete and tested.
> Table 2 (calibration) and Table 3 (judge pairs) will be populated once
> the real-model corpus and judging runs are approved and executed.
> All numbers in the paper come from logged runs; none are hardcoded.

---

## Quick start

```bash
pip install -e .
pip install pylint ruff flake8   # optional: for the linter probe

# Verify the meter works on synthetic data (required before trusting any result)
judge-blindspot selftest

# Check your environment
judge-blindspot doctor

# Build and verify the smoke corpus
judge-blindspot build-corpus --smoke

# Run the mock pipeline end-to-end
judge-blindspot run --corpus data/synthetic_smoke.jsonl --judges mock --out results/smoke/

# Phase 5 calibration gate (mock)
judge-blindspot calibrate --corpus data/synthetic_smoke.jsonl --judges mock
```

---

## Repository layout

```
judge_blindspot/    core package
  stats.py          phi, ratio, bootstrap CI, pairwise report
  verdict.py        pre-registered decision rule (INDEPENDENT / OVERLAP / DUPLICATE / INCONCLUSIVE)
  corpus.py         CorpusItem schema, GT verification via hidden tests
  mutate.py         mutation verification (defective vs equivalent mutant)
  smoke_corpus.py   10 hand-authored tasks, 20 items (50/50 balance)
  mock_judges.py    deterministic mock judge (same seed → phi=+1; different seeds → phi≈0)
  runner.py         run_judges, build_miss_matrix
  power.py          Phase 6 power simulation (vectorized bootstrap)
  cli.py            CLI entry point

prompts/            judging prompt templates
  strict_passfail.txt   main prompt (Phase 2)
  rubric.txt            rubric variant (Phase 7)
  confidence.txt        confidence variant (Phase 7)

data/
  synthetic_smoke.jsonl   verified smoke corpus (20 items)

scripts/
  run_power_sim.py    Phase 6 power simulation script

results/power_sim/  Phase 6 output (committed)
  power_results.csv
  power_table.md
  power_summary.json

experiment.yaml     single source of truth for all run parameters
```

---

## Pre-registered decision rule

Verdict for a judge pair (locked before any data collection):

| Verdict | Condition |
|---|---|
| **DUPLICATE** | φ ≥ 0.7 AND CI half-width ≤ 0.15 |
| **INDEPENDENT** | φ CI covers 0 AND ratio CI covers 1 |
| **OVERLAP** | CI lower bound > 0 AND φ < 0.7 |
| **INCONCLUSIVE** | CI half-width > 0.20 OR n < 30 |

Bootstrap: 2000 resamples, percentile method, seed = 0.

---

## Phase 6 power simulation

Target n per defect class: **170** (worst-case scenario: φ=0.3, marginals 0.2/0.2).
Full results in `results/power_sim/power_table.md`.

---

## License

MIT — see [LICENSE](LICENSE).
