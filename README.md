# judge-blindspot

**Do two clones of the same LLM judge miss the same defects?**

This repository is the measurement behind one claim: when you ask "the same model
again" for a second opinion on generated code, you are not buying a second opinion.
On the classes measured here, the two runs missed **exactly the same items** —
φ = +1.000, zero disagreements out of 158 judged items.

Companion code and data for the paper *"Do LLM Judges Share Blind Spots?"*
and for the pipeline article that cites this repo as its correlation measurement.

> **Кратко (RU):** замер коррелированности ошибок двух клонов одного LLM-судьи.
> Результат H1: φ = +1.000, ни одного расхождения между прогонками. Основной,
> обеспеченный мощностью класс — `wrong_operator` (n = 82, вердикт DUPLICATE);
> второй класс `off_by_one` (n = 76) даёт ту же картину, но UNDERPOWERED, и по
> пред-регистрации его вердикт — INCONCLUSIVE. H2 — NOT COMPLETED, H3 — NOT
> EVALUABLE; причины ниже, ничего не спрятано. Все числа берутся verbatim из
> `results/`, таблица источников — в конце.

---

## What is measured

For a pair of judges (i, j) and a fixed defect class, every corpus item is scored as
a **miss** or **not-miss** against ground truth (hidden tests). This gives two binary
miss-vectors, and their 2×2 contingency table:

|            | j not-miss | j miss |
|------------|-----------|--------|
| **i not-miss** | N00 | N01 |
| **i miss**     | N10 | N11 |

The statistic is the **φ coefficient** (Matthews correlation on the miss-vectors):

* **φ = 0** — the judges' misses are independent. A second judge genuinely adds
  information, and stacking N judges drives the false-accept risk down.
* **φ = +1** — the judges miss the *same* items. The second judge is an echo. You
  pay twice and learn nothing.

A bootstrap CI (2000 resamples, percentile method, seed = 0) and a pre-registered
decision rule turn φ into a verdict.

### Pre-registered decision rule (locked before any data collection)

| Verdict | Condition |
|---|---|
| **DUPLICATE** | φ ≥ 0.7 AND CI half-width ≤ 0.15 |
| **INDEPENDENT** | φ CI covers 0 AND ratio CI covers 1 |
| **OVERLAP** | CI lower bound > 0 AND φ < 0.7 |
| **INCONCLUSIVE** | CI half-width > 0.20 OR n < 30 |

---

## Results

### H1 — intra-model axis (same model, two independent runs): **DUPLICATE**

Judge pair: `qwen2.5-coder:7b` × `qwen2.5-coder:7b_clone` (same Ollama model, same
prompt, `judge_seed = 42`, run twice as two separate judges).

| Defect class | φ | φ 95% CI | ratio | N00 | N01 | N10 | N11 | n | verdict (artifact) | registered verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `wrong_operator` | +1.000 | [+1.000, +1.000] | 1.49 [1.30, 1.74] | 27 | 0 | 0 | 55 | 82 | DUPLICATE | **DUPLICATE** (powered, 82 ≥ 80) |
| `off_by_one` | +1.000 | [+1.000, +1.000] | 1.55 [1.33, 1.85] | 27 | 0 | 0 | 49 | 76 | DUPLICATE | **INCONCLUSIVE** (UNDERPOWERED, 76 < 80) |

Sources: [`results/h1_wrong_operator/results_wrong_operator.json`](results/h1_wrong_operator/results_wrong_operator.json),
[`results/h1_off_by_one/results_off_by_one.json`](results/h1_off_by_one/results_off_by_one.json).

**The number that matters is N01 = N10 = 0.** Across both classes, in 158 judged
items, the two runs never disagreed once — not a single item where one caught the
defect and the other did not. The judges missed 104 items and they were the *same*
104 items.

**The two verdict columns differ, and the difference is disclosed rather than
resolved in our favour.** The `verdict` field written into the results JSON applies
only the φ/CI decision rule, which returns DUPLICATE for both classes. The
pre-registration adds a second, stricter rule on top of it:

> Classes with n < `target_n_h1` (80) are flagged as UNDERPOWERED […] The verdict
> for underpowered classes is automatically INCONCLUSIVE regardless of observed phi.
> — `PREREG_AMENDMENT_2026_05_31.md`, "Honest underpowered reporting rule"

That override is **not implemented in the reporting code**, so the artifact for
`off_by_one` (n = 76 < 80) carries DUPLICATE while the pre-registered verdict for it
is **INCONCLUSIVE**. The registered verdict governs. `off_by_one` is therefore a
*supporting* observation, not a second confirmed result.

**The headline claim rests on `wrong_operator` alone** (n = 82, powered): φ = +1.000,
verdict DUPLICATE under both rules. `off_by_one` points the same way with the same
zero-disagreement pattern, but is underpowered and must not be counted as
independent confirmation.

Supporting numbers, same sources: miss marginals 0.671 / 0.671 (`wrong_operator`)
and 0.645 / 0.645 (`off_by_one`); shuffle-null φ 0.0028 and 0.0010 respectively
(the label-shuffled control lands at ≈ 0, as it should).

Per-call reasoning traces for every judged item are in
`results/h1_*/logs/*.jsonl` — one JSON record per call with `item_id`,
`judge_id`, `verdict`, and the model's full `raw_output`. Nothing is summarized
away; you can re-derive the tables from the logs.

### H2 — inter-family axis: **NOT COMPLETED**

Registered pair (Amendment 4): `qwen2.5-coder:7b` × `deepseek-coder:6.7b`.
**No H2 measurement was ever taken.** Reason: instrument failure — every candidate
second judge failed validation *before* measurement, so there was no reliable
instrument to measure with.

| Candidate | Smoke run | Outcome |
|---|---|---|
| `deepseek-coder:6.7b` | 5 items | **2/5 INVALID** — deferred judgment: the model describes what a judge *should* do instead of committing to PASS/FAIL. `qwen` on the same 5 items: 5/5 clean. Excluded. |
| `codegen:latest` (replacement) | 20 items | **2/20** code-content hallucinations (`factorial_00`, `count_in_range_01`). Excluded. |
| `qwen2.5-coder:7b` (incumbent) | 20 items | **2/20** — 1 code-content hallucination (`count_in_range_00`) + 1 evaluated-own-fix (`fibonacci_00`: correctly identifies the `a-b` defect, then judges a hypothetical *corrected* version). Excluded. |

Both parser-invisible failure modes produce syntactically valid `VERDICT: PASS/FAIL`
tokens — `_parse_verdict` cannot see them; they are only detectable by reading the
reasoning by hand.

The original H2 axis (inter-capability, Claude Haiku × Sonnet) became unavailable
for an unglamorous reason: zero API credit balance. The axis was re-registered to
inter-family *before* any H2 data existed (Amendment 4), and then that pair failed
validation too (Amendment 5). We did not substitute a third pair and we did not
report a number.

Details: [`PREREG_AMENDMENT_2026_05_31.md`](PREREG_AMENDMENT_2026_05_31.md)
(Amendments 4 and 5), [`findings/judge_reliability_6b.md`](findings/judge_reliability_6b.md).
Smoke artifacts: `results/h2_smoke/`, `results/h2_smoke20/`, `results/h2_smoke_codegen/`.

> A φ value does appear in `results/h2_smoke20/results_all.json` (φ = +0.567,
> CI [+0.185, +0.899], n = 19, verdict INCONCLUSIVE). **This is not an H2 result.**
> It is an instrument-validation smoke run on a 20-item corpus with a judge pair
> that failed validation. It is published for completeness, not as a finding, and
> must not be cited as a measurement of the inter-family axis.

### H3 — cross-type axis (LLM judge × deterministic checker): **NOT EVALUABLE**

Expected φ ≈ 0 — this was designed as the orthogonal control. It cannot be run on
this corpus, and the reason is structural, not fixable by trying harder:

* **Test runner (`verify_gt`)** — zero misses *by construction*: it defines ground
  truth. Its miss-vector is all-zeros → zero variance → φ undefined.
* **Static checker (ruff / pylint)** — **0/20 detections** across all four defect
  classes (logged 2026-06-04). The corpus is semantic mutations: syntactically valid
  Python where the bug is in the logic. That is outside a linter's scope entirely →
  constant miss-vector → φ degenerate.

Both candidate second judges yield NaN or degenerate φ. H3 is documented as a
**limitation of the original design**; no post-hoc metric was substituted for it.

Consequence, locked as Variant B: `calibrate --judges <real>` prints the orthogonal
skip verbatim and reports `gate: POSITIVE-ONLY` — never `PASS`. The one-sidedness of
the gate is visible in the output, not buried.

Details: [`PREREG_AMENDMENT_2026_05_31.md`](PREREG_AMENDMENT_2026_05_31.md), Amendment 2.

### What this does and does not show

* It **does** show that two runs of one 7B code model are a single judge wearing two
  hats — on one powered semantic-defect class (`wrong_operator`, n = 82), with no
  disagreement at all, and the same pattern on a second, underpowered class.
* It does **not** show anything about judges from different families — that is
  exactly the measurement that failed (H2), and the honest status is *unknown*.
* It does **not** establish that φ measures error-diversity cleanly in general. If
  both judges hallucinate on the same item, φ conflates shared hallucination with
  shared blind spots; the two cannot be separated post-hoc without ground-truth
  reasoning labels. See `findings/judge_reliability_6b.md`.
* The H1 pair is intra-model by design, so φ = +1 is the *expected* direction. The
  informative part is the magnitude: not "high correlation" but **zero disagreement**.

---

## Reproducing

```bash
git clone https://github.com/SergeiUstiugov/judge-blindspot
cd judge-blindspot

python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 1. Verify the meter before trusting any result

```bash
judge-blindspot selftest      # mock judges: same seed -> phi=+1, different seeds -> phi~0
judge-blindspot doctor        # environment check
python -m pytest              # 177 tests
```

`selftest` is the point of the whole exercise: if the instrument cannot produce
φ ≈ 0 on data engineered to be independent, no φ = +1 from it means anything.

### 2. Re-derive the published tables from the published logs

No model calls, no GPU — this reads the committed JSONL traces:

```bash
judge-blindspot report results/h1_wrong_operator/results_wrong_operator.json \
  --out results/h1_wrong_operator/ --class wrong_operator
judge-blindspot report results/h1_off_by_one/results_off_by_one.json \
  --out results/h1_off_by_one/ --class off_by_one
```

### 3. Re-run the measurement end to end

Requires [Ollama](https://ollama.com) with `qwen2.5-coder:7b` pulled locally.

```bash
ollama pull qwen2.5-coder:7b

judge-blindspot run \
  --corpus data/full_corpus.jsonl \
  --judges ollama:qwen2.5-coder:7b \
  --class wrong_operator \
  --judge-seed 42 --seed 0 --n-boot 2000 \
  --out results/repro_wrong_operator/
```

> **Reproducibility caveat — read this before comparing n.**
> The H1 runs were executed against the **276-item** version of
> `data/full_corpus.jsonl` (`run_manifest.json` → `"n_items": 276`, runs dated
> 2026-06-05 and 2026-06-09). The corpus at `HEAD` was later expanded to **509
> items** under Amendment 3 (195 `wrong_operator` / 97 `off_by_one` / 90
> `dropped_guard` / 37 `swapped_args` / 90 correct). Running the command above on
> today's corpus will **not** reproduce n = 82 / n = 76.
> To reproduce the published H1 numbers exactly, restore the corpus as it was:
>
> ```bash
> git show 40030c1:data/full_corpus.jsonl > data/full_corpus_276.jsonl
> ```
>
> and pass `--corpus data/full_corpus_276.jsonl`. LLM judging is not bit-exact even
> at fixed seed, so expect small drift in n_valid; the published logs are the
> authoritative record of what was actually observed.

All run parameters live in [`experiment.yaml`](experiment.yaml) — the single source
of truth. Every number in this README comes from a committed artifact; none are
hardcoded anywhere.

---

## Repository layout

```
judge_blindspot/    core package (the code; installed as the judge-blindspot CLI)
  stats.py            phi, ratio, bootstrap CI, pairwise report
  verdict.py          pre-registered decision rule
  corpus.py           CorpusItem schema, GT verification via hidden tests
  mutate.py           mutation verification (defective vs equivalent mutant)
  judges.py           Ollama + Anthropic adapters, verdict parsing
  mock_judges.py      deterministic mock judge (selftest instrument)
  deterministic_checker.py  test-runner-as-judge (H3 candidate)
  runner.py           run_judges, build_miss_matrix
  power.py            power simulation
  cli.py              CLI entry point
scripts/            standalone scripts (run_power_sim.py)
tests/              177 tests
prompts/            judging prompt templates (strict_passfail / rubric / confidence)
data/               corpora (full_corpus.jsonl + smoke sets)
results/            measurement artifacts — see table below
findings/           post-hoc, hypothesis-generating notes (explicitly not registered)
PREREG_AMENDMENT_2026_05_31.md   pre-registration and all five amendments
experiment.yaml     run parameters
```

The code lives in `judge_blindspot/` and `scripts/` rather than a `code/` folder,
because `judge_blindspot/` is an installable package with a console entry point —
renaming it would break `pip install -e .`, the CLI, and every test import.

### Published artifacts

| Directory | What it is |
|---|---|
| `results/h1_wrong_operator/` | H1, `wrong_operator` — **primary result**. JSON, table, forest plot, full per-call logs |
| `results/h1_off_by_one/` | H1, `off_by_one` — **primary result**. Same artifact set |
| `results/h2_smoke/` | deepseek-coder:6.7b validation smoke (5 items) — why it was excluded |
| `results/h2_smoke20/` | qwen × codegen validation smoke (20 items) — **not** an H2 result |
| `results/h2_smoke_codegen/` | codegen replacement smoke |
| `results/pilot_intra3/` | pilot that motivated `target_n_h1 = 80` (n = 10, INCONCLUSIVE by rule) |
| `results/power_sim/` | power simulation: 160-row table, `target_n = 170` |

Not published, and why: `results/calib_smoke/` and `results/calib_real/` (calibration
runs predating the Amendment 2 Variant B gate semantics — the stored `gate` value no
longer means what the current code means by it, so publishing it would mislead);
`results/pilot_intra/`, `results/pilot_intra2/` (aborted pilots, call logs only, no
summary artifact — superseded by `pilot_intra3`); one zero-byte log from the aborted
Anthropic attempt.

---

## Number → source

Every figure in this README, and where it comes from. Nothing is recomputed here.

| Number | Source file |
|---|---|
| φ = +1.000, CI [+1.000, +1.000], n = 82, N00 = 27 / N01 = 0 / N10 = 0 / N11 = 55, ratio 1.49 [1.30, 1.74], verdict DUPLICATE, marginals 0.671 / 0.671, shuffle-null 0.0028 | `results/h1_wrong_operator/results_wrong_operator.json`, rendered in `tables_wrong_operator.md` |
| φ = +1.000, CI [+1.000, +1.000], n = 76, N00 = 27 / N01 = 0 / N10 = 0 / N11 = 49, ratio 1.55 [1.33, 1.85], verdict DUPLICATE, marginals 0.645 / 0.645, shuffle-null 0.0010 | `results/h1_off_by_one/results_off_by_one.json`, rendered in `tables_off_by_one.md` |
| 158 judged items, 104 shared misses, zero disagreements | 82 + 76 and 55 + 49 from the two files above |
| Corpus version used by the H1 runs: 276 items | `results/h1_*/run_manifest.json` → `n_items` |
| n_boot = 2000, seed = 0, judge_seed = 42, prompt `prompts/strict_passfail.txt` | `results/h1_*/run_manifest.json` |
| Corpus at HEAD: 509 items (195 / 97 / 90 / 37 / 90 correct) | `data/full_corpus.jsonl` |
| `target_n_h1` = 80 (H1); underpowered classes → verdict automatically INCONCLUSIVE | `PREREG_AMENDMENT_2026_05_31.md`, "Decision (locked 2026-05-31)" table + "Honest underpowered reporting rule" |
| `off_by_one` n = 76 < 80 → UNDERPOWERED → registered verdict INCONCLUSIVE (artifact field says DUPLICATE; override not implemented in code) | `results/h1_off_by_one/results_off_by_one.json` (n, verdict) vs. the rule above |
| `target_n` = 170 | `results/power_sim/power_summary.json` → `target_n` |
| deepseek-coder:6.7b — 2/5 INVALID; qwen 5/5 clean on the same items | `findings/judge_reliability_6b.md`; `results/h2_smoke/run_manifest.json` → `n_items` 5, `n_valid` 3 |
| qwen 2/20 (1 hallucination + 1 evaluated-own-fix); codegen 2/20 hallucinations | `findings/judge_reliability_6b.md`, rates table |
| H2 smoke φ = +0.567, CI [+0.185, +0.899], n = 19, INCONCLUSIVE (**not** an H2 result) | `results/h2_smoke20/results_all.json` |
| ruff 0/20 detections across all four defect classes | `PREREG_AMENDMENT_2026_05_31.md`, Amendment 2 |
| 177 tests | `python -m pytest --collect-only -q` |

---

## License

MIT — see [LICENSE](LICENSE).
