# ⚠ NOT AN H2 RESULT — instrument-validation smoke run

**`tables_all.md` in this directory is titled "Table 3 — Judge pairs" and shows
φ = nan, n = 3. Do not cite it as a measurement.**

It is not the paper's Table 3. It is the 5-item smoke run that excluded
`deepseek-coder:6.7b` from the registered H2 pair: **2 of 5 verdicts came back
INVALID** (deferred judgment — the model describes what a judge *should* do
instead of committing to PASS/FAIL), leaving n = 3 valid items and a degenerate
φ of `nan`. `qwen2.5-coder:7b` returned 5/5 clean verdicts on the same items.

**The pair was excluded, so no H2 measurement was ever taken.** H2 is reported as
**NOT COMPLETED**.

The empty Anthropic call log from the original inter-capability axis (Haiku ×
Sonnet, abandoned at zero credit balance) is not published — it contains no calls.

See: [`../../README.md`](../../README.md) § H2,
[`../../PREREG_AMENDMENT_2026_05_31.md`](../../PREREG_AMENDMENT_2026_05_31.md)
Amendments 4–5, [`../../findings/judge_reliability_6b.md`](../../findings/judge_reliability_6b.md).
