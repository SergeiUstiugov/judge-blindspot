# ⚠ NOT AN H2 RESULT — instrument-validation smoke run

**`tables_all.md` in this directory is titled "Table 3 — Judge pairs" and shows
φ = +0.567, CI [+0.185, +0.899], n = 19. Do not cite it as a measurement.**

It is not the paper's Table 3. It is a 20-item smoke run used to decide whether
`qwen2.5-coder:7b` × `codegen:latest` was usable as a judge pair at all.

**The pair failed validation and was excluded** (qwen: 1 code-content
hallucination + 1 evaluated-own-fix; codegen: 2 code-content hallucinations —
2/20 each). Because the instrument failed, **no H2 measurement was ever taken**,
with this pair or any other. H2 is reported as **NOT COMPLETED**.

n = 19 is also below the pre-registered minimum of n ≥ 30, so the verdict is
INCONCLUSIVE by rule regardless of φ.

This run is published because deleting inconvenient artifacts is how repositories
start lying — not because it measures anything.

See: [`../../README.md`](../../README.md) § H2,
[`../../PREREG_AMENDMENT_2026_05_31.md`](../../PREREG_AMENDMENT_2026_05_31.md)
Amendments 4–5, [`../../findings/judge_reliability_6b.md`](../../findings/judge_reliability_6b.md).
