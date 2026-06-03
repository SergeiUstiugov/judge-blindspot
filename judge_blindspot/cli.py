# -*- coding: utf-8 -*-
"""CLI: judge-blindspot {selftest | doctor | build-corpus | run | calibrate | report}."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .stats import independence_report, bootstrap_ci, pairwise_report, _phi, _ratio
from .verdict import verdict_for_pair, apply_verdicts, VerdictLabel


# ─────────────────────────── selftest ─────────────────────────────────────────

def _run_selftest(n_boot: int = 2000, seed: int = 0) -> int:
    """Synthetic calibration gate: must both pass and fail by design."""
    rng = np.random.default_rng(seed)
    n = 200
    print("=" * 65)
    print("JUDGE-BLINDSPOT SELFTEST (synthetic; must both pass and fail)")
    print("=" * 65)

    # A: known-independent
    Ma = np.column_stack([
        (rng.random(n) < 0.30).astype(int),
        (rng.random(n) < 0.40).astype(int),
    ])
    phi_a, ratio_a, null_a = independence_report(Ma, n_boot=n_boot, seed=seed)
    phi_a_ci   = bootstrap_ci(Ma, lambda m: _phi  (m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed)
    ratio_a_ci = bootstrap_ci(Ma, lambda m: _ratio(m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed+1)
    v_a  = verdict_for_pair(phi_a, phi_a_ci, ratio_a_ci, n)
    ok_a = v_a in (VerdictLabel.INDEPENDENT, VerdictLabel.INCONCLUSIVE)

    print(f"\n--- A: deliberately INDEPENDENT (expected: INDEPENDENT or INCONCLUSIVE) ---")
    print(f"  phi   = {phi_a:+.3f}  CI [{phi_a_ci[0]:+.3f}, {phi_a_ci[1]:+.3f}]  shuffle≈{null_a:+.3f}")
    print(f"  ratio = {ratio_a:.3f}  CI [{ratio_a_ci[0]:.3f}, {ratio_a_ci[1]:.3f}]")
    print(f"  marginals: A={Ma[:,0].mean():.2f}  B={Ma[:,1].mean():.2f}")
    print(f"  → verdict: {v_a.value}  ({'OK' if ok_a else 'FAIL'})")

    # B: known-dependent
    hard = rng.random(n) < 0.40
    Mb = np.column_stack([
        (rng.random(n) < 0.15).astype(int),
        (rng.random(n) < 0.15).astype(int),
    ])
    Mb[hard, :] = 1
    phi_b, ratio_b, null_b = independence_report(Mb, n_boot=n_boot, seed=seed)
    phi_b_ci   = bootstrap_ci(Mb, lambda m: _phi  (m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed)
    ratio_b_ci = bootstrap_ci(Mb, lambda m: _ratio(m[:,0].astype(int), m[:,1].astype(int)), n_boot=n_boot, seed=seed+1)
    v_b  = verdict_for_pair(phi_b, phi_b_ci, ratio_b_ci, n)
    ok_b = v_b in (VerdictLabel.OVERLAP, VerdictLabel.DUPLICATE)

    print(f"\n--- B: deliberately DEPENDENT (expected: OVERLAP or DUPLICATE) ---")
    print(f"  phi   = {phi_b:+.3f}  CI [{phi_b_ci[0]:+.3f}, {phi_b_ci[1]:+.3f}]  shuffle≈{null_b:+.3f}")
    print(f"  ratio = {ratio_b:.3f}  CI [{ratio_b_ci[0]:.3f}, {ratio_b_ci[1]:.3f}]")
    print(f"  marginals: A={Mb[:,0].mean():.2f}  B={Mb[:,1].mean():.2f}")
    print(f"  → verdict: {v_b.value}  ({'OK' if ok_b else 'FAIL'})")

    ok = ok_a and ok_b
    print(f"\n{'=' * 65}")
    if ok:
        print("SELFTEST PASSED: tool distinguishes independent from dependent.")
        print("  (A→INDEPENDENT/INCONCLUSIVE, B→OVERLAP/DUPLICATE — as required)")
    else:
        print("SELFTEST FAILED: tool does NOT distinguish cases → do NOT run on data.")
    print("=" * 65)
    return 0 if ok else 1


# ─────────────────────────── doctor ───────────────────────────────────────────

def _run_doctor() -> int:
    print("=" * 55)
    print("JUDGE-BLINDSPOT DOCTOR — environment check")
    print("=" * 55)
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    enc = (sys.stdout.encoding or "").lower()
    print(f"Encoding: stdout={sys.stdout.encoding}")
    if enc and "utf" not in enc:
        print("  ! WARNING: stdout not UTF-8 — set PYTHONUTF8=1")

    all_ok = True
    for pkg in ("numpy", "pandas", "matplotlib", "yaml"):
        try:
            mod = __import__(pkg)
            print(f"  [OK ] {pkg}=={getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  [    ] {pkg} — NOT installed (required)")
            all_ok = False
    for pkg in ("httpx", "pytest"):
        try:
            mod = __import__(pkg)
            print(f"  [OK ] {pkg}=={getattr(mod, '__version__', '?')} (optional)")
        except ImportError:
            print(f"  [    ] {pkg} — not installed (optional)")

    print("")
    if all_ok:
        print("Environment OK — run `judge-blindspot selftest` to verify the meter.")
        return 0
    print("Install missing packages: pip install -e .[dev]")
    return 1


# ─────────────────────────── build-corpus ────────────────────────────────────

def _run_build_corpus(smoke: bool, full: bool, output: str) -> int:
    if smoke and full:
        print("Specify either --smoke or --full, not both.")
        return 1
    if smoke:
        from .smoke_corpus import build_smoke_corpus
        try:
            stats = build_smoke_corpus(output_path=output, verbose=True)
            print(f"\nBuild complete. Stats: {stats}")
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
    if full:
        from .full_corpus import build_full_corpus
        try:
            stats = build_full_corpus(output_path=output, verbose=True)
            if stats["task_errors"]:
                print(f"\nWARNING: {len(stats['task_errors'])} task errors:")
                for err in stats["task_errors"]:
                    print(f"  {err}")
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: {e}")
            return 1
    print("Specify --smoke or --full.")
    return 1


# ─────────────────────────── judge spec parser ────────────────────────────────

def _parse_judge_specs(
    judges_spec: str,
    prompt_template: str,
    judge_seed: int,
    api_key: str | None,
) -> list:
    """Parse --judges spec into exactly 2 Judge objects.

    Format: comma-separated list of ``provider:model_id``.
    Supported providers: ``mock``, ``ollama``, ``anthropic``.

    Rules:
    - ``mock`` → two MockJudges (seeds judge_seed and judge_seed+1).
    - Single ``ollama:`` or ``anthropic:`` spec → intra-model pair; the second
      judge gets a ``_clone`` suffix on its judge_id (same model + seed).
    - Two specs → independent pair; identical model_ids get ``_0``/``_1`` suffix
      to keep judge_ids unique in the results dict.

    Raises ValueError on unknown providers, bad format, or wrong count.
    """
    from .mock_judges import MockJudge
    from .judges import make_ollama_judge, make_anthropic_judge

    if judges_spec == "mock":
        return [
            MockJudge(error_rate=0.25, seed=judge_seed,     judge_id="mock_0"),
            MockJudge(error_rate=0.25, seed=judge_seed + 1, judge_id="mock_1"),
        ]

    parts = [p.strip() for p in judges_spec.split(",") if p.strip()]
    if not parts or len(parts) > 2:
        raise ValueError(
            f"Need 1 or 2 judge specs (got {len(parts)}). "
            "Format: provider:model_id[,provider:model_id]"
        )

    def _make_one(spec: str, suffix: str) -> object:
        if ":" not in spec:
            raise ValueError(
                f"Bad judge spec {spec!r}. Expected format: provider:model_id"
            )
        provider, model_id = spec.split(":", 1)
        provider = provider.lower()
        if provider == "ollama":
            return make_ollama_judge(
                model_id, prompt_template,
                decoding={"temperature": 0.0, "max_tokens": 768, "seed": judge_seed},
                judge_id_suffix=suffix,
            )
        if provider == "anthropic":
            return make_anthropic_judge(
                model_id, prompt_template,
                api_key=api_key,
                judge_id_suffix=suffix,
            )
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: mock, ollama, anthropic"
        )

    if len(parts) == 1:
        return [_make_one(parts[0], ""), _make_one(parts[0], "_clone")]

    # Two specs: add _0/_1 only when they would produce identical judge_ids.
    j0 = _make_one(parts[0], "")
    j1 = _make_one(parts[1], "")
    if j0.judge_id == j1.judge_id:
        j0 = _make_one(parts[0], "_0")
        j1 = _make_one(parts[1], "_1")
    return [j0, j1]


# ─────────────────────────── run ─────────────────────────────────────────────

def _run_pipeline(
    corpus_path: str,
    judges_spec: str,
    out_dir: str,
    defect_class: str | None,
    dry_run: bool,
    force: bool,
    n_boot: int,
    seed: int,
    prompt_path: str,
    api_key: str | None,
    judge_seed: int,
) -> int:
    from .corpus import load_corpus
    from .runner import run_judges, build_miss_matrix
    from .report import save_results, save_tables_md, save_forest_plot, save_manifest
    from .judges import load_prompt, is_ollama_available
    import datetime

    # Load prompt template (not needed for mock)
    prompt_template = ""
    if judges_spec != "mock":
        try:
            prompt_template = load_prompt(prompt_path)
        except FileNotFoundError:
            print(f"ERROR: prompt file not found: {prompt_path}")
            return 1

    # Pre-flight: Ollama availability
    if "ollama:" in judges_spec:
        if not is_ollama_available():
            print("ERROR: Ollama server not reachable. Start it with: ollama serve")
            return 1

    try:
        judges = _parse_judge_specs(judges_spec, prompt_template, judge_seed, api_key)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    corpus = load_corpus(corpus_path)
    out = Path(out_dir)

    n_calls = len(corpus) * len(judges)
    print(f"[info] corpus={len(corpus)}  judges={[j.judge_id for j in judges]}")
    print(f"[info] planned calls: {n_calls}")

    results = run_judges(corpus, judges, out / "logs", dry_run=dry_run,
                         budget_calls=None, force=force)
    if dry_run:
        return 0

    M, item_ids, judge_ids = build_miss_matrix(corpus, results, defect_class)
    valid_rows = ~np.any(np.isnan(M), axis=1)
    M_clean = M[valid_rows].astype(int)
    n = M_clean.shape[0]

    if n < 2:
        print(f"Too few valid rows ({n}) for pairwise analysis.")
        return 1

    pairwise = pairwise_report(M_clean, judge_ids, n_boot=n_boot, seed=seed)
    pairwise = apply_verdicts(pairwise)

    label = defect_class or "all"
    save_results(pairwise, out, label)
    save_tables_md(pairwise, out, label)
    save_forest_plot(pairwise, out, label)
    manifest = {
        "corpus":       corpus_path,
        "judges":       [j.judge_id for j in judges],
        "judges_spec":  judges_spec,
        "prompt_path":  prompt_path,
        "n_items":      len(corpus),
        "n_valid":      int(n),
        "defect_class": label,
        "n_boot":       n_boot,
        "seed":         seed,
        "judge_seed":   judge_seed,
        "ts":           datetime.datetime.utcnow().isoformat(),
    }
    save_manifest(manifest, out)

    print(f"\nResults saved to {out}/")
    for pair, p in pairwise.items():
        v   = p.get("verdict", "?")
        phi_s = f"{p['phi']:+.3f}" if p["phi"] == p["phi"] else "nan"
        print(f"  {pair}: phi={phi_s}  verdict={v}  n={p['n']}")

    return 0


# ─────────────────────────── calibrate ───────────────────────────────────────

def _run_calibrate(corpus_path: str, judges_spec: str, out_dir: str,
                   pos_threshold: float, n_boot: int, seed: int,
                   prompt_path: str = "prompts/strict_passfail.txt",
                   api_key: str | None = None,
                   judge_seed: int = 42) -> int:
    """Phase 5 calibration gate. Exits non-zero if either control fails.

    Positive control  (same model+seed → identical errors): phi must be >= pos_threshold.
    Orthogonal control: mock → different seeds; real judges → LLM × DeterministicChecker.
    phi CI must cover 0. No auto-PASS — both controls are always evaluated.
    """
    from .corpus import load_corpus
    from .runner import run_judges, build_miss_matrix
    import datetime

    corpus = load_corpus(corpus_path)
    out = Path(out_dir)

    if judges_spec == "mock":
        from .mock_judges import make_calibration_judges
        pos_a, pos_b, orth_a, orth_b = make_calibration_judges(error_rate=0.25, seed=seed)
    else:
        from .judges import load_prompt, is_ollama_available
        from .deterministic_checker import DeterministicChecker

        try:
            prompt_template = load_prompt(prompt_path)
        except FileNotFoundError:
            print(f"ERROR: prompt file not found: {prompt_path}")
            return 1

        if "ollama:" in judges_spec:
            if not is_ollama_available():
                print("ERROR: Ollama server not reachable. Start it with: ollama serve")
                return 1

        try:
            pos_judges = _parse_judge_specs(judges_spec, prompt_template, judge_seed, api_key)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

        pos_a, pos_b = pos_judges[0], pos_judges[1]
        # Orthogonal pair: LLM judge × deterministic GT checker (structurally independent).
        # Expected φ ≈ 0 (H3 axis). Gate FAILS if CI does not cover 0 — this is correct.
        orth_a = pos_a
        orth_b = DeterministicChecker()

    # --- positive control ---
    res_pos = run_judges(corpus, [pos_a, pos_b], out / "positive", force=True)
    M_pos, _, jids_pos = build_miss_matrix(corpus, res_pos, all_items=True)
    valid = ~np.any(np.isnan(M_pos), axis=1)
    M_pos = M_pos[valid].astype(int)
    n_pos = M_pos.shape[0]
    phi_pos = _phi(M_pos[:, 0], M_pos[:, 1]) if n_pos >= 2 else float("nan")

    # --- orthogonal control ---
    res_orth = run_judges(corpus, [orth_a, orth_b], out / "orthogonal", force=True)
    M_orth, _, jids_orth = build_miss_matrix(corpus, res_orth, all_items=True)
    valid2 = ~np.any(np.isnan(M_orth), axis=1)
    M_orth = M_orth[valid2].astype(int)
    n_orth = M_orth.shape[0]
    if n_orth >= 2:
        phi_orth = _phi(M_orth[:, 0], M_orth[:, 1])
        phi_orth_ci = bootstrap_ci(
            M_orth, lambda m: _phi(m[:, 0].astype(int), m[:, 1].astype(int)),
            n_boot=n_boot, seed=seed,
        )
    else:
        phi_orth, phi_orth_ci = float("nan"), (float("nan"), float("nan"))

    # --- gate ---
    pos_pass  = (phi_pos == phi_pos) and (phi_pos >= pos_threshold)
    orth_pass = (phi_orth_ci[0] == phi_orth_ci[0]) and (phi_orth_ci[0] <= 0 <= phi_orth_ci[1])

    print("=" * 65)
    print("CALIBRATION GATE")
    print("=" * 65)
    phi_pos_s  = f"{phi_pos:+.3f}" if phi_pos == phi_pos else "nan"
    phi_orth_s = f"{phi_orth:+.3f}" if phi_orth == phi_orth else "nan"
    lo, hi = phi_orth_ci
    ci_s = f"[{lo:+.3f}, {hi:+.3f}]" if lo == lo else "—"
    print(f"\nPositive control  (same seed, n={n_pos})")
    print(f"  phi = {phi_pos_s}  (threshold >= {pos_threshold})  → {'PASS' if pos_pass else 'FAIL'}")
    print(f"\nOrthogonal control (different seeds, n={n_orth})")
    print(f"  phi = {phi_orth_s}  CI {ci_s}  (CI must cover 0)  → {'PASS' if orth_pass else 'FAIL'}")
    print(f"\n{'=' * 65}")

    manifest = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "corpus": corpus_path,
        "judges": "mock",
        "positive_phi": phi_pos,
        "positive_pass": pos_pass,
        "orthogonal_phi": phi_orth,
        "orthogonal_phi_ci": list(phi_orth_ci),
        "orthogonal_pass": orth_pass,
        "gate": "PASS" if (pos_pass and orth_pass) else "FAIL",
    }
    from .report import save_manifest
    save_manifest(manifest, out)

    if pos_pass and orth_pass:
        print("CALIBRATION PASSED — proceed to Table 3.")
        return 0
    else:
        if not pos_pass:
            print("FAIL: positive control phi below threshold — pipeline or parsing broken.")
            print("      Do NOT proceed to Table 3 until this is fixed.")
        if not orth_pass:
            print("FAIL: orthogonal control CI does not cover 0 — independent judges appear correlated.")
            print("      Do NOT proceed to Table 3 until this is fixed.")
        print("=" * 65)
        return 1


# ─────────────────────────── main ─────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="judge-blindspot",
        description="Measures whether LLM judges share blind spots (miss-correlation per defect class).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="synthetic calibration — must both pass and fail")
    st.add_argument("--n-boot", type=int, default=2000)
    st.add_argument("--seed",   type=int, default=0)

    sub.add_parser("doctor", help="check environment: packages, Python, encoding")

    bc = sub.add_parser("build-corpus", help="build and verify a corpus")
    bc.add_argument("--smoke",  action="store_true", help="build 20-item synthetic smoke corpus")
    bc.add_argument("--full",   action="store_true", help="build full corpus (40 tasks, target ≥170/class)")
    bc.add_argument("--output", default=None, help="output JSONL path (default: data/synthetic_smoke.jsonl or data/full_corpus.jsonl)")

    run_p = sub.add_parser("run", help="run judges on corpus and compute pairwise stats")
    run_p.add_argument("--corpus",     required=True, help="path to corpus JSONL")
    run_p.add_argument(
        "--judges", default="mock",
        help=(
            "Judge spec(s). Examples:\n"
            "  mock                                      (two deterministic mocks)\n"
            "  ollama:llama3.1:8b                        (intra-model pair, same seed)\n"
            "  ollama:llama3.1:8b,ollama:mistral:7b      (two Ollama models)\n"
            "  anthropic:claude-haiku-4-5-20251001,anthropic:claude-sonnet-4-6\n"
            "  ollama:llama3.1:8b,anthropic:claude-haiku-4-5-20251001  (cross-type)"
        ),
    )
    run_p.add_argument("--prompt",     default="prompts/strict_passfail.txt",
                       help="path to judging prompt template (default: prompts/strict_passfail.txt)")
    run_p.add_argument("--api-key",    dest="api_key", default=None,
                       help="Anthropic API key (default: $ANTHROPIC_API_KEY env var)")
    run_p.add_argument("--judge-seed", dest="judge_seed", type=int, default=42,
                       help="seed for Ollama deterministic output (default: 42)")
    run_p.add_argument("--class",      dest="defect_class", default=None)
    run_p.add_argument("--out",        default="results/")
    run_p.add_argument("--dry-run",    action="store_true")
    run_p.add_argument("--force",      action="store_true")
    run_p.add_argument("--n-boot",     type=int, default=500)
    run_p.add_argument("--seed",       type=int, default=0,
                       help="bootstrap seed (default: 0)")

    cal_p = sub.add_parser("calibrate", help="Phase 5 calibration gate (run before Table 3)")
    cal_p.add_argument("--corpus",        required=True)
    cal_p.add_argument("--judges",        default="mock")
    cal_p.add_argument("--out",           default="results/calibration/")
    cal_p.add_argument("--pos-threshold", type=float, default=0.8,
                       help="minimum phi for positive control (default 0.8)")
    cal_p.add_argument("--n-boot",        type=int, default=500)
    cal_p.add_argument("--seed",          type=int, default=0)
    cal_p.add_argument("--prompt",        default="prompts/strict_passfail.txt",
                       help="path to judging prompt template (real judges only)")
    cal_p.add_argument("--api-key",       dest="api_key", default=None,
                       help="Anthropic API key (default: $ANTHROPIC_API_KEY env var)")
    cal_p.add_argument("--judge-seed",    dest="judge_seed", type=int, default=42,
                       help="seed for Ollama deterministic output (default: 42)")

    rep_p = sub.add_parser("report", help="emit tables.md and forest plot from results.json")
    rep_p.add_argument("results_json")
    rep_p.add_argument("--out",   default="results/")
    rep_p.add_argument("--class", dest="class_label", default="")

    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        sys.exit(_run_selftest(a.n_boot, a.seed))

    if a.cmd == "doctor":
        sys.exit(_run_doctor())

    if a.cmd == "build-corpus":
        default_output = "data/full_corpus.jsonl" if a.full else "data/synthetic_smoke.jsonl"
        out = a.output if a.output else default_output
        sys.exit(_run_build_corpus(a.smoke, a.full, out))

    if a.cmd == "run":
        sys.exit(_run_pipeline(
            a.corpus, a.judges, a.out, a.defect_class,
            a.dry_run, a.force, a.n_boot, a.seed,
            a.prompt, a.api_key, a.judge_seed,
        ))

    if a.cmd == "calibrate":
        sys.exit(_run_calibrate(
            a.corpus, a.judges, a.out,
            a.pos_threshold, a.n_boot, a.seed,
            a.prompt, a.api_key, a.judge_seed,
        ))

    if a.cmd == "report":
        from .report import save_tables_md, save_forest_plot
        with open(a.results_json, encoding="utf-8") as f:
            pairwise = json.load(f)
        out = Path(a.out)
        t = save_tables_md(pairwise, out, a.class_label)
        print(f"Tables : {t}")
        fp = save_forest_plot(pairwise, out, a.class_label)
        if fp:
            print(f"Forest plot: {fp}")
        sys.exit(0)


if __name__ == "__main__":
    main()
