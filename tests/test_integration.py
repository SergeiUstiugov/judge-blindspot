# -*- coding: utf-8 -*-
"""Integration tests: smoke pipeline and calibration gate end-to-end.

Uses an in-memory corpus (no subprocess) so tests stay fast.
The corpus-building subprocess tests are in test_mutate.py.
"""
import json
import pytest
import numpy as np
from pathlib import Path

from judge_blindspot.corpus import CorpusItem, save_corpus
from judge_blindspot.mock_judges import MockJudge, make_calibration_judges
from judge_blindspot.runner import run_judges, build_miss_matrix
from judge_blindspot.stats import pairwise_report, _phi
from judge_blindspot.verdict import apply_verdicts
from judge_blindspot.report import save_results, save_tables_md, save_forest_plot


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_corpus(n_def: int = 12, n_cor: int = 12) -> list:
    """In-memory corpus with no hidden tests — no subprocess needed."""
    items = []
    for i in range(n_def):
        items.append(CorpusItem(
            item_id=f"def_{i:02d}", spec="spec", candidate_code="def f(): pass",
            gt_label="defective", defect_type="off_by_one",
            source="test_fixture", contamination_flag=False,
        ))
    for i in range(n_cor):
        items.append(CorpusItem(
            item_id=f"cor_{i:02d}", spec="spec", candidate_code="def f(): return 1",
            gt_label="correct", defect_type=None,
            source="test_fixture", contamination_flag=False,
        ))
    return items


# ── smoke pipeline ─────────────────────────────────────────────────────────────

def test_smoke_pipeline_produces_artifacts(tmp_path):
    """Full pipeline: corpus → run_judges → miss_matrix → pairwise → artifacts."""
    corpus = _make_corpus()
    judges = [
        MockJudge(error_rate=0.25, seed=0, judge_id="mock_0"),
        MockJudge(error_rate=0.25, seed=1, judge_id="mock_1"),
    ]
    results = run_judges(corpus, judges, tmp_path / "logs", force=True)
    M, item_ids, judge_ids = build_miss_matrix(corpus, results)
    valid = ~np.any(np.isnan(M), axis=1)
    M_clean = M[valid].astype(int)

    assert M_clean.shape[0] == 12     # 12 defective items
    assert M_clean.shape[1] == 2      # 2 judges

    pairwise = pairwise_report(M_clean, judge_ids, n_boot=200, seed=0)
    pairwise = apply_verdicts(pairwise)

    assert "mock_0|mock_1" in pairwise
    p = pairwise["mock_0|mock_1"]
    assert "verdict" in p
    assert p["N00"] + p["N01"] + p["N10"] + p["N11"] == 12

    # artifacts
    r_path = save_results(pairwise, tmp_path, "smoke")
    t_path = save_tables_md(pairwise, tmp_path, "smoke")
    assert r_path.exists()
    assert t_path.exists()

    with open(r_path) as f:
        loaded = json.load(f)
    assert "mock_0|mock_1" in loaded


def test_smoke_pipeline_logs_written(tmp_path):
    corpus = _make_corpus(n_def=6, n_cor=6)
    judges = [MockJudge(seed=0, judge_id="j0"), MockJudge(seed=1, judge_id="j1")]
    run_judges(corpus, judges, tmp_path / "logs", force=True)
    log_files = list((tmp_path / "logs").glob("calls_*.jsonl"))
    assert len(log_files) == 2
    lines = (tmp_path / "logs" / "calls_j0.jsonl").read_text().strip().splitlines()
    assert len(lines) == 12  # 6 def + 6 cor


def test_smoke_tables_md_contains_verdict(tmp_path):
    corpus = _make_corpus()
    judges = [MockJudge(seed=0, judge_id="A"), MockJudge(seed=1, judge_id="B")]
    results = run_judges(corpus, judges, tmp_path / "logs", force=True)
    M, _, jids = build_miss_matrix(corpus, results)
    M = M[~np.any(np.isnan(M), axis=1)].astype(int)
    pairwise = apply_verdicts(pairwise_report(M, jids, n_boot=100))
    t = save_tables_md(pairwise, tmp_path, "test")
    content = t.read_text()
    assert "A|B" in content
    assert any(v in content for v in ("INDEPENDENT", "OVERLAP", "DUPLICATE", "INCONCLUSIVE"))


# ── calibration gate ──────────────────────────────────────────────────────────

def test_calibration_positive_control_passes(tmp_path):
    """Same seed → identical miss columns → phi = +1 → gate PASSES."""
    corpus = _make_corpus(n_def=30, n_cor=30)
    save_corpus(corpus, tmp_path / "corpus.jsonl")
    from judge_blindspot.cli import _run_calibrate
    rc = _run_calibrate(
        str(tmp_path / "corpus.jsonl"), "mock",
        str(tmp_path / "cal"), pos_threshold=0.8, n_boot=200, seed=0,
    )
    assert rc == 0


def test_calibration_fails_when_positive_control_broken(tmp_path):
    """Two judges with DIFFERENT seeds used as 'positive control' → phi << 1 → gate FAILS."""
    corpus = _make_corpus(n_def=30, n_cor=30)
    corpus_path = tmp_path / "corpus.jsonl"
    save_corpus(corpus, corpus_path)

    # Patch make_calibration_judges to return non-identical pair for positive control
    from judge_blindspot import mock_judges as _mj
    original = _mj.make_calibration_judges

    def _broken(error_rate=0.25, seed=0):
        # positive pair uses different seeds → will NOT give phi=+1
        pos_a  = MockJudge(error_rate=error_rate, seed=seed,    judge_id="cal_pos_a")
        pos_b  = MockJudge(error_rate=error_rate, seed=seed+99, judge_id="cal_pos_b")  # WRONG
        orth_a = MockJudge(error_rate=error_rate, seed=seed,    judge_id="cal_orth_a")
        orth_b = MockJudge(error_rate=error_rate, seed=seed+1,  judge_id="cal_orth_b")
        return pos_a, pos_b, orth_a, orth_b

    _mj.make_calibration_judges = _broken
    try:
        from judge_blindspot.cli import _run_calibrate
        rc = _run_calibrate(
            str(corpus_path), "mock",
            str(tmp_path / "broken_cal"), pos_threshold=0.8, n_boot=200, seed=0,
        )
        assert rc != 0, "Broken positive control must cause non-zero exit"
    finally:
        _mj.make_calibration_judges = original


def test_calibration_manifest_written(tmp_path):
    # n=160 keeps sampling variance low enough that phi of two independent
    # judges reliably sits near 0 with CI covering 0 (SE ≈ 1/sqrt(160) ≈ 0.08).
    # n=40 can produce phi ≈ ±0.17 by chance, causing a tight CI to miss 0.
    corpus = _make_corpus(n_def=80, n_cor=80)
    save_corpus(corpus, tmp_path / "corpus.jsonl")
    from judge_blindspot.cli import _run_calibrate
    _run_calibrate(
        str(tmp_path / "corpus.jsonl"), "mock",
        str(tmp_path / "cal"), pos_threshold=0.8, n_boot=500, seed=0,
    )
    manifest = json.loads((tmp_path / "cal" / "run_manifest.json").read_text())
    assert "gate" in manifest
    assert "positive_phi" in manifest
    assert manifest["gate"] == "PASS"


# ── CLI integration ───────────────────────────────────────────────────────────

def test_cli_run_command(tmp_path):
    # Call _run_pipeline directly: main() calls sys.exit() which pytest
    # raises as SystemExit — test via the internal function instead.
    corpus = _make_corpus()
    corpus_path = tmp_path / "corpus.jsonl"
    save_corpus(corpus, corpus_path)
    from judge_blindspot.cli import _run_pipeline
    rc = _run_pipeline(str(corpus_path), "mock", str(tmp_path / "out"),
                       None, False, True, 100, 0,
                       prompt_path="prompts/strict_passfail.txt",
                       api_key=None, judge_seed=42)
    assert rc == 0
    assert (tmp_path / "out" / "results_all.json").exists()
    assert (tmp_path / "out" / "tables_all.md").exists()


def test_cli_calibrate_command(tmp_path):
    corpus = _make_corpus(n_def=30, n_cor=30)
    corpus_path = tmp_path / "corpus.jsonl"
    save_corpus(corpus, corpus_path)
    from judge_blindspot.cli import _run_calibrate
    rc = _run_calibrate(str(corpus_path), "mock", str(tmp_path / "cal"),
                        pos_threshold=0.8, n_boot=200, seed=0)
    assert rc == 0
