# -*- coding: utf-8 -*-
"""Emit results.json, tables.md, forest plot (Figure 3), run_manifest.json."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def save_results(pairwise: dict, output_dir: str | Path, class_label: str = "") -> Path:
    """Write results.json (raw pairwise stats + verdicts)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{class_label}" if class_label else ""
    path = output_dir / f"results{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairwise, f, indent=2, default=str)
    return path


def save_tables_md(pairwise: dict, output_dir: str | Path, class_label: str = "") -> Path:
    """Write Table 3 (judge pairs) as Markdown with phi, CI, ratio, verdict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{class_label}" if class_label else ""
    path = output_dir / f"tables{suffix}.md"

    lines = [f"## Table 3 — Judge pairs ({class_label or 'all'})\n"]
    lines.append("| Pair | φ | φ 95% CI | ratio | ratio CI | N00 | N01 | N10 | N11 | verdict | n |")
    lines.append("|------|---|----------|-------|----------|-----|-----|-----|-----|---------|---|")

    for pair, p in sorted(pairwise.items()):
        phi = p["phi"]
        phi_s = f"{phi:+.3f}" if phi == phi else "nan"
        lo_p, hi_p = p["phi_ci"]
        phi_ci_s = f"[{lo_p:+.3f}, {hi_p:+.3f}]" if lo_p == lo_p else "—"
        r = p["ratio"]
        ratio_s = f"{r:.2f}" if r == r else "nan"
        lo_r, hi_r = p["ratio_ci"]
        ratio_ci_s = f"[{lo_r:.2f}, {hi_r:.2f}]" if lo_r == lo_r else "—"
        verdict = p.get("verdict", "—")
        n = p["n"]
        n00, n01, n10, n11 = p.get("N00","?"), p.get("N01","?"), p.get("N10","?"), p.get("N11","?")
        lines.append(
            f"| {pair} | {phi_s} | {phi_ci_s} | {ratio_s} | {ratio_ci_s} "
            f"| {n00} | {n01} | {n10} | {n11} | {verdict} | {n} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_forest_plot(pairwise: dict, output_dir: str | Path, class_label: str = "") -> Optional[Path]:
    """Save forest plot of phi with 95% CIs across pairs (mirrors Figure 3)."""
    if not _HAS_MPL:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{class_label}" if class_label else ""
    path = output_dir / f"forest_plot{suffix}.png"

    pairs = sorted(pairwise.keys())
    phis  = [pairwise[p]["phi"] for p in pairs]
    los   = [pairwise[p]["phi_ci"][0] for p in pairs]
    his   = [pairwise[p]["phi_ci"][1] for p in pairs]

    fig, ax = plt.subplots(figsize=(7, max(3, len(pairs) * 0.6 + 1)))
    for i, (phi, lo, hi) in enumerate(zip(phis, los, his)):
        if phi == phi and lo == lo:
            ax.plot([lo, hi], [i, i], color="steelblue", linewidth=2)
            ax.plot(phi, i, "o", color="steelblue", markersize=6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="φ=0")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("φ miss-correlation")
    ax.set_xlim(-1.1, 1.1)
    ax.set_title(f"Judge pairs — φ with 95% CI ({class_label or 'all'})")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Write run_manifest.json (live source of truth cited in the paper)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
