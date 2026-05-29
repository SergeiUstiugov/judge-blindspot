# -*- coding: utf-8 -*-
"""Run judges on corpus, build miss matrix, persist per-call logs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def run_judges(
    corpus: list,
    judges: list,
    output_dir: str | Path,
    dry_run: bool = False,
    budget_calls: Optional[int] = None,
    force: bool = False,
) -> Dict[str, list]:
    """Run each judge on every corpus item. Logs every call to JSONL.

    Returns {judge_id: [JudgeResult, ...]} ordered as corpus.
    Raises RuntimeError if planned calls exceed budget_calls (unless force=True).
    """
    output_dir = Path(output_dir)
    n_calls = len(corpus) * len(judges)

    if dry_run:
        print(f"[dry-run] planned calls : {n_calls}")
        print(f"[dry-run] judges        : {[j.judge_id for j in judges]}")
        print(f"[dry-run] corpus items  : {len(corpus)}")
        return {}

    if budget_calls and n_calls > budget_calls and not force:
        raise RuntimeError(
            f"Planned {n_calls} calls exceeds budget {budget_calls}. "
            "Use --force to override."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, list] = {j.judge_id: [] for j in judges}

    for judge in judges:
        safe_id = judge.judge_id.replace("/", "_").replace(":", "_")
        log_path = output_dir / f"calls_{safe_id}.jsonl"
        with open(log_path, "w", encoding="utf-8") as log_f:
            for item in corpus:
                result = judge.judge(item)
                results[judge.judge_id].append(result)
                log_f.write(json.dumps(result.to_dict()) + "\n")
                log_f.flush()

    return results


def build_miss_matrix(
    corpus: list,
    results: Dict[str, list],
    defect_class: Optional[str] = None,
) -> tuple:
    """Build binary miss matrix M[n_items, n_judges] for gt_label='defective' items.

    - Drops INVALID verdicts (NaN in float matrix; callers filter per pair).
    - Filters to defect_class if given (matches item.defect_type).
    - Returns (M: ndarray[n, k], item_ids: list[str], judge_ids: list[str]).
    """
    judge_ids = list(results.keys())

    items = corpus
    if defect_class is not None:
        items = [it for it in corpus if it.defect_type == defect_class]

    # per the pre-registered rule: only defective items enter the miss matrix
    defective = [it for it in items if it.gt_label == "defective"]
    if not defective:
        defective = items  # fallback for synthetic corpora without explicit gt

    item_ids = [it.item_id for it in defective]
    n, k = len(item_ids), len(judge_ids)
    M = np.full((n, k), fill_value=np.nan)

    for j_idx, judge_id in enumerate(judge_ids):
        by_item = {r.item_id: r for r in results[judge_id]}
        for i_idx, item in enumerate(defective):
            res = by_item.get(item.item_id)
            if res is not None:
                val = res.miss(item.gt_label)
                if val is not None:
                    M[i_idx, j_idx] = val

    return M, item_ids, judge_ids
