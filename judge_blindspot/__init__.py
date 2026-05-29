# -*- coding: utf-8 -*-
"""judge-blindspot: measures whether LLM judges share blind spots
(miss-correlation per defect class)."""
from .stats   import independence_report, bootstrap_ci, pairwise_report
from .verdict import verdict_for_pair, apply_verdicts, VerdictLabel
from .corpus  import CorpusItem, load_corpus, save_corpus

__version__ = "0.1.0"
__all__ = [
    "independence_report", "bootstrap_ci", "pairwise_report",
    "verdict_for_pair", "apply_verdicts", "VerdictLabel",
    "CorpusItem", "load_corpus", "save_corpus",
]
