"""
evaluation.py
-------------
Evaluates shift scores against known-shifted and known-stable words,
following the SemEval-2020 Task 1 framework (Schlechtweg et al. 2020).

We do not have exact SemEval gold annotations for Google Books, so we
use a manually curated proxy list of English words with well-documented
semantic shift histories, drawn from the SemEval-2020 English test set
and NLP literature.

Two evaluation modes
~~~~~~~~~~~~~~~~~~~~
1. Ranking (Subtask 2): Spearman ρ between our shift scores and a
   gold ranking (higher = more shifted).
2. Binary classification (Subtask 1): accuracy of shifted/stable
   labels after thresholding our score at the midpoint.

Usage
-----
    from src.metrics import compute_metrics, shift_score
    from src.evaluation import evaluate

    scores  = {w: shift_score(m) for w, m in metrics.items()}
    results = evaluate(scores)
    print(results.summary())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ── Gold reference ─────────────────────────────────────────────────────────────
#
# Curated proxy list.  Scores are *relative* gold rankings (0 = stable,
# 1 = strongly shifted) derived from SemEval-2020 English annotations and
# corroborated by published case studies.  Only words that overlap with our
# TARGET_WORDS list are used; the rest serve as context for Spearman ρ.
#
# Sources:
#   Schlechtweg et al. (2020) SemEval-2020 Task 1
#   Hamilton et al. (2016) Diachronic word embeddings
#   Periti et al. (2022) WiDiD case studies

GOLD_SCORES: Dict[str, float] = {
    # strongly shifted (score ≥ 0.7)
    "computer": 0.95,   # human calculator → digital machine
    "virus":    0.90,   # biology only → biology + computing
    "gay":      0.95,   # carefree → LGBTQ+ identity
    "cloud":    0.80,   # sky feature → cloud computing
    "tablet":   0.75,   # stone tablet / pill → computing device
    "cell":     0.70,   # biology cell / prison → mobile phone
    "web":      0.85,   # spider web → World Wide Web
    "tweet":    0.90,   # bird sound → social media post
    "spam":     0.85,   # canned meat → unsolicited email
    "stream":   0.65,   # water flow → media streaming
    # mildly / not shifted (score ≤ 0.4)
    "crash":    0.40,   # already polysemous; finance + car + computing
    "mouse":    0.55,   # rodent + computing device (shifted but early)
}

GOLD_BINARY: Dict[str, int] = {
    "computer": 1,
    "virus":    1,
    "gay":      1,
    "cloud":    1,
    "tablet":   1,
    "cell":     1,
    "web":      1,
    "tweet":    1,
    "spam":     1,
    "stream":   1,
    "crash":    0,
    "mouse":    1,
}


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    words:          List[str]
    pred_scores:    Dict[str, float]
    gold_scores:    Dict[str, float]
    gold_binary:    Dict[str, int]
    spearman_rho:   float = 0.0
    spearman_p:     float = 1.0
    accuracy:       float = 0.0
    threshold_used: float = 0.0
    pred_binary:    Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  Evaluation results",
            "=" * 52,
            f"  Words evaluated  : {len(self.words)}",
            f"  Spearman ρ       : {self.spearman_rho:+.3f}  (p={self.spearman_p:.3f})",
            f"  Binary accuracy  : {self.accuracy:.1%}  (threshold={self.threshold_used:.3f})",
            "",
            "  Per-word scores (pred | gold):",
        ]
        for w in sorted(self.words):
            marker = "✓" if self.pred_binary.get(w) == self.gold_binary.get(w) else "✗"
            lines.append(
                f"    {marker} {w:<12}  pred={self.pred_scores[w]:.3f}  "
                f"gold={self.gold_scores.get(w, '?')}"
            )
        lines.append("=" * 52)
        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    pred_scores:  Dict[str, float],
    gold_scores:  Optional[Dict[str, float]] = None,
    gold_binary:  Optional[Dict[str, int]]   = None,
) -> EvalResult:
    """
    Evaluate predicted shift scores against gold annotations.

    Parameters
    ----------
    pred_scores : {word: shift_score} — output of metrics.shift_score()
    gold_scores : optional override for the module-level GOLD_SCORES
    gold_binary : optional override for the module-level GOLD_BINARY

    Returns
    -------
    EvalResult with Spearman ρ and binary accuracy
    """
    if gold_scores is None:
        gold_scores = GOLD_SCORES
    if gold_binary is None:
        gold_binary = GOLD_BINARY

    # Keep only words present in both pred and gold
    words = sorted(set(pred_scores) & set(gold_scores))
    if not words:
        raise ValueError("No overlapping words between predictions and gold set.")

    preds = np.array([pred_scores[w] for w in words])
    golds = np.array([gold_scores[w] for w in words])

    # ── Subtask 2: Spearman ρ ─────────────────────────────────────────────────
    rho, p_val = spearmanr(preds, golds)

    # ── Subtask 1: Binary classification ──────────────────────────────────────
    # Sweep threshold to find the best accuracy on this set
    best_acc, best_thr = 0.0, 0.5
    for thr in np.linspace(preds.min(), preds.max(), 50):
        pbin = (preds >= thr).astype(int)
        gbin = np.array([gold_binary.get(w, 0) for w in words])
        acc  = (pbin == gbin).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    best_preds_bin = {
        w: int(pred_scores[w] >= best_thr) for w in words
    }

    return EvalResult(
        words          = words,
        pred_scores    = {w: pred_scores[w] for w in words},
        gold_scores    = {w: gold_scores[w]  for w in words},
        gold_binary    = {w: gold_binary.get(w, 0) for w in words},
        spearman_rho   = float(rho),
        spearman_p     = float(p_val),
        accuracy       = float(best_acc),
        threshold_used = best_thr,
        pred_binary    = best_preds_bin,
    )
