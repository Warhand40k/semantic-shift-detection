"""
metrics.py
----------
Cluster analysis metrics, directly adapted from WiDiD (Periti et al. 2022).

Four metrics, two perspectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synchronic (status of a word *at* a time period):
  polysemy  (π)   — number of active clusters at decade t
  prominence (ρ)  — share of decades assigned to cluster k, up to t

Diachronic (change *between* consecutive time periods):
  semantic_shift (S)  — JSD between prominence distributions at t-1 and t
  sense_shift    (T)  — cosine distance between prototypes at t-1 and t

Usage
-----
    from src.metrics import compute_metrics, MetricResult

    metrics = compute_metrics(results)   # results from cluster_all()
    m = metrics["computer"]

    print(m.polysemy)           # {1850: 1, …, 1980: 2}
    print(m.semantic_shift)     # {1860: 0.00, …, 1980: 0.31}
    print(m.top_shifted_decades(n=3))
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.clustering import ClusterResult

PROCESSED_DIR = os.path.join("data", "processed")


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    word:           str
    polysemy:       Dict[int, int]          = field(default_factory=dict)
    prominence:     Dict[int, Dict[int, float]] = field(default_factory=dict)
    semantic_shift: Dict[int, float]        = field(default_factory=dict)
    sense_shift:    Dict[int, Dict[int, float]] = field(default_factory=dict)

    def top_shifted_decades(self, n: int = 3) -> List[Tuple[int, float]]:
        """Return the *n* decades with the highest semantic shift score."""
        ranked = sorted(self.semantic_shift.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def shift_series(self) -> Tuple[List[int], List[float]]:
        """Chronological (decade, shift) pairs, suitable for plotting."""
        pairs = sorted(self.semantic_shift.items())
        return [d for d, _ in pairs], [s for _, s in pairs]


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_metrics(
    results:       Dict[str, ClusterResult],
    processed_dir: str = PROCESSED_DIR,
    force:         bool = False,
) -> Dict[str, MetricResult]:
    """
    Compute all four metrics for every word.

    Parameters
    ----------
    results       : output of ``clustering.cluster_all()``
    processed_dir : cache directory
    force         : recompute even if a cache exists

    Returns
    -------
    dict mapping each word to its MetricResult
    """
    cache_path = os.path.join(processed_dir, "metrics.pkl")
    if not force and os.path.exists(cache_path):
        print(f"[metrics] Loading cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    metrics = {word: _compute_word_metrics(r) for word, r in results.items()}

    os.makedirs(processed_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(metrics, fh)
    print(f"[metrics] Saved cache: {cache_path}")
    return metrics


def shift_score(metric: MetricResult) -> float:
    """
    Single scalar shift score for ranking/evaluation purposes.

    Defined as the weighted combination of:
      - mean semantic shift (JSD) across all decades  →  captures sustained change
      - max semantic shift                            →  captures abrupt change
      - final polysemy normalised by decades          →  rewards gaining new senses

    This composite avoids the all-words-same-score collapse that occurs
    when using max JSD alone (all words that ever change hit the same peak).
    """
    if not metric.semantic_shift:
        return 0.0

    shifts = list(metric.semantic_shift.values())
    mean_s = float(np.mean(shifts))
    max_s  = float(np.max(shifts))

    decades     = sorted(metric.polysemy.keys())
    final_poly  = metric.polysemy[decades[-1]]
    num_decades = len(decades)
    poly_norm   = (final_poly - 1) / max(num_decades - 1, 1)

    # Weighted sum — weights can be tuned; these reflect the WiDiD priority
    # of gradual shift (mean) over abrupt peaks, with polysemy as a bonus.
    return float(0.5 * mean_s + 0.3 * max_s + 0.2 * poly_norm)


# ── Internal computation ───────────────────────────────────────────────────────

def _compute_word_metrics(r: ClusterResult) -> MetricResult:
    m = MetricResult(word=r.word)
    decades = r.decades

    # Snapshot of which clusters are active and how many members each has,
    # computed incrementally as we move forward in time.
    running_counts: Dict[int, int] = {}   # cluster_id -> count of decades assigned

    prev_prominence: Optional[Dict[int, float]] = None
    prev_prototypes: Dict[int, np.ndarray]      = {}

    for t_idx, decade in enumerate(decades):
        cid = r.assignments[decade]
        running_counts[cid] = running_counts.get(cid, 0) + 1

        total = sum(running_counts.values())

        # ── Synchronic ────────────────────────────────────────────────────────

        # Polysemy: number of distinct clusters seen so far
        m.polysemy[decade] = len(running_counts)

        # Prominence: share of total assignments for each cluster
        prom = {k: v / total for k, v in running_counts.items()}
        m.prominence[decade] = prom

        # ── Diachronic ────────────────────────────────────────────────────────

        if prev_prominence is not None:
            # All cluster IDs seen across both periods
            all_cids = sorted(set(prev_prominence) | set(prom))

            p_prev = np.array([prev_prominence.get(k, 0.0) for k in all_cids])
            p_curr = np.array([prom.get(k, 0.0)            for k in all_cids])

            m.semantic_shift[decade] = _jsd(p_prev, p_curr)

            # Sense shift per cluster: cosine distance between consecutive prototypes
            sense = {}
            for k in all_cids:
                if k in prev_prototypes and k in r.prototypes:
                    sense[k] = _cosine_dist(prev_prototypes[k], r.prototypes[k])
            m.sense_shift[decade] = sense

        prev_prominence = prom
        # Snapshot current prototypes (deep copy)
        prev_prototypes = {k: v.copy() for k, v in r.prototypes.items()}

    return m


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions (log base 2)."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / (q[mask] + 1e-12))))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
