"""
clustering.py
-------------
WiDiD-inspired incremental clustering adapted for distributional vectors.

Core idea (from Periti et al. 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instead of re-clustering from scratch at every time period, we maintain a
"memory" of past cluster prototypes.  At each new decade:

  1.  Compute the centroid (sense prototype) of every existing cluster.
  2.  Run Affinity Propagation on {new_decade_vector} ∪ {prototypes}.
  3.  Assign the new decade's vector to whichever cluster absorbed it.
  4.  Past assignments are never revised — "What is Done is Done".

Adaptation from the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~
The original WiDiD clusters *multiple* contextualised embeddings per time
period (one per sentence containing the target word).  Here we have *one*
PPMI+SVD vector per (word, decade).  We therefore treat each decade-vector
as a single data point.  The "cluster" across time is the sequence of
decades that get assigned to the same sense prototype.

Because we have very few data points per word (≤ 16 decades), we replace
Affinity Propagation with a simpler threshold-based assignment that is
more robust at small sample sizes, while keeping the same incremental,
memory-based semantics.  Full AP is also available via use_ap=True.

Output
~~~~~~
ClusterResult(word) gives:
  .assignments  dict[decade -> cluster_id]
  .prototypes   dict[cluster_id -> np.ndarray]  (centroid in embedding space)
  .history      list of (decade, cluster_id, vector) in chronological order

Usage
-----
    from src.clustering import cluster_word, cluster_all

    results = cluster_all(vectors)
    r = results["computer"]
    print(r.assignments)       # {1850: 0, 1860: 0, …, 1980: 1, …}
    print(r.num_clusters)      # 2
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PROCESSED_DIR = os.path.join("data", "processed")

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    word:        str
    assignments: Dict[int, int]                  = field(default_factory=dict)
    prototypes:  Dict[int, np.ndarray]           = field(default_factory=dict)
    history:     List[Tuple[int, int, np.ndarray]] = field(default_factory=list)

    @property
    def num_clusters(self) -> int:
        return len(self.prototypes)

    @property
    def decades(self) -> List[int]:
        return sorted(self.assignments.keys())

    def cluster_ids_at(self, decade: int) -> int:
        return self.assignments.get(decade, -1)

    def active_clusters_at(self, decade: int) -> List[int]:
        """Cluster IDs that have received at least one assignment up to *decade*."""
        return sorted({cid for dec, cid in self.assignments.items() if dec <= decade})


# ── Public API ─────────────────────────────────────────────────────────────────

def cluster_all(
    vectors:      Dict[str, Dict[int, np.ndarray]],
    threshold:    float = None,
    use_ap:       bool  = False,
    processed_dir: str  = PROCESSED_DIR,
    force:        bool  = False,
) -> Dict[str, ClusterResult]:
    """
    Run incremental clustering for every word in *vectors*.

    Parameters
    ----------
    vectors    : output of ``representation.build_vectors()``
    threshold  : cosine-distance threshold for creating a new cluster.
                 If None, auto-calibrated to the 40th percentile of all
                 pairwise inter-decade distances across all words — this
                 ensures ~40% of transitions stay in the same cluster,
                 capturing only genuine distributional jumps as new senses.
    use_ap     : use sklearn's AffinityPropagation instead of threshold rule
    processed_dir : cache directory
    force      : recompute even if a cache exists

    Returns
    -------
    dict mapping each word to its ClusterResult
    """
    cache_path = os.path.join(processed_dir, "clusters.pkl")
    if not force and os.path.exists(cache_path):
        print(f"[clustering] Loading cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    if threshold is None:
        threshold = _auto_threshold(vectors)
        print(f"[clustering] Auto-calibrated threshold: {threshold:.4f}")

    results = {}
    for word in sorted(vectors.keys()):
        print(f"[clustering] Processing '{word}' …", end=" ")
        results[word] = cluster_word(
            word, vectors[word], threshold=threshold, use_ap=use_ap
        )
        print(f"{results[word].num_clusters} cluster(s) found")

    os.makedirs(processed_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"[clustering] Saved cache: {cache_path}")
    return results


def _auto_threshold(vectors: Dict[str, Dict[int, np.ndarray]]) -> float:
    """
    Set threshold at the 40th percentile of consecutive inter-decade
    cosine distances across all words.  Words below this distance are
    considered semantically stable across that transition; above it a
    potential sense change is flagged.
    """
    dists = []
    for word_vecs in vectors.values():
        decades = sorted(word_vecs.keys())
        for d1, d2 in zip(decades[:-1], decades[1:]):
            v1, v2 = word_vecs[d1], word_vecs[d2]
            dists.append(float(1.0 - np.dot(v1, v2)))
    return float(np.percentile(dists, 40))


def cluster_word(
    word:      str,
    vecs:      Dict[int, np.ndarray],
    threshold: float = 0.25,
    use_ap:    bool  = False,
) -> ClusterResult:
    """
    Incrementally cluster the decade-vectors for a single *word*.

    The first decade always seeds cluster 0.  Each subsequent decade is
    either assigned to the closest existing cluster (if within *threshold*)
    or starts a new one.
    """
    if use_ap:
        return _cluster_ap(word, vecs)
    return _cluster_threshold(word, vecs, threshold)


# ── Threshold-based incremental clustering ─────────────────────────────────────

def _cluster_threshold(
    word:      str,
    vecs:      Dict[int, np.ndarray],
    threshold: float,
) -> ClusterResult:
    """
    Greedy, threshold-based incremental clustering.

    At each step we compare the new vector to the *prototype* (centroid)
    of every existing cluster and assign to the nearest one if its
    cosine distance is below *threshold*.  Otherwise we open a new cluster.

    Prototype update: running mean — keeps memory bounded and avoids
    recomputing from scratch.
    """
    result       = ClusterResult(word=word)
    # cluster_id -> list of all member vectors (for centroid computation)
    members: Dict[int, List[np.ndarray]] = {}
    next_id = 0

    for decade in sorted(vecs.keys()):
        vec = vecs[decade]

        if not members:
            # First decade seeds cluster 0
            cid = 0
            members[cid] = [vec]
            result.prototypes[cid] = vec.copy()
            next_id = 1
        else:
            # Find nearest prototype
            best_cid, best_dist = _nearest_prototype(vec, result.prototypes)

            if best_dist <= threshold:
                cid = best_cid
            else:
                # Open a new cluster
                cid     = next_id
                next_id += 1
                members[cid] = []

            # Accumulate member and update centroid
            members[cid].append(vec)
            proto = np.mean(members[cid], axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            result.prototypes[cid] = proto

        result.assignments[decade] = cid
        result.history.append((decade, cid, vec))

    return result


def _nearest_prototype(
    vec:        np.ndarray,
    prototypes: Dict[int, np.ndarray],
) -> Tuple[int, float]:
    """Return (cluster_id, cosine_distance) of the closest prototype."""
    best_cid  = -1
    best_dist = float("inf")
    for cid, proto in prototypes.items():
        dist = float(1.0 - np.dot(vec, proto))
        if dist < best_dist:
            best_dist = dist
            best_cid  = cid
    return best_cid, best_dist


# ── Affinity Propagation variant ───────────────────────────────────────────────

def _cluster_ap(
    word: str,
    vecs: Dict[int, np.ndarray],
) -> ClusterResult:
    """
    A Posteriori Affinity Propagation (APP) as described in WiDiD.

    At t=0  : run standard AP over the first decade's vector (trivial — 1 cluster).
    At t > 0: run AP over {new_vector} ∪ {all existing prototypes}, then
              assign the new vector to whichever cluster it was grouped with.

    Note: AP is designed for larger datasets.  With very few data points
    (as here) it often returns a single cluster.  Use threshold mode for
    more meaningful results with small corpora.
    """
    from sklearn.cluster import AffinityPropagation

    result       = ClusterResult(word=word)
    members: Dict[int, List[np.ndarray]] = {}
    next_id = 0

    decades = sorted(vecs.keys())

    # t = 0 — seed the memory
    d0 = decades[0]
    v0 = vecs[d0]
    result.assignments[d0] = 0
    result.prototypes[0]   = v0.copy()
    members[0]             = [v0]
    result.history.append((d0, 0, v0))
    next_id = 1

    for decade in decades[1:]:
        vec = vecs[decade]

        # Build input matrix: new vector + one prototype per existing cluster
        proto_ids   = sorted(result.prototypes.keys())
        proto_vecs  = np.stack([result.prototypes[cid] for cid in proto_ids])
        input_mat   = np.vstack([vec.reshape(1, -1), proto_vecs])  # (1+K, D)

        # Similarity matrix (cosine similarity via dot product on unit vectors)
        sim = input_mat @ input_mat.T

        try:
            ap = AffinityPropagation(affinity="precomputed", random_state=42)
            ap.fit(sim)
            label_of_new = ap.labels_[0]  # label assigned to the new vector

            # Map AP label → our cluster id based on which prototype is in same group
            ap_to_our: Dict[int, int] = {}
            for i, proto_id in enumerate(proto_ids, start=1):
                ap_label = ap.labels_[i]
                if ap_label not in ap_to_our:
                    ap_to_our[ap_label] = proto_id

            if label_of_new in ap_to_our:
                cid = ap_to_our[label_of_new]
            else:
                cid     = next_id
                next_id += 1

        except Exception:
            # AP failed (e.g. convergence) — fall back to nearest prototype
            cid, _ = _nearest_prototype(vec, result.prototypes)

        if cid not in members:
            members[cid] = []
        members[cid].append(vec)
        proto = np.mean(members[cid], axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        result.prototypes[cid] = proto

        result.assignments[decade] = cid
        result.history.append((decade, cid, vec))

    return result
