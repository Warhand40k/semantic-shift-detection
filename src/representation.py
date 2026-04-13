"""
representation.py
-----------------
Converts raw co-occurrence counts (from data_loader) into dense, comparable
word vectors using PPMI weighting followed by Truncated SVD.

Key design choices
~~~~~~~~~~~~~~~~~~
* Shared SVD — all (word, decade) vectors share the same low-dimensional axes.
* PPMI weighting — down-weights very frequent context words.
* Temporal smoothing — reduces decade-to-decade sampling noise.
* L2 normalisation — unit-length vectors; cosine similarity = dot product.

Usage
-----
    from src.representation import build_vectors
    vectors = build_vectors(cooc)
    # vectors["computer"][1980] -> np.ndarray of shape (100,)
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

PROCESSED_DIR = os.path.join("data", "processed")


def build_vectors(
    cooc:              Dict[str, Dict[int, Dict[str, int]]],
    svd_dim:           int  = 100,
    min_context_total: int  = 50,
    smooth_window:     int  = 2,
    processed_dir:     str  = PROCESSED_DIR,
    force:             bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Build PPMI + SVD word vectors for every (word, decade) pair.

    Parameters
    ----------
    cooc              : output of data_loader.load_cooccurrences()
    svd_dim           : output dimensionality after SVD (0 = skip SVD)
    min_context_total : drop context words with total count below this
    smooth_window     : half-width of temporal averaging window (1 = off)
    processed_dir     : cache directory
    force             : ignore cache and recompute
    """
    cache_path = os.path.join(processed_dir, "vectors.pkl")
    if not force and os.path.exists(cache_path):
        print(f"[representation] Loading cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
        
    # ── Filter sparse decades before any computation ──────────────────────────
    # Decades with fewer than 50 total bigrams produce unreliable PPMI vectors
    # and cause the 1860s bootstrapping artifact seen in results.
    for word in list(cooc.keys()):
        cooc[word] = {dec: ctx for dec, ctx in cooc[word].items()
                    if sum(ctx.values()) >= 50}

    vocab, ctx_index = _build_vocab(cooc, min_context_total)
    V = len(vocab)
    print(f"[representation] Context vocabulary: {V} words")

    total_count   = sum(cnt for wd in cooc.values()
                        for dc in wd.values() for cnt in dc.values())
    ctx_marginals = _context_marginals(cooc, vocab, ctx_index, total_count)
    word_marginals = _word_marginals(cooc, total_count)

    keys:     List[Tuple[str, int]] = []
    raw_vecs: List[np.ndarray]      = []

    for word in sorted(cooc.keys()):
        p_w = word_marginals[word]
        for decade in sorted(cooc[word].keys()):
            v = _ppmi_vector(cooc[word][decade], ctx_index, V,
                             p_w, ctx_marginals, total_count)
            keys.append((word, decade))
            raw_vecs.append(v)
        print(f"[representation] PPMI done: '{word}' ({len(cooc[word])} decades)")

    matrix = np.stack(raw_vecs)

    if svd_dim and svd_dim < V:
        matrix = _svd_reduce(matrix, svd_dim)

    matrix = normalize(matrix, norm="l2")

    # Pack into per-word dict
    vectors: Dict[str, Dict[int, np.ndarray]] = {}
    for (word, decade), vec in zip(keys, matrix):
        vectors.setdefault(word, {})[decade] = vec

    # Temporal smoothing: average each vector with its nearest neighbours.
    # Suppresses sampling noise — standard in diachronic distributional semantics.
    if smooth_window > 1:
        for word in vectors:
            dec_sorted = sorted(vectors[word].keys())
            smoothed   = {}
            for i, dec in enumerate(dec_sorted):
                nbrs = dec_sorted[max(0, i - smooth_window): i + smooth_window + 1]
                avg  = np.mean([vectors[word][d] for d in nbrs], axis=0)
                smoothed[dec] = avg / (np.linalg.norm(avg) + 1e-12)
            vectors[word] = smoothed
        print(f"[representation] Temporal smoothing applied (window={smooth_window})")

    os.makedirs(processed_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(vectors, fh)
    print(f"[representation] Saved cache: {cache_path}")
    return vectors


# ── Distance utilities ────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (_kl(p, m) + _kl(q, m)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / (q[mask] + 1e-12))))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_vocab(cooc, min_total):
    totals: Dict[str, int] = {}
    for wd in cooc.values():
        for dc in wd.values():
            for ctx, cnt in dc.items():
                totals[ctx] = totals.get(ctx, 0) + cnt
    vocab     = sorted(c for c, n in totals.items() if n >= min_total)
    ctx_index = {c: i for i, c in enumerate(vocab)}
    return vocab, ctx_index


def _context_marginals(cooc, vocab, ctx_index, total_count):
    totals = np.zeros(len(vocab), dtype=np.float64)
    for wd in cooc.values():
        for dc in wd.values():
            for ctx, cnt in dc.items():
                if ctx in ctx_index:
                    totals[ctx_index[ctx]] += cnt
    return totals / total_count


def _word_marginals(cooc, total_count):
    return {w: sum(sum(d.values()) for d in dm.values()) / total_count
            for w, dm in cooc.items()}


def _ppmi_vector(dec_ctx, ctx_index, V, p_word, p_ctx, total_count):
    raw = np.zeros(V, dtype=np.float64)
    for ctx, cnt in dec_ctx.items():
        if ctx in ctx_index:
            raw[ctx_index[ctx]] = cnt
    p_joint = raw / total_count
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_joint / (p_word * p_ctx + 1e-30))
    return np.where(np.isfinite(pmi), np.maximum(0.0, pmi), 0.0)


def _svd_reduce(matrix, svd_dim):
    svd     = TruncatedSVD(n_components=svd_dim, n_iter=10, random_state=42)
    reduced = svd.fit_transform(matrix)
    print(f"[representation] SVD {matrix.shape[1]} → {svd_dim}d  "
          f"(var explained: {svd.explained_variance_ratio_.sum():.1%})")
    return reduced
