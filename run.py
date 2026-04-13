"""
run.py
------
End-to-end pipeline: data → vectors → clusters → metrics → evaluation.

Run:
    python run.py                     # full pipeline
    python run.py --force             # ignore all caches, recompute everything
    python run.py --threshold 0.30    # tune the clustering distance threshold
    python run.py --ap                # use A Posteriori AP instead of threshold rule

Output:
    results/summary.txt   — per-word shift scores + evaluation
    results/clusters.txt  — decade-by-decade cluster assignments
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Semantic Shift Detection via WiDiD-inspired clustering")
    parser.add_argument("--force",     action="store_true", help="Ignore caches")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Cosine-distance threshold for new cluster creation (default 0.25)")
    parser.add_argument("--ap",        action="store_true",
                        help="Use A Posteriori Affinity Propagation instead of threshold rule")
    parser.add_argument("--svd-dim",   type=int, default=100,
                        help="SVD dimensionality for PPMI vectors (default 100)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # ── Phase 1: Data ──────────────────────────────────────────────────────────
    print("\n── Phase 1: Loading co-occurrence data ──")
    from src.data_loader import load_cooccurrences
    cooc = load_cooccurrences(force=args.force)

    # ── Phase 2: Vectors ───────────────────────────────────────────────────────
    print("\n── Phase 2: Building PPMI + SVD vectors ──")
    from src.representation import build_vectors
    vectors = build_vectors(cooc, svd_dim=args.svd_dim, force=args.force)

    # ── Phase 3: Clustering ────────────────────────────────────────────────────
    print("\n── Phase 3: Incremental clustering ──")
    from src.clustering import cluster_all
    cluster_results = cluster_all(
        vectors,
        threshold=args.threshold,
        use_ap=args.ap,
        force=args.force,
    )

    # ── Phase 4: Metrics ───────────────────────────────────────────────────────
    print("\n── Phase 4: Computing metrics ──")
    from src.metrics import compute_metrics, shift_score
    metrics = compute_metrics(cluster_results, force=args.force)

    # ── Phase 5: Evaluation ────────────────────────────────────────────────────
    print("\n── Phase 5: Evaluation ──")
    from src.evaluation import evaluate
    pred_scores = {w: shift_score(m) for w, m in metrics.items()}
    eval_result = evaluate(pred_scores)
    print(eval_result.summary())

    # ── Save results ───────────────────────────────────────────────────────────
    summary_path = os.path.join("results", "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(eval_result.summary())
        fh.write("\n\n")
        fh.write("Per-decade cluster assignments\n")
        fh.write("=" * 52 + "\n")
        for word, r in sorted(cluster_results.items()):
            fh.write(f"\n{word} ({r.num_clusters} cluster(s))\n")
            for decade, cid in sorted(r.assignments.items()):
                shift = metrics[word].semantic_shift.get(decade, 0.0)
                fh.write(f"  {decade}  cluster={cid}  shift={shift:.4f}\n")

    print(f"\n[run] Results saved to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
