"""
visualize.py
------------
WiDiD-style visualisations for semantic shift analysis.

Four plot types
~~~~~~~~~~~~~~~
1. shift_timeline   — semantic shift score per decade for one word
                      (mirrors Fig. 1 / Fig. 4a in Periti et al. 2022)
2. cluster_evolution — cluster prominence per decade for one word
                      (mirrors Fig. 4b in Periti et al. 2022)
3. ranking_bar      — horizontal bar chart ranking all words by shift score
4. heatmap          — shift scores across all words × all decades

Usage
-----
    from src.visualize import plot_word, plot_ranking, plot_heatmap

    plot_word("computer", cluster_results, metrics, save_dir="results/")
    plot_ranking(pred_scores, save_dir="results/")
    plot_heatmap(metrics, save_dir="results/")
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.clustering import ClusterResult
from src.metrics import MetricResult

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        120,
})


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_word(
    word:            str,
    cluster_results: Dict[str, ClusterResult],
    metrics:         Dict[str, MetricResult],
    save_dir:        Optional[str] = None,
    show:            bool          = True,
) -> None:
    """
    Two-panel WiDiD visualisation for a single word.

    Top panel    : polysemy (bar height) + semantic shift (line) over decades.
    Bottom panel : prominence of each cluster per decade (stacked area).
    """
    r = cluster_results[word]
    m = metrics[word]

    decades = sorted(r.assignments.keys())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.6]}
    )
    fig.suptitle(f'"{word}" — semantic shift & cluster evolution', fontsize=14)

    # ── Top panel: shift + polysemy ───────────────────────────────────────────
    shift_decades = [d for d in decades if d in m.semantic_shift]
    shift_vals    = [m.semantic_shift[d] for d in shift_decades]
    polysemy_vals = [m.polysemy[d]       for d in decades]

    ax1.bar(decades, polysemy_vals, color="#BFBFBF", alpha=0.5,
            width=8, label="polysemy (# active clusters)")
    ax1_r = ax1.twinx()
    ax1_r.plot(shift_decades, shift_vals, color="#C44E52",
               linewidth=2, marker="o", markersize=5, label="semantic shift (JSD)")
    ax1_r.set_ylabel("Semantic shift (JSD)", color="#C44E52", fontsize=10)
    ax1_r.tick_params(axis="y", colors="#C44E52")
    ax1_r.set_ylim(bottom=0)

    ax1.set_ylabel("Polysemy", fontsize=10)
    ax1.set_xticks(decades)
    ax1.set_xticklabels([str(d) for d in decades], rotation=45, ha="right", fontsize=8)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # ── Bottom panel: cluster prominence (stacked area) ────────────────────────
    cluster_ids = sorted(r.prototypes.keys())
    prom_matrix = np.zeros((len(cluster_ids), len(decades)))

    for j, decade in enumerate(decades):
        prom = m.prominence.get(decade, {})
        for i, cid in enumerate(cluster_ids):
            prom_matrix[i, j] = prom.get(cid, 0.0)

    ax2.stackplot(
        decades,
        prom_matrix,
        labels=[f"Cluster {cid}" for cid in cluster_ids],
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(cluster_ids))],
        alpha=0.80,
    )
    ax2.set_ylabel("Prominence (proportion)", fontsize=10)
    ax2.set_xlabel("Decade", fontsize=10)
    ax2.set_xticks(decades)
    ax2.set_xticklabels([str(d) for d in decades], rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.7)

    plt.tight_layout()
    _save_or_show(fig, save_dir, f"{word}_shift.png", show)


def plot_ranking(
    pred_scores: Dict[str, float],
    gold_scores: Optional[Dict[str, float]] = None,
    save_dir:    Optional[str] = None,
    show:        bool          = True,
) -> None:
    """
    Horizontal bar chart ranking words by predicted shift score.
    Overlays gold score markers if provided.
    """
    words_sorted = sorted(pred_scores, key=pred_scores.get)
    preds        = [pred_scores[w] for w in words_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(words_sorted) * 0.45)))
    bars = ax.barh(words_sorted, preds, color="#4C72B0", alpha=0.8)

    if gold_scores:
        gold_vals = [gold_scores.get(w, None) for w in words_sorted]
        for i, (w, gv) in enumerate(zip(words_sorted, gold_vals)):
            if gv is not None:
                # Normalise gold to pred scale for visual comparison
                max_pred = max(preds) if max(preds) > 0 else 1
                ax.plot(gv * max_pred, i, marker="D", color="#C44E52",
                        markersize=7, zorder=5)

    ax.set_xlabel("Shift score", fontsize=11)
    ax.set_title("Word ranking by semantic shift score", fontsize=13)

    if gold_scores:
        from matplotlib.lines import Line2D
        legend_els = [
            plt.Rectangle((0, 0), 1, 1, fc="#4C72B0", alpha=0.8),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#C44E52",
                   markersize=8),
        ]
        ax.legend(legend_els, ["predicted score", "gold score (scaled)"],
                  loc="lower right", fontsize=9)

    for bar, val in zip(bars, preds):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "ranking.png", show)


def plot_heatmap(
    metrics:  Dict[str, MetricResult],
    save_dir: Optional[str] = None,
    show:     bool          = True,
) -> None:
    """
    Heatmap of semantic shift scores: words (rows) × decades (columns).
    Bright cells = high shift at that decade for that word.
    """
    # Collect all decades across all words
    all_decades = sorted({d for m in metrics.values() for d in m.semantic_shift})
    words       = sorted(metrics.keys())

    mat = np.zeros((len(words), len(all_decades)))
    for i, word in enumerate(words):
        for j, dec in enumerate(all_decades):
            mat[i, j] = metrics[word].semantic_shift.get(dec, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(all_decades) * 0.7),
                                    max(4, len(words) * 0.55)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(all_decades)))
    ax.set_xticklabels([str(d) for d in all_decades], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_title("Semantic shift (JSD) across words and decades", fontsize=13)
    ax.set_xlabel("Decade")

    plt.colorbar(im, ax=ax, label="JSD shift score", fraction=0.03, pad=0.04)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "heatmap.png", show)


def plot_sense_shift(
    word:            str,
    cluster_results: Dict[str, ClusterResult],
    metrics:         Dict[str, MetricResult],
    save_dir:        Optional[str] = None,
    show:            bool          = True,
) -> None:
    """
    Bubble chart: each cluster is a row; each decade a column.
    Bubble size = prominence, bubble colour intensity = sense shift magnitude.
    Mirrors Fig. 3b / Fig. 4b of Periti et al. 2022.
    """
    r       = cluster_results[word]
    m       = metrics[word]
    decades = sorted(r.assignments.keys())
    cids    = sorted(r.prototypes.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(decades) * 0.7),
                                    max(3, len(cids) * 0.6)))
    ax.set_title(f'"{word}" — sense-nodule prominence & sense shift', fontsize=13)

    for row, cid in enumerate(cids):
        for col, dec in enumerate(decades):
            prom = m.prominence.get(dec, {}).get(cid, 0.0)
            if prom < 0.001:
                continue
            ss = m.sense_shift.get(dec, {}).get(cid, 0.0)
            size  = max(20, prom * 1800)
            color = plt.cm.RdYlGn_r(min(1.0, ss * 4))
            ax.scatter(dec, row, s=size, color=color, alpha=0.85,
                       edgecolors="gray", linewidths=0.4, zorder=3)
            if prom > 0.05:
                ax.text(dec, row, f"{prom:.2f}", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")

    ax.set_xticks(decades)
    ax.set_xticklabels([str(d) for d in decades], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels([f"cluster {c}" for c in cids], fontsize=9)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Sense cluster")
    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                norm=plt.Normalize(vmin=0, vmax=0.25))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="sense shift (cosine distance)", fraction=0.03)
    plt.tight_layout()
    _save_or_show(fig, save_dir, f"{word}_sense_shift.png", show)


# ── Internal ──────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_dir, filename, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"[visualize] Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)
