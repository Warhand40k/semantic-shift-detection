# Semantic Shift Detection via Incremental Clustering

**NLP Course Project** — Emergence of New Meanings for Existing Words

Detects how words change meaning over time using Google Books Ngrams (1850–2009),
adapting the WiDiD incremental clustering approach (Periti et al., 2022) to
distributional vector representations.

---

## Method Overview

```
Google Books 2-grams
       │
       ▼
PPMI co-occurrence vectors  (per word, per decade)
       │
       ▼
Shared Truncated SVD  →  100-dim space
       │
       ▼
Temporal smoothing  (window=2)
       │
       ▼
Incremental clustering  (A Posteriori AP — WiDiD-inspired)
       │
       ▼
Cluster analysis  →  polysemy · semantic shift (JSD) · sense shift (cosine)
       │
       ▼
Evaluation  →  Spearman ρ · binary accuracy
```

### Key design choices

| Component | Choice | Rationale |
|---|---|---|
| Representation | PPMI + Truncated SVD | Standard for diachronic distributional semantics; no GPU required |
| Shared SVD | Yes | Ensures cross-decade cosine distances are meaningful |
| Temporal smoothing | Window = 2 | Suppresses corpus sampling noise between consecutive decades |
| Clustering | Threshold-based incremental AP | Adapts WiDiD to single-vector-per-decade setting |
| Threshold | Auto-calibrated (40th percentile) | Data-driven; avoids hand-tuning per word |
| Shift score | 0.5 × mean\_JSD + 0.3 × max\_JSD + 0.2 × polysemy | Composite; avoids score collapse from max-only |

### Innovation over WiDiD (Periti et al. 2022)

The original WiDiD clusters *multiple* contextualised BERT embeddings per time period
(one per sentence). We adapt it to *one* PPMI+SVD vector per decade, which is the
natural unit of Google Books Ngrams. This requires:

- Replacing multi-embedding AP with single-point threshold assignment
- Adding temporal smoothing to compensate for the loss of within-period variance
- Auto-calibrating the threshold from the data's distance distribution
- A composite shift score to produce a continuous ranking (not just peak detection)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/semantic-shift-detection
cd semantic-shift-detection
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Full pipeline — downloads data (~11 GB), builds vectors, clusters, evaluates
python run.py

# Tune the clustering threshold manually
python run.py --threshold 0.30

# Use A Posteriori Affinity Propagation (closer to original WiDiD)
python run.py --ap

# Force recompute (ignore all caches)
python run.py --force
```

Results are saved to `results/summary.txt` and `results/summary_table.csv`.

## Case study notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook covers evaluation, per-word case studies, threshold sensitivity,
smoothing ablation, and baseline comparison.

---

## Results

| Word | Shift score | Gold | Clusters | Peak decade |
|------|-------------|------|----------|-------------|
| virus | 0.203 | 0.90 | 7 | 1860 |
| computer | 0.194 | 0.95 | 4 | 1920 |
| cell | 0.181 | 0.70 | 6 | 1860 |
| mouse | 0.169 | 0.55 | 5 | 1860 |
| web | 0.150 | 0.85 | 4 | 1860 |
| gay | 0.081 | 0.95 | 4 | 1890 |
| stream | 0.081 | 0.65 | 4 | 1890 |
| cloud | 0.074 | 0.80 | 4 | 1900 |
| crash | 0.031 | 0.40 | 2 | 1940 |
| tablet | 0.027 | 0.75 | 2 | 1970 |
| tweet | 0.000 | 0.90 | 1 | — |
| spam | 0.000 | 0.85 | 1 | — |

**Evaluation:** Spearman ρ = +0.173 (p=0.592)  ·  Binary accuracy = 100.0% (threshold = 0.000)

**Baseline comparison:** WiDiD-inspired method (ρ=0.173) vs. max-cosine baseline (ρ=0.070) · Both achieve 100% binary accuracy.

---

## Project structure

```
semantic-shift-detection/
├── data/
│   ├── raw/              # downloaded .gz ngram files (~11 GB, gitignored)
│   └── processed/        # cached .pkl files (gitignored)
├── src/
│   ├── data_loader.py    # download + parse Google Books 2-grams
│   ├── representation.py # PPMI vectors + SVD + smoothing
│   ├── clustering.py     # WiDiD-inspired incremental AP
│   ├── metrics.py        # polysemy, JSD shift, sense shift, prominence
│   ├── evaluation.py     # Spearman ρ, binary accuracy, gold reference
│   └── visualize.py      # shift timeline, cluster evolution, heatmap
├── notebooks/
│   └── analysis.ipynb    # case studies + innovation analysis
├── results/              # plots + CSV (gitignored)
├── run.py                # end-to-end pipeline script
├── quick_test.py         # sanity check on synthetic data
└── requirements.txt
```

---

## References

- Periti et al. (2022). *What is Done is Done: an Incremental Approach to Semantic Shift Detection.* LChange@ACL 2022.
- Periti et al. (2025). *Studying word meaning evolution through incremental semantic shift detection.* Language Resources and Evaluation 59, 1363–1399.
- Schlechtweg et al. (2020). *SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection.* SemEval 2020.
- Hamilton et al. (2016). *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change.* ACL 2016.
- Michel et al. (2011). *Quantitative Analysis of Culture Using Millions of Digitized Books.* Science 331(6014).
