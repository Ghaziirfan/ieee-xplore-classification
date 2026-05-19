# A TF-IDF and Logistic Regression Pipeline for Scholarly Article Classification and Recommendation

A unified content-based framework that uses a single TF-IDF representation of scholarly abstracts to drive both multi-class topical classification and top-*k* article recommendation, evaluated on a new five-domain IEEE *Xplore* benchmark of 11,744 abstracts.

This repository contains the dataset, preprocessing pipeline, trained models, and replication notebooks for the paper:

> **A TF-IDF and Logistic Regression Pipeline for Scholarly Article Classification and Recommendation: A Five-Domain IEEE Xplore Benchmark Study**
> Ghazi Irfan and Ghulam Mustafa
> *IEEE Access* (under review), 2026

---

## Headline results

Reproducible bit-exactly from the notebooks below on the same library versions.

**Classification (held-out 2,349-abstract test set, abstract-only TF-IDF):**

| Model | Hold-out accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression (tuned, *C* = 5.0) | **85.01%** | 0.850 |
| Linear SVM | 84.55% | — |
| SGD Classifier | 85.65% | — |
| Soft-voting ensemble (LR + SVM + SGD) | **85.57%** | 0.855 |
| *k*-Nearest Neighbours (cosine) | 75.01% | — |
| Decision Tree (CART) | 78.25% | — |

SGD has the best 10-fold cross-validation accuracy (86.59% ± 0.73). Logistic Regression is used as the headline classifier because its calibrated probabilities also drive the recommendation re-ranker.

**Top-10 recommendation (2,349 queries against 9,395 candidates):**

| Metric | Pure cosine | LR-reranked | Δ |
|---|---|---|---|
| Precision@10 | 0.6467 | **0.7715** | +12.48 pp |
| MAP@10 | 0.7664 | **0.8428** | +7.64 pp |
| NDCG@10 | 0.8525 | **0.8902** | +3.77 pp |
| MRR | 0.8502 | **0.8871** | +3.69 pp |

---

## Repository structure

```
.
├── README.md                          # this file
├── requirements.txt                   # pinned Python dependencies
├── data/
│   ├── BigDataAnalysis.csv            # 2,000 abstracts, source class 0
│   ├── Cloudcomputing.csv             # 2,000 abstracts, source class 0 (merged with BDA)
│   ├── DataScience.csv                # 1,744 abstracts, source class 2
│   ├── Robotics.csv                   # 2,000 abstracts, source class 3
│   ├── Wireless.csv                   # 2,000 abstracts, source class 4
│   └── BreastCancer.csv               # 2,000 abstracts, source class 5
├── notebooks/
│   ├── classification_pipeline.ipynb  # end-to-end classification pipeline
│   └── recommendation_system.ipynb    # top-k recommender + LR re-ranking
└── figures/
    ├── Figure1.png                    # methodology pipeline diagram
    ├── Figure2.png                    # class-distribution histogram
    ├── Figure3.png                    # multi-classifier accuracy comparison
    ├── Figure4.png                    # confusion matrix (tuned LR)
    ├── Figure5.png                    # one-vs-rest ROC curves
    └── Figure6.png                    # recommendation comparison
```

---

## Dataset

Six per-class CSV files, exported from the IEEE *Xplore* digital library via its built-in CSV export, totalling 11,744 records. Each file contains only two fields: the article `Abstract` text and the `Class` label assigned by the source topical query. All other metadata (titles, authors, indexer-supplied terms, citation count, year, DOI, etc.) has been dropped — the entire pipeline operates on abstract text alone, with no recourse to author keywords or human-curated indexer tags.

**Source-CSV class labels (non-contiguous, as exported):**

| File | Class in CSV | Domain |
|---|---|---|
| `BigDataAnalysis.csv` | 0 | Big Data Analysis |
| `Cloudcomputing.csv` | 0 | Cloud Computing (already merged with BDA in source) |
| `DataScience.csv` | 2 | Data Science |
| `Robotics.csv` | 3 | Robotics |
| `Wireless.csv` | 4 | Wireless Communication |
| `BreastCancer.csv` | 5 | Breast Cancer |

The notebooks remap these to a contiguous `0..4` set during loading:

```python
LABEL_MAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
LABEL_NAMES = {
    0: "Big Data & Cloud Computing",
    1: "Data Science",
    2: "Robotics",
    3: "Wireless Communication",
    4: "Breast Cancer",
}
```

After remap and concatenation, the working dataset has the distribution `{0: 4000, 1: 1744, 2: 2000, 3: 2000, 4: 2000}` — total 11,744 rows.

**Why Big Data Analysis and Cloud Computing are merged.** A diagnostic six-class confusion analysis revealed that, at the abstract level, IEEE *Xplore*'s "Big Data Analysis" and "Cloud Computing" queries return papers with near-identical vocabulary — a symmetric off-diagonal mass of ~295–300 articles per direction in the confusion matrix and per-class F1 < 0.10 for both classes. Section 3.3 of the paper documents the full diagnostic.

---

## Setup

### Requirements

- Python 3.9
- scikit-learn 1.0
- NLTK 3.6 (with `stopwords`, `punkt`, `punkt_tab`, and `wordnet` data files)
- pandas 1.3
- NumPy 1.21
- matplotlib 3.5+ (for figure generation)
- Jupyter (for notebook execution)

### Installation

```bash
git clone https://github.com/ghazi-irfan/ieee-xplore-classification.git
cd ieee-xplore-classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NLTK data is downloaded automatically the first time a notebook runs (the relevant `nltk.download(...)` calls are at the top of each notebook).

---

## Reproducing the results

The pipelines fix `random_state=100` everywhere a stochastic operation is involved, so the numbers reported below should reproduce bit-exactly on the pinned library versions.

### Classification (≈ 15 minutes on commodity CPU)

```bash
jupyter notebook notebooks/classification_pipeline.ipynb
```

Then *Restart Kernel and Run All*. The notebook performs:

1. Loading and concatenation of the six per-class CSVs
2. Label remap to contiguous `0..4`
3. Seven-stage text cleaning (case normalisation, punctuation removal, digit removal, NLTK tokenisation, stop-word removal, short-token removal, WordNet lemmatisation)
4. TF-IDF vectorisation (unigrams + bigrams, sublinear TF, ℓ₂ normalisation, `min_df=3`, `max_df=0.90`, `max_features=50000`)
5. 80 / 20 stratified train / test split
6. Five classifiers under hold-out + 10-fold cross-validation
7. Grid search over Logistic Regression's *C*
8. Soft-voting ensemble of LR + SVM + SGD
9. Per-class precision / recall / F1 + confusion matrix

Expected output: tuned LR achieves 85.01% on hold-out (weighted F1 = 0.850); ensemble reaches 85.57% (weighted F1 = 0.855); SGD has the best 10-fold mean (86.59% ± 0.73).

### Recommendation

```bash
jupyter notebook notebooks/recommendation_system.ipynb
```

This notebook reuses the same TF-IDF substrate and `random_state=100` split, then evaluates two ranking strategies for top-10 retrieval:

- **Method A** — pure cosine similarity over abstract-only TF-IDF vectors
- **Method B** — cosine similarity re-weighted multiplicatively by Logistic Regression's calibrated class probabilities

Relevance proxy: same-class-as-query (the standard for content-based scholarly recommendation).

Expected output: cosine-only MAP@10 = 0.7664; LR-reranked MAP@10 = 0.8428 (+7.64 pp lift). Precision@10 sees the largest gain (+12.48 pp).

---

## Pipeline configuration (TF-IDF and classifiers)

These settings are applied identically in both notebooks so the classifier and the recommender share the same TF-IDF substrate.

**TF-IDF Vectoriser (scikit-learn `TfidfVectorizer`):**

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    norm="l2",
    max_features=50_000,
)
```

**Logistic Regression (used both standalone and in the ensemble + re-ranker):**

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=5.0,             # selected by 5-fold inner GridSearchCV over {0.1, 0.5, 1, 2, 5, 10}
    solver="lbfgs",
    multi_class="multinomial",
)
```

The full hyper-parameter table for all five classifiers is in Table III of the paper.

---

## Citation

If you use this dataset or code, please cite the paper:

```bibtex
@article{irfan2026tfidf,
  author  = {Irfan, Ghazi and Mustafa, Ghulam},
  title   = {A {TF-IDF} and Logistic Regression Pipeline for Scholarly
             Article Classification and Recommendation: A Five-Domain
             {IEEE} {Xplore} Benchmark Study},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

- **Code (notebooks)** — MIT License. See `LICENSE`.
- **Dataset** — the abstracts were exported from the IEEE *Xplore* digital library and are subject to IEEE's terms of use. Redistribution is provided for academic-research reproducibility under fair-use terms; contact IEEE for commercial-use licensing.

---

## Contact

For questions about the dataset, code, or paper:

- **Ghazi Irfan** — `ghazi.irfan@ucp.edu.pk` *(Department of Computer Science, University of Central Punjab, Lahore, Pakistan)*
- **Ghulam Mustafa** — `ghulammustafa02@ucp.edu.pk`

For bug reports or reproducibility issues, please open a GitHub issue at <https://github.com/ghazi-irfan/ieee-xplore-classification/issues>.

---

## Acknowledgements

This work was originally conducted at Bahria University Lahore Campus, Pakistan, as part of the first author's Master's research. The authors thank Dr. Muhammad Aasim Qureshi for co-supervisory input during the initial dataset construction.
#   i e e e - x p l o r e - c l a s s i f i c a t i o n  
 #   i e e e - x p l o r e - c l a s s i f i c a t i o n  
 