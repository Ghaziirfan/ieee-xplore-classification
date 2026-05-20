# TF-IDF and Logistic Regression Pipeline for Scholarly Article Classification and Recommendation

A unified content-based framework that uses a single TF-IDF representation of scholarly abstracts to perform both:
- Multi-class topical classification  
- Top-k scholarly article recommendation  

Evaluated on a 11,744-abstract five-domain IEEE Xplore benchmark dataset.

---

## 📄 Paper

> **A TF-IDF and Logistic Regression Pipeline for Scholarly Article Classification and Recommendation: IEEE Xplore Benchmark Study**  
> Ghazi Irfan and Ghulam Mustafa  
> *IEEE Access (under review), 2026*

---

## 📁 Repository Structure

```
.
├── data/
├── notebooks/
├── figures/
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- Total: **11,744 abstracts**
- Source: IEEE Xplore CSV export
- 5-class classification problem

### Label Mapping

```python
LABEL_MAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
```

Final distribution:

```
{0: 4000, 1: 1744, 2: 2000, 3: 2000, 4: 2000}
```

---

## ⚙️ TF-IDF Pipeline

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    max_features=50000,
    norm="l2"
)
```

---

## 🤖 Models Used

- Logistic Regression  
- Linear SVM  
- SGD Classifier  
- KNN (cosine)  
- Decision Tree  
- Soft Voting Ensemble  

---

## 🔁 Recommendation Methods

- Cosine similarity (baseline)
- Logistic Regression probability re-ranking

---

## 🚀 How to Run

```bash
git clone https://github.com/ghazi-irfan/ieee-xplore-classification.git
cd ieee-xplore-classification
pip install -r requirements.txt
jupyter notebook
```

Run:
- classification_pipeline.ipynb  
- recommendation_system.ipynb  

---

## 📌 Reproducibility

- random_state = 100
- Stratified split (80/20)
- 10-fold CV included

---

## 📜 Citation

```bibtex
@article{irfan2026tfidf,
  author  = {Irfan, Ghazi and Mustafa, Ghulam},
  title   = {A TF-IDF and Logistic Regression Pipeline for Scholarly Article Classification and Recommendation},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📧 Contact

- Ghazi Irfan — ghazi.irfan@ucp.edu.pk  
- Ghulam Mustafa — ghulammustafa02@ucp.edu.pk  

---

## 🙏 Acknowledgements

Bahria University Lahore Campus and University of Central Punjab for research support.
```
