# amazon-review-sentiment-analysis
# Sentiment Analysis on Amazon Product Reviews

Binary sentiment classification on Amazon product **review text** (`reviewText`) with labels **`Positive`** ∈ {0, 1}.
The notebook walks through dataset → preprocessing → vectorization → multiple models → evaluation → hyperparameter tuning → comparison → conclusions.

**Dataset (public CSV):**
`https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv`

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Environment & Setup](#environment--setup)
* [Data](#data)
* [Workflow](#workflow)
* [Quickstart](#quickstart)
* [Models](#models)
* [Training & Tuning](#training--tuning)
* [Evaluation](#evaluation)
* [Comparative Analysis](#comparative-analysis)
* [Reproducibility](#reproducibility)
* [Troubleshooting](#troubleshooting)
* [Save / Load Model](#save--load-model)
* [License & Use](#license--use)

---

## Project Overview

* **Goal:** Predict **binary sentiment** (1 = positive, 0 = negative) from the free-text review.
* **Why:** Classify product feedback at scale to improve product insights and customer experience.
* **Approach:** TF-IDF features with several lightweight classifiers; tune best models; report metrics and confusion matrices.

---

## Repository Structure

```
.
├─ notebooks/
│  └─ Sentiment Analysis on Amazon Product Reviews.ipynb   # main, step-by-step notebook
├─ models/                                                 # saved models (optional)
├─ data/                                                   # optional local cache
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> Submitting coursework? Keeping everything inside `notebooks/` is fine.

---

## Environment & Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

**Suggested requirements:**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
nltk>=3.8
```

*(Optional)* `imbalanced-learn`, `optuna`, `xgboost`, `lightgbm`, `catboost`

---

## Data

* Columns used:

  * **`reviewText`**: raw review (string)
  * **`Positive`**: label (1 = positive, 0 = negative)
* The notebook:

  * fills missing reviews with empty strings,
  * **cleans text** (lowercase, strip URLs/HTML, non-letters),
  * tokenizes + **Porter stemming** (no internet corpora required).

---

## Workflow

1. **Dataset overview** — shape, head, label distribution, missingness
2. **Preprocessing** — cleaning + stemming → train/test split
3. **Vectorization** — **TF-IDF** with `stop_words='english'`, bigrams, `min_df`, `max_df`
4. **Model selection** — Logistic Regression, Linear SVM, Multinomial NB, Random Forest
5. **Training** — fit models on TF-IDF features
6. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
7. **Tuning** — GridSearchCV for SVM & LR (example)
8. **Comparison** — table + bar chart of F1
9. **Conclusions** — findings, challenges, next steps

---

## Quickstart

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load
url = "https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv"
df = pd.read_csv(url).fillna({"reviewText": ""})

# Minimal clean (same logic as notebook uses, simplified here)
import re
def clean(x):
    x = x.lower()
    x = re.sub(r'https?://\\S+|www\\.\\S+',' ', x)
    x = re.sub(r'<.*?>',' ', x)
    x = re.sub(r'[^a-z\\s]',' ', x)
    return re.sub(r'\\s+',' ', x).strip()

df["clean"] = df["reviewText"].map(clean)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["Positive"], test_size=0.2, random_state=42, stratify=df["Positive"]
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# Train a strong baseline (TF-IDF + LR)
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)

print(classification_report(y_test, pred, zero_division=0))
```

---

## Models

* **Logistic Regression** (strong, fast, interpretable; outputs probabilities)
* **Linear SVM (LinearSVC)** (often top performer on sparse TF-IDF; needs calibration for probabilities)
* **Multinomial Naïve Bayes** (very fast; solid baseline)
* **Random Forest** (included for comparison; not ideal for high-dim, sparse text)

*(Extensions)* Try Gradient Boosting (XGBoost/LightGBM/CatBoost) or neural models (LSTM/GRU/Transformers) if compute allows.

---

## Training & Tuning

* Vectorization: `TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=2, max_df=0.9)`
* Baseline training for each model, then **GridSearchCV** examples for SVM & LR:

  * `C`, `loss` for LinearSVC
  * `C`, `solver` for Logistic Regression

---

## Evaluation

* Metrics reported:

  * **Accuracy**, **Precision**, **Recall**, **F1** (binary)
  * **Confusion Matrix** heatmaps per model
* (Optional) Cross-validation via `StratifiedKFold` and `cross_val_score`.

---

## Comparative Analysis

* Consolidated table of **accuracy/precision/recall/F1** for all models (base + tuned)
* Bar plot of F1 scores
* Notes on **speed, accuracy, interpretability**, and suitability for sparse text

---

## Reproducibility

* Fixed `random_state=42` for splits and CV
* Deterministic CV (`StratifiedKFold`)
* Pinned environment via `requirements.txt`

---

## Troubleshooting

* **Imbalanced labels:** check label counts; consider class weighting or stratified sampling
* **Slow training:** reduce n-gram range, increase `min_df`, or sample subset for prototyping
* **Overfitting:** tighten regularization (`C`↓ for LR/SVM), simplify features
* **No internet corpora:** use PorterStemmer and sklearn stopwords (as in the notebook)

---

## Save / Load Model

```python
import joblib
joblib.dump((tfidf, clf), "models/tfidf_logreg.joblib")

# later
tfidf_loaded, clf_loaded = joblib.load("models/tfidf_logreg.joblib")
```

---

## License & Use

* For educational purposes only.
* Ensure data privacy/compliance when handling user reviews.
