# Sentiment Analysis on Amazon Product Reviews

Binary sentiment classification on Amazon product **review text** (`reviewText`) with labels **`Positive` ∈ {0,1}**.
This repo includes a complete, reproducible notebook covering: dataset → preprocessing → TF-IDF vectorization → multiple models → evaluation → tuning → comparison → conclusions.

**Dataset (public CSV):**
`https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv`

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Environment & Setup](#environment--setup)
* [Data](#data)
* [Methodology](#methodology)
* [Results](#results)
* [How to Reproduce](#how-to-reproduce)
* [Model Cards (Concise)](#model-cards-concise)
* [Troubleshooting](#troubleshooting)
* [Next Steps](#next-steps)
* [License](#license)

---

## Project Overview

* **Goal:** Predict review sentiment (0 = negative, 1 = positive) from text.
* **Approach:** Clean & stem text → TF-IDF (uni/bi-grams) → classical ML (LinearSVC, Logistic Regression, MultinomialNB, RandomForest) → evaluate with Accuracy/Precision/Recall/F1 + confusion matrices → tune best models with GridSearchCV.

---

## Repository Structure

```
.
├─ notebooks/
│  └─ Sentiment Analysis on Amazon Product Reviews.ipynb   # main end-to-end notebook
├─ models/                                                 # saved models (optional)
├─ data/                                                   # optional local cache
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> For coursework submission, keeping everything under `notebooks/` is fine.

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

**Suggested versions:**
`pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `seaborn>=0.12`, `nltk>=3.8`
*(Optional)* `imbalanced-learn`, `optuna`, `xgboost`, `lightgbm`, `catboost`

---

## Data

* **Columns:**

  * `reviewText` — raw review text (string)
  * `Positive` — binary label (1 positive, 0 negative)
* **Preprocessing in notebook:**

  * Fill missing reviews with empty strings
  * Lowercasing, strip URLs/HTML/non-letters
  * Tokenize + **Porter stemming** (no internet corpora needed)
  * Train/test split with stratification

---

## Methodology

1. **Text vectorization:** `TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.9)`
2. **Models trained:** LinearSVC, Logistic Regression, Multinomial Naïve Bayes, RandomForest
3. **Tuning:** GridSearchCV for LinearSVC (`C`, `loss`) and Logistic Regression (`C`, `solver`)
4. **Metrics:** Accuracy, Precision, Recall, F1 (binary); confusion matrices for each model
5. **Comparison:** Tabular & bar-chart comparison of F1 scores

---

## Results

| model             | acc     | prec     | rec      | f1       |
| ----------------- | ------- | -------- | -------- | -------- |
| LinearSVC         | 0.90575 | 0.925702 | 0.952740 | 0.939026 |
| LinearSVC (tuned) | 0.90575 | 0.925702 | 0.952740 | 0.939026 |
| LogReg (tuned)    | 0.89850 | 0.907939 | 0.964555 | 0.935391 |
| LogReg            | 0.88950 | 0.895056 | 0.968494 | 0.930328 |
| RandomForest      | 0.87650 | 0.876882 | 0.974729 | 0.923220 |
| MultinomialNB     | 0.82925 | 0.817571 | 0.998687 | 0.899099 |

**Takeaways**

* **LinearSVC** achieves the best F1 (≈ **0.939**), typical for sparse TF-IDF text.
* **Logistic Regression** is a very strong, interpretable baseline (F1 up to **0.935** after tuning).
* **MultinomialNB** has very high recall but lower precision (predicts positive too often).
* **RandomForest** underperforms on high-dimensional sparse text and can be slower.

**Strengths / Weaknesses (quick)**

* **LinearSVC:** strong on sparse TF-IDF, fast; needs calibration for probabilities.
* **LogisticRegression:** robust, interpretable; supports calibrated probabilities.
* **MultinomialNB:** simple/fast; may struggle with large n-gram space (precision drop).
* **RandomForest:** less suited to sparse, high-dimensional text; slower; risk of overfit.

---

## How to Reproduce

Minimal training block from the notebook:

```python
import pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load
url = "https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv"
df = pd.read_csv(url).fillna({"reviewText": ""})

# Simple clean
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
Xtr = tfidf.fit_transform(X_train); Xte = tfidf.transform(X_test)

# Train + report
clf = LinearSVC()
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)
print(classification_report(y_test, pred, zero_division=0))
```

> For Logistic Regression, replace `LinearSVC()` with `LogisticRegression(max_iter=2000, n_jobs=-1)`.

---

## Model Cards (Concise)

**LinearSVC**

* **Features:** TF-IDF (1–2 grams, English stopwords, min_df=2, max_df=0.9)
* **Key hyperparams:** `C` tuned ∈ {0.5, 1.0, 2.0}, `loss` ∈ {hinge, squared_hinge}
* **Pros:** top F1, fast; **Cons:** needs calibration for probabilities

**Logistic Regression**

* **Key hyperparams:** `C` tuned ∈ {0.5, 1.0, 2.0}; `solver` ∈ {liblinear, lbfgs}
* **Pros:** interpretable, robust, probability outputs; **Cons:** slightly behind LinearSVC

**MultinomialNB**

* **Pros:** ultra fast; **Cons:** precision drop with bigrams/high-dim

**RandomForest**

* **Pros:** non-linear; **Cons:** slower and weaker on sparse TF-IDF

---

## Troubleshooting

* **Slow training:** reduce n-gram range to (1,1), increase `min_df`, or sample data for prototyping.
* **Imbalance:** check label counts; use stratified splits; consider class weights / threshold tuning.
* **Overfitting:** increase regularization (lower `C`), simplify features.
* **Need probabilities for SVM:** use **CalibratedClassifierCV** (Platt scaling).

---

## Next Steps

* Try **character n-grams** and/or tri-grams.
* Add **probability calibration** and threshold tuning for precision/recall trade-offs.
* Explore **transformer models** (DistilBERT) for potential gains.
* Add cross-domain validation and error analysis (top false positives/negatives).

---

## License

Educational use only. Ensure data privacy and compliance when handling user-generated content.
