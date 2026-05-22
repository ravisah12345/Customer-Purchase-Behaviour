# Customer Purchase Behaviour Using E-Commerce Data

**BSc Dissertation — York St John University — 2026**
**Student:** Ravi Kumar Sah | **Student No:** 230209875

---

## Overview

An end-to-end machine learning pipeline that predicts whether a retail customer will make a repeat purchase, segments customers by purchase behaviour, and generates personalised product recommendations using a three-stage hybrid architecture.

Built on the UCI Online Retail dataset covering 541,909 transactions from 4,332 customers across 37 countries between December 2010 and December 2011.

---

## System Architecture

```
Raw Dataset (541,909 transactions)
            │
            ▼
┌───────────────────────────┐
│  Data Cleaning            │
│  - Drop missing CustomerID│
│  - Remove cancellations   │
│  - Remove UnitPrice <= 0  │
│  - Clip top 0.1% outliers │
│  397,484 rows retained    │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  Feature Engineering      │
│  4,332 customer profiles  │
│  - tenure_days            │
│  - recency_days           │
│  - total_spend            │
│  - unique_products        │
│  - items, avg_line_value  │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────────────────────────────┐
│            Repeat Buyer Classification            │
│                                                   │
│  Logistic Regression | Decision Tree | Random Forest│
│       96.31%         |    97.46%     |   98.50% │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│          Critical Evaluation (Section 4.6)        │
│  Standard split 98.50% -> Temporal split 99.15%  │
│  Cold-start -> confirms hybrid design needed      │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│         Customer Segmentation (3 Stages)          │
│                                                   │
│  New (1 invoice)      -> 1,497 customers (34.6%) │
│  Early (2-3 invoices) -> 1,334 customers (30.8%) │
│  Established (4+)     -> 1,501 customers (34.6%) │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│         Hybrid Recommendation Pipeline            │
│                                                   │
│  Not repeat buyer  --> Popularity (country)       │
│  New stage         --> Popularity (country)       │
│  Early stage       --> Apriori Market Basket      │
│  Established       --> Collaborative Filtering    │
│                                                   │
│  Fallback: Global popularity if method returns 0  │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
        Interactive Streamlit Dashboard
           (6 sections — see below)
```

---

## What the system does

When you run the pipeline, here is what happens in order:

1. Raw data is cleaned — cancellations removed, missing customer IDs dropped, prices fixed, outliers clipped. 397,484 rows survive out of 541,909.
2. Each customer gets a profile — tenure, recency, total spend, number of products, average order value.
3. Three classifiers are trained to predict whether a customer will return: Logistic Regression, Decision Tree, and Random Forest.
4. The best model (Random Forest, 98.5% accuracy) is saved and evaluated three ways — standard split, temporal split, and cold-start.
5. Customers are grouped into three stages: new (1 invoice), early (2-3 invoices), and established (4+).
6. Each stage gets a recommendation method — popularity for new, Apriori market basket for early, collaborative filtering for established.
7. All three recommenders are evaluated using Precision@K, Recall@K, NDCG, MAP, Lift and Coverage at K=5 and K=10.
8. Everything gets written to CSV files and figures in the outputs folder.

---

## Results

### How the classifiers compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **98.50%** | **99.64%** | **98.06%** | **98.84%** |
| Decision Tree | 97.46% | 97.89% | 98.24% | 98.06% |
| Logistic Regression | 96.31% | 99.81% | 94.53% | 97.10% |

Random Forest won comfortably and was saved as the production model.

---

### Is the model actually good or just lucky?

This was a fair question from supervision — 98.5% looks suspiciously high. So three evaluations were run:

| Test | Accuracy | F1 | What it means |
|---|---|---|---|
| Standard random split | 98.50% | 0.9884 | Baseline |
| Temporal split (train early, test later) | 99.15% | 0.9952 | Model works across time too |
| Cold-start (brand new customers) | N/A | N/A | Cannot be evaluated — explains why the hybrid pipeline exists |

The temporal result is actually higher than the standard one, which means the model is not just memorising patterns — it genuinely generalises. The cold-start test failing is the honest finding: the model needs purchase history to work, and that is the whole reason new customers get a popularity-based fallback instead.

---

### What drives the predictions

| Feature | MDI Importance | Permutation Importance |
|---|---|---|
| tenure_days | 55.7% | 0.339 |
| total_spend | 16.7% | 0.001 |
| items | 10.8% | ~0.000 |
| unique_products | 7.8% | ~0.000 |
| recency_days | 5.6% | ~0.000 |

How long someone has been a customer (tenure) is by far the strongest signal. Country makes no difference at all — permutation importance for every country feature came out at zero.

---

### Customer segments

| Stage | Customers | Avg Spend | Avg Orders | Recommendation method |
|---|---|---|---|---|
| New (1 invoice) | 1,497 (34.6%) | £357.95 | 1.00 | Country-based popularity |
| Early (2-3 invoices) | 1,334 (30.8%) | £814.12 | 2.38 | Apriori market basket |
| Established (4+ invoices) | 1,501 (34.6%) | £4,219.29 | 9.18 | Collaborative filtering |

Established customers spend about 12x more than new ones on average. That gap is why it is worth using a more sophisticated recommendation method for them.

---

### How the recommenders performed

| Method | K | Precision@K | Recall@K | Lift | NDCG@K | MAP@K | Coverage |
|---|---|---|---|---|---|---|---|
| Popularity-based | 5 | 0.1312 | 0.0124 | 96.10x | 0.1320 | 0.0767 | 100% |
| Apriori (Market Basket) | 5 | 0.1333 | 0.0260 | 97.68x | **0.2304** | **0.1375** | 1.39% |
| **Collaborative Filtering** | **5** | **0.1460** | 0.0249 | **106.93x** | 0.1640 | 0.1168 | **100%** |
| Popularity-based | 10 | 0.1055 | 0.0213 | 38.66x | 0.1154 | 0.0523 | 100% |
| Apriori (Market Basket) | 10 | 0.0667 | 0.0260 | 24.42x | 0.1609 | 0.0792 | 1.39% |
| **Collaborative Filtering**  | **10** | 0.1024 | **0.0320** | 37.52x | 0.1307 | 0.0763 | **100%** |

Collaborative filtering comes out on top for precision and lift. Apriori actually ranks items better when it works (NDCG 0.23) but its coverage is only 1.39% — it can only recommend for customers whose purchases match the generated rules, which is a known sparsity problem. Popularity is the safe universal option and covers everyone.

For context — recommending randomly from 3,663 products would give roughly 0.001 precision. All three methods are dramatically better than that.

---

### Sample recommendations from the hybrid pipeline

| Customer | Repeat buyer? | Stage | Top 5 recommendations | Method used |
|---|---|---|---|---|
| 12347 | Yes | Established | 84952C, 84952B, 23509, 16237, 21499 | Collaborative filtering |
| 12348 | Yes | Established | 21975, 84991, 21213, 21212, 21977 | Collaborative filtering |
| 12349 | No | New | 51014A, 23445, 51014C, 22197, 51014L | Popularity (country) |
| 12350 | No | New | 16008, 22693, 22197, 72232, 84050 | Popularity (country) |
| 12352 | Yes | Established | 47590B, 23139, 23138, 22759, 22090 | Collaborative filtering |

---

## Files in this repo

```
project root/
│
├── src/
│   ├── pipeline.py               ← run this first — does everything end to end
│   ├── apriori_rules.py          ← builds baskets and generates association rules
│   ├── cf_recommender.py         ← cosine similarity item-based CF
│   ├── hybrid_pipeline.py        ← decides which method each customer gets
│   └── recommender_metrics.py    ← Precision@K, Recall@K, NDCG, MAP, Lift, Coverage
│
├── data/
│   ├── raw/                      ← put online_retail.xlsx here (not included)
│   └── processed/                ← generated automatically when pipeline runs
│
├── outputs/
│   ├── figures/                  ← charts saved as PNG
│   ├── models/                   ← saved joblib model files (not included, too large)
│   └── tables/                   ← all CSV result files
│
├── dashboard.py                  ← streamlit dashboard (6 sections)
├── demonstrate.py                ← quick terminal summary of results
├── hybrid_pipeline_notebook.ipynb← step by step notebook walkthrough
├── requirements.txt
├── run_dashboard.sh              ← mac/linux launcher
├── run_dashboard.bat             ← windows launcher
└── README.md
```

**Not included in this repo (too large or not redistributable):**
- `data/raw/online_retail.xlsx` — download from Kaggle (link below)
- `data/processed/transactions_clean.csv` — regenerated by pipeline
- `outputs/tables/item_similarity_matrix.csv` — 183MB, regenerated by pipeline
- `outputs/models/*.joblib` — regenerated by pipeline

Run `python src/pipeline.py` and everything gets rebuilt from scratch.

---

## How to run it

```bash
# clone the repo
git clone https://github.com/ravisah12345/Customer-Purchase-Behaviour
cd Customer-Purchase-Behaviour

# set up environment
python -m venv venv
source venv/bin/activate        # mac/linux
venv\Scripts\activate           # windows

# install dependencies
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at `data/raw/online_retail.xlsx`:
> https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

```bash
# run the full pipeline (takes about 8-12 minutes)
python src/pipeline.py

# launch the dashboard
streamlit run dashboard.py

# or just see a terminal summary
python demonstrate.py

# or open the notebook walkthrough
jupyter notebook hybrid_pipeline_notebook.ipynb
```

---

## The dashboard

Six sections, launched with `streamlit run dashboard.py`:

- **Overview** — dataset numbers, cleaning stats, country breakdown
- **Classification** — model comparison, confusion matrix, feature importance
- **Segmentation** — stage distribution, spend by stage
- **Recommenders** — metric comparison across all three methods
- **Interactive demo** — pick any customer ID and see their recommendations live
- **Business impact** — revenue by segment, practical implications

---

## Tech stack

| Library | Version | What it is used for |
|---|---|---|
| pandas | 2.3.3 | data wrangling |
| scikit-learn | 1.6.1 | classifiers, cosine similarity, preprocessing |
| mlxtend | 0.23.4 | Apriori algorithm |
| streamlit | 1.50.0 | dashboard |
| plotly | 6.7.0 | charts in dashboard |
| matplotlib | 3.9.4 | static figures |
| joblib | 1.5.3 | saving and loading models |
| openpyxl | 3.1.5 | reading the xlsx file |
| numpy | 2.0.2 | numerical operations |

---

## Reproducibility

Every random operation uses `random_state=42` so results are identical every time:

```python
train_test_split(..., random_state=42)
RandomForestClassifier(..., random_state=42)
DecisionTreeClassifier(..., random_state=42)
cust.sample(frac=0.2, random_state=42)
```

---

## Dataset

UCI Online Retail II — 541,909 transactions, December 2010 to December 2011, 37 countries.

Download: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Place at `data/raw/online_retail.xlsx` before running the pipeline.

---

*Submitted for academic assessment — York St John University, 2026.*

---




