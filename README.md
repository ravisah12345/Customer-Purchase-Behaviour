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
│       96.31%         |    97.46%     |   98.50% ⭐  │
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

## Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** ⭐ | **98.50%** | **99.64%** | **98.06%** | **98.84%** |
| Decision Tree | 97.46% | 97.89% | 98.24% | 98.06% |
| Logistic Regression | 96.31% | 99.81% | 94.53% | 97.10% |

Random Forest was selected as the best model and saved to `outputs/models/`.

---

### Critical Evaluation

| Evaluation Type | Accuracy | F1 | Notes |
|---|---|---|---|
| Standard Random Split | 98.50% | 0.9884 | Baseline — same time period |
| **Temporal Split** | **99.15%** | **0.9952** | Train on early customers, test on later |
| Cold-Start | N/A | N/A | Structurally not evaluable — confirms cold-start problem |

The temporal split result (99.15%) is higher than the standard split, confirming the model generalises well across time. The cold-start limitation directly motivated the hybrid recommendation design.

---

### Feature Importance (Random Forest)

| Rank | Feature | MDI Importance | Permutation Importance |
|---|---|---|---|
| 1 | tenure_days | 55.7% | 0.339 |
| 2 | total_spend | 16.7% | 0.001 |
| 3 | items | 10.8% | ~0.000 |
| 4 | unique_products | 7.8% | ~0.000 |
| 5 | recency_days | 5.6% | ~0.000 |
| 6 | avg_line_value | 2.4% | ~0.000 |

Permutation importance confirms `tenure_days` as the dominant predictive signal. All country features have zero permutation importance, confirming that behavioural patterns — not geography — drive the classifier.

---

### Customer Segmentation

| Stage | Customers | % of Total | Avg Invoices | Avg Spend | Avg Products | Rec. Method |
|---|---|---|---|---|---|---|
| New (1 invoice) | 1,497 | 34.6% | 1.00 | £357.95 | 21.8 | Popularity (country) |
| Early (2-3 invoices) | 1,334 | 30.8% | 2.38 | £814.12 | 46.2 | Apriori Market Basket |
| **Established (4+)** | **1,501** | **34.6%** | **9.18** | **£4,219.29** | **114.9** | Collaborative Filtering |

Established customers spend 11.8x more than new customers on average, justifying personalised collaborative filtering for this segment.

---

### Recommender System Evaluation

| Method | K | Precision@K | Recall@K | Lift | NDCG@K | MAP@K | Coverage |
|---|---|---|---|---|---|---|---|
| Popularity-based | 5 | 0.1312 | 0.0124 | 96.10x | 0.1320 | 0.0767 | 100% |
| Market Basket (Apriori) | 5 | 0.1333 | 0.0260 | 97.68x | **0.2304** | **0.1375** | 1.39% |
| **Collaborative Filtering** ⭐ | **5** | **0.1460** | 0.0249 | **106.93x** | 0.1640 | 0.1168 | **100%** |
| Popularity-based | 10 | 0.1055 | 0.0213 | 38.66x | 0.1154 | 0.0523 | 100% |
| Market Basket (Apriori) | 10 | 0.0667 | 0.0260 | 24.42x | 0.1609 | 0.0792 | 1.39% |
| **Collaborative Filtering** ⭐ | **10** | 0.1024 | **0.0320** | 37.52x | 0.1307 | 0.0763 | **100%** |

**Key findings:**
- Collaborative Filtering achieves best Precision@5 (0.146), best Lift@5 (106.93x), and best MAP@5 (0.117)
- Apriori achieves best NDCG@5 (0.230) but only 1.39% coverage due to dataset sparsity
- Popularity-based achieves 100% coverage — reliable universal fallback for new customers
- All three methods substantially outperform random recommendation

---

### Top Apriori Association Rules (by Lift)

| Antecedent | Consequent | Confidence | Lift |
|---|---|---|---|
| Product 23171 + 23170 | Product 23172 | 81.1% | 66.80x |
| Product 23172 | Products 23171 + 23170 | 82.6% | 66.80x |
| Product 23172 + 23170 | Product 23171 | 93.9% | 64.14x |
| Product 23171 | Product 23172 | 74.8% | 61.59x |

337 rules generated in total. Full rule set: `outputs/tables/apriori_rules.csv`

---

### Hybrid Pipeline Sample Output

| Customer | Repeat Predicted | Stage | Top 5 Recommendations | Method |
|---|---|---|---|---|
| 12347 | Yes | Established | 84952C, 84952B, 23509, 16237, 21499 | Collaborative Filtering |
| 12348 | Yes | Established | 21975, 84991, 21213, 21212, 21977 | Collaborative Filtering |
| 12349 | No | New | 51014A, 23445, 51014C, 22197, 51014L | Popularity (Country) |
| 12350 | No | New | 16008, 22693, 22197, 72232, 84050 | Popularity (Country) |
| 12352 | Yes | Established | 47590B, 23139, 23138, 22759, 22090 | Collaborative Filtering |

---

## Project Structure

```
project root/
│
├── src/
│   ├── pipeline.py                <- Main pipeline — run this first
│   ├── apriori_rules.py           <- Apriori market basket (mlxtend)
│   ├── cf_recommender.py          <- Item-based CF (cosine similarity)
│   ├── hybrid_pipeline.py         <- Stage-based routing logic
│   └── recommender_metrics.py     <- Precision@K, Recall@K, NDCG, MAP, Lift
│
├── data/
│   ├── raw/                       <- Place online_retail.xlsx here (not included)
│   └── processed/                 <- Auto-generated by pipeline.py
│       ├── transactions_clean.csv
│       ├── customer_features.csv
│       └── customer_features_segmented.csv
│
├── outputs/
│   ├── figures/                   <- Auto-generated charts
│   │   ├── monthly_revenue_trend.png
│   │   ├── top_countries_revenue.png
│   │   ├── rf_feature_importance_builtin_top15.png
│   │   └── rf_feature_importance_permutation_top15.png
│   │
│   ├── models/                    <- Not included — regenerate by running pipeline
│   │   └── best_model_RandomForest.joblib
│   │
│   └── tables/                    <- Key CSVs included, large files excluded
│       ├── model_comparison.csv
│       ├── recommender_evaluation.csv
│       ├── critical_evaluation_comparison.csv
│       ├── apriori_rules.csv
│       ├── hybrid_recommendations_sample.csv
│       ├── rf_feature_importance_builtin.csv
│       ├── rf_feature_importance_permutation.csv
│       └── [confusion matrices + classification reports]
│
├── dashboard.py                   <- Streamlit interactive dashboard
├── demonstrate.py                 <- Terminal demo script
├── hybrid_pipeline_notebook.ipynb <- Jupyter notebook walkthrough
├── requirements.txt               <- All dependencies with versions
├── run_dashboard.sh               <- Mac/Linux launcher
├── run_dashboard.bat              <- Windows launcher
└── README.md
```

> **Note:** The following large files are excluded and must be regenerated locally:
> - `outputs/tables/item_similarity_matrix.csv` (183MB)
> - `outputs/models/*.joblib`
> - `data/raw/online_retail.xlsx` (Kaggle dataset — not redistributable)
> - `data/processed/transactions_clean.csv`
> - `data/processed/customer_features.csv`
>
> Run `python src/pipeline.py` to regenerate all of these automatically.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-retention-dissertation.git
cd customer-retention-dissertation
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Dataset
Download the UCI Online Retail II dataset from Kaggle:
> https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Place the downloaded file at:
```
data/raw/online_retail.xlsx
```

### 5. Run the Full Pipeline
```bash
python src/pipeline.py
```

Expected terminal output:
```
Loading data...          Raw shape: (541909, 8)
Cleaning data...         Clean shape: (397484, 9)
Creating customer feature table...   Customer table shape: (4332, 11)
Saving EDA outputs...
Exporting processed datasets...
Training and evaluating models...
Running critical evaluation (temporal split)...
Exporting feature importance for RandomForest...
Generating Apriori rules...
Computing item similarity for Collaborative Filtering...
Running hybrid recommendation pipeline...
Evaluating recommender systems...
All done! Enhanced pipeline completed.
```

Total runtime: approximately 8-12 minutes depending on machine.

### 6. Launch the Interactive Dashboard
```bash
streamlit run dashboard.py

# Or use the launcher scripts:
bash run_dashboard.sh       # Mac / Linux
run_dashboard.bat           # Windows
```

Opens automatically at http://localhost:8501

### 7. Run the Terminal Demo
```bash
python demonstrate.py
```

Prints a structured summary of all results to the terminal.

### 8. Open the Jupyter Notebook
```bash
jupyter notebook hybrid_pipeline_notebook.ipynb
```

Step-by-step walkthrough of the hybrid pipeline with explanations and visualisations.

---

## Dashboard Sections

| Section | What It Shows |
|---|---|
| **Overview** | Dataset scale, cleaning statistics, country breakdown, key figures |
| **Classification** | Model comparison, best model summary, confusion matrix, feature importance |
| **Segmentation** | Stage distribution, average spend by stage, behavioural characteristics |
| **Recommenders** | Metric comparison across all 3 methods at K=5 and K=10 |
| **Interactive Demo** | Select any customer ID to see stage, prediction, top 5 recommendations and method |
| **Business Impact** | Revenue by segment, ROI implications, deployment considerations |

---

## Generated Output Files

| File | Description |
|---|---|
| `outputs/tables/model_comparison.csv` | Accuracy, precision, recall, F1 for all 3 classifiers |
| `outputs/tables/critical_evaluation_comparison.csv` | Standard vs temporal vs cold-start evaluation |
| `outputs/tables/recommender_evaluation.csv` | 6 metrics at K=5 and K=10 for all 3 methods |
| `outputs/tables/apriori_rules.csv` | 337 association rules with support, confidence, lift |
| `outputs/tables/hybrid_recommendations_sample.csv` | Sample output for 5 customers |
| `outputs/tables/rf_feature_importance_builtin.csv` | MDI feature importance (all features) |
| `outputs/tables/rf_feature_importance_permutation.csv` | Permutation importance (F1 scoring) |
| `outputs/tables/RandomForest_classification_report.txt` | Full per-class precision, recall, F1 |
| `outputs/tables/monthly_revenue.csv` | Monthly revenue Dec 2010 – Dec 2011 |
| `outputs/tables/top_countries_revenue.csv` | Top 10 countries by total revenue |
| `outputs/figures/rf_feature_importance_builtin_top15.png` | Top 15 features bar chart |
| `outputs/figures/rf_feature_importance_permutation_top15.png` | Permutation importance bar chart |
| `outputs/figures/monthly_revenue_trend.png` | Revenue trend line chart |
| `outputs/figures/top_countries_revenue.png` | Revenue by country bar chart |

---

## Technologies

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.3.3 | Data processing and feature engineering |
| scikit-learn | 1.6.1 | Classification, cosine similarity, preprocessing |
| mlxtend | 0.23.4 | Apriori algorithm and association rule mining |
| streamlit | 1.50.0 | Interactive dashboard |
| plotly | 6.7.0 | Interactive charts in dashboard |
| matplotlib | 3.9.4 | Static figure exports |
| joblib | 1.5.3 | Model serialisation and loading |
| openpyxl | 3.1.5 | Reading .xlsx dataset |
| numpy | 2.0.2 | Numerical operations |
| scipy | 1.13.1 | Statistical functions |

Full pinned dependency list: `requirements.txt`

---

## Reproducibility

All random seeds fixed at `42` across every experiment:

```python
train_test_split(..., random_state=42)
RandomForestClassifier(..., random_state=42)
DecisionTreeClassifier(..., random_state=42)
cust.sample(frac=0.2, random_state=42)
```

Running `python src/pipeline.py` on the same dataset produces identical results every time.

---

## Dataset

| Stat | Value |
|---|---|
| Source | Kaggle — https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci |
| Raw size | 541,909 transactions x 8 columns |
| After cleaning | 397,484 transactions x 9 columns |
| Customers | 4,332 unique |
| Products | 3,663 unique stock codes |
| Countries | 37 |
| Date range | 1 Dec 2010 – 9 Dec 2011 |
| Repeat buyers | 2,835 (65.4%) |
| Non-repeat buyers | 1,497 (34.6%) |

The dataset is not included in this repository. Download from Kaggle and place at `data/raw/online_retail.xlsx`.

---

## .gitignore

```
# Large generated files
data/raw/
data/processed/
outputs/tables/item_similarity_matrix.csv
outputs/models/

# Python
venv/
__pycache__/
*.pyc
.DS_Store

# Jupyter
.ipynb_checkpoints/
```

---

## Academic Context

Submitted as a BSc final-year dissertation at York St John University (2026). The system demonstrates:

- Multi-paradigm machine learning — supervised classification combined with unsupervised recommendation
- Critical evaluation methodology — temporal split and cold-start analysis
- Real-world deployment considerations — cold-start problem, coverage vs precision trade-off, fallback design
- End-to-end reproducible research pipeline with fixed random seeds and automated output generation

---

## License

Submitted for academic assessment at York St John University. Available for reference and educational purposes.

