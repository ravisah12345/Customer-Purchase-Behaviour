import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Config
# ----------------------------
RAW_PATH = "data/raw/online_retail.xlsx"  # XLSX Kaggle dataset
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MOD_DIR = os.path.join(OUT_DIR, "models")
PROC_DIR = "data/processed"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV or XLSX safely."""
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)  # requires: pip install openpyxl
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1")

    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning rules aligned with Online Retail practice:
    - Drop rows with missing CustomerID
    - Remove cancellations/refunds: InvoiceNo starts with 'C' OR Quantity <= 0
    - Remove UnitPrice <= 0
    - Parse InvoiceDate
    - Create TotalPrice
    """
    df = df.copy()

    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])

    if "InvoiceNo" in df.columns:
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]

    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]

    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()

    # optional outlier clip
    df = df[df["TotalPrice"] < df["TotalPrice"].quantile(0.999)]

    return df


def create_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level features.
    Target: RepeatBuyer (1 if customer has >1 distinct invoice, else 0)
    """
    df = df.copy()
    grp = df.groupby("CustomerID")

    cust = grp.agg(
        invoices=("InvoiceNo", "nunique"),
        items=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        total_spend=("TotalPrice", "sum"),
        avg_line_value=("TotalPrice", "mean"),
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
    ).reset_index()

    dataset_end = df["InvoiceDate"].max()
    cust["recency_days"] = (dataset_end - cust["last_purchase"]).dt.days
    cust["tenure_days"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days.clip(lower=0)
    cust["purchase_rate"] = cust["invoices"] / (cust["tenure_days"].replace(0, 1))

    if "Country" in df.columns:
        country_mode = (
            df.groupby(["CustomerID", "Country"]).size().reset_index(name="n")
            .sort_values(["CustomerID", "n"], ascending=[True, False])
            .drop_duplicates("CustomerID")[["CustomerID", "Country"]]
        )
        cust = cust.merge(country_mode, on="CustomerID", how="left")
    else:
        cust["Country"] = "Unknown"

    cust["RepeatBuyer"] = (cust["invoices"] > 1).astype(int)
    cust = cust.drop(columns=["last_purchase", "first_purchase"])

    return cust


def save_basic_eda(df: pd.DataFrame, cust: pd.DataFrame) -> None:
    """Save charts + summary tables."""
    if "Country" in df.columns:
        top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        top_countries.to_csv(os.path.join(TAB_DIR, "top_countries_revenue.csv"))

        plt.figure()
        top_countries.plot(kind="bar")
        plt.title("Top 10 Countries by Revenue")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "top_countries_revenue.png"), dpi=200)
        plt.close()

    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
    monthly.to_csv(os.path.join(TAB_DIR, "monthly_revenue.csv"))

    plt.figure()
    monthly.plot()
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "monthly_revenue_trend.png"), dpi=200)
    plt.close()

    balance = cust["RepeatBuyer"].value_counts().rename_axis("RepeatBuyer").reset_index(name="count")
    balance.to_csv(os.path.join(TAB_DIR, "repeat_buyer_balance.csv"), index=False)


def export_for_dashboard(df: pd.DataFrame, cust: pd.DataFrame) -> None:
    df.to_csv(os.path.join(PROC_DIR, "transactions_clean.csv"), index=False)
    cust.to_csv(os.path.join(PROC_DIR, "customer_features.csv"), index=False)


def export_feature_importance(best_model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Export feature importance for RandomForest:
    1) Built-in impurity-based importance
    2) Permutation importance (recommended)
    """
    pipe = joblib.load(best_model_path)
    feature_names = list(X_test.columns)

    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        imp.to_csv(os.path.join(TAB_DIR, "rf_feature_importance_builtin.csv"), index=False)

        top = imp.head(15).iloc[::-1]
        plt.figure()
        plt.barh(top["feature"], top["importance"])
        plt.title("Random Forest Feature Importance (Built-in) - Top 15")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "rf_feature_importance_builtin_top15.png"), dpi=200)
        plt.close()

    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1, scoring="f1"
    )

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std
    }).sort_values("perm_importance_mean", ascending=False)

    perm_df.to_csv(os.path.join(TAB_DIR, "rf_feature_importance_permutation.csv"), index=False)

    top2 = perm_df.head(15).iloc[::-1]
    plt.figure()
    plt.barh(top2["feature"], top2["perm_importance_mean"])
    plt.title("Random Forest Feature Importance (Permutation, F1) - Top 15")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rf_feature_importance_permutation_top15.png"), dpi=200)
    plt.close()


def train_and_evaluate(cust: pd.DataFrame):
    """Train LR, DT, RF and export evaluation results."""
    data = cust.copy()

    # Remove leakage features (they directly define RepeatBuyer)
    leak_cols = [c for c in ["invoices", "purchase_rate"] if c in data.columns]
    data = data.drop(columns=leak_cols)

    data = pd.get_dummies(data, columns=["Country"], drop_first=True)

    y = data["RepeatBuyer"]
    X = data.drop(columns=["CustomerID", "RepeatBuyer"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop"
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    rows = []
    best_name, best_pipe, best_f1 = None, None, -1.0

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        rows.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

        report = classification_report(y_test, preds, zero_division=0)
        with open(os.path.join(TAB_DIR, f"{name}_classification_report.txt"), "w") as f:
            f.write(report)

        cm = confusion_matrix(y_test, preds)
        pd.DataFrame(cm, index=["Actual0", "Actual1"], columns=["Pred0", "Pred1"]) \
            .to_csv(os.path.join(TAB_DIR, f"{name}_confusion_matrix.csv"))

        if f1 > best_f1:
            best_f1 = f1
            best_name, best_pipe = name, pipe

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(os.path.join(TAB_DIR, "model_comparison.csv"), index=False)

    best_path = os.path.join(MOD_DIR, f"best_model_{best_name}.joblib")
    joblib.dump(best_pipe, best_path)

    return results, best_name, best_path, X_test, y_test


def critical_evaluation(df_clean: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    """
    Critical evaluation using temporal split to address overfitting concerns.
    Compares three evaluation strategies:
      1. Standard random split  (existing result — optimistic)
      2. Temporal split         (train on early customers, test on later ones)
      3. Cold-start only        (train on repeat buyers, test on single-invoice customers)
    Outputs: outputs/tables/critical_evaluation_comparison.csv
    """
    print("\nRunning critical evaluation (temporal split)...")

    data = cust.copy()

    # Drop leakage features
    leak_cols = [c for c in ["invoices", "purchase_rate"] if c in data.columns]
    data = data.drop(columns=leak_cols)

    # Keep CustomerID aside before encoding
    customer_ids = cust["CustomerID"].values

    data = pd.get_dummies(data, columns=["Country"], drop_first=True)

    y = data["RepeatBuyer"]
    X = data.drop(columns=["CustomerID", "RepeatBuyer"])

    numeric_features = X.columns.tolist()

    def make_rf_pipeline():
        return Pipeline(steps=[
            ("prep", ColumnTransformer(
                transformers=[("num", StandardScaler(), numeric_features)],
                remainder="drop"
            )),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
            ))
        ])

    rows = []

    # ------------------------------------------------------------------
    # 1. Standard random split (replicates existing train_and_evaluate)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = make_rf_pipeline()
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    rows.append({
        "evaluation_type": "Standard (Random Split)",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "notes": "Baseline — same time period train/test, optimistic estimate"
    })

    # ------------------------------------------------------------------
    # 2. Temporal split — train on earliest 70% of customers by last
    #    purchase date, test on the most recent 30%
    # ------------------------------------------------------------------
    df_dates = df_clean.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    df_dates.columns = ["CustomerID", "last_date"]
    cutoff = df_dates["last_date"].quantile(0.70)

    early_ids = set(df_dates[df_dates["last_date"] <= cutoff]["CustomerID"].tolist())
    late_ids  = set(df_dates[df_dates["last_date"] >  cutoff]["CustomerID"].tolist())

    # Build boolean masks aligned with X/y index
    cust_id_series = pd.Series(customer_ids, index=X.index)
    train_mask = cust_id_series.isin(early_ids)
    test_mask  = cust_id_series.isin(late_ids)

    X_train_t, y_train_t = X[train_mask], y[train_mask]
    X_test_t,  y_test_t  = X[test_mask],  y[test_mask]

    if len(X_train_t) > 10 and len(X_test_t) > 10:
        rf_t = make_rf_pipeline()
        rf_t.fit(X_train_t, y_train_t)
        preds_t = rf_t.predict(X_test_t)

        rows.append({
            "evaluation_type": "Temporal Split (Train early → Test later)",
            "train_size": len(X_train_t),
            "test_size": len(X_test_t),
            "accuracy": round(accuracy_score(y_test_t, preds_t), 4),
            "precision": round(precision_score(y_test_t, preds_t, zero_division=0), 4),
            "recall": round(recall_score(y_test_t, preds_t, zero_division=0), 4),
            "f1": round(f1_score(y_test_t, preds_t, zero_division=0), 4),
            "notes": "Realistic — simulates predicting behaviour of future customers"
        })
    else:
        print("  Warning: Not enough data for temporal split — skipping.")

    # ------------------------------------------------------------------
    # 3. Cold-start simulation — use only FIRST transaction features
    #    to predict whether customer became a repeat buyer
    # ------------------------------------------------------------------
    if "invoices" in cust.columns:
        # Only test on customers who had MORE than 1 invoice
        # Use only their first-visit features (recency will be high,
        # tenure=0, spend=first order only) to simulate cold-start
        cold_test = cust[cust["invoices"] > 1].copy()

        if len(cold_test) > 10:
            # Simulate first-visit: zero out tenure, set spend to avg_line_value
            cold_test_sim = cold_test.copy()
            cold_test_sim["tenure_days"] = 0
            cold_test_sim["recency_days"] = cold_test_sim["recency_days"] * 2  # simulate distant
            cold_test_sim["items"] = cold_test_sim["avg_line_value"]
            cold_test_sim["unique_products"] = 1
            cold_test_sim["total_spend"] = cold_test_sim["avg_line_value"]

            data_cold = cold_test_sim.copy()
            data_cold = data_cold.drop(columns=[c for c in leak_cols if c in data_cold.columns])
            data_cold = pd.get_dummies(data_cold, columns=["Country"], drop_first=True)
            # Align columns with training data
            for col in X.columns:
                if col not in data_cold.columns:
                    data_cold[col] = 0
            data_cold = data_cold[X.columns]

            y_cold = cust[cust["invoices"] > 1]["RepeatBuyer"].values

            # Train on all data, test on simulated cold-start
            rf_c = make_rf_pipeline()
            rf_c.fit(X_train, y_train)
            preds_c = rf_c.predict(data_cold)

            rows.append({
                "evaluation_type": "Cold-Start Simulation (First-visit features only)",
                "train_size": len(X_train),
                "test_size": len(data_cold),
                "accuracy": round(accuracy_score(y_cold, preds_c), 4),
                "precision": round(precision_score(y_cold, preds_c, zero_division=0), 4),
                "recall": round(recall_score(y_cold, preds_c, zero_division=0), 4),
                "f1": round(f1_score(y_cold, preds_c, zero_division=0), 4),
                "notes": "Simulates predicting repeat behaviour from first visit only"
            })

    # ------------------------------------------------------------------
    # Save & print
    # ------------------------------------------------------------------
    results = pd.DataFrame(rows)
    out_path = os.path.join(TAB_DIR, "critical_evaluation_comparison.csv")
    results.to_csv(out_path, index=False)

    print("\nCritical Evaluation Results:")
    print(results[["evaluation_type", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
    print(f"\nSaved to {out_path}")

    return results


def evaluate_recommenders(df_clean: pd.DataFrame, cust: pd.DataFrame,
                          apriori_rules: pd.DataFrame, item_similarity: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Evaluate all recommender methods using precision@k, recall@k, and lift metrics.
    """
    from recommender_metrics import (
        precision_at_k, recall_at_k, lift, calculate_random_baseline,
        ndcg_at_k, map_at_k
    )
    from apriori_rules import get_product_recommendations as get_apriori_recommendations
    from cf_recommender import get_item_based_recommendations
    from hybrid_pipeline import (
        get_popularity_based_recommendations, get_country_based_recommendations
    )

    # Split customers into train/test
    test_size = 0.2
    cust_test = cust.sample(frac=test_size, random_state=42)

    # Separate test transactions
    test_customer_ids = set(cust_test['CustomerID'].tolist())
    test_transactions = df_clean[df_clean['CustomerID'].isin(test_customer_ids)].copy()

    # For each customer, the test set is their purchases; train set is for recommendations
    train_transactions = df_clean[~df_clean['CustomerID'].isin(test_customer_ids)].copy()

    if len(test_transactions) == 0:
        print("Warning: No test transactions found. Skipping recommender evaluation.")
        return pd.DataFrame()

    k_values = [5, 10]
    results = []

    n_products = df_clean['StockCode'].nunique()

    for k in k_values:
        print(f"\n  Evaluating at K={k}...")

        # 1. Popularity-based recommender
        print(f"    - Popularity-based...")
        pop_scores = []
        pop_ndcgs = []
        pop_maps = []
        pop_covered = 0

        for _, customer in cust_test.iterrows():
            cust_id = customer['CustomerID']
            cust_transactions = test_transactions[test_transactions['CustomerID'] == cust_id]
            if len(cust_transactions) == 0:
                continue

            relevant = set(cust_transactions['StockCode'].unique())
            recommended = get_popularity_based_recommendations(train_transactions, top_n=k)

            prec = precision_at_k(recommended, relevant, k)
            rec = recall_at_k(recommended, relevant, k)
            ndcg = ndcg_at_k(recommended, relevant, k)
            map_score = map_at_k(recommended, relevant, k)

            pop_scores.append({'precision': prec, 'recall': rec})
            pop_ndcgs.append(ndcg)
            pop_maps.append(map_score)
            pop_covered += 1

        if pop_covered > 0:
            precision_random = calculate_random_baseline(n_products, k)
            avg_prec_pop = np.mean([s['precision'] for s in pop_scores])
            avg_rec_pop = np.mean([s['recall'] for s in pop_scores])
            avg_lift_pop = np.mean([lift(s['precision'], precision_random) for s in pop_scores])

            results.append({
                'recommender': 'Popularity-based',
                'k': k,
                'precision_at_k': avg_prec_pop,
                'recall_at_k': avg_rec_pop,
                'lift': avg_lift_pop,
                'ndcg_at_k': np.mean(pop_ndcgs),
                'map_at_k': np.mean(pop_maps),
                'coverage': pop_covered / len(cust_test)
            })

        # 2. Apriori (Market Basket Analysis) recommender
        print(f"    - Market Basket Analysis (Apriori)...")
        apriori_scores = []
        apriori_ndcgs = []
        apriori_maps = []
        apriori_covered = 0

        for _, customer in cust_test.iterrows():
            cust_id = customer['CustomerID']
            cust_transactions = test_transactions[test_transactions['CustomerID'] == cust_id]
            if len(cust_transactions) == 0:
                continue

            relevant = set(cust_transactions['StockCode'].unique())
            if len(relevant) > 0:
                base_product = list(relevant)[0]
                try:
                    recommended = get_apriori_recommendations(apriori_rules, base_product, top_n=k)
                except:
                    recommended = []

                if len(recommended) > 0:
                    prec = precision_at_k(recommended, relevant, k)
                    rec = recall_at_k(recommended, relevant, k)
                    ndcg = ndcg_at_k(recommended, relevant, k)
                    map_score = map_at_k(recommended, relevant, k)

                    apriori_scores.append({'precision': prec, 'recall': rec})
                    apriori_ndcgs.append(ndcg)
                    apriori_maps.append(map_score)
                    apriori_covered += 1

        if apriori_covered > 0:
            precision_random = calculate_random_baseline(n_products, k)
            avg_prec_apriori = np.mean([s['precision'] for s in apriori_scores])
            avg_rec_apriori = np.mean([s['recall'] for s in apriori_scores])
            avg_lift_apriori = np.mean([lift(s['precision'], precision_random) for s in apriori_scores])

            results.append({
                'recommender': 'Market Basket (Apriori)',
                'k': k,
                'precision_at_k': avg_prec_apriori,
                'recall_at_k': avg_rec_apriori,
                'lift': avg_lift_apriori,
                'ndcg_at_k': np.mean(apriori_ndcgs),
                'map_at_k': np.mean(apriori_maps),
                'coverage': apriori_covered / len(cust_test)
            })

        # 3. Collaborative Filtering (item-based) recommender
        print(f"    - Collaborative Filtering (Item-based)...")
        cf_scores = []
        cf_ndcgs = []
        cf_maps = []
        cf_covered = 0

        for _, customer in cust_test.iterrows():
            cust_id = customer['CustomerID']
            cust_transactions = test_transactions[test_transactions['CustomerID'] == cust_id]
            if len(cust_transactions) == 0:
                continue

            relevant = set(cust_transactions['StockCode'].unique())
            if len(relevant) > 0:
                base_product = list(relevant)[0]
                try:
                    recommended = get_item_based_recommendations(item_similarity, base_product, top_n=k)
                except:
                    recommended = []

                if len(recommended) > 0:
                    prec = precision_at_k(recommended, relevant, k)
                    rec = recall_at_k(recommended, relevant, k)
                    ndcg = ndcg_at_k(recommended, relevant, k)
                    map_score = map_at_k(recommended, relevant, k)

                    cf_scores.append({'precision': prec, 'recall': rec})
                    cf_ndcgs.append(ndcg)
                    cf_maps.append(map_score)
                    cf_covered += 1

        if cf_covered > 0:
            precision_random = calculate_random_baseline(n_products, k)
            avg_prec_cf = np.mean([s['precision'] for s in cf_scores])
            avg_rec_cf = np.mean([s['recall'] for s in cf_scores])
            avg_lift_cf = np.mean([lift(s['precision'], precision_random) for s in cf_scores])

            results.append({
                'recommender': 'Collaborative Filtering (Item-based)',
                'k': k,
                'precision_at_k': avg_prec_cf,
                'recall_at_k': avg_rec_cf,
                'lift': avg_lift_cf,
                'ndcg_at_k': np.mean(cf_ndcgs),
                'map_at_k': np.mean(cf_maps),
                'coverage': cf_covered / len(cust_test)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    results_df.to_csv(os.path.join(output_dir, "recommender_evaluation.csv"), index=False)

    print("\nRecommender Evaluation Results:")
    print(results_df.to_string(index=False))

    return results_df


def main():
    print("Loading data...")
    df = load_data(RAW_PATH)
    print("Raw shape:", df.shape)

    print("Cleaning data...")
    df_clean = clean_data(df)
    print("Clean shape:", df_clean.shape)

    print("Creating customer feature table...")
    cust = create_customer_table(df_clean)
    print("Customer table shape:", cust.shape)

    print("Saving EDA outputs...")
    save_basic_eda(df_clean, cust)

    print("Exporting processed datasets...")
    export_for_dashboard(df_clean, cust)

    print("Training and evaluating models...")
    results, best_name, best_path, X_test, y_test = train_and_evaluate(cust)

    # Critical evaluation — temporal split (addresses professor feedback)
    critical_evaluation(df_clean, cust)

    if best_name == "RandomForest":
        print("\nExporting feature importance for RandomForest...")
        export_feature_importance(best_path, X_test, y_test)
        print("Feature importance exported.")
    else:
        print(f"\nBest model is {best_name}. Feature importance export is set for RandomForest only.")

    # Generate Apriori rules
    print("Generating Apriori rules...")
    from apriori_rules import create_transaction_baskets, encode_baskets, generate_apriori_rules, save_rules
    baskets = create_transaction_baskets(df_clean)
    df_encoded = encode_baskets(baskets)
    apriori_rules = generate_apriori_rules(df_encoded)
    save_rules(apriori_rules, TAB_DIR)

    # Compute item similarity for CF
    print("Computing item similarity for Collaborative Filtering...")
    from cf_recommender import create_customer_product_matrix, compute_item_similarity, save_similarity_matrix
    matrix = create_customer_product_matrix(df_clean)
    item_similarity = compute_item_similarity(matrix)
    save_similarity_matrix(item_similarity, TAB_DIR)

    # Segment customers and run hybrid recommendations
    print("Running hybrid recommendation pipeline...")
    from hybrid_pipeline import segment_customers, run_hybrid_pipeline
    cust_segmented = segment_customers(cust)
    cust_segmented.to_csv(os.path.join(PROC_DIR, "customer_features_segmented.csv"), index=False)
    run_hybrid_pipeline()

    # Evaluate recommenders using metrics
    print("\nEvaluating recommender systems...")
    evaluate_recommenders(df_clean, cust, apriori_rules, item_similarity, TAB_DIR)

    print("\nAll done! Enhanced pipeline completed.")


if __name__ == "__main__":
    main()
