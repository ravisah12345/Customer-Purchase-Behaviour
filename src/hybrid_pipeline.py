import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict
from apriori_rules import generate_apriori_rules, create_transaction_baskets, encode_baskets, get_product_recommendations
from cf_recommender import create_customer_product_matrix, compute_item_similarity, get_item_based_recommendations

def segment_customers(cust: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer stage segmentation based on number of invoices.
    Stages: new (1), early (2-3), established (4+)
    """
    cust = cust.copy()
    cust['stage'] = pd.cut(
        cust['invoices'],
        bins=[0, 1, 3, np.inf],
        labels=['new', 'early', 'established'],
        right=True
    )
    return cust

def get_popularity_based_recommendations(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Get most popular products based on total quantity sold.
    """
    popular = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    return popular

def get_country_based_recommendations(df: pd.DataFrame, country: str, top_n: int = 5) -> List[str]:
    """
    Get popular products in a specific country.
    """
    country_df = df[df['Country'] == country]
    if country_df.empty:
        return get_popularity_based_recommendations(df, top_n)
    popular = country_df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    return popular

def load_models_and_data():
    """
    Load the best classifier model and necessary data.
    """
    # Load best model
    mod_dir = "outputs/models"
    model_files = [f for f in os.listdir(mod_dir) if f.startswith("best_model_") and f.endswith(".joblib")]
    if not model_files:
        raise FileNotFoundError("No best model found.")
    best_model_path = os.path.join(mod_dir, model_files[0])
    model = joblib.load(best_model_path)

    # Load processed data
    proc_dir = "data/processed"
    transactions = pd.read_csv(os.path.join(proc_dir, "transactions_clean.csv"))
    customers = pd.read_csv(os.path.join(proc_dir, "customer_features.csv"))

    return model, transactions, customers

def hybrid_recommend(customer_id: int, transactions: pd.DataFrame, customers: pd.DataFrame,
                     model, apriori_rules: pd.DataFrame, item_similarity: pd.DataFrame,
                     top_n: int = 5) -> Dict:
    """
    Hybrid recommendation system.
    1. Predict if repeat buyer using classifier.
    2. If not repeat, recommend popular products.
    3. If repeat, segment by stage and recommend accordingly.
    """
    # Get customer data
    cust_data = customers[customers['CustomerID'] == customer_id]
    if cust_data.empty:
        # Cold start: no customer data
        country = "United Kingdom"  # Default
        recs = get_country_based_recommendations(transactions, country, top_n)
        return {
            "customer_id": customer_id,
            "is_repeat_predicted": None,
            "stage": "cold_start",
            "recommendations": recs,
            "method": "popularity_country"
        }

    stage = cust_data['stage'].iloc[0] if 'stage' in cust_data.columns else "unknown"

    # Build prediction row aligned to the trained pipeline feature schema.
    # The classifier was trained on data with Country one-hot columns already expanded.
    expected_cols = []
    if hasattr(model, "named_steps") and "prep" in model.named_steps:
        prep = model.named_steps["prep"]
        if hasattr(prep, "feature_names_in_"):
            expected_cols = list(prep.feature_names_in_)

    if not expected_cols:
        raise ValueError("Model schema unavailable: could not determine expected feature columns.")

    row = cust_data.iloc[0]
    features = pd.DataFrame([[0.0] * len(expected_cols)], columns=expected_cols)

    # Fill direct numeric/continuous features.
    for col in expected_cols:
        if col in row.index:
            features.at[0, col] = row[col]

    # Fill the one-hot country indicator if this specific country column exists.
    country_value = str(row["Country"]) if "Country" in row.index else None
    if country_value:
        country_col = f"Country_{country_value}"
        if country_col in features.columns:
            features.at[0, country_col] = 1.0

    is_repeat_pred = model.predict(features)[0]

    if is_repeat_pred == 0:
        # Not predicted as repeat: recommend popular
        country = cust_data['Country'].iloc[0] if 'Country' in cust_data.columns else "United Kingdom"
        recs = get_country_based_recommendations(transactions, country, top_n)
        method = "popularity_country"
    else:
        # Predicted as repeat: segment and recommend
        stage = cust_data['stage'].iloc[0] if 'stage' in cust_data.columns else 'early'  # Default

        if stage == 'new':
            # New: popularity
            country = cust_data['Country'].iloc[0] if 'Country' in cust_data.columns else "United Kingdom"
            recs = get_country_based_recommendations(transactions, country, top_n)
            method = "popularity_country"
        elif stage == 'early':
            # Early: MBA (Apriori)
            # Get customer's purchased products
            cust_products = transactions[transactions['CustomerID'] == customer_id]['StockCode'].unique()
            if len(cust_products) == 0:
                recs = get_popularity_based_recommendations(transactions, top_n)
                method = "popularity_fallback"
            else:
                # Recommend based on one of the products
                product = cust_products[0]  # Pick first
                recs = get_product_recommendations(apriori_rules, product, top_n)
                if not recs:
                    recs = get_popularity_based_recommendations(transactions, top_n)
                    method = "popularity_fallback"
                else:
                    method = "apriori"
        else:  # established
            # Established: CF
            cust_products = transactions[transactions['CustomerID'] == customer_id]['StockCode'].unique()
            if len(cust_products) == 0:
                recs = get_popularity_based_recommendations(transactions, top_n)
                method = "popularity_fallback"
            else:
                # Get recommendations based on one product
                product = cust_products[0]
                recs = get_item_based_recommendations(item_similarity, product, top_n)
                if not recs:
                    recs = get_popularity_based_recommendations(transactions, top_n)
                    method = "popularity_fallback"
                else:
                    method = "collaborative_filtering"

    return {
        "customer_id": customer_id,
        "is_repeat_predicted": int(is_repeat_pred),
        "stage": stage if 'stage' in cust_data.columns else "unknown",
        "recommendations": recs,
        "method": method
    }

def run_hybrid_pipeline():
    """
    Run the full hybrid pipeline.
    """
    print("Loading models and data...")
    model, transactions, customers = load_models_and_data()

    # Segment customers
    customers = segment_customers(customers)

    # Prepare Apriori rules
    baskets = create_transaction_baskets(transactions)
    df_encoded = encode_baskets(baskets)
    apriori_rules = generate_apriori_rules(df_encoded, min_support=0.03)

    # Prepare item similarity
    matrix = create_customer_product_matrix(transactions)
    item_similarity = compute_item_similarity(matrix)

    # Example: Recommend for a few customers
    sample_customers = customers['CustomerID'].head(5).tolist()
    results = []
    for cid in sample_customers:
        rec = hybrid_recommend(cid, transactions, customers, model, apriori_rules, item_similarity)
        results.append(rec)
        print(f"Customer {cid}: {rec}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/tables/hybrid_recommendations_sample.csv", index=False)

    print("Hybrid pipeline completed.")

if __name__ == "__main__":
    run_hybrid_pipeline()