import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_customer_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a customer-product matrix where rows are customers, columns are products (StockCode),
    and values are the quantity purchased.
    """
    matrix = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    return matrix

def compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between items (columns).
    """
    # Transpose to make items as rows
    item_matrix = matrix.T
    similarity = cosine_similarity(item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=item_matrix.index,
        columns=item_matrix.index
    )
    return similarity_df

def get_item_based_recommendations(similarity_df: pd.DataFrame, product: str, top_n: int = 5) -> list:
    """
    Get top similar products for a given product.
    """
    if product not in similarity_df.index:
        return []
    sim_scores = similarity_df.loc[product].sort_values(ascending=False)
    # Exclude self
    sim_scores = sim_scores.drop(product, errors='ignore')
    recommendations = sim_scores.head(top_n).index.tolist()
    return recommendations

def get_user_based_recommendations(matrix: pd.DataFrame, customer_id: int, top_n: int = 5) -> list:
    """
    Get recommendations for a customer based on similar customers' purchases.
    Note: This is user-based CF, but the plan mentions item-based. Keeping for completeness.
    """
    if customer_id not in matrix.index:
        return []
    user_similarity = cosine_similarity(matrix)
    user_sim_df = pd.DataFrame(
        user_similarity,
        index=matrix.index,
        columns=matrix.index
    )
    similar_users = user_sim_df.loc[customer_id].sort_values(ascending=False).drop(customer_id, errors='ignore').head(10)
    similar_user_purchases = matrix.loc[similar_users.index].sum(axis=0)
    user_purchases = matrix.loc[customer_id]
    recommendations = similar_user_purchases[user_purchases == 0].sort_values(ascending=False).head(top_n).index.tolist()
    return recommendations

def save_similarity_matrix(similarity_df: pd.DataFrame, output_dir: str = "outputs/tables") -> None:
    """
    Save the item similarity matrix to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    similarity_df.to_csv(os.path.join(output_dir, "item_similarity_matrix.csv"))

if __name__ == "__main__":
    from pipeline import load_data, clean_data

    df = load_data("../data/raw/online_retail.xlsx")
    df_clean = clean_data(df)
    matrix = create_customer_product_matrix(df_clean)
    similarity_df = compute_item_similarity(matrix)
    save_similarity_matrix(similarity_df)
    print("Item similarity matrix computed and saved.")