import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

def create_transaction_baskets(df: pd.DataFrame) -> list:
    """
    Create transaction baskets from cleaned transaction data.
    Each basket is a list of StockCodes for a single invoice.
    """
    baskets = df.groupby('InvoiceNo')['StockCode'].apply(list).tolist()
    return baskets

def encode_baskets(baskets: list) -> pd.DataFrame:
    """
    Encode baskets into a binary matrix for Apriori.
    """
    # TransactionEncoder sorts item labels internally, so mixed types
    # (e.g., int and str StockCodes) will raise TypeError in Python 3.
    # Normalize all items to cleaned strings and drop null/empty values.
    normalized_baskets = []
    for basket in baskets:
        normalized = []
        for item in basket:
            if pd.isna(item):
                continue
            item_str = str(item).strip()
            if item_str:
                normalized.append(item_str)
        if normalized:
            normalized_baskets.append(normalized)

    if not normalized_baskets:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(normalized_baskets).transform(normalized_baskets)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded

def generate_apriori_rules(df_encoded: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.5) -> pd.DataFrame:
    """
    Generate association rules using Apriori algorithm.
    """
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

def save_rules(rules: pd.DataFrame, output_dir: str = "outputs/tables") -> None:
    """
    Save the association rules to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    rules.to_csv(os.path.join(output_dir, "apriori_rules.csv"), index=False)

def get_product_recommendations(rules: pd.DataFrame, product: str, top_n: int = 5) -> list:
    """
    Get top product recommendations based on association rules for a given product.
    """
    antecedents = rules[rules['antecedents'].apply(lambda x: product in x)]
    recommendations = antecedents.nlargest(top_n, 'confidence')['consequents'].tolist()
    # Flatten the frozensets
    flat_recs = [list(rec) for rec in recommendations]
    # Flatten further if needed
    flat_recs = [item for sublist in flat_recs for item in sublist]
    return list(set(flat_recs))[:top_n]

if __name__ == "__main__":
    # Example usage
    from pipeline import load_data, clean_data

    df = load_data("../data/raw/online_retail.xlsx")
    df_clean = clean_data(df)
    baskets = create_transaction_baskets(df_clean)
    df_encoded = encode_baskets(baskets)
    rules = generate_apriori_rules(df_encoded)
    save_rules(rules)
    print("Apriori rules generated and saved.")
    print("Apriori rules generated and saved.")