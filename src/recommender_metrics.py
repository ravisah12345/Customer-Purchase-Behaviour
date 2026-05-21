"""
Recommender System Evaluation Metrics

This module calculates evaluation metrics for recommendation systems:
- Precision@K: Of the top K recommendations, how many were relevant?
- Recall@K: Of all relevant items, how many appear in the top K recommendations?
- Lift: How much better is the recommender compared to random recommendations?
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set


def precision_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# of relevant items in top K recommendations) / K
    
    Args:
        recommended: List of recommended item IDs (in order of recommendation)
        relevant: Set of relevant item IDs (items the user actually purchased/interacted with)
        k: Number of top recommendations to consider
    
    Returns:
        Precision@K score (0 to 1)
    """
    if len(recommended) == 0 or k == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant])
    return relevant_in_top_k / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# of relevant items in top K recommendations) / (total # of relevant items)
    
    Args:
        recommended: List of recommended item IDs (in order of recommendation)
        relevant: Set of relevant item IDs (items the user actually purchased/interacted with)
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score (0 to 1)
    """
    if len(relevant) == 0:
        return 0.0
    
    if len(recommended) == 0 or k == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant])
    return relevant_in_top_k / len(relevant)


def lift(precision_recommender: float, precision_random: float) -> float:
    """
    Calculate Lift.
    
    Lift = (Precision of recommender) / (Precision of random recommender)
    Lift > 1 means the recommender performs better than random.
    
    Args:
        precision_recommender: Precision of the recommender
        precision_random: Precision of a random recommender
    
    Returns:
        Lift score (should be > 1 for good recommenders)
    """
    if precision_random == 0:
        return 0.0
    return precision_recommender / precision_random


def calculate_random_baseline(n_products: int, k: int = 5) -> float:
    """
    Calculate expected precision of a random recommender.
    
    Random recommender precision = K / (total number of products)
    
    Args:
        n_products: Total number of unique products in the dataset
        k: Number of recommendations
    
    Returns:
        Expected precision of random recommender
    """
    if n_products == 0:
        return 0.0
    return min(k / n_products, 1.0)


def evaluate_recommender(
    test_customers: pd.DataFrame,
    transactions: pd.DataFrame,
    recommender_func,
    k: int = 5,
    sample_size: int = None
) -> Dict[str, float]:
    """
    Evaluate a recommender function on test customers.
    
    Args:
        test_customers: DataFrame with customer info (must have 'CustomerID' column)
        transactions: DataFrame with cleaned transactions (must have 'CustomerID' and 'StockCode')
        recommender_func: Function that takes customer_id and returns list of recommended items
        k: Number of top recommendations to consider
        sample_size: If provided, evaluate only on this many customers
    
    Returns:
        Dictionary with metrics: precision_at_k, recall_at_k, lift, coverage
    """
    if sample_size is not None:
        test_customers = test_customers.sample(min(sample_size, len(test_customers)), random_state=42)
    
    precisions = []
    recalls = []
    lifts = []
    n_covered = 0
    n_total = len(test_customers)
    
    # Get total number of unique products for random baseline
    n_products = transactions['StockCode'].nunique()
    precision_random = calculate_random_baseline(n_products, k)
    
    for _, customer in test_customers.iterrows():
        customer_id = customer['CustomerID']
        
        # Get relevant items (items purchased by this customer in transactions)
        customer_transactions = transactions[transactions['CustomerID'] == customer_id]
        if len(customer_transactions) == 0:
            continue
        
        relevant_items = set(customer_transactions['StockCode'].unique())
        
        try:
            # Get recommendations
            recommended = recommender_func(customer_id)
            if recommended is None or len(recommended) == 0:
                continue
            
            n_covered += 1
            
            # Calculate metrics
            prec_k = precision_at_k(recommended, relevant_items, k)
            rec_k = recall_at_k(recommended, relevant_items, k)
            lift_score = lift(prec_k, precision_random)
            
            precisions.append(prec_k)
            recalls.append(rec_k)
            lifts.append(lift_score)
        
        except Exception as e:
            # Skip customers where recommendation fails
            continue
    
    if n_covered == 0:
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'lift': 0.0,
            'coverage': 0.0,
            'n_evaluated': 0
        }
    
    coverage = n_covered / n_total if n_total > 0 else 0.0
    
    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'lift': np.mean(lifts),
        'coverage': coverage,
        'n_evaluated': n_covered,
        'precision_at_k_std': np.std(precisions),
        'recall_at_k_std': np.std(recalls),
        'lift_std': np.std(lifts)
    }


def compare_recommenders(
    test_customers: pd.DataFrame,
    transactions: pd.DataFrame,
    recommenders: Dict[str, callable],
    k: int = 5,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Compare multiple recommenders using standard metrics.
    
    Args:
        test_customers: DataFrame with test customers
        transactions: DataFrame with transactions
        recommenders: Dictionary of {recommender_name: recommender_function}
        k: Number of recommendations
        sample_size: Number of customers to evaluate on
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for recommender_name, recommender_func in recommenders.items():
        print(f"Evaluating {recommender_name}...")
        metrics = evaluate_recommender(
            test_customers,
            transactions,
            recommender_func,
            k=k,
            sample_size=sample_size
        )
        metrics['recommender'] = recommender_name
        results.append(metrics)
    
    return pd.DataFrame(results)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG considers both relevance and position in the ranking.
    Items appearing earlier in the recommendation list have higher weight.
    
    Args:
        recommended: List of recommended item IDs (in ranking order)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        NDCG@K score (0 to 1)
    """
    if len(relevant) == 0 or len(recommended) == 0 or k == 0:
        return 0.0
    
    top_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            # Relevance = 1 if item is relevant, 0 otherwise
            rel = 1.0
            # Discount = 1 / log2(i + 2) (standard discount function)
            discount = 1.0 / np.log2(i + 2)
            dcg += rel * discount
    
    # Calculate ideal DCG (all relevant items ranked first)
    ideal_dcg = 0.0
    n_relevant = min(len(relevant), k)
    for i in range(n_relevant):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def map_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Calculate Mean Average Precision (MAP@K).
    
    MAP considers the position of relevant items in the ranking.
    
    Args:
        recommended: List of recommended item IDs (in ranking order)
        relevant: Set of relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        MAP@K score (0 to 1)
    """
    if len(relevant) == 0 or len(recommended) == 0 or k == 0:
        return 0.0
    
    top_k = recommended[:k]
    
    score = 0.0
    n_relevant_found = 0
    
    for i, item in enumerate(top_k):
        if item in relevant:
            n_relevant_found += 1
            # Precision at position i+1
            precision_at_i = n_relevant_found / (i + 1)
            score += precision_at_i
    
    if len(relevant) == 0:
        return 0.0
    
    # Average over all relevant items (not just found ones)
    return score / min(len(relevant), k)


if __name__ == "__main__":
    # Example usage
    print("Precision@5:", precision_at_k(['A', 'B', 'C', 'D', 'E'], {'A', 'C', 'F'}, k=5))
    print("Recall@5:", recall_at_k(['A', 'B', 'C', 'D', 'E'], {'A', 'C', 'F'}, k=5))
    print("Lift:", lift(0.4, 0.1))
    print("Random Baseline:", calculate_random_baseline(3000, k=5))
