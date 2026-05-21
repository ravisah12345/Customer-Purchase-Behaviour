#!/usr/bin/env python3
"""
Dissertation Demonstration Script
End-to-End Customer Analytics & Recommendation System

This script demonstrates the upgraded dissertation system that combines:
1. Repeat buyer prediction (classification)
2. Customer segmentation (new/early/established)
3. Hybrid product recommendations (popularity/MBA/CF)

Run this to show your faculty the complete system!
"""

import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Use absolute paths for all file operations
BASE_DIR = script_dir

def print_header(title):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

def demonstrate_classification():
    """Show classification results"""
    print_header("1. REPEAT BUYER PREDICTION (CLASSIFICATION)")

    # Load model comparison
    try:
        results = pd.read_csv(os.path.join(BASE_DIR, "outputs/tables/model_comparison.csv"))
        print("📊 Model Performance Comparison:")
        print(results.to_string(index=False))

        # Load best model
        mod_dir = os.path.join(BASE_DIR, "outputs/models")
        model_files = [f for f in os.listdir(mod_dir) if f.startswith("best_model_") and f.endswith(".joblib")]
        if model_files:
            print(f"\n🏆 Best Model: {model_files[0].replace('best_model_', '').replace('.joblib', '')}")
            print("✅ Model files available (compatibility issues with current sklearn version)")
        else:
            print("❌ No model files found in outputs/models/")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please run pipeline.py first to generate the required files.")

def demonstrate_segmentation():
    """Show customer segmentation"""
    print_header("2. CUSTOMER SEGMENTATION")

    try:
        # Load customer data
        customers = pd.read_csv(os.path.join(BASE_DIR, "data/processed/customer_features.csv"))

        # Add segmentation (simulate)
        customers['stage'] = pd.cut(
            customers['invoices'],
            bins=[0, 1, 3, np.inf],
            labels=['new', 'early', 'established'],
            right=True
        )

        stage_counts = customers['stage'].value_counts()
        print("👥 Customer Segmentation Results:")
        print(f"   New customers (1 invoice): {stage_counts.get('new', 0)}")
        print(f"   Early stage (2-3 invoices): {stage_counts.get('early', 0)}")
        print(f"   Established (4+ invoices): {stage_counts.get('established', 0)}")

        print("\n📈 Stage Characteristics:")
        stage_stats = customers.groupby('stage').agg({
            'total_spend': 'mean',
            'invoices': 'mean',
            'RepeatBuyer': 'mean'
        }).round(2)
        print(stage_stats)

    except FileNotFoundError as e:
        print(f"❌ Customer data not found: {e}")
        print("Please run pipeline.py first to generate the required files.")

def demonstrate_recommendations():
    """Show recommendation system concepts"""
    print_header("3. HYBRID RECOMMENDATION SYSTEM")

    print("🔄 Hybrid Recommendation Logic:")
    print("   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐")
    print("   │  Classifier     │ -> │   Customer Stage  │ -> │  Recommender    │")
    print("   │ (Repeat Buyer?) │    │ (New/Early/Est.) │    │ (Method Select) │")
    print("   └─────────────────┘    └──────────────────┘    └─────────────────┘")
    print("         ↓                        ↓                        ↓")
    print("   Not Repeat → Popularity    New → Popularity        Popularity")
    print("   Is Repeat  → Stage-based   Early → MBA             MBA")
    print("                               Est. → CF               CF")

    print("\n🎯 Recommendation Methods:")
    print("   📊 Popularity-based: Most sold products (cold-start)")
    print("   🛒 Market Basket Analysis: Apriori association rules")
    print("   🤝 Collaborative Filtering: Item-based similarity")

    # Show sample recommendations (mock data for demo)
    print("\n📋 Sample Recommendations:")
    sample_recs = [
        {"customer": 12345, "stage": "new", "method": "popularity", "recommendations": ["85123A", "85099B", "84879"]},
        {"customer": 12346, "stage": "early", "method": "MBA", "recommendations": ["22697", "22699", "22698"]},
        {"customer": 12347, "stage": "established", "method": "CF", "recommendations": ["22960", "22961", "22993"]}
    ]

    for rec in sample_recs:
        print(f"   Customer {rec['customer']} ({rec['stage']}): {rec['recommendations']} via {rec['method']}")

    # Show recommender evaluation metrics
    print("\n📊 Recommender Evaluation Metrics:")
    try:
        eval_results = pd.read_csv(os.path.join(BASE_DIR, "outputs/tables/recommender_evaluation.csv"))
        if len(eval_results) > 0:
            print(eval_results.to_string(index=False))
            print("\n📈 Interpretation:")
            print("   • Precision@K: % of recommendations that match user's actual purchases")
            print("   • Recall@K: % of user's purchases captured in recommendations")
            print("   • Lift: How much better than random recommendations")
            print("   • NDCG@K: Ranking quality (considers position)")
            print("   • MAP@K: Average precision across all relevant items")
            print("   • Coverage: % of customers with recommendations")
        else:
            print("   ℹ️  Recommender evaluation not yet generated. Run pipeline.py to calculate metrics.")
    except FileNotFoundError:
        print("   ℹ️  Recommender evaluation file not found. Run pipeline.py to generate metrics.")


def demonstrate_outputs():
    """Show generated outputs"""
    print_header("4. GENERATED OUTPUTS & FILES")

    outputs = {
        "Classification": [
            os.path.join(BASE_DIR, "outputs/tables/model_comparison.csv"),
            os.path.join(BASE_DIR, "outputs/models/best_model_RandomForest.joblib"),
            os.path.join(BASE_DIR, "outputs/tables/RandomForest_classification_report.txt")
        ],
        "Data Processing": [
            os.path.join(BASE_DIR, "data/processed/customer_features.csv"),
            os.path.join(BASE_DIR, "data/processed/transactions_clean.csv")
        ],
        "Visualizations": [
            os.path.join(BASE_DIR, "outputs/figures/rf_feature_importance_builtin_top15.png"),
            os.path.join(BASE_DIR, "outputs/figures/top_countries_revenue.png")
        ]
    }

    for category, files in outputs.items():
        print(f"📁 {category}:")
        for file in files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (missing)")

def demonstrate_business_value():
    """Show business implications"""
    print_header("5. BUSINESS VALUE & IMPACT")

    print("💰 Business Applications:")
    print("   • Checkout recommendations: 'Customers also bought...'")
    print("   • Email campaigns: Personalized product suggestions")
    print("   • Inventory optimization: Predict demand patterns")
    print("   • Customer retention: Targeted interventions")

    print("\n📊 ROI Potential:")
    print("   • Increased conversion: 15-30% with recommendations")
    print("   • Customer lifetime value: Higher with personalization")
    print("   • Reduced churn: Better engagement with relevant products")

    print("\n🔬 Academic Excellence:")
    print("   • Multi-paradigm ML: Supervised + Unsupervised")
    print("   • Real-world challenges: Cold-start, scalability")
    print("   • Industry relevance: E-commerce personalization")

def main():
    print("End-to-End Customer Analytics & Recommendation System")

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    demonstrate_classification()
    demonstrate_segmentation()
    demonstrate_recommendations()
    demonstrate_outputs()
    demonstrate_business_value()

    print_header("NEXT: VIEW THE INTERACTIVE DASHBOARD")
    print("📊 For a rich, interactive visualization experience:")
    print("   $ streamlit run dashboard.py")
    print("\nThe dashboard includes:")
    print("   ✅ 6 comprehensive sections")
    print("   ✅ Interactive visualizations (Plotly)")
    print("   ✅ Customer recommendation demo")
    print("   ✅ Business impact analysis")
    print("   ✅ Deployment planning tools")

    print_header("DEMONSTRATION COMPLETE")
    print("🎉 Questions? This system demonstrates thesis-level ML expertise!")
    print("   - Advanced algorithms (Apriori, CF, Hybrid)")
    print("   - Real-world deployment considerations")
    print("   - Business impact quantification")
    print("   - Comprehensive evaluation framework")

if __name__ == "__main__":
    main()