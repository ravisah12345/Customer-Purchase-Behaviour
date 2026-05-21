#!/usr/bin/env python3
"""
Interactive Dashboard for End-to-End Customer Analytics & Recommendation System

This dashboard provides:
1. Classification Performance Summary
2. Customer Segmentation Insights
3. Recommender System Comparison
4. Interactive Recommendations
5. Business Impact Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Analytics & Recommendation Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling


# Set working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Cache data loading
@st.cache_data
def load_data():
    """Load all required data files"""
    data = {}
    
    # Load processed data
    try:
        data['customer_features'] = pd.read_csv("data/processed/customer_features.csv")
        data['transactions'] = pd.read_csv("data/processed/transactions_clean.csv")
    except FileNotFoundError:
        st.error("❌ Processed data not found. Please run `python src/pipeline.py` first.")
        return None
    
    # Load model comparison
    try:
        data['model_comparison'] = pd.read_csv("outputs/tables/model_comparison.csv")
    except FileNotFoundError:
        data['model_comparison'] = None
    
    # Load recommender evaluation
    try:
        data['recommender_eval'] = pd.read_csv("outputs/tables/recommender_evaluation.csv")
    except FileNotFoundError:
        data['recommender_eval'] = None
    
    # Load apriori rules
    try:
        data['apriori_rules'] = pd.read_csv("outputs/tables/apriori_rules.csv")
    except FileNotFoundError:
        data['apriori_rules'] = None
    
    # Load best model
    try:
        mod_dir = "outputs/models"
        model_files = [f for f in os.listdir(mod_dir) if f.startswith("best_model_") and f.endswith(".joblib")]
        if model_files:
            data['model'] = joblib.load(os.path.join(mod_dir, model_files[0]))
            data['best_model_name'] = model_files[0].replace('best_model_', '').replace('.joblib', '')
    except Exception as e:
        st.warning(f"⚠️  Could not load model: {e}")
    
    # Add segmentation
    if 'customer_features' in data:
        data['customer_features']['stage'] = pd.cut(
            data['customer_features']['invoices'],
            bins=[0, 1, 3, np.inf],
            labels=['new', 'early', 'established'],
            right=True
        )
    
    return data

# Header
st.title("Customer Analytics & Recommendation System Dashboard")
st.markdown("**End-to-End System**: Prediction • Segmentation • Personalization")

# Load data
data = load_data()

if data is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    [
        "Overview",
        "Classification",
        "Segmentation",
        "Recommenders",
        "Interactive Demo",
        "Business Impact"
    ]
)

# ===========================
# 1. OVERVIEW PAGE
# ===========================
if page == "Overview":
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{len(data['customer_features']):,}",
            "From Online Retail Dataset"
        )
    
    with col2:
        st.metric(
            "Total Transactions",
            f"{len(data['transactions']):,}",
            "2010-2011 Period"
        )
    
    with col3:
        repeat_buyers = data['customer_features']['RepeatBuyer'].sum()
        repeat_pct = (repeat_buyers / len(data['customer_features']) * 100)
        st.metric(
            "Repeat Buyers",
            f"{repeat_buyers:,}",
            f"{repeat_pct:.1f}% of customers"
        )
    
    st.divider()
    
    # System Architecture
    st.subheader("System Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Part A: Predict
        - **Task**: Identify repeat buyers
        - **Methods**: 3 classifiers
        - **Metrics**: Accuracy, Precision, Recall, F1
        - **Best**: Random Forest (F1: 98.8%)
        """)
    
    with col2:
        st.markdown("""
        ### Part B: Recommend
        - **Methods**: 3 techniques
        - **Popularity**: Cold-start baseline
        - **Apriori**: Market basket analysis
        - **CF**: Item-based similarity
        """)
    
    with col3:
        st.markdown("""
        ### Part C: Hybrid
        - **Driver**: Classifier output
        - **Logic**: Stage-based selection
        - **Result**: Personalized recommendations
        - **Benefit**: Best method per customer
        """)
    
    st.divider()
    
    # Key Statistics
    st.subheader("Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_products = data['transactions']['StockCode'].nunique()
        st.metric("Unique Products", f"{unique_products:,}")
    
    with col2:
        avg_spend = data['customer_features']['total_spend'].mean()
        st.metric("Avg Spend per Customer", f"£{avg_spend:.2f}")
    
    with col3:
        countries = data['transactions']['Country'].nunique()
        st.metric("Countries", countries)
    
    with col4:
        date_range = pd.to_datetime(data['transactions']['InvoiceDate'])
        days = (date_range.max() - date_range.min()).days
        st.metric("Time Period", f"{days} days")


# ===========================
# 2. CLASSIFICATION PAGE
# ===========================
elif page == "Classification":
    st.header("Repeat Buyer Classification Results")
    
    if data['model_comparison'] is not None:
        st.subheader("Model Performance Comparison")
        
        # Display table
        st.dataframe(
            data['model_comparison'].sort_values('f1', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                data['model_comparison'],
                x='model',
                y=['accuracy', 'precision', 'recall', 'f1'],
                barmode='group',
                title='Model Comparison: All Metrics',
                labels={'value': 'Score', 'model': 'Model'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            radar_df = data['model_comparison'].melt(
                id_vars='model',
                value_vars=['accuracy', 'precision', 'recall', 'f1'],
                var_name='metric',
                value_name='score'
            )
            fig = px.line_polar(
                radar_df,
                r='score',
                theta='metric',
                color='model',
                line_close=True,
                title='Model Performance Radar',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model info
        best_row = data['model_comparison'].loc[data['model_comparison']['f1'].idxmax()]
        
        st.markdown("---")
        st.subheader("Best Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Name", best_row['model'])
        with col2:
            st.metric("F1 Score", f"{best_row['f1']:.4f}")
        with col3:
            st.metric("Precision", f"{best_row['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{best_row['recall']:.4f}")
        
        # Interpretation
        st.markdown("""
        #### What This Means:
        - **F1 Score** (~98.8%): Excellent balance between precision and recall
        - **Precision** (~99.6%): Very few false positives - reliable predictions
        - **Recall** (~98.1%): Catches most actual repeat buyers
        - **Verdict**: Random Forest is the winner - use for production!
        """)
    else:
        st.warning("⚠️  Model comparison data not available. Run pipeline first.")


# ===========================
# 3. SEGMENTATION PAGE
# ===========================
elif page == "Segmentation":
    st.header("Customer Segmentation Analysis")
    
    # Stage distribution
    st.subheader("Customer Distribution by Stage")
    
    stage_counts = data['customer_features']['stage'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        count = stage_counts.get('new', 0)
        pct = (count / len(data['customer_features']) * 100)
        st.metric("🆕 New", f"{count:,}", f"{pct:.1f}%")
    
    with col2:
        count = stage_counts.get('early', 0)
        pct = (count / len(data['customer_features']) * 100)
        st.metric("📈 Early", f"{count:,}", f"{pct:.1f}%")
    
    with col3:
        count = stage_counts.get('established', 0)
        pct = (count / len(data['customer_features']) * 100)
        st.metric("Established", f"{count:,}", f"{pct:.1f}%")
    
    # Pie chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.pie(
            values=stage_counts.values,
            names=stage_counts.index,
            title="Customer Stage Distribution",
            color_discrete_map={'new': '#3498db', 'early': '#f39c12', 'established': '#27ae60'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stage characteristics
        stage_stats = data['customer_features'].groupby('stage').agg({
            'total_spend': ['mean', 'median'],
            'invoices': ['mean', 'median'],
            'RepeatBuyer': 'mean'
        }).round(2)
        
        st.dataframe(stage_stats, use_container_width=True)
    
    # Detailed statistics
    st.subheader("Stage Characteristics")
    
    for stage in ['new', 'early', 'established']:
        stage_data = data['customer_features'][data['customer_features']['stage'] == stage]
        
        if len(stage_data) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"{stage.title()} - Avg Spend",
                    f"£{stage_data['total_spend'].mean():.2f}"
                )
            
            with col2:
                st.metric(
                    f"{stage.title()} - Avg Invoices",
                    f"{stage_data['invoices'].mean():.1f}"
                )
            
            with col3:
                st.metric(
                    f"{stage.title()} - Avg Products",
                    f"{stage_data['unique_products'].mean():.1f}"
                )
            
            with col4:
                repeat_pct = (stage_data['RepeatBuyer'].mean() * 100)
                st.metric(
                    f"{stage.title()} - Repeat Rate",
                    f"{repeat_pct:.1f}%"
                )
            
            st.divider()


# ===========================
# 4. RECOMMENDERS PAGE
# ===========================
elif page == "Recommenders":
    st.header("Recommender System Comparison")
    
    if data['recommender_eval'] is not None:
        st.subheader("Evaluation Metrics at Different K Values")
        
        # Full table
        st.dataframe(
            data['recommender_eval'].round(4),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                data['recommender_eval'][data['recommender_eval']['k'] == 5],
                x='recommender',
                y='precision_at_k',
                title='Precision@5 by Method',
                height=400,
                labels={'precision_at_k': 'Precision@5', 'recommender': 'Method'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                data['recommender_eval'][data['recommender_eval']['k'] == 5],
                x='recommender',
                y='lift',
                title='Lift@5 vs Random Baseline',
                height=400,
                labels={'lift': 'Lift', 'recommender': 'Method'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive comparison
        st.subheader("Comprehensive Metrics Comparison (K=5)")
        
        eval_k5 = data['recommender_eval'][data['recommender_eval']['k'] == 5]
        
        fig = go.Figure()
        
        metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k']
        
        for _, row in eval_k5.iterrows():
            fig.add_trace(go.Bar(
                x=metrics,
                y=[row['precision_at_k'], row['recall_at_k'], row['ndcg_at_k'], row['map_at_k']],
                name=row['recommender']
            ))
        
        fig.update_layout(
            title='All Metrics Comparison (K=5)',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.subheader("Metric Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Precision@K**
            - % of recommendations user bought
            - Higher = more accurate
            - Best for: Avoiding wasted suggestions
            """)
        
        with col2:
            st.markdown("""
            **Recall@K**
            - % of user's purchases captured
            - Higher = more coverage
            - Best for: Finding all relevant items
            """)
        
        with col3:
            st.markdown("""
            **Lift**
            - How much better than random
            - >1 = better than guessing
            - Best for: ROI justification
            """)
        
        # Key insights
        best_precision = eval_k5.loc[eval_k5['precision_at_k'].idxmax()]
        best_lift = eval_k5.loc[eval_k5['lift'].idxmax()]
        
        st.markdown("---")
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Best Precision**: {best_precision['recommender']}
            - Precision@5: {best_precision['precision_at_k']:.4f}
            - Most accurate recommendations
            """)
        
        with col2:
            st.markdown(f"""
            **Best Lift**: {best_lift['recommender']}
            - Lift: {best_lift['lift']:.2f}x better than random
            - Best value for business
            """)
    
    else:
        st.warning("⚠️  Recommender evaluation data not available. Run pipeline first.")


# ===========================
# 5. INTERACTIVE DEMO PAGE
# ===========================
elif page == "Interactive Demo":
    st.header("Interactive Recommendation Demo")
    
    st.markdown("""
    Select a customer and see what recommendations they would receive based on the hybrid system.
    """)
    
    # Customer selector
    customer_ids = data['customer_features']['CustomerID'].unique()
    selected_customer_id = st.selectbox(
        "Choose a customer:",
        customer_ids,
        format_func=lambda x: f"Customer {int(x)}"
    )
    
    # Get customer info
    customer = data['customer_features'][data['customer_features']['CustomerID'] == selected_customer_id].iloc[0]
    customer_trans = data['transactions'][data['transactions']['CustomerID'] == selected_customer_id]
    
    st.divider()
    
    # Customer profile
    st.subheader("👤 Customer Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customer ID", int(selected_customer_id))
    
    with col2:
        st.metric("Total Spend", f"£{customer['total_spend']:.2f}")
    
    with col3:
        st.metric("# Invoices", int(customer['invoices']))
    
    with col4:
        st.metric("# Products", int(customer['unique_products']))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stage = customer['stage']
        st.metric("Customer Stage", stage.upper() if stage else "Unknown")
    
    with col2:
        repeat = "Yes" if customer['RepeatBuyer'] == 1 else "No"
        st.metric("Repeat Buyer", repeat)
    
    with col3:
        country = customer['Country']
        st.metric("Country", country)
    
    with col4:
        recency = int(customer['recency_days'])
        st.metric("Recency (days)", recency)
    
    st.divider()
    
    # Purchase history
    st.subheader("Recent Purchases")
    
    if len(customer_trans) > 0:
        purchase_summary = customer_trans.groupby('Description').agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum'
        }).sort_values('Quantity', ascending=False).head(10)
        
        st.dataframe(purchase_summary, use_container_width=True)
    else:
        st.info("No purchase history found.")
    
    st.divider()
    
    # Hybrid recommendation logic
    st.subheader("Hybrid Recommendation Logic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if customer['RepeatBuyer'] == 1:
            pred_text = "REPEAT BUYER"
            pred_color = "green"
        else:
            pred_text = "NON-REPEAT"
            pred_color = "red"
        st.markdown(f"### Step 1: Classification\n**{pred_text}**")
    
    with col2:
        stage_upper = str(customer['stage']).upper()
        st.markdown(f"### Step 2: Segmentation\n**{stage_upper}**")
    
    with col3:
        if customer['stage'] == 'new' or customer['RepeatBuyer'] == 0:
            method = "POPULARITY"
        elif customer['stage'] == 'early':
            method = "APRIORI (MBA)"
        else:
            method = "COLLABORATIVE FILTERING"
        st.markdown(f"### Step 3: Method Selection\n**{method}**")
    
    st.divider()
    
    # Show recommendation method
    st.subheader("Recommendations")
    
    st.markdown("""
    ### Method Selection Explanation:
    """)
    
    if customer['stage'] == 'new' or customer['RepeatBuyer'] == 0:
        st.info("""
        **Popularity-Based Recommendations**
        - Used for: New customers, non-repeat buyers
        - Logic: Recommend most popular products (proven sellers)
        - Benefit: No personalization data needed (cold-start)
        - Approach: Top products in customer's country or globally
        """)
        rec_method = "Popularity"
    
    elif customer['stage'] == 'early':
        st.info("""
        **Market Basket Analysis (Apriori)**
        - Used for: Early-stage repeat customers (2-3 purchases)
        - Logic: "Customers who bought A also bought B"
        - Benefit: Learn from association patterns
        - Approach: Find association rules from past transactions
        """)
        rec_method = "Apriori"
    
    else:  # established
        st.info("""
        **Collaborative Filtering (Item-Based)**
        - Used for: Established customers (4+ purchases)
        - Logic: "Products similar to what you bought"
        - Benefit: Leverage full purchase history
        - Approach: Find items similar to past purchases
        """)
        rec_method = "Collaborative Filtering"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        #### Why This Method?
        - Customer is **{customer['stage']}** stage
        - Repeat buyer: **{"Yes" if customer['RepeatBuyer'] == 1 else "No"}**
        - Personalization level: **{"High" if customer['stage'] == 'established' else "Medium" if customer['stage'] == 'early' else "Low"}**
        """)
    
    with col2:
        st.markdown(f"""
        #### Expected Results
        - **Precision@5**: Higher for {rec_method}
        - **Coverage**: Best for this customer stage
        - **Business value**: Optimized recommendations
        """)


# ===========================
# 6. BUSINESS IMPACT PAGE
# ===========================
elif page == "Business Impact":
    st.header("Business Impact & Value Analysis")
    
    st.subheader("Strategic Value Propositions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Part A: Prediction
        **Value**: Identify repeat buyers
        - **Accuracy**: 98.5%
        - **Use**: Target best customers
        - **ROI**: Reduce marketing waste
        """)
    
    with col2:
        st.markdown("""
        ### Part B: Segmentation
        **Value**: Tier customers
        - **New**: Cold-start handling
        - **Early**: Growing relationship
        - **Est.**: High-value targets
        """)
    
    with col3:
        st.markdown("""
        ### Part C: Personalization
        **Value**: Right recommendations
        - **Precision**: 41.6% (CF)
        - **Lift**: 8.3x better
        - **Engagement**: Higher CTR
        """)
    
    st.divider()
    
    # Revenue impact
    st.subheader("Estimated Revenue Impact")
    
    total_customers = len(data['customer_features'])
    repeat_buyers = data['customer_features']['RepeatBuyer'].sum()
    avg_spend = data['customer_features']['total_spend'].mean()
    
    # Scenarios
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        #### Baseline (No System)
        - Customers: {total_customers:,}
        - Repeat buyers: {repeat_buyers:,} ({repeat_buyers/total_customers*100:.1f}%)
        - Avg spend/customer: £{avg_spend:.2f}
        - **Total revenue: £{total_customers * avg_spend:,.0f}**
        """)
    
    with col2:
        # Assume 10% improvement from recommendations
        improvement = 0.10
        revenue_increase = total_customers * avg_spend * improvement
        st.markdown(f"""
        #### With System (+{improvement*100:.0f}% improvement)
        - Additional revenue: £{revenue_increase:,.0f}
        - From: Better recommendations
        - Impact: Higher repeat purchases
        - **New revenue: £{total_customers * avg_spend * (1 + improvement):,.0f}**
        """)
    
    with col3:
        # Assume CF method has 41.6% precision vs 21.4% popularity
        cf_precision = 0.416
        pop_precision = 0.214
        improvement_cf = (cf_precision - pop_precision) / pop_precision
        revenue_increase_cf = total_customers * avg_spend * improvement_cf * 0.5  # Conservative
        st.markdown(f"""
        #### With Hybrid System
        - CF vs Popularity: {improvement_cf*100:.0f}% better
        - Implementation: Phased rollout
        - Revenue impact: £{revenue_increase_cf:,.0f}
        - **Total with hybrid: £{total_customers * avg_spend + revenue_increase_cf:,.0f}**
        """)
    
    st.divider()
    
    # Key metrics
    st.subheader("Key Metrics for Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Classification Metrics
        - **F1 Score**: 98.8% (excellent balance)
        - **Precision**: 99.6% (few false positives)
        - **Recall**: 98.1% (catches most repeats)
        - **Action**: Use for targeting
        """)
    
    with col2:
        st.markdown("""
        ### Recommendation Metrics
        - **Precision@5**: 41.6% (CF method)
        - **Recall@5**: 36.5% (captures purchases)
        - **Lift**: 8.31x better than random
        - **Coverage**: 68% (room to optimize)
        """)
    
    st.divider()
    
    # Risk & Mitigation
    st.subheader("Risk Mitigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Data Privacy (GDPR)
        - **Risk**: Personal data in CF
        - **Mitigation**: Anonymization
        - **Solution**: Aggregate-level similarity
        - **Compliance**: Data retention policies
        """)
    
    with col2:
        st.markdown("""
        ### Filter Bubble / Bias
        - **Risk**: Show only similar items
        - **Mitigation**: Diversity boost
        - **Solution**: Mix popular + recommended
        - **Monitoring**: Track recommendation diversity
        """)
    



# Footer
st.divider()
st.markdown("""
---
**Dashboard Version**: 1.0 | **Last Updated**: May 2026 | **Project**: Dissertation - Customer Analytics & Recommendation System
""")
