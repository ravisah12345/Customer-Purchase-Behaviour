#!/bin/bash
# Dashboard Launcher
# Quick script to run the interactive dashboard

echo "🚀 Starting Customer Analytics & Recommendation Dashboard..."
echo ""
echo "Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit
fi

# Run dashboard
streamlit run dashboard.py
