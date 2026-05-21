@echo off
REM Dashboard Launcher for Windows
REM Quick script to run the interactive dashboard

echo.
echo ========================================
echo 🚀 Starting Dashboard...
echo ========================================
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Streamlit not found. Installing...
    pip install streamlit
)

REM Run dashboard
streamlit run dashboard.py

pause
