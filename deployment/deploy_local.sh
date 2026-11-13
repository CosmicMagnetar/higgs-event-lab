#!/usr/bin/env bash
# Local development and launch script for Higgs Event Lab Dashboard
set -euo pipefail

echo "=================================================="
echo "Higgs Event Lab: Local Dashboard Launch"
echo "=================================================="

# Ensure PYTHONPATH points to src
export PYTHONPATH="${PYTHONPATH:-}:src"

# Verify that streamlit is available
if ! command -v streamlit >/dev/null 2>&1; then
    echo "Streamlit not found in current environment. Installing..."
    pip install streamlit
fi

echo "Launching dashboard on http://localhost:8501..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
