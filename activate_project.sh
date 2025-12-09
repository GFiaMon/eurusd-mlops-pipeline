#!/bin/bash
# Quick activation script for EUR/USD project

PROJECT_DIR="$HOME/eurusd-capstone"
VENV_PATH="$HOME/venvs/venv_eurusd"

echo "Activating EUR/USD Capstone Project..."
echo ""

# Navigate to project
cd "$PROJECT_DIR" || { echo "Error: Project directory not found"; exit 1; }

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run setup script first"
    exit 1
fi

# Show project info
echo ""
echo "📁 Project: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📓 Kernel: EUR/USD Capstone (venv_eurusd)"
echo ""
echo "Available commands:"
echo "  python ml_pipeline.py    - Run ML pipeline"
echo "  jupyter notebook         - Start Jupyter"
echo "  mlflow ui                - Start MLflow UI"
echo ""
echo "Notebook workflow:"
echo "  1. cd notebooks/"
echo "  2. jupyter notebook"
echo "  3. Select 'EUR/USD Capstone' kernel"
