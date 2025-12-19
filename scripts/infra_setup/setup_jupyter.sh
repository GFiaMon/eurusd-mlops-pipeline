#!/bin/bash
# Setup Jupyter kernel for VS Code and other IDEs

echo "Setting up Jupyter kernel for EUR/USD Capstone..."

# Activate virtual environment
source ~/venvs/venv_eurusd/bin/activate

# Install ipykernel if not already installed
pip install ipykernel

# Create kernel specification
python -m ipykernel install --user --name="venv_eurusd" --display-name="EUR/USD Capstone"

echo ""
echo "✅ Jupyter kernel setup complete!"
echo ""
echo "To use in VS Code:"
echo "1. Open a .ipynb file"
echo "2. Click on the kernel selector (top right)"
echo "3. Select 'EUR/USD Capstone'"
echo ""
echo "To use in Jupyter Lab/Notebook:"
echo "1. Start Jupyter: jupyter lab"
echo "2. Create new notebook"
echo "3. Select 'EUR/USD Capstone' kernel"
