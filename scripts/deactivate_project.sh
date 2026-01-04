#!/bin/bash
# Deactivate project environment

echo "Deactivating EUR/USD project environment..."
deactivate 2>/dev/null || true
echo "✅ Environment deactivated"
echo ""
echo "To reactivate:"
echo "  source ~/venvs/venv_eurusd/bin/activate"
echo "  or"
echo "  cd ~/eurusd-capstone && ./activate_project.sh"
