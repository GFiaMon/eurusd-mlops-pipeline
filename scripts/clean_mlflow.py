#!/usr/bin/env python3
"""
Clean MLflow Registry - Delete all models and start fresh
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to MLflow server
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://98.84.5.27:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

print(f"Connected to MLflow: {tracking_uri}")

# Safety Confirmation
confirm = input(f"⚠️  This will delete all versions and metadata for ['EURUSD_LSTM', 'EURUSD_LinearRegression', 'EURUSD_ARIMA'] from the Model Registry. \nUnderlying experiment runs will NOT be deleted. \nContinue? (y/n): ")
if confirm.lower() != 'y':
    print("❌ Aborted.")
    exit()

print("\n🗑️  Deleting all registered models...\n")

# Models to delete
models = ["EURUSD_LSTM", "EURUSD_LinearRegression", "EURUSD_ARIMA"]

for model_name in models:
    try:
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"⚠️  {model_name}: No versions found in registry")
            continue
            
        # Delete all versions
        for version in versions:
            client.delete_model_version(model_name, version.version)
            print(f"  ✅ Deleted {model_name} v{version.version}")
        
        # Delete the model itself
        client.delete_registered_model(model_name)
        print(f"✅ Deleted model entry: {model_name}\n")
        
    except Exception as e:
        print(f"❌ Error deleting {model_name}: {e}\n")

print("\n🎉 Cleanup complete!")
print("\nNext steps to restore registry with latest format:")
print("1. python src/03_train_models.py    (Optional: only if you want new runs)")
print("2. python src/04_evaluate_select.py (Required: will re-register the best runs and assign @champion/@challenger)")

