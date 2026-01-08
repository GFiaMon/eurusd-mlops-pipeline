#!/usr/bin/env python3
"""
Delete Model Versions by Metric Criteria
==========================================
This script allows you to delete registered model versions based on run metrics.

Usage:
    python tests/delete_models_by_metric.py

Customize the FILTER_CRITERIA section below to change what gets deleted.
"""

import mlflow
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION - Modify these to change filtering behavior
# ============================================================================

# Model names to scan (leave empty to scan all models)
MODEL_NAMES = [
    "EURUSD_LinearRegression",
    "EURUSD_ARIMA", 
    "EURUSD_LSTM"
]

# FILTER CRITERIA - Customize this function to define what should be deleted
def should_delete_version(run_metrics, run_params):
    """
    Define your deletion criteria here.
    
    Args:
        run_metrics: dict of metrics from the run (e.g., {'rmse': 0.005, 'directional_accuracy': 0.85})
        run_params: dict of params from the run (e.g., {'model_type': 'LinearRegression'})
    
    Returns:
        True if the version should be deleted, False otherwise
    """
    # Example 1: Delete if Directional Accuracy > 0.8 (overfitting indicator)
    da = run_metrics.get('directional_accuracy', 0)
    if da > 0.8:
        return True
    
    # Example 2: Delete if Directional Accuracy > 0.7 (more aggressive)
    # if da > 0.7:
    #     return True
    
    # Example 3: Delete if RMSE is suspiciously low (< 0.003)
    # rmse = run_metrics.get('rmse', float('inf'))
    # if rmse < 0.003:
    #     return True
    
    # Example 4: Delete specific model types with high DA
    # model_type = run_params.get('model_type', '')
    # if 'LinearRegression' in model_type and da > 0.75:
    #     return True
    
    return False

# ============================================================================
# MAIN SCRIPT - No need to modify below unless changing core logic
# ============================================================================

def main():
    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"📡 Using Remote MLflow: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("💾 Using Local MLflow")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = mlflow.tracking.MlflowClient()
    
    # Confirmation
    print("\n" + "="*70)
    print("MODEL VERSION DELETION BY METRIC CRITERIA")
    print("="*70)
    print("\nThis script will DELETE model versions where:")
    print("  - Directional Accuracy > 0.8 (current criteria)")
    print("\nYou can modify the 'should_delete_version()' function to change criteria.")
    print("\n⚠️  WARNING: This will PERMANENTLY delete model versions from the registry.")
    print("   (The underlying runs will NOT be deleted)")
    
    confirm = input("\nContinue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return
    
    total_scanned = 0
    total_deleted = 0
    total_kept = 0
    total_errors = 0
    
    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"Scanning: {model_name}")
        print("="*70)
        
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            print(f"   ⚠️ Could not search versions: {e}")
            continue
        
        if not versions:
            print(f"   No versions found.")
            continue
        
        print(f"   Found {len(versions)} versions.")
        
        for version_info in versions:
            total_scanned += 1
            version_num = version_info.version
            run_id = version_info.run_id
            
            try:
                # Get the run to access metrics
                run = client.get_run(run_id)
                
                # Check if run is deleted
                if run.info.lifecycle_stage == 'deleted':
                    print(f"   Version {version_num}: Run is deleted, skipping.")
                    continue
                
                # Extract metrics and params
                run_metrics = run.data.metrics
                run_params = run.data.params
                
                # Apply filter
                if should_delete_version(run_metrics, run_params):
                    # Show what we found
                    da = run_metrics.get('directional_accuracy', 'N/A')
                    rmse = run_metrics.get('rmse', 'N/A')
                    print(f"\n   Version {version_num} (Run {run_id[:8]})")
                    print(f"      DA: {da}, RMSE: {rmse}")
                    
                    # Ask for confirmation for each
                    choice = input(f"      Delete this version? (y/n/all): ")
                    
                    if choice.lower() == 'y' or choice.lower() == 'all':
                        client.delete_model_version(name=model_name, version=version_num)
                        print(f"      ✅ Deleted.")
                        total_deleted += 1
                        
                        # If 'all', set auto-confirm for rest (simplified - just for this model)
                        # For true 'all', you'd need a global flag
                    else:
                        print(f"      ⏭️  Skipped.")
                        total_kept += 1
                else:
                    # Doesn't match criteria, keep it
                    total_kept += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error processing version {version_num}: {e}")
                total_errors += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Scanned:  {total_scanned}")
    print(f"Total Deleted:  {total_deleted}")
    print(f"Total Kept:     {total_kept}")
    print(f"Total Errors:   {total_errors}")
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
