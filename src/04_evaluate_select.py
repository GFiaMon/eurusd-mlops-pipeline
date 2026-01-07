import pandas as pd
import numpy as np
import mlflow
import os
import json
import warnings
from dotenv import load_dotenv

load_dotenv()

# Suppress annoying "pkg_resources is deprecated" warning from mlflow dependencies
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
from sklearn.metrics import mean_squared_error

# Constants
# Constants
MODELS_DIR = "models"
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.json")
EXPERIMENT_NAME = "EURUSD_Experiments"

def evaluate_and_select():
    # 1. Load Test Data for Baseline Comparison
    print("Loading test data for baseline comparison...")
    try:
        from utils.data_manager import DataManager
    except ImportError:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from utils.data_manager import DataManager

    dm = DataManager(data_type='processed')
    _, test_df, _ = dm.load_processed()

    if test_df is None:
        print("Error: Test data not found (DataManager load failed).")
        return
    # Naive Baseline: Prediction = Previous day's return
    # In our dataset, 'Return' is current day return. 'Target' is Next Day return.
    # So Naive prediction for Target(t) (which is Return(t+1)) is Return(t).
    # Since our 'Return' column corresponds to the feature inputs for predicting 'Target',
    # we can just use `test_df['Return']` as the naive prediction for `test_df['Target']`.
    
    y_test = test_df['Target']
    y_pred_naive = test_df['Return_Unscaled']
    
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    
    # Calculate Naive Directional Accuracy
    # Use values to avoid index mismatch
    start_sign = np.sign(y_test.values)
    pred_sign = np.sign(y_pred_naive.values)
    naive_da = np.mean(start_sign == pred_sign)
    
    print(f"Naive Baseline RMSE: {naive_rmse}")
    print(f"Naive Baseline Dir. Acc: {naive_da}")

    # 2. Setup MLflow Client
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"Using Remote MLflow Tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("Using Local MLflow (sqlite:///mlflow.db)")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("Error: Experiment not found. Run training script first.")
        return

    # 3. Find LATEST Run Per Model Type (Focus on what we just trained)
    print("Querying MLflow runs (Latest first)...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        # order_by=["metrics.rmse ASC"]
        order_by=["attribute.start_time DESC"]
    )
    
    if not runs:
        print("No runs found.")
        return

    # Categorize runs by model type
    best_runs_per_type = {}
    global_best_run = None
    global_best_rmse = float('inf')

    for run in runs:
        model_type = run.data.params.get('model_type') or run.data.tags.get('model_type')
        rmse = run.data.metrics.get('rmse')
        
        if model_type and rmse is not None:
            # Update global best
            if rmse < global_best_rmse:
                global_best_rmse = rmse
                global_best_run = run
            
            # Update per-type best
            if model_type not in best_runs_per_type:
                best_runs_per_type[model_type] = run
            else:
                current_type_best = best_runs_per_type[model_type].data.metrics.get('rmse')
                if rmse < current_type_best:
                    best_runs_per_type[model_type] = run

    print("\nBest Models per Type:")
    registered_models = []
    
    for model_type, run in best_runs_per_type.items():
        rmse = run.data.metrics['rmse']
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = f"EURUSD_{model_type}"
        
        print(f"  {model_type}: RMSE={rmse:.5f} (Run ID: {run_id})")
        
        # Register the model (for history/provenance)
        reg_model = mlflow.register_model(model_uri, model_name)
        
        registered_models.append({
            "model_type": model_type,
            "name": model_name,
            "version": reg_model.version,
            "run_id": run_id,
            "rmse": rmse
        })

    # 4. Championship Ranking & Aliasing
    if registered_models:
        print("\n*** CHAMPIONSHIP RANKING (Lower RMSE is Better) ***")
        
        # Sort finalists by RMSE (ascending: lower is better)
        ranked_models = sorted(registered_models, key=lambda x: x['rmse'])
        
        alias_map = {
            0: "champion",
            1: "challenger",
            2: "candidate"
        }
        
        for rank, model_info in enumerate(ranked_models):
            model_name = model_info['name']
            version = model_info['version']
            rmse = model_info['rmse']
            model_type = model_info['model_type']
            
            # Determine alias based on rank
            alias = alias_map.get(rank, f"rank_{rank+1}")
            
            print(f"{rank+1}. {model_type} (RMSE: {rmse:.5f}) -> {model_name} (v{version}) gets alias '@{alias}'")
            
            # Assign the alias to the specific model
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version
            )

        # 5. Save Champion Info for Deployment
        # The JSON file will act as the pointer to the current champion
        champion = ranked_models[0]
        champion_uri = f"models:/{champion['name']}@champion"
        
        print(f"\nDeployment Choice: {champion['name']} (v{champion['version']})")
        
        # Compare Champion with Baseline
        if champion['rmse'] < naive_rmse:
            print(f"Validation: Champion beats Baseline ({champion['rmse']:.5f} < {naive_rmse:.5f})")
        else:
            print(f"WARNING: Champion performs worse than Baseline")
            
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        info = {
            "run_id": champion['run_id'],
            "model_type": champion['model_type'],
            "model_name": champion['name'],
            "model_version": champion['version'],
            "model_alias": "champion",
            "model_uri": champion_uri,
            "metrics": {
                "rmse": champion['rmse']
            },
            "baseline_metrics": {
                "rmse": naive_rmse,
                "directional_accuracy": naive_da
            }
        }
        
        with open(BEST_MODEL_INFO_PATH, 'w') as f:
            json.dump(info, f, indent=4)
            
    if registered_models:
        # ... (previous code) ...
        print(f"Best model info saved to {BEST_MODEL_INFO_PATH}")
        
        # Upload best_model_info.json to S3
        s3_bucket = os.getenv("S3_BUCKET")
        if s3_bucket:
            try:
                import boto3
                s3 = boto3.client('s3')
                s3_key = BEST_MODEL_INFO_PATH # models/best_model_info.json
                print(f"Uploading to S3: s3://{s3_bucket}/{s3_key}...")
                s3.upload_file(BEST_MODEL_INFO_PATH, s3_bucket, s3_key)
                print("  ✅ Upload successful.")
            except Exception as e:
                print(f"  ❌ Upload failed: {e}")
        else:
            print("  ⚠️ S3_BUCKET not set in .env, skipping upload.")
            
    else:
        print("No valid runs found to evaluate.")

if __name__ == "__main__":
    evaluate_and_select()
