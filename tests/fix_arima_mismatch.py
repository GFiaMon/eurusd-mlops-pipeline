import mlflow
import mlflow.sklearn
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Setup MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    print(f"📡 Using Remote MLflow: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
else:
    print("💾 Using Local MLflow")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("EURUSD_Experiments")

if not experiment:
    print("Experiment not found.")
    sys.exit(1)

print("\n🔍 Searching for ALL ARIMA runs to validate...")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.model_type = 'ARIMA'",
    order_by=["attribute.start_time DESC"]
)

if not runs:
    print("No ARIMA runs found.")
    sys.exit(0)

print(f"Found {len(runs)} ARIMA runs. Checking for sample size mismatches...")
print("Expected Training Samples: 1004")
print("-" * 60)

deleted_count = 0
kept_count = 0
error_count = 0

for i, run in enumerate(runs):
    run_id = run.info.run_id
    start_time = run.info.start_time
    
    print(f"[{i+1}/{len(runs)}] Checking Run {run_id}...", end=" ", flush=True)
    
    try:
        # Load model
        model_uri = f"runs:/{run_id}/model"
        # Suppress benign warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = mlflow.sklearn.load_model(model_uri)
        
        # Check size
        in_sample_preds = arima_model.predict_in_sample()
        n_samples = len(in_sample_preds)
        
        if n_samples != 1004:
            print(f"❌ MISMATCH ({n_samples} samples). Deleting...", end=" ")
            client.delete_run(run_id)
            print("Done.")
            deleted_count += 1
        else:
            print(f"✅ OK ({n_samples} samples). Keeping.")
            kept_count += 1
            
    except Exception as e:
        print(f"⚠️ Error: {e}")
        error_count += 1

print("-" * 60)
print("CLEANUP COMPLETE")
print(f"Deleted: {deleted_count}")
print(f"Kept:    {kept_count}")
print(f"Errors:  {error_count}")
