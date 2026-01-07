import mlflow
import os
import shutil
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
    exit()

print(f"Checking runs in experiment: {experiment.name} (ID: {experiment.experiment_id})")

# Get ARIMA runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.model_type = 'ARIMA'",
    order_by=["attribute.start_time DESC"]
)

print(f"Found {len(runs)} ARIMA runs.")

for run in runs:
    run_id = run.info.run_id
    start_time = run.info.start_time
    rmse = run.data.metrics.get('rmse', 'N/A')
    
    print(f"\nRun ID: {run_id}")
    print(f"  Date: {start_time}")
    print(f"  RMSE: {rmse}")
    
    # Check artifacts
    try:
        model_uri = f"runs:/{run_id}/model"
        # We can't easily check sample size without loading, which takes time/dependencies.
        # But we can ask user if they want to delete it.
    except:
        pass

print("\nTo delete a specific mismatched run, you can use:")
print("mlflow.delete_run('RUN_ID')")
print("\nOr I can create a script to delete the latest one if you confirm.")
