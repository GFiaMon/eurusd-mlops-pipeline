import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# Set the Tracking URI correctly (same as in your training script)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    print(f"Using Remote MLflow: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
else:
    print("Using Local MLflow: sqlite:///mlflow.db")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()

# Get the experiment
experiment_name = "EURUSD_Experiments"
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    print(f"Experiment '{experiment_name}' not found.")
else:
    print(f"Found Experiment: {experiment.name} (ID: {experiment.experiment_id})")
    
    # Get runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )
    
    print(f"Found {len(runs)} runs.")
    
    for run in runs:
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model Type: {run.data.tags.get('model_type')}")
        
        # List artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        print("Artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")
            # If it's a directory, list children (like 'model' or 'scaler')
            if artifact.is_dir:
                try:
                    children = client.list_artifacts(run.info.run_id, artifact.path)
                    for child in children:
                        print(f"    - {child.path}")
                except:
                    pass
