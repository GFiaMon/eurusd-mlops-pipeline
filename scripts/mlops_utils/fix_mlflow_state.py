#!/usr/bin/env python3
"""
Fix MLflow Deleted State - Restores experiments so they can be reused.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI") #, "http://98.84.5.27:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

print(f"🔍 Checking for deleted experiments on {tracking_uri}...")

# Search for ALL experiments (including deleted ones)
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

count = 0
for exp in experiments:
    if exp.lifecycle_stage == "deleted":
        print(f"♻️  Restoring experiment: {exp.name} (ID: {exp.experiment_id})")
        client.restore_experiment(exp.experiment_id)
        
        # Now that it's restored, let's make sure it's empty
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            client.delete_run(run.info.run_id)
        
        count += 1

if count == 0:
    print("✅ No deleted experiments found. Your database is ready.")
else:
    print(f"\n✨ Fixed {count} experiments. You can now run your training scripts!")
