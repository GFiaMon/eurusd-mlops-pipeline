#!/usr/bin/env python3
"""
MLflow Hard Reset - The Nuclear Option
Deletes all Registered Models, all Experiments (and Runs), and wipes the S3 artifact storage.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# Configuration
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://98.84.5.27:5000")
s3_bucket = os.getenv("S3_BUCKET", "eurusd-ml-models")
s3_prefix = "mlflow-artifacts/"

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

print(f"🚀 Starting Hard Reset for MLflow at {tracking_uri}")
print(f"🪣 Target S3 Bucket: {s3_bucket}/{s3_prefix}")

# Safety Confirmation
print("\n🔥 WARNING: This will PERMANENTLY delete:")
print("1. ALL Registered Models and their versions.")
print("2. ALL Experiment history and Runs.")
print("3. ALL Artifact files in S3.")
confirm = input("\nType 'DELETE EVERYTHING' to proceed: ")

if confirm != "DELETE EVERYTHING":
    print("❌ Aborted. Nothing was deleted.")
    exit()

# 1. Delete Registered Models
print("\n🗑️  Deleting Registered Models...")
try:
    for model in client.search_registered_models():
        name = model.name
        print(f"   Deleting model: {name}")
        # Delete all versions first
        for version in model.latest_versions:
            client.delete_model_version(name, version.version)
        client.delete_registered_model(name)
except Exception as e:
    print(f"   ⚠️ Error deleting models: {e}")

# 2. Delete Experiments
print("\n🗑️  Deleting Experiments...")
try:
    for exp in client.search_experiments(view_type=mlflow.entities.ViewType.ALL):
        if exp.name == "Default" and exp.experiment_id == "0":
            # Can't delete experiment 0 usually, but can delete its runs
            runs = client.search_runs(experiment_ids=["0"])
            for run in runs:
                client.delete_run(run.info.run_id)
            print("   Cleared runs in Default experiment (ID 0)")
            continue
            
        print(f"   Deleting experiment: {exp.name} (ID: {exp.experiment_id})")
        client.delete_experiment(exp.experiment_id)
except Exception as e:
    print(f"   ⚠️ Error deleting experiments: {e}")

# 3. Wipe S3 Bucket
print(f"\n🗑️  Wiping S3 artifacts in s3://{s3_bucket}/{s3_prefix}...")
try:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket)
    
    # Use objects.filter to only delete stuff under our prefix
    objects_to_delete = bucket.objects.filter(Prefix=s3_prefix)
    
    count = 0
    # Batch delete
    for obj in objects_to_delete:
        obj.delete()
        count += 1
    
    print(f"   ✅ Deleted {count} objects from S3.")
except Exception as e:
    print(f"   ⚠️ Error wiping S3: {e}")

print("\n✨ MLflow has been factory reset!")
print("The UI might still show 'deleted' items for a few minutes due to caching.")
print("\nNext steps to start fresh:")
print("1. python src/03_train_models.py")
print("2. python src/04_evaluate_select.py")
