simport mlflow
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
MODEL_NAMES = ["EURUSD_ARIMA", "EURUSD_LSTM", "EURUSD_LinearRegression"]
EXPECTED_SAMPLES = 1004

def check_and_clean_registry():
    print("search_model_versions...")
    
    for model_name in MODEL_NAMES:
        print(f"\nScanning Registry for: {model_name}")
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            print(f"   ⚠️ Could not search versions for {model_name}: {e}")
            continue

        if not versions:
            print(f"   No registered versions found for {model_name}.")
            continue
            
        print(f"   Found {len(versions)} versions.")
        
        for v in versions:
            version_num = v.version
            run_id = v.run_id
            
            # 1. Check if Run exists AND is active
            try:
                run = client.get_run(run_id)
                if run.info.lifecycle_stage == 'deleted':
                    print(f"   ❌ Version {version_num} (Run {run_id}): Run is DELETED. Deleting version...")
                    client.delete_model_version(name=model_name, version=version_num)
                    continue
            except Exception:
                print(f"   ❌ Version {version_num} (Run {run_id}): Run NOT found (orphaned). Deleting version...")
                client.delete_model_version(name=model_name, version=version_num)
                continue
            
            print(f"   ✅ Version {version_num} (Run {run_id}): Run active.", end=" ")
            print("Skipping deep check.")

if __name__ == "__main__":
    check_and_clean_registry()
