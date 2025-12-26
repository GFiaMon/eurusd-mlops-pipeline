
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Optional S3 support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not available - S3 functionality disabled")

def download_s3_json(bucket, key, region='us-east-1'):
    """Download and parse JSON from S3"""
    if not S3_AVAILABLE: return None
    try:
        s3 = boto3.client('s3', region_name=region)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"S3 Load Error: {e}")
        return None

def load_model_from_mlflow(project_root, use_s3=False, s3_bucket=None, s3_key=None, tracking_uri=None, model_name_override=None):
    """
    Load model, scaler, and config with robust alias handling and explicit tracking URI.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Explicitly pass tracking_uri to Client
    client = MlflowClient(tracking_uri=tracking_uri)
    
    model_name = ""
    target_aliases = ["champion", "challenger", "candidate"]
    
    # 1. Resolve Target Model
    if model_name_override and model_name_override != "best":
        model_key = model_name_override.lower()
        if 'lstm' in model_key:
            model_name = "EURUSD_LSTM"
        elif 'arima' in model_key:
            model_name = "EURUSD_ARIMA"
        elif 'linear' in model_key:
            model_name = "EURUSD_LinearRegression"
        else:
            model_name = f"EURUSD_{model_name_override.upper()}"
    else:
        # Load Best Model Info
        model_info = None
        if use_s3 and s3_bucket and s3_key:
            model_info = download_s3_json(s3_bucket, s3_key)
        
        if not model_info:
            local_info_path = os.path.join(project_root, 'models', 'best_model_info.json')
            if os.path.exists(local_info_path):
                with open(local_info_path, 'r') as f:
                    model_info = json.load(f)
        
        if model_info:
            model_name = model_info.get('model_name', "EURUSD_LinearRegression")
            json_alias = model_info.get('model_alias')
            if json_alias: target_aliases = [json_alias] + target_aliases

    if not model_name:
        return None, None, None, False

    # 2. Try to find the model and its run_id
    model = None
    run_id = None
    
    print(f"--- Searching Registry for: {model_name} ---")
    
    for alias in target_aliases:
        try:
            model_uri = f"models:/{model_name}@{alias}"
            # Verify alias exists
            v_info = client.get_model_version_by_alias(model_name, alias)
            run_id = v_info.run_id
            
            print(f"Attempting load: {model_uri}")
            if 'arima' in model_name.lower():
                model = mlflow.sklearn.load_model(model_uri)
            else:
                model = mlflow.pyfunc.load_model(model_uri)
            print(f"SUCCESS: Loaded via alias '{alias}'")
            break
        except Exception:
            continue
            
    # Fallback to latest version
    if not model:
        try:
            print(f"No aliases matched. Trying latest version of {model_name}...")
            versions = client.get_latest_versions(model_name)
            if versions:
                latest = versions[0]
                run_id = latest.run_id
                model_uri = f"models:/{model_name}/{latest.version}"
                if 'arima' in model_name.lower():
                    model = mlflow.sklearn.load_model(model_uri)
                else:
                    model = mlflow.pyfunc.load_model(model_uri)
                print(f"SUCCESS: Loaded version {latest.version}")
        except Exception as e:
            print(f"CRITICAL: Failed to load {model_name} from registry: {e}")

    if not model:
        return None, None, None, False

    # 3. Load Artifacts
    feature_config = None
    scaler = None
    
    if run_id:
        try:
            p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_config.json")
            with open(p, 'r') as f:
                feature_config = json.load(f)
        except: pass
        
        try:
            p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler/scaler.pkl")
            scaler = joblib.load(p)
        except: pass

    # Robust Metadata
    if not feature_config:
        feature_config = {
            "features": ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5'],
            "model_type": "LSTM" if "LSTM" in model_name else "ARIMA" if "ARIMA" in model_name else "LinearRegression",
            "time_steps": 60 if "LSTM" in model_name else 1
        }
    
    # Ensure model_type is set for UI labels
    if 'model_type' not in feature_config:
        feature_config['model_type'] = "LSTM" if "LSTM" in model_name else "ARIMA" if "ARIMA" in model_name else "Linear"

    return model, scaler, feature_config, True
