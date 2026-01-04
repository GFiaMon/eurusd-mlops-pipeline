
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
project_root = os.getcwd()
sys.path.append(project_root)

# Explicitly import from api.src
try:
    from api.src.model_loader import load_model_from_mlflow
    from api.src.feature_engineer import FeatureEngineer
    print("Imported from api.src")
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.path.join(project_root, 'api'))
    from src.model_loader import load_model_from_mlflow
    from src.feature_engineer import FeatureEngineer
    print("Imported from src (after adding api to path)")

from utils.data_manager import DataManager

def test_model(name):
    print(f"\n--- Testing {name} ---")
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    model, scaler, config, success = load_model_from_mlflow(
        project_root, 
        tracking_uri=tracking_uri,
        model_name_override=name
    )
    
    if not success:
        print(f"FAILED to load {name}")
        return
        
    print(f"SUCCESS: Loaded {name}. Type: {config.get('model_type')}")
    
    fe = FeatureEngineer(config, scaler)
    dm = DataManager(mode='local')
    df = dm.get_latest_data()
    
    try:
        X, price = fe.preprocess(df)
        print(f"Input Shape: {X.shape if hasattr(X, 'shape') else type(X)}")
        
        if 'arima' in name.lower():
            # Check if model has predict method with n_periods
            try:
                pred = model.predict(n_periods=1)
                if hasattr(pred, 'values'): pred = pred.values
                print(f"Prediction (ARIMA): {pred}")
            except Exception as e:
                print(f"ARIMA prediction failed: {e}")
        else:
            pred = model.predict(X)
            print(f"Prediction: {pred}")
    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model("best")
    test_model("lstm")
    test_model("arima")
