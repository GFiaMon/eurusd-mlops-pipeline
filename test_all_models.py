"""
Test script to verify the Flask app works with all three model types
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

load_dotenv()

# Set MLflow tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

from api.app_cloud import FeatureEngineer, load_data_file
import joblib

# Load test data
SCALER_PATH = os.path.join("data", "processed", "scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# Load data
df_data = None
local_data_paths = [
    'data/raw/eurusd_raw.csv',
    'data/processed/test.csv',
    'data/processed/train.csv'
]
for path in local_data_paths:
    if os.path.exists(path):
        if load_data_file(path):
            print(f"✓ Loaded data from: {path}")
            from api.app_cloud import df_data
            break

if df_data is None:
    print("❌ Could not load data")
    sys.exit(1)

# Test configurations for each model type
test_configs = [
    {
        "name": "Linear Regression",
        "model_uri": "models:/EURUSD_LinearRegression@champion",
        "config": {
            "features": ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5'],
            "target": "Target",
            "model_type": "LinearRegression"
        }
    },
    {
        "name": "ARIMA",
        "model_uri": "models:/EURUSD_ARIMA@challenger",
        "config": {
            "features": ['Return_Unscaled'],
            "target": "Target",
            "model_type": "ARIMA"
        }
    },
    {
        "name": "LSTM",
        "model_uri": "models:/EURUSD_LSTM@candidate",
        "config": {
            "features": ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5'],
            "target": "Target",
            "time_steps": 1,
            "n_features": 6,
            "model_type": "LSTM"
        }
    }
]

print("\n" + "=" * 70)
print("TESTING ALL MODEL TYPES")
print("=" * 70)

all_passed = True

for test_config in test_configs:
    print(f"\n{'=' * 70}")
    print(f"Testing: {test_config['name']}")
    print(f"{'=' * 70}")
    
    try:
        # Load model - special handling for ARIMA
        print(f"Loading model from: {test_config['model_uri']}")
        
        if 'arima' in test_config['name'].lower():
            print("Using sklearn loader for ARIMA")
            model = mlflow.sklearn.load_model(test_config['model_uri'])
        else:
            model = mlflow.pyfunc.load_model(test_config['model_uri'])
            
        print(f"✓ Model loaded successfully")
        
        # Create feature engineer
        fe = FeatureEngineer(test_config['config'], scaler)
        
        # Preprocess data
        X, current_price = fe.preprocess(df_data)
        
        print(f"✓ Preprocessed data type: {type(X)}")
        print(f"✓ Preprocessed data shape: {X.shape}")
        
        if isinstance(X, pd.DataFrame):
            print(f"✓ Column names: {list(X.columns)}")
        elif isinstance(X, np.ndarray):
            print(f"✓ Array dimensions: {X.ndim}D")
            
        print(f"✓ Current price: ${current_price:.4f}")
        
        # Make prediction - handle ARIMA specially
        if 'arima' in test_config['name'].lower():
            pred_return = model.predict(n_periods=1)[0]
        else:
            pred_return = model.predict(X)
            
        print(f"✓ Prediction successful!")
        print(f"  Raw prediction: {pred_return}")
        
        # Handle varying output shapes
        if hasattr(pred_return, 'values'):
            pred_return = pred_return.values
        if isinstance(pred_return, (list, np.ndarray)) and len(np.shape(pred_return)) > 0:
            pred_return = pred_return.flatten()[0]
        else:
            pred_return = float(pred_return)
            
        predicted_price = current_price * (1 + pred_return)
        change_pct = pred_return * 100
        
        print(f"✓ Predicted return: {pred_return:.6f}")
        print(f"✓ Predicted price: ${predicted_price:.4f}")
        print(f"✓ Change: {change_pct:+.2f}%")
        
        print(f"\n✅ {test_config['name']} - PASSED")
        
    except Exception as e:
        print(f"\n❌ {test_config['name']} - FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL MODEL TYPES WORK CORRECTLY!")
else:
    print("❌ SOME MODELS FAILED - SEE ERRORS ABOVE")
print("=" * 70)

sys.exit(0 if all_passed else 1)
