# api/app_cloud.py - Enhanced version with S3/MLflow support
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Try to import boto3 for S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

# Global variables
model = None
scaler = None
feature_config = None
df_data = None
load_success = False

# Configuration
USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
S3_PREFIX = os.getenv('S3_PREFIX', 'data/raw/')
S3_MODEL_INFO_KEY = os.getenv('S3_MODEL_INFO_KEY', 'models/best_model_info.json') # Key to JSON
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

def download_s3_json(bucket, key):
    """Download and parse JSON from S3"""
    if not S3_AVAILABLE:
        print("boto3 not available")
        return None
    
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        print(f"Fetching {key} from {bucket}...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"Error fetching JSON from S3: {e}")
        return None

def load_from_mlflow():
    """Load model, scaler, and config from MLflow"""
    global model, scaler, feature_config
    
    # 1. Get Model Info
    model_info = None
    if USE_S3:
        model_info = download_s3_json(S3_BUCKET, S3_MODEL_INFO_KEY)
    
    if not model_info:
        print("Could not retrieve model info from S3. Checking local fallback...")
        local_info_path = os.path.join(project_root, 'models', 'best_model_info.json')
        if os.path.exists(local_info_path):
             with open(local_info_path, 'r') as f:
                 model_info = json.load(f)
        else:
            print("No local model info found.")
            return False

    print(f"Model Info: {model_info}")
    model_uri = model_info.get('model_uri')
    model_type = model_info.get('model_type', 'Unknown')
    
    if not model_uri:
        print("Invalid model info: missing model_uri")
        return False
        
    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            print(f"MLflow URI set to: {MLFLOW_TRACKING_URI}")
        
        # 2. Load Model - special handling for ARIMA
        print(f"Loading model from {model_uri}...")
        
        # For ARIMA, we need to load the underlying sklearn model directly
        # because MLflow's pyfunc wrapper doesn't support ARIMA's predict(n_periods=1) interface
        if 'arima' in model_type.lower():
            print("Detected ARIMA model - using sklearn loader")
            model = mlflow.sklearn.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)
            
        print("Model loaded successfully")
        
        # 3. Get the run_id from the loaded model's metadata
        # The model object has metadata that includes the run_id
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Parse model name and version/alias from URI
        # Format: models:/MODEL_NAME@ALIAS or models:/MODEL_NAME/VERSION
        model_name = model_info.get('model_name')
        model_alias = model_info.get('model_alias')
        
        if model_alias:
            # Get model version by alias
            model_version_info = client.get_model_version_by_alias(model_name, model_alias)
        else:
            model_version = model_info.get('model_version')
            model_version_info = client.get_model_version(model_name, model_version)
        
        run_id = model_version_info.run_id
        print(f"Retrieved run_id from model version: {run_id}")
        
        # 4. Load Feature Config
        try:
            local_config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_config.json")
            with open(local_config_path, 'r') as f:
                feature_config = json.load(f)
            print(f"Feature Config loaded: {feature_config}")
        except Exception as e:
            print(f"Warning: Could not load feature_config.json: {e}")
            # Fallback to default config
            feature_config = {
                "features": ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5'],
                "target": "Target",
                "model_type": model_type,
                "time_steps": 1,  # Default for LSTM
                "n_features": 6
            }
            print(f"Using fallback feature config: {feature_config}")
        
        # 5. Load Scaler
        try:
            local_scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler/scaler.pkl")
            scaler = joblib.load(local_scaler_path)
            print("Scaler loaded")
        except Exception as e:
            print(f"Warning: Could not load scaler: {e}")
            # Try local fallback
            local_scaler = os.path.join(project_root, 'data', 'processed', 'scaler.pkl')
            if os.path.exists(local_scaler):
                scaler = joblib.load(local_scaler)
                print("Loaded scaler from local fallback")
            else:
                print("No scaler available - predictions may be incorrect")
        
        return True
        
    except Exception as e:
        print(f"Error loading from MLflow: {e}")
        import traceback
        traceback.print_exc()
        return False

class FeatureEngineer:
    def __init__(self, config, scaler):
        self.config = config
        self.scaler = scaler
        self.feature_cols = config.get('features', [])
        self.model_type = config.get('model_type', 'Unknown')
        # Default fallback
        if not self.feature_cols:
             self.feature_cols = ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']

    def preprocess(self, df):
        """
        Takes raw dataframe with 'Close' (or 'close') and returns:
        - X input for model
        - Current price (for ref)
        """
        # copy to avoid modifying original
        data = df.copy()
        
        # Normalize columns
        if 'Close' not in data.columns and 'close' in data.columns:
            data = data.rename(columns={'close': 'Close'})
            
        if 'Close' not in data.columns:
            raise ValueError("Data must have a 'Close' column")
            
        # Calculate Returns
        data['Return'] = data['Close'].pct_change()
        
        # Calculate features dynamically based on what's in config keys
        # We assume standard naming convention: MA_X, Lag_X
        # But for now, let's just compute the standard set and select what's needed.
        # In a robust system, we would parse the feature names.

        # Calculate features dynamically based on what's in self.feature_cols
        # For simplicity, we assume the standard set is available
        
        # Moving Averages
        for ma in [5, 10, 20, 50]:
            data[f'MA_{ma}'] = data['Close'].rolling(window=ma).mean()
            
        # Shifted Returns (Lags)
        for lag in [1, 2, 3, 5]:
            data[f'Lag_{lag}'] = data['Return'].shift(lag)

        # Longer Horizon Returns (Momentum)
        for r in [5, 20]:
            data[f'Return_{r}d'] = data['Close'].pct_change(periods=r)
            
        # Ensure OHLC are available (renaming if lower case)
        for col in ['Open', 'High', 'Low']:
            if col not in data.columns and col.lower() in data.columns:
                 data = data.rename(columns={col.lower(): col})
            
        # Drop NaNs
        data = data.dropna()
        
        # Create unscaled return for specific model types (like ARIMA)
        data['Return_Unscaled'] = data['Return']
        
        if len(data) == 0:
            return None, None
            
        # Select features
        # Filter feature_cols to only those present in data
        available_features = [c for c in self.feature_cols if c in data.columns]
        if not available_features:
            # Fallback if config features not found
            available_features = ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
            
        # Get the LAST row for prediction
        last_row = data.iloc[[-1]][available_features]
        current_price = data.iloc[-1]['Close']
        
        # Scale - keep as DataFrame for MLflow schema validation
        if self.scaler:
             try:
                 X_scaled = self.scaler.transform(last_row)
                 # Convert back to DataFrame with column names for MLflow
                 X = pd.DataFrame(X_scaled, columns=available_features, index=last_row.index)
             except Exception as e:
                 print(f"Scaling failed: {e}. Using raw values.")
                 X = last_row
        else:
             X = last_row
             
        # Reshape based on model type
        # LSTM expects (samples, time_steps, features) as numpy array
        # Linear Regression and others expect DataFrame with named columns
        if 'lstm' in self.model_type.lower():
            time_steps = self.config.get('time_steps', 1)
            # For LSTM, convert to numpy and reshape
            X = X.values.reshape((1, time_steps, X.shape[1]))
            
        return X, current_price

def load_data_file(file_path):
    """Load data from a specific file"""
    global df_data
    try:
        # Load with skiprows to handle metadata
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, skiprows=[1, 2])
        
        # Ensure 'Close' column exists and is numeric
        if 'Close' in df.columns:
            # Convert to numeric, coercing errors to NaN
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            # Drop rows with NaN in Close
            df = df.dropna(subset=['Close'])
        elif 'close' in df.columns:
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            df = df.rename(columns={'close': 'Close'})
        else:
            print(f"No 'Close' column in {file_path}")
            return False
        
        # Ensure we have enough history for lags (at least 10 rows)
        if len(df) > 10:
            df_data = df
            return True
        return False
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

# Initialize
print("Starting App...")
load_success = load_from_mlflow()

# Try loading some data for default view (S3 or local)
if USE_S3:
    # Use the data loader to fetch and merge partitioned data
    try:
        from api.src.data_loader import load_data_from_s3
        print(f"Loading data from S3 bucket: {S3_BUCKET}...")
        # Pass skiprows to handle the specific yfinance multi-header format
        df_data = load_data_from_s3(S3_BUCKET, prefix=S3_PREFIX if S3_PREFIX else "data/raw/", skiprows=[1, 2])
        if not df_data.empty:
            print(f"Successfully loaded {len(df_data)} rows from S3")
        else:
            print("S3 load returned empty DataFrame.")
    except Exception as e:
        print(f"Failed to load data from S3: {e}")
        import traceback
        traceback.print_exc()

# Try local files if data not loaded yet
if df_data is None or df_data.empty:
    local_data_paths = [
        'data/raw/eurusd_raw.csv',
        'data/processed/test.csv',
        'data/processed/train.csv'
    ]
    for path in local_data_paths:
        if os.path.exists(path):
            if load_data_file(path):
                print(f"Loaded data from local file: {path}")
                break

fe = None
if load_success:
    fe = FeatureEngineer(feature_config, scaler)

@app.route('/')
def home():
    if not load_success:
        return render_template('index.html', success=False, error="Model not loaded")
    
    if df_data is None:
         return render_template('index.html', success=False, error="Data not loaded")

    # Predict
    try:
        X, current_price = fe.preprocess(df_data)
        if X is None:
             return render_template('index.html', success=False, error="Not enough data to generate features")
             
        # Prediction - handle ARIMA specially
        model_type = feature_config.get('model_type', 'Unknown')
        
        if 'arima' in model_type.lower():
            # ARIMA models use predict(n_periods=1) interface
            pred_return = model.predict(n_periods=1)[0]
        else:
            # Standard prediction for Linear Regression and LSTM
            pred_return = model.predict(X)
        
        # Handle varying output shapes (numpy, pandas, etc.)
        if hasattr(pred_return, 'values'):
            pred_return = pred_return.values
        if isinstance(pred_return, (list, np.ndarray)) and len(np.shape(pred_return)) > 0:
            pred_return = pred_return.flatten()[0]
        else:
            pred_return = float(pred_return)
            
        predicted_price = current_price * (1 + pred_return)
        change_pct = pred_return * 100
        
        # Model description for UI
        model_type = feature_config.get('model_type', 'Unknown')
        if 'lstm' in model_type.lower():
            model_desc = "LSTM (Deep Learning)"
            method = f"using the last {feature_config.get('time_steps', 1)} days and {len(feature_config.get('features', []))} features"
        elif 'arima' in model_type.lower():
            model_desc = "ARIMA (Statistical)"
            method = "using historical return patterns"
        else:
            model_desc = "Linear Regression"
            method = f"using {len(feature_config.get('features', []))} technical indicators"

        stats = {
            'total_days': len(df_data),
            'current': current_price,
            'min': df_data['Close'].min(),
            'max': df_data['Close'].max(),
            'avg': df_data['Close'].mean(),
            'last_date': df_data.index[-1].strftime('%Y-%m-%d'),
            'date_range': f"{df_data.index[0].strftime('%Y-%m-%d')} to {df_data.index[-1].strftime('%Y-%m-%d')}"
        }

        context = {
            'success': True,
            'predicted_price': round(predicted_price, 4),
            'current_price': round(current_price, 4),
            'change_pct': round(change_pct, 4),
            'model_type': model_desc,
            'model_method': method,
            'stats': stats,
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return render_template('index.html', **context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', success=False, error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    if not load_success:
        return jsonify({'error': 'Model not loaded'}), 500
        
    # Get data from request or global
    data = None
    if request.method == 'POST':
        json_data = request.get_json()
        if json_data:
            data = pd.DataFrame(json_data)
            
    if data is None:
        data = df_data
        
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
        
    X, current_price = fe.preprocess(data)
    
    # Prediction - handle ARIMA specially
    model_type = feature_config.get('model_type', 'Unknown')
    
    if 'arima' in model_type.lower():
        # ARIMA models use predict(n_periods=1) interface
        pred_return = model.predict(n_periods=1)[0]
    else:
        # Standard prediction for Linear Regression and LSTM
        pred_return = model.predict(X)
        
    if hasattr(pred_return, 'values'):
         pred_return = pred_return.values
    if isinstance(pred_return, (list, np.ndarray)) and len(np.shape(pred_return)) > 0:
        pred_return = pred_return.flatten()[0]
        
    predicted_price = float(current_price * (1 + pred_return))
    
    return jsonify({
        'predicted_price': predicted_price,
        'predicted_return': float(pred_return),
        'current_price': float(current_price),
        'model_type': feature_config.get('model_type', 'Unknown')
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy' if load_success else 'unhealthy'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
