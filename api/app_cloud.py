# api/app_cloud.py
import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

# Import components
try:
    from api.src.feature_engineer import FeatureEngineer
    from api.src.model_loader import load_model_from_mlflow
except ImportError:
    if os.path.join(project_root, 'api') not in sys.path: sys.path.append(os.path.join(project_root, 'api'))
    from src.feature_engineer import FeatureEngineer
    from src.model_loader import load_model_from_mlflow

try:
    from utils.data_manager import DataManager
except ImportError:
    DataManager = None

app = Flask(__name__, template_folder='frontend')

# Globals
current_model_name = "best"
model = None
scaler = None
feature_config = None
df_data = None
load_success = False
fe = None
last_load_time = None

# Config
USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', 'eurusd-ml-models')
S3_PREFIX = os.getenv('S3_PREFIX', 'data/raw/')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

def initialize_model(model_name="best"):
    global model, scaler, feature_config, load_success, fe, current_model_name
    print(f"--- LOADING MODEL: {model_name} ---")
    
    m, s, c, success = load_model_from_mlflow(
        project_root, 
        use_s3=USE_S3, 
        s3_bucket=S3_BUCKET, 
        s3_key=os.getenv('S3_MODEL_INFO_KEY', 'models/best_model_info.json'),
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name_override=model_name
    )
    
    if success:
        model, scaler, feature_config, load_success = m, s, c, True
        fe = FeatureEngineer(feature_config, scaler)
        current_model_name = model_name
        print(f"SUCCESS: {model_name} is now active.")
    else:
        print(f"ERROR: Could not load {model_name}. System state preserved.")
        # If this was the initial load, we are still in "failure" state
        # but if it was a switch, we keep the old model.

def load_data(force=False):
    global df_data, last_load_time
    if DataManager:
        try:
            # Check TTL - Daily update (UTC)
            is_stale = False
            if last_load_time:
                # Reload if current UTC day is different from last load UTC day
                if datetime.utcnow().date() > last_load_time.date():
                    is_stale = True
                    print("New UTC day detected. Reloading data...")

            if df_data is not None and not force and not is_stale:
                return

            dm = DataManager(mode='cloud' if USE_S3 else 'auto', s3_bucket=S3_BUCKET, s3_prefix=S3_PREFIX)
            new_data = dm.get_latest_data()
            
            if new_data is not None and not new_data.empty:
                df_data = new_data
                # Store time in UTC to match the check above
                last_load_time = datetime.utcnow()
                print(f"Data Loaded: {len(df_data)} rows. Last date: {df_data.index[-1]}")
            else:
                 print("Warning: Reload attempt returned no data. Keeping old data.")
                 
        except Exception as e: print(f"Data Load Error: {e}")

# Init
initialize_model()
load_data()

@app.route('/')
def home():
    # Ensure data is fresh
    load_data()
    
    req_model = request.args.get('model')
    if req_model and req_model != current_model_name:
        initialize_model(req_model)
    
    if not load_success:
        return render_template('index.html', success=False, error="Model load failed. Check tracker connection.")
    if df_data is None:
         load_data()
         if df_data is None: return render_template('index.html', success=False, error="Historical data not found.")

    try:
        X, current_price = fe.preprocess(df_data)
        if X is None: return render_template('index.html', success=False, error="Insufficient history (need 60+ days for LSTM).")
             
        # Predict
        m_type = feature_config.get('model_type', 'Unknown')
        if 'arima' in m_type.lower():
            # Robustly handle prediction output (Series vs Array)
            p = model.predict(n_periods=1)
            if hasattr(p, 'iloc'):
                pred_val = p.iloc[0]
            elif isinstance(p, (list, np.ndarray)) and len(p) > 0:
                pred_val = p[0]
            else:
                pred_val = p # Fallback for scalar
        else:
            pred_val = model.predict(X)
        
        # Parse output
        if hasattr(pred_val, 'values'): pred_val = pred_val.values
        if isinstance(pred_val, (list, np.ndarray)) and len(np.shape(pred_val)) > 0:
            pred_val = pred_val.flatten()[0]
        
        pred_return = float(pred_val)
        predicted_price = current_price * (1 + pred_return)
        change_pct = pred_return * 100
        
        context = {
            'success': True,
            'predicted_price': round(predicted_price, 4),
            'current_price': round(current_price, 4),
            'change_pct': round(change_pct, 4),
            'pred_return': round(pred_return, 6),
            'model_type': m_type,
            'selected_model': current_model_name,
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stats': {
                'current': current_price,
                'total_days': len(df_data),
                'last_date': df_data.index[-1].strftime('%Y-%m-%d'),
                'min': df_data['Close'].min(),
                'max': df_data['Close'].max()
            },
            'data_last_date': df_data.index[-1].strftime('%Y-%m-%d') # For footer
        }
        return render_template('index.html', **context)
    except Exception as e:
        return render_template('index.html', success=False, error=f"Prediction Error: {str(e)}")

@app.route('/history')
def history():
    # Model switching logic
    req_model = request.args.get('model')
    if req_model and req_model != current_model_name:
        initialize_model(req_model)

    # Ensure data is fresh
    load_data()
    
    if not load_success or df_data is None: return render_template('index.html', error="Not ready.")
    try:
        lookback = 60
        # Check if we have enough data for lookback + window
        if len(df_data) < lookback + 10:
             return f"Not enough data history. Has {len(df_data)} rows.", 400

        X_hist, y_actual_returns, prices, dates = fe.preprocess_history(df_data, lookback=lookback)
        
        m_type = feature_config.get('model_type', 'Unknown')
        
        # Performance comparison: Linear Regression can do batch 2D prediction. 
        # LSTM/ARIMA require sliding windows or stateful steps, which we skip for the history graph for now.
        if 'linear' in m_type.lower():
            try:
                y_pred = model.predict(X_hist)
                y_pred = np.array(y_pred).flatten()
            except:
                y_pred = np.zeros(len(X_hist))
        else:
            # For LSTM/ARIMA, show zeros on the graph to avoid shape errors
            y_pred = np.zeros(len(X_hist))
        
        mae = np.mean(np.abs(y_actual_returns.values - y_pred))
        mae_price = np.mean(np.abs(prices.values - (prices.shift(1)*(1+y_pred)).fillna(prices)))
        mae_pips = mae_price * 10000
        # Accuracy is random (50%) if y_pred is all zeros, which reflects 'no prediction'
        acc = np.mean(np.sign(y_actual_returns.values) == np.sign(y_pred))
        
        plot_json = {
            'data': [
                {'x': [d.strftime('%Y-%m-%d') for d in dates], 'y': list(prices.values), 'name': 'Actual', 'line': {'color': '#2ecc71'}},
                {'x': [d.strftime('%Y-%m-%d') for d in dates], 'y': list((prices.shift(1)*(1+y_pred)).fillna(prices).values), 'name': 'Predicted', 'line': {'color': '#e74c3c', 'dash': 'dot'}}
            ],
            'layout': {'title': f'Backtest Analysis ({m_type})', 'autosize': True}
        }
        
        return render_template('history.html', 
                             metrics={'mae': mae, 'mae_price': mae_price, 'mae_pips': mae_pips, 'accuracy': acc}, 
                             plot_json=json.dumps(plot_json), 
                             last_updated=df_data.index[-1].strftime('%Y-%m-%d'), # Use data date
                             lookback=lookback, 
                             model_type=m_type,
                             current_model=current_model_name) # Pass current model name
    except Exception as e: return f"History Error: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)), debug=True)
