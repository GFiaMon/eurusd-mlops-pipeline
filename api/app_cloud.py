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
        # Check if we have enough data
        if len(df_data) < lookback + 10:
             return f"Not enough data history. Has {len(df_data)} rows.", 400

        # Get FULL history features (now returns full DF)
        X_full, y_full_returns, prices_full, dates_full = fe.preprocess_history(df_data, lookback=lookback)
        
        # Define the target range (last 'lookback' days)
        # Note: X_full aligns with y_full_returns and prices_full
        total_len = len(X_full)
        start_idx = total_len - lookback
        
        # Target slices for metrics
        y_actual_returns = y_full_returns.iloc[-lookback:]
        prices = prices_full.iloc[-lookback:]
        dates = dates_full[-lookback:]
        
        m_type = feature_config.get('model_type', 'Unknown')
        y_pred = []

        if 'lstm' in m_type.lower():
            time_steps = feature_config.get('time_steps', 60)
            # Sliding window prediction on the last 'lookback' days
            # We need at least 'time_steps' of history to make a prediction
            preds = []
            
            # For each day in the target window, construct a sequence and predict
            for idx in range(lookback):
                # Absolute position in X_full
                abs_pos = start_idx + idx
                
                # Check if we have enough history (need 'time_steps' rows before abs_pos)
                if abs_pos < time_steps:
                    preds.append(0.0)  # Not enough history
                else:
                    # Window: [abs_pos - time_steps : abs_pos]
                    # This gives us the 60 days BEFORE the target day
                    window = X_full.iloc[abs_pos - time_steps:abs_pos].values
                    window = window.reshape((1, time_steps, window.shape[1]))
                    try:
                        p = model.predict(window)  # No verbose parameter for pyfunc models
                        if isinstance(p, (list, np.ndarray)): 
                            p = p.flatten()[0]
                        preds.append(float(p))
                    except Exception as e:
                        print(f"LSTM prediction error at idx {idx}: {e}")
                        preds.append(0.0)
            
            y_pred = np.array(preds)

        elif 'arima' in m_type.lower():
            # ARIMA: Try predict_in_sample for the specific historical window
            try:
                n_train = model.nobs_
                # We want the LAST 'lookback' points from the training history
                # start index (inclusive)
                start_p = max(0, n_train - lookback)
                # end index (inclusive)
                end_p = n_train - 1
                
                preds = model.predict_in_sample(start=start_p, end=end_p)
                
                # Ensure preds is flat numpy array
                if hasattr(preds, 'values'): preds = preds.values
                preds = np.array(preds).flatten()
                
                # Align lengths
                if len(preds) < lookback:
                    preds = np.concatenate([np.zeros(lookback-len(preds)), preds])
                elif len(preds) > lookback:
                    preds = preds[-lookback:]
                y_pred = preds
                
            except Exception as e:
                print(f"ARIMA History Error: {e}")
                y_pred = np.zeros(lookback)

        else:
            # Linear / standard sklearn
            # slicing X for the target range
            X_target = X_full.iloc[start_idx:]
            try:
                y_pred = model.predict(X_target)
                if hasattr(y_pred, 'values'): y_pred = y_pred.values
                y_pred = np.array(y_pred).flatten()
            except:
                y_pred = np.zeros(len(X_target))
        
        # Final safety normalization
        if len(y_pred) != lookback:
             # Force length match if strictly needed (though y_actual is also defined by lookback)
             if len(y_pred) > lookback: y_pred = y_pred[-lookback:]
             else: y_pred = np.concatenate([np.zeros(lookback-len(y_pred)), y_pred])
             
        # Double check against actuals length (should be same as lookback)
        if len(y_pred) != len(y_actual_returns):
            min_l = min(len(y_pred), len(y_actual_returns))
            y_pred = y_pred[-min_l:] # Take prediction from end
            y_actual_returns = y_actual_returns.iloc[-min_l:]
            prices = prices.iloc[-min_l:]
            dates = dates[-min_l:]
        
        # Formatting metrics
        if len(y_pred) != len(y_actual_returns):
            # Safety checks for shape mismatch
            min_l = min(len(y_pred), len(y_actual_returns))
            y_pred = y_pred[:min_l]
            y_actual_returns = y_actual_returns.iloc[:min_l]
            prices = prices.iloc[:min_l]
            dates = dates[:min_l]

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
