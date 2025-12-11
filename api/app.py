# api/app.py - SIMPLE WORKING VERSION
import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Try to import TensorFlow
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

# Global variables
model = None
scaler = None
SEQ_LENGTH = 100
df_data = None

def load_model_and_data():
    """Load model, scaler, and data"""
    global model, scaler, df_data
    
    print("Loading model and data...")
    
    # Load model
    model_path = os.path.join(project_root, 'models', 'lstm_trained_model.keras')
    scaler_path = os.path.join(project_root, 'models', 'lstm_scaler.joblib')
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"Scaler not found: {scaler_path}")
        return False
    
    if TF_AVAILABLE:
        model = load_model(model_path)
        print("Model loaded")
    else:
        print("TensorFlow not available")
        return False
    
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")
    
    # Load data - try multiple possible files
    data_files = [
        os.path.join(project_root, 'data', 'processed', 'lstm_simple_test_data.csv'),
        os.path.join(project_root, 'data', 'processed', 'lstm_simple_train_data.csv'),
        os.path.join(project_root, 'data', 'raw', 'eurusd_raw.csv')
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Find price column
                if 'close' in df.columns:
                    price_col = 'close'
                elif 'Close' in df.columns:
                    price_col = 'Close'
                    df = df.rename(columns={'Close': 'close'})
                else:
                    print(f"No 'close' column in {file_path}")
                    continue
                
                df_data = df[['close']].copy()
                df_data = df_data.dropna()
                
                if len(df_data) >= SEQ_LENGTH:
                    print(f"Data loaded: {len(df_data)} rows from {os.path.basename(file_path)}")
                    return True
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    print("Could not load sufficient data")
    return False

def make_prediction():
    """Make prediction using loaded data"""
    if model is None or scaler is None or df_data is None:
        return None, None, None
    
    if len(df_data) < SEQ_LENGTH:
        return None, None, None
    
    # Get recent prices
    recent_prices = df_data['close'].tail(SEQ_LENGTH).values
    
    # Prepare for LSTM
    prices_array = np.array(recent_prices).reshape(-1, 1)
    scaled_prices = scaler.transform(prices_array)
    sequence = scaled_prices.reshape(1, SEQ_LENGTH, 1)
    
    # Predict
    scaled_prediction = model.predict(sequence, verbose=0)
    prediction = scaler.inverse_transform(scaled_prediction)[0][0]
    
    current_price = recent_prices[-1]
    change_pct = ((prediction - current_price) / current_price) * 100
    
    return float(prediction), float(current_price), float(change_pct)

def get_stats():
    """Get basic statistics"""
    if df_data is None:
        return None
    
    return {
        'total_days': len(df_data),
        'date_range': f"{df_data.index[0].strftime('%Y-%m-%d')} to {df_data.index[-1].strftime('%Y-%m-%d')}",
        'current': float(df_data['close'].iloc[-1]),
        'min': float(df_data['close'].min()),
        'max': float(df_data['close'].max()),
        'avg': float(df_data['close'].mean()),
        'last_date': df_data.index[-1].strftime('%Y-%m-%d')
    }

# Load everything at startup
print("Starting EUR/USD Prediction App...")
load_success = load_model_and_data()

@app.route('/')
def home():
    """Main page"""
    if not load_success:
        return render_template('index.html',
                             success=False,
                             error="Failed to load model or data")
    
    # Get prediction
    predicted_price, current_price, change_pct = make_prediction()
    
    if predicted_price is None:
        return render_template('index.html',
                             success=False,
                             error="Could not make prediction")
    
    # Get stats
    stats = get_stats()
    
    # Prepare context
    context = {
        'success': True,
        'predicted_price': round(predicted_price, 4),
        'current_price': round(current_price, 4),
        'change_pct': round(change_pct, 2),
        'change_up': change_pct > 0,
        'change_down': change_pct < 0,
        'stats': stats,
        'seq_length': SEQ_LENGTH,
        'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('index.html', **context)

@app.route('/api/predict')
def api_predict():
    """API endpoint"""
    if not load_success:
        return jsonify({'error': 'Model not loaded'}), 500
    
    predicted_price, current_price, change_pct = make_prediction()
    
    if predicted_price is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({
        'predicted_price': predicted_price,
        'current_price': current_price,
        'change_percent': change_pct,
        'prediction_for': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def api_stats():
    """Stats endpoint"""
    if not load_success:
        return jsonify({'error': 'Data not loaded'}), 500
    
    stats = get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    print(f"App running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)