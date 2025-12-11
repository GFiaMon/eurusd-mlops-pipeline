# api/app.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, request, render_template, jsonify

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Try to import TensorFlow
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

# Global variables
model = None
scaler = None
SEQ_LENGTH = 100

def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler
    
    # Paths to model files
    model_path = os.path.join(project_root, 'models', 'lstm_trained_model.keras')
    scaler_path = os.path.join(project_root, 'models', 'lstm_scaler.joblib')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    # Load the model
    if TF_AVAILABLE:
        model = load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    else:
        print("❌ TensorFlow not available")
        model = None
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded from: {scaler_path}")
    
    return True

def predict_next_price(recent_prices):
    """Predict next day's EUR/USD price"""
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded")
    
    # Convert to numpy array
    prices_array = np.array(recent_prices).reshape(-1, 1)
    
    # Scale
    scaled_prices = scaler.transform(prices_array)
    
    # Reshape for LSTM
    sequence = scaled_prices.reshape(1, SEQ_LENGTH, 1)
    
    # Predict
    scaled_prediction = model.predict(sequence, verbose=0)
    
    # Convert back to actual price
    prediction = scaler.inverse_transform(scaled_prediction)[0][0]
    
    return float(prediction)

# Load model when app starts
print("🚀 Starting EUR/USD Prediction Web App...")
try:
    load_model_and_scaler()
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    """Render the main page"""
    # Get some sample data for display
    sample_prices = []
    if scaler is not None:
        # Load recent data to show
        try:
            data_path = os.path.join(project_root, 'data', 'processed', 'lstm_simple_train_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                if len(df) >= 5:
                    # Get last 5 prices
                    sample_prices = df['close'].tail(5).round(4).tolist()
        except:
            pass
    
    return render_template('index.html', 
                          model_loaded=model is not None,
                          sample_prices=sample_prices,
                          seq_length=SEQ_LENGTH)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction form submission"""
    try:
        # Get the recent prices from form
        recent_prices_str = request.form.get('recent_prices', '')
        
        if not recent_prices_str:
            return render_template('index.html',
                                 model_loaded=model is not None,
                                 error="Please enter recent prices",
                                 recent_prices_input=recent_prices_str)
        
        # Convert string to list of floats
        recent_prices = []
        for price_str in recent_prices_str.split(','):
            try:
                price = float(price_str.strip())
                recent_prices.append(price)
            except ValueError:
                return render_template('index.html',
                                     model_loaded=model is not None,
                                     error=f"Invalid price value: {price_str}",
                                     recent_prices_input=recent_prices_str)
        
        # Check we have exactly SEQ_LENGTH prices
        if len(recent_prices) != SEQ_LENGTH:
            return render_template('index.html',
                                 model_loaded=model is not None,
                                 error=f"Need exactly {SEQ_LENGTH} prices, got {len(recent_prices)}",
                                 recent_prices_input=recent_prices_str)
        
        # Make prediction
        predicted_price = predict_next_price(recent_prices)
        current_price = recent_prices[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Prepare result
        result = {
            'predicted_price': round(predicted_price, 4),
            'current_price': round(current_price, 4),
            'change_pct': round(change_pct, 2),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('index.html',
                             model_loaded=model is not None,
                             result=result,
                             recent_prices_input=recent_prices_str)
    
    except Exception as e:
        return render_template('index.html',
                             model_loaded=model is not None,
                             error=f"Prediction error: {str(e)}")

@app.route('/predict/latest', methods=['GET'])
def predict_latest():
    """Predict using latest data from file"""
    try:
        if model is None or scaler is None:
            return render_template('index.html',
                                 model_loaded=False,
                                 error="Model not loaded")
        
        # Find the latest data file
        data_files = [
            os.path.join(project_root, 'data', 'processed', 'lstm_simple_test_data.csv'),
            os.path.join(project_root, 'data', 'processed', 'lstm_simple_train_data.csv')
        ]
        
        data_path = None
        for file_path in data_files:
            if os.path.exists(file_path):
                data_path = file_path
                break
        
        if data_path is None:
            return render_template('index.html',
                                 model_loaded=True,
                                 error="No data file found")
        
        # Load data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        if len(df) < SEQ_LENGTH:
            return render_template('index.html',
                                 model_loaded=True,
                                 error=f"Not enough data (need {SEQ_LENGTH}, have {len(df)})")
        
        # Get latest prices
        latest_prices = df['close'].tail(SEQ_LENGTH).tolist()
        latest_prices_str = ', '.join([str(round(p, 4)) for p in latest_prices])
        
        # Make prediction
        predicted_price = predict_next_price(latest_prices)
        current_price = latest_prices[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        result = {
            'predicted_price': round(predicted_price, 4),
            'current_price': round(current_price, 4),
            'change_pct': round(change_pct, 2),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': os.path.basename(data_path),
            'used_latest_data': True
        }
        
        return render_template('index.html',
                             model_loaded=True,
                             result=result,
                             recent_prices_input=latest_prices_str)
    
    except Exception as e:
        return render_template('index.html',
                             model_loaded=True,
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    print(f"🌐 Web app running on http://localhost:5001")
    print("📊 Open your browser and go to the above URL")
    app.run(host='0.0.0.0', port=5001, debug=True)