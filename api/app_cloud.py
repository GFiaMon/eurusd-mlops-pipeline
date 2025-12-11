# api/app_cloud.py - Enhanced version with S3 support
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
SEQ_LENGTH = 100
df_data = None

# Configuration
USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
S3_MODEL_KEY = os.getenv('S3_MODEL_KEY', 'models/lstm_trained_model.keras')
S3_SCALER_KEY = os.getenv('S3_SCALER_KEY', 'models/lstm_scaler.joblib')
S3_DATA_KEY = os.getenv('S3_DATA_KEY', 'data/processed/lstm_simple_test_data.csv')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

def download_from_s3(bucket, key, local_path):
    """Download file from S3 to local path"""
    if not S3_AVAILABLE:
        print("boto3 not available for S3 operations")
        return False
    
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        print(f"Successfully downloaded {key}")
        return True
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading from S3: {e}")
        return False

def load_model_and_data():
    """Load model, scaler, and data from local storage or S3"""
    global model, scaler, df_data
    
    print("Loading model and data...")
    print(f"USE_S3: {USE_S3}")
    
    # Define local paths
    model_path = os.path.join(project_root, 'models', 'lstm_trained_model.keras')
    scaler_path = os.path.join(project_root, 'models', 'lstm_scaler.joblib')
    
    # If using S3, download files first
    if USE_S3:
        print(f"Downloading from S3 bucket: {S3_BUCKET}")
        
        if not download_from_s3(S3_BUCKET, S3_MODEL_KEY, model_path):
            print("Failed to download model from S3")
            return False
        
        if not download_from_s3(S3_BUCKET, S3_SCALER_KEY, scaler_path):
            print("Failed to download scaler from S3")
            return False
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"Scaler not found: {scaler_path}")
        return False
    
    # Load model
    if TF_AVAILABLE:
        model = load_model(model_path)
        print("Model loaded")
    else:
        print("TensorFlow not available")
        return False
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")
    
    # Load data
    if USE_S3:
        # Download data from S3
        data_path = os.path.join(project_root, 'data', 'processed', 'lstm_simple_test_data.csv')
        if not download_from_s3(S3_BUCKET, S3_DATA_KEY, data_path):
            print("Failed to download data from S3, trying local files")
            return load_local_data()
        
        return load_data_file(data_path)
    else:
        return load_local_data()

def load_local_data():
    """Load data from local files"""
    global df_data
    
    # Try multiple possible files
    data_files = [
        os.path.join(project_root, 'data', 'processed', 'lstm_simple_test_data.csv'),
        os.path.join(project_root, 'data', 'processed', 'lstm_simple_train_data.csv'),
        os.path.join(project_root, 'data', 'raw', 'eurusd_raw.csv')
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            if load_data_file(file_path):
                return True
    
    print("Could not load sufficient data from local files")
    return False

def load_data_file(file_path):
    """Load data from a specific file"""
    global df_data
    
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
            return False
        
        df_data = df[['close']].copy()
        df_data = df_data.dropna()
        
        if len(df_data) >= SEQ_LENGTH:
            print(f"Data loaded: {len(df_data)} rows from {os.path.basename(file_path)}")
            return True
        else:
            print(f"Insufficient data in {file_path}: {len(df_data)} rows")
            return False
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
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
        'last_date': df_data.index[-1].strftime('%Y-%m-%d'),
        'storage_type': 'S3' if USE_S3 else 'Local'
    }

# Load everything at startup
print("Starting EUR/USD Prediction App...")
print(f"Storage mode: {'S3' if USE_S3 else 'Local'}")
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
        'timestamp': datetime.now().isoformat(),
        'storage_type': 'S3' if USE_S3 else 'Local'
    })

@app.route('/api/stats')
def api_stats():
    """Stats endpoint"""
    if not load_success:
        return jsonify({'error': 'Data not loaded'}), 500
    
    stats = get_stats()
    return jsonify(stats)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if load_success else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'storage_type': 'S3' if USE_S3 else 'Local'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"App running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
