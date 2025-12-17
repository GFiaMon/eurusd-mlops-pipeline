import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# Constants
RAW_DATA_PATH = os.path.join("data", "raw", "eurusd_raw.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")

def preprocess_data():
    """
    Loads raw data, performs feature engineering, splits, scales, and saves processed data.
    """
    print("Loading raw data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found. Run ingest_data.py first.")
        return

    # Load data, skipping the 2nd and 3rd lines which contain Ticker info and blank Date line
    df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True, skiprows=[1, 2])
    
    # Verify we have numeric data
    try:
        df['Close'] = df['Close'].astype(float)
    except ValueError:
        # Fallback if the skipping schema didn't match perfectly, try reading all and cleaning
        print("Warning: Simple skip failed, attempting robust cleaning...")
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        # Drop rows that contain metadata (strings)
        df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
        df = df.astype(float)
    
    # Feature Engineering
    print("Feature Engineering...")
    # Daily Returns (Percentage)
    df['Return'] = df['Close'].pct_change()
    
    # 5-day SMA
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    
    # Lagged Returns
    for lag in [1, 2, 3, 5]:
        df[f'Lag_{lag}'] = df['Return'].shift(lag)
        
    # Drop NaNs created by rolling and shifting and pct_change
    df = df.dropna()
    
    # Define features and target
    # We are predicting Tomorrow's Return, so we need to shift the target or features.
    # Standard approach: Predict Return_t using information available at t-1.
    # But description says "Forecast Tomorrow's EUR/USD Daily Return". 
    # So if we are at row `t` (today), we want to predict `Return_{t+1}`.
    # Let's create a 'Target' column which is 'Return' shifted by -1.
    
    df['Target'] = df['Return'].shift(-1)
    df = df.dropna() # Drop the last row which has no target
    
    # Split Chronologically (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Create a copy of Return for ARIMA (unscaled)
    train_df['Return_Unscaled'] = train_df['Return']
    test_df['Return_Unscaled'] = test_df['Return']
    
    # Scaling
    print("Scaling data...")
    feature_cols = ['Return', 'MA_5', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    # Note: We scale features. Target usually doesn't need scaling for Regression/ARIMA but might helpful for LSTM. 
    # However task says "Forecast Daily Returns", and we convert back to price later. 
    # Let's scale features only for now as is typical for inputs. 
    # Actually, for LSTM it's often good to scale everything. 
    # Let's scale features. Target is a percentage, already small scale, so maybe fine. 
    # Requirement: "Fit a MinMaxScaler on Train data only; transform both Train and Test."
    
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv(TEST_DATA_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    preprocess_data()
