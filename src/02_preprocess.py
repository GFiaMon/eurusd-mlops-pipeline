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
    
    # Moving Averages
    for ma in [5, 10, 20, 50]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
        
    # Lagged Returns (1 day is 'Return')
    # Returns over longer periods (Momentum)
    for r in [5, 20]:
        df[f'Return_{r}d'] = df['Close'].pct_change(periods=r)

    # Lagged Features for 1-step prediction (Standard Regression features)
    # We still keep these for the granular regression model
    for lag in [1, 2, 3, 5]:
        df[f'Lag_{lag}'] = df['Return'].shift(lag)
        
    # Drop NaNs created by rolling and shifting (max rolling is 50)
    df = df.dropna()
    
    # Define features and target
    # Target: Return_{t+1}
    df['Target'] = df['Return'].shift(-1)
    df = df.dropna()
    
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
    # Update feature list to include OHLC and new features
    # Note: OHLC are in 'df' if read correctly. 
    # Let's verify we have them. If not, we might need to handle it.
    # Assuming 'Open', 'High', 'Low' are available from yfinance download.
    
    essential_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [c for c in essential_cols if c not in df.columns]
    if missing_cols:
         print(f"Warning: Columns {missing_cols} missing. Using available.")
         
    # Construct feature list
    feature_cols = ['Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Return_5d', 'Return_20d', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    
    # Add OHLC to features if present (scaled)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            feature_cols.append(col)
            
    print(f"Features to scale: {feature_cols}")

    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv(TEST_DATA_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")

    # Upload to S3
    s3_bucket = os.getenv("S3_BUCKET")
    if s3_bucket:
        try:
            import boto3
            s3 = boto3.client('s3')
            
            artifacts = [TRAIN_DATA_PATH, TEST_DATA_PATH, SCALER_PATH]
            
            print(f"Uploading artifacts to s3://{s3_bucket}...")
            for artifact in artifacts:
                s3_key = artifact # Keep same path structure: data/processed/...
                s3.upload_file(artifact, s3_bucket, s3_key)
                print(f"  ✅ Uploaded {artifact}")
                
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    else:
        print("  ⚠️ S3_BUCKET not set in .env, skipping upload.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    preprocess_data()
