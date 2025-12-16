import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath="data/raw/eurusd_data.csv"):
    """Loads raw data from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def create_features(df):
    """
    Creates basic technical features:
    - Daily Returns
    - Lagged Prices (1, 2, 3, 5 days)
    - Moving Averages (SMA 7, SMA 30)
    """
    df = df.copy()
    
    # Ensure Date is datetime and sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df.sort_index(inplace=True)

    # Ensure Close is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    
    # 1. Target Variables (What we want to predict)
    # We want to predict TOMORROW's Price and TOMORROW's Return
    # So we shift data BACKWARDS by 1 to align today's features with tomorrow's value
    df['Target_Close'] = df['Close'].shift(-1)
    
    # 2. Daily Returns (Percentage change)
    df['Return'] = df['Close'].pct_change()
    df['Target_Return'] = df['Return'].shift(-1)
    
    # 3. Lagged Features (Past history)
    # Price lags
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
        
    # 4. Moving Averages (Trend)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 5. Drop NaNs created by lagging/rolling and shifting
    # The last row will have NaN targets (because we don't know tomorrow yet), 
    # and first rows will have NaN features.
    
    # Save the last row separately for "Live Inference" (predicting tomorrow)
    df_latest = df.iloc[[-1]].copy()
    
    # Drop NaNs for training data
    df_processed = df.dropna()
    
    logging.info(f"Features created. Data shape: {df_processed.shape}")
    return df_processed, df_latest

def split_data(df, train_ratio=0.8):
    """Splits data into Train and Test sets chronologically."""
    train_size = int(len(df) * train_ratio)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    return train, test

def main():
    INPUT_PATH = "data/raw/eurusd_data.csv"
    PROCESSED_DIR = "data/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    try:
        logging.info("Loading data...")
        df = load_data(INPUT_PATH)
        
        logging.info("Engineering features...")
        df_processed, df_latest = create_features(df)
        
        logging.info("Splitting data...")
        train, test = split_data(df_processed)
        
        # Save files
        train.to_csv(f"{PROCESSED_DIR}/train.csv")
        test.to_csv(f"{PROCESSED_DIR}/test.csv")
        df_latest.to_csv(f"{PROCESSED_DIR}/latest.csv") # For predicting tomorrow really
        
        logging.info(f"Data saved to {PROCESSED_DIR}/")
        logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
