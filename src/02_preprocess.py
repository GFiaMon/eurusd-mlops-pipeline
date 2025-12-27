import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Add project root to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import DataManager

def preprocess_data():
    """
    Loads raw data via DataManager, performs feature engineering, and saves processed artifacts via DataManager.
    """
    print("🔄 Starting preprocessing via DataManager...")
    
    # 1. Load Raw Data
    dm_raw = DataManager(data_type='raw')
    print(f"📥 Loading raw data (Environment: {dm_raw.get_environment()})...")
    
    # This handles checking local mirror and syncing from S3 if needed
    df = dm_raw.get_latest_data(force_refresh=False)
    
    if df.empty:
        print("❌ Error: No data returned from DataManager. Run ingest_data.py first.")
        return

    print(f"✅ Loaded {len(df)} rows. Range: {df.index.min()} to {df.index.max()}")
    
    # Fix: Remove MultiIndex tuple columns if present (both actual tuples and string representations)
    tuple_cols = [col for col in df.columns if isinstance(col, tuple) or (isinstance(col, str) and col.startswith('(') and col.endswith(')'))]
    if tuple_cols:
        print(f"🔧 Removing {len(tuple_cols)} MultiIndex tuple columns...")
        df = df.drop(columns=tuple_cols)
    
    # Drop Volume column (always 0 for forex pairs)
    if 'Volume' in df.columns:
        print("🔧 Dropping Volume column (not relevant for forex)...")
        df = df.drop(columns=['Volume'])
    
    # 2. Feature Engineering (Original Logic)
    print("🔧 Feature Engineering...")
    
    # Ensure numeric Close
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    # Daily Returns (Percentage)
    df['Return'] = df['Close'].pct_change()
    
    # Moving Averages
    # Superset: Generating all potentially useful MAs (Important one is MA_50 based on feature importance)
    for ma in [5, 10, 20, 50]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()
        
    # Lagged Returns (1 day is 'Return')
    # Returns over longer periods (Momentum) (Important one is Return_20d based on feature importance)
    for r in [5, 20]:
        df[f'Return_{r}d'] = df['Close'].pct_change(periods=r)

    # Lagged Features for 1-step prediction (Standard Regression features)
    # Important one is Lag_2, Lag_3 based on feature importance. Filtered: Lag_1, Lag_5 (weak importance).

    for lag in [1, 2, 3, 5]:
        df[f'Lag_{lag}'] = df['Return'].shift(lag)
        
    # Drop NaNs created by rolling and shifting (max rolling is 50)
    print(f"🔍 Before dropna: {len(df)} rows")
    print(f"🔍 NaN counts per column:")
    nan_counts = df.isna().sum()
    for col in nan_counts[nan_counts > 0].index:
        print(f"   {col}: {nan_counts[col]} NaNs")
    print(f"🔍 Columns: {list(df.columns)}")
    df = df.dropna()
    print(f"🔍 After dropna: {len(df)} rows")
    
    # Define features and target
    # Target: Return_{t+1}
    df['Target'] = df['Return'].shift(-1)
    print(f"🔍 After creating Target: {len(df)} rows")
    df = df.dropna()
    print(f"🔍 After final dropna: {len(df)} rows")
    
    # Split Chronologically (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"📊 Train set size: {len(train_df)}")
    print(f"📊 Test set size: {len(test_df)}")
    
    # Create a copy of Return for ARIMA (unscaled)
    train_df['Return_Unscaled'] = train_df['Return']
    test_df['Return_Unscaled'] = test_df['Return']
    
    # Scaling
    print("⚖️  Scaling data...")
    # Update feature list to include OHLC and new features
    # Note: OHLC are in 'df' if read correctly. 
    # 'Open', 'High', 'Low' are available from yfinance download.

    # Determine columns present
    essential_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [c for c in essential_cols if c not in df.columns]
    if missing_cols:
         print(f"⚠️  Warning: Columns {missing_cols} missing. Using available.")
         
    # Construct feature list (Superset) (Important features are Return, MA_50, Return_20d, Lag_2, Lag_3)
    feature_cols = ['Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Return_5d', 'Return_20d', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5']
    
    # Add OHLC to features if present (scaled)
    for col in essential_cols:
        if col in df.columns:
            feature_cols.append(col)
            
    print(f"Features Generated & Scaled: {feature_cols}")

    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 3. Save Processed Data via DataManager
    print("💾 Saving processed artifacts...")
    dm_processed = DataManager(data_type='processed')
    
    success = dm_processed.save_processed(
        train_df, 
        test_df, 
        scaler, 
        metadata={'version': '1.0', 'source_rows': len(df)}
    )
    
    if success:
        print(f"✅ Processed data saved to {dm_processed.local_dir}/")
        if dm_processed.is_cloud_available():
            print("☁️  Synced artifacts to S3.")
        else:
            print("params: Local only (S3 not configured or offline).")
    else:
        print("❌ Failed to save processed data.")
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    preprocess_data()
