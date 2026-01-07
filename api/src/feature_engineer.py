
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config, scaler):
        self.config = config
        self.scaler = scaler
        self.feature_cols = config.get('features', [])
        # Fallback to standard 15 features if not specified
        if not self.feature_cols:
             self.feature_cols = ['Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Return_5d', 'Return_20d', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Open', 'High', 'Low', 'Close']
        
        self.model_type = config.get('model_type', 'Unknown')
        
    def preprocess(self, df):
        """
        Takes raw dataframe (OHLC) and generates features for the model.
        """
        if df is None or df.empty:
            return None, None
            
        data = df.copy()
        
        # Consistent OHLC naming
        name_map = {c: c.capitalize() for c in data.columns if c.lower() in ['open', 'high', 'low', 'close']}
        data = data.rename(columns=name_map)
        
        if 'Close' not in data.columns:
            return None, None
            
        # --- FEATURE CALCULATION (Deriving from Close) ---
        # 1. Base Return
        data['Return'] = data['Close'].pct_change()
        data['Return_Unscaled'] = data['Return']
        
        # 2. Indicators (Moving Averages & Multi-day Returns)
        for ma in [5, 10, 20, 50]:
            data[f'MA_{ma}'] = data['Close'].rolling(window=ma).mean()
            
        for d in [5, 20]:
            data[f'Return_{d}d'] = data['Close'].pct_change(periods=d)
            
        # 3. Lags
        for lag in [1, 2, 3, 5]:
            data[f'Lag_{lag}'] = data['Return'].shift(lag)
            
        # Clean up NaNs created by rolling/shifting
        data = data.dropna()
        if len(data) == 0:
            return None, data['Close'].iloc[-1] if not data.empty else None
            
        # --- PREPARE FOR MODEL ---
        # Ensure we have all columns the model expects (fills with 0 as safety, but calc covers 15)
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = 0
                
        # Filter and order exactly as trained
        X_df = data[self.feature_cols]
        current_price = data['Close'].iloc[-1]
        
        # Scaling
        if self.scaler:
            try:
                # Only transform if shape matches
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == len(self.feature_cols):
                    X_scaled = self.scaler.transform(X_df)
                    X_df = pd.DataFrame(X_scaled, columns=self.feature_cols, index=X_df.index)
            except:
                pass

        # Formatting for specific model types
        if 'lstm' in self.model_type.lower() or 'lstm' in str(self.config).lower():
            time_steps = self.config.get('time_steps', 60)
            if len(X_df) < time_steps:
                return None, current_price
            
            # Extract last N rows for sequence
            X_seq = X_df.iloc[-time_steps:].values
            X = X_seq.reshape((1, time_steps, len(self.feature_cols)))
        else:
            # Linear / ARIMA
            X = X_df.iloc[[-1]]
            
        return X, current_price

    def preprocess_history(self, df, lookback=60):
        """Batch preprocess for history page - returns FULL dataset features"""
        # 1. Generate all base features on the full dataframe
        data = df.copy()
        
        # Ensure consistent naming
        name_map = {c: c.capitalize() for c in data.columns if c.lower() in ['open', 'high', 'low', 'close']}
        data = data.rename(columns=name_map)
        
        data['Return'] = data['Close'].pct_change()
        for ma in [5, 10, 20, 50]: data[f'MA_{ma}'] = data['Close'].rolling(window=ma).mean()
        for d in [5, 20]: data[f'Return_{d}d'] = data['Close'].pct_change(periods=d)
        for lag in [1, 2, 3, 5]: data[f'Lag_{lag}'] = data['Return'].shift(lag)
        
        # Drop NaNs from feature gen
        data = data.dropna()
        
        # Ensure model features exist
        for col in self.feature_cols:
            if col not in data.columns: data[col] = 0
            
        X_df = data[self.feature_cols]
        
        # Scale
        if self.scaler:
            try:
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == len(self.feature_cols):
                    X_scaled = self.scaler.transform(X_df)
                    X_df = pd.DataFrame(X_scaled, columns=self.feature_cols, index=X_df.index)
            except: pass
            
        return X_df, data['Return'], data['Close'], data.index
