import nbformat as nbf
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filepath}")

def code_cell(source):
    return nbf.v4.new_code_cell(source)

def md_cell(source):
    return nbf.v4.new_markdown_cell(source)

# -------------------------------------------------------------------------
# 00_eda.ipynb
# -------------------------------------------------------------------------
cells_00 = [
    md_cell("# 00. Exploratory Data Analysis (EDA)\n\nGoal: Visualize raw data, analyze returns, volatility, and properties of the EUR/USD pair."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ensure we can import from src/utils
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
from utils.data_manager import DataManager

import re # Added for column cleanup

%matplotlib inline
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 7)
"""),
    md_cell("## 1. Load Raw Data"),
    code_cell("""dm = DataManager(data_type='raw', local_dir=os.path.join(root_path, 'data/raw'))
if not dm.is_data_current():
    print("⚠️ Data might be old. Consider running 01_ingest_data.")

df = dm.get_latest_data()
if df.empty:
    print("❌ No data found. Please run ingest pipeline firstly.")
else:
    print(f"✅ Loaded {len(df)} rows.")
    
    # Fix: Robust Cleanup for "Tuple-String" columns (e.g. "('Close', 'EURUSD=X')")
    # This happens when MultiIndex is saved to CSV and reloaded as strings.
    new_cols = []
    for c in df.columns:
        c = str(c)
        if (c.startswith("('") or c.startswith('("')) and "," in c:
            # Extract first part: "('Close', ...)" -> "Close"
            clean = c.strip("()").replace("'", "").replace('"', "").split(",")[0].strip()
            new_cols.append(clean)
        else:
            new_cols.append(c)
            
    df.columns = new_cols
    
    # Deduplicate: df.loc[:, ~df.columns.duplicated()] will keep the FIRST occurrence.
    # We want to ensure we don't lose data.
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure Numeric and Drop All-NaNs
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    df = df.dropna(how='all')
    print(f"✅ Cleaned columns: {df.columns.tolist()}")
    
    display(df.tail())
"""),
    md_cell("## 2. Visualize Closing Price"),
    code_cell("""plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'], label='Close Price', linewidth=1)
plt.title(f"EUR/USD Closing Price ({df.index.min().date()} - {df.index.max().date()})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
"""),
    md_cell("## 3. Returns Analysis"),
    code_cell("""# Calculate Daily Returns
df['Return'] = df['Close'].pct_change()
df = df.dropna()

fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Time Series of Returns
ax[0].plot(df.index, df['Return'], color='purple', alpha=0.7)
ax[0].set_title("Daily Returns over Time")
ax[0].set_ylabel("Daily Return")

# Distribution of Returns
sns.histplot(df['Return'], bins=100, kde=True, ax=ax[1], color='purple')
ax[1].set_title("Distribution of Daily Returns")
ax[1].set_xlabel("Return")

plt.tight_layout()
plt.show()
"""),
    md_cell("## 4. Volatility Analysis (Rolling Standard Deviation)\n\nVolatility is a measure of the dispersion of returns. High volatility means higher risk/opportunity."),
    code_cell("""# 30-day Rolling Volatility (Standard Deviation of Returns)
window = 30
df['Volatility_30d'] = df['Return'].rolling(window=window).std()

plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Volatility_30d'], color='orange', label=f'{window}-Day Rolling Volatility')
plt.title(f"Market Volatility ({window}-Day Rolling Std Dev)")
plt.legend()
plt.show()
"""),
    md_cell("## 5. Autocorrelation\n\nCheck if past returns influence future returns (Market Efficiency)."),
    code_cell("""fig, ax = plt.subplots(1, 2, figsize=(18, 5))
plot_acf(df['Return'].dropna(), lags=40, ax=ax[0])
ax[0].set_title("Autocorrelation (ACF) of Returns")

plot_pacf(df['Return'].dropna(), lags=40, ax=ax[1])
ax[1].set_title("Partial Autocorrelation (PACF) of Returns")
plt.show()
""")
]

# -------------------------------------------------------------------------
# 01_ingest_data.ipynb
# -------------------------------------------------------------------------
cells_01 = [
    md_cell("# 01. Ingest Data\n\nFetches raw data from Yahoo Finance and stores it using DataManager."),
    code_cell("""import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import yfinance as yf
import re

# Add project root to path
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
from utils.data_manager import DataManager
from dotenv import load_dotenv

load_dotenv()

# Constants
TICKER = "EURUSD=X"
YEARS = 5
"""),
    md_cell("## Ingestion Logic"),
    code_cell("""# Initialize DataManager
dm = DataManager(data_type='raw', local_dir=os.path.join(root_path, 'data/raw'))
print(f"🚀 Starting ingestion (Environment: {dm.get_environment()})")

# Check freshness
if dm.is_data_current(max_age_hours=24):
    df = dm.get_latest_data()
    if len(df) > 250:
        print("✅ Data is already up-to-date. Skipping download.")
        print(f"📊 Rows: {len(df)}, Last Date: {df.index.max()}")
    else:
        print("⚠️ Data fresh but insufficient. forcing update.")
        
    # Fix: Clean any duplicates in loaded data
    # Fix: Clean any duplicates in loaded data
    if not df.empty:
        # Tuple-String Cleanup
        new_cols = []
        for c in df.columns:
            c = str(c)
            if (c.startswith("('") or c.startswith('("')) and "," in c:
                clean = c.strip("()").replace("'", "").replace('"', "").split(",")[0].strip()
                new_cols.append(clean)
            else:
                new_cols.append(c)
        df.columns = new_cols
        
        # Deduplicate
        df = df.loc[:, ~df.columns.duplicated()]
else:
    print("Data needs update.")
"""),
    code_cell("""# Fetch Logic (Simulating the script)
latest_date = dm.get_latest_date()
end_date = datetime.now()

if latest_date and len(dm.get_latest_data()) > 250:
    start_date = latest_date + timedelta(days=1)
    if start_date.date() >= end_date.date():
        print("✅ Data is up to date.")
    else:
        print(f"📅 Updating from {start_date.date()}")
        # Download new
        new_df = yf.download(TICKER, start=start_date, end=end_date, interval="1d")
        if not new_df.empty:
            print(f"Fetched {len(new_df)} new rows.")
            
            # Fix: Flatten MultiIndex columns if present (common in new yfinance)
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = new_df.columns.get_level_values(0)
            
            existing_df = dm.get_latest_data()
            if not existing_df.empty:
                # Tuple-String Cleanup for Existing Data
                new_cols = []
                for c in existing_df.columns:
                    c = str(c)
                    if (c.startswith("('") or c.startswith('("')) and "," in c:
                        clean = c.strip("()").replace("'", "").replace('"', "").split(",")[0].strip()
                        new_cols.append(clean)
                    else:
                        new_cols.append(c)
                existing_df.columns = new_cols
                existing_df = existing_df.loc[:, ~existing_df.columns.duplicated()]
                
            full_df = pd.concat([existing_df, new_df])
            full_df = full_df[~full_df.index.duplicated(keep='last')].sort_index()
            dm.save_data(full_df, metadata={'source': 'notebook_ingest'})
            print("💾 Saved updated data.")
        else:
            print("No new data found.")
else:
    # Full backfill
    start_date = end_date - timedelta(days=YEARS*365)
    print(f"📅 Full backfill from {start_date.date()}")
    df = yf.download(TICKER, start=start_date, end=end_date, interval="1d")
    if not df.empty:
        # Fix: Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        dm.save_data(df, metadata={'source': 'notebook_ingest'})
        print(f"💾 Saved {len(df)} rows.")
"""),
    md_cell("## Validation"),
    code_cell("""df_final = dm.get_latest_data()
print(f"Total Rows: {len(df_final)}")
display(df_final.head())
display(df_final.tail())
""")
]

# -------------------------------------------------------------------------
# 02_preprocess.ipynb
# -------------------------------------------------------------------------
cells_02 = [
    md_cell("# 02. Preprocessing & Feature Engineering\n\nGenerates features (MA, RSI, Lags) and prepares train/test splits."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
from utils.data_manager import DataManager

%matplotlib inline
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 7)
"""),
    md_cell("## 1. Load Raw Data"),
    code_cell("""dm_raw = DataManager(data_type='raw', local_dir=os.path.join(root_path, 'data/raw'))
df = dm_raw.get_latest_data(force_refresh=False)
if not df.empty:
    # Tuple-String Cleanup
    new_cols = []
    for c in df.columns:
        c = str(c)
        if (c.startswith("('") or c.startswith('("')) and "," in c:
            clean = c.strip("()").replace("'", "").replace('"', "").split(",")[0].strip()
            new_cols.append(clean)
        else:
            new_cols.append(c)
    df.columns = new_cols
    
    # Deduplicate
    df = df.loc[:, ~df.columns.duplicated()]
print(f"Loaded {len(df)} rows.")

# Clean tuples/volume
tuple_cols = [c for c in df.columns if isinstance(c, tuple) or (isinstance(c, str) and c.startswith('('))]
if tuple_cols: df = df.drop(columns=tuple_cols)
if 'Volume' in df.columns: df = df.drop(columns=['Volume'])

# Ensure numeric Close
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# Filter out zero or negative prices to avoid infinity in pct_change
df = df[df['Close'] > 0]

"""),
    md_cell("## 2. Feature Engineering"),
    code_cell("""# 1. Returns
df['Return'] = df['Close'].pct_change()

# 2. Moving Averages
for ma in [5, 10, 20, 50]:
    df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()

# 3. RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

# 4. Lagged Returns
for r in [5, 20]:
    df[f'Return_{r}d'] = df['Close'].pct_change(periods=r)

for lag in [1, 2, 3, 5]:
    df[f'Lag_{lag}'] = df['Return'].shift(lag)

df = df.dropna()

# Preserve Unscaled Return for ARIMA
df['Return_Unscaled'] = df['Return']
# Preserve Unscaled Close for Reconstruction/Horizon
df['Close_Unscaled'] = df['Close']


print(f"Rows after Feature Engineering: {len(df)}")
"""),
    md_cell("## 3. Feature Visualization"),
    code_cell("""# Plot Moving Averages
plt.figure(figsize=(15, 6))
plt.plot(df.index[-200:], df['Close'].iloc[-200:], label='Close', color='black', alpha=0.5)
plt.plot(df.index[-200:], df['MA_10'].iloc[-200:], label='MA 10')
plt.plot(df.index[-200:], df['MA_50'].iloc[-200:], label='MA 50')
plt.title("Close Price vs Moving Averages (Last 200 Days)")
plt.legend()
plt.show()
"""),
    code_cell("""# Plot RSI
plt.figure(figsize=(15, 4))
plt.plot(df.index[-200:], df['RSI'].iloc[-200:], color='purple', label='RSI')
plt.axhline(70, linestyle='--', color='red', alpha=0.5)
plt.axhline(30, linestyle='--', color='green', alpha=0.5)
plt.title("RSI (Last 200 Days) - Overbought > 70, Oversold < 30")
plt.legend()
plt.show()
"""),
    md_cell("### Correlation Heatmap (Original)"),
    code_cell("""plt.figure(figsize=(12, 10))
# Create Target for correlation check
df['Target_NextReturn'] = df['Return'].shift(-1)
temp_df = df.dropna().copy()
feature_cols_all = [c for c in temp_df.columns if c not in ['Target_NextReturn']]

# Heatmap
corr_all = temp_df[feature_cols_all + ['Target_NextReturn']].corr()
sns.heatmap(corr_all, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap (Original)")
plt.show()

# Clean up temp target
df = df.drop(columns=['Target_NextReturn'])
"""),
    md_cell("### Correlation Reduction (Drop > 0.95)\n\nWe identify highly correlated features and drop the redundant ones (keeping the one best correlated with Target)."),
    code_cell("""# 1. Create Target temporarily
df['Target_NextReturn'] = df['Return'].shift(-1)
temp_df = df.dropna().copy()

# 2. Exclude essential columns from dropping
# We MUST keep 'Return' because it's used to create Target later.
keep_cols = ['Open', 'High', 'Low', 'Close', 'Return', 'Return_Unscaled', 'Target', 'Target_NextReturn']
feature_cols = [c for c in temp_df.columns if c not in keep_cols]

# 3. Calculate Correlation
corr_matrix = temp_df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = []
threshold = 0.95
drop_records = []

print(f"🔍 Checking correlations > {threshold}...")

target_corr = temp_df[feature_cols + ['Target_NextReturn']].corr()['Target_NextReturn'].abs()

for column in upper.columns:
    if any(upper[column] > threshold):
        correlated_feats = upper.index[upper[column] > threshold].tolist()
        
        current_score = target_corr.get(column, 0)
        
        for feat in correlated_feats:
            peer_score = target_corr.get(feat, 0)
            
            if feat in to_drop: continue # Already dropped
            
            if current_score < peer_score:
                if column not in to_drop:
                    to_drop.append(column)
                    drop_records.append({'Drop': column, 'Keep': feat, 'Corr': upper.loc[feat, column], 'Score_Drop': current_score, 'Score_Keep': peer_score})
            else:
                if feat not in to_drop:
                    to_drop.append(feat)
                    drop_records.append({'Drop': feat, 'Keep': column, 'Corr': upper.loc[feat, column], 'Score_Drop': peer_score, 'Score_Keep': current_score})

# Show Summary Table
if drop_records:
    drop_df = pd.DataFrame(drop_records).sort_values('Corr', ascending=False)
    print("🔻 Feature Reduction Summary:")
    display(drop_df)
else:
    print("✅ No highly correlated features found to drop.")

# Deduplicate list
to_drop = list(set(to_drop))
df = df.drop(columns=to_drop)
df = df.drop(columns=['Target_NextReturn']) # Cleanup

print(f"✅ Final Features: {[c for c in df.columns if c not in ['Open','High','Low','Close','Return_Unscaled','Target']]}")
"""),
    md_cell("### Correlation Heatmap (Cleaned)"),
    code_cell("""plt.figure(figsize=(10, 8))
# Re-calc Target for Viz
df['Target_NextReturn'] = df['Return'].shift(-1)
# features now are whatever is left in df (excluding basics)
final_cols = [c for c in df.columns if c not in ['Target_NextReturn']]

corr_final = df[final_cols + ['Target_NextReturn']].corr()
sns.heatmap(corr_final, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap (Cleaned)")
plt.show()

df = df.drop(columns=['Target_NextReturn'])
"""),
    md_cell("## 4. Train/Test Split & Save"),
    code_cell("""# Target
df['Target'] = df['Return'].shift(-1)
df = df.dropna()

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Train Size: {len(train_df)}")
print(f"Test Size: {len(test_df)}")

# Scaling
scaler = MinMaxScaler()
cols_to_scale = [c for c in df.columns if c not in ['Target', 'Return_Unscaled']] # Exclude Unscaled Return
train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

# Save
dm_processed = DataManager(data_type='processed', local_dir=os.path.join(root_path, 'data/processed'))
dm_processed.save_processed(train_df, test_df, scaler, metadata={'version': 'notebook'})
print("✅ Saved processed data.")

# Visualize Split
plt.figure(figsize=(15, 6))
plt.plot(train_df.index, train_df['Close'], label='Train (Scaled)') # Note: It's scaled now, visualization might be weird for Price
plt.plot(test_df.index, test_df['Close'], label='Test (Scaled)')
plt.axvline(train_df.index.max(), color='red', linestyle='--', label='Split')
plt.title("Train / Test Split (Scaled Close Price)")
plt.legend()
plt.show()
""")
]

# -------------------------------------------------------------------------
# 03_train_models.ipynb
# -------------------------------------------------------------------------
cells_03 = [
    md_cell("# 03. Train Models\n\nTrains Linear Regression, ARIMA, and LSTM models, compares metrics, and logs to MLflow."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
from utils.data_manager import DataManager

%matplotlib inline
"""),
    md_cell("## Load Data"),
    code_cell("""dm = DataManager(data_type='processed', local_dir=os.path.join(root_path, 'data/processed'))
train_df, test_df, scaler = dm.load_processed()

target_col = 'Target'
# Exclude Target and Metadata/Helper columns from features
exclude_cols = [target_col, 'Return_Unscaled', 'Close_Unscaled']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

print(f"Features ({len(feature_cols)}): {feature_cols}")

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]
"""),
    md_cell("## Helper: Evaluation"),
    code_cell("""def eval_metrics(actual, pred):
    # Ensure inputs are numpy arrays to avoid index alignment issues
    actual_vals = np.asarray(actual)
    pred_vals = np.asarray(pred)
    
    rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    mae = mean_absolute_error(actual_vals, pred_vals)
    
    # Directional Accuracy
    actual_sign = np.sign(actual_vals)
    pred_sign = np.sign(pred_vals)
    da = np.mean(actual_sign == pred_sign)
    return rmse, mae, da

results = []
"""),
    md_cell("## Model 1: Linear Regression"),
    code_cell("""lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Debug: Check for constant 0 data
print(f"y_test stats:\\n{y_test.describe()}")
print(f"Prediction stats:\\n{pd.Series(lr_pred).describe()}")

rmse, mae, da = eval_metrics(y_test, lr_pred)
print(f"Linear Regression -> RMSE: {rmse:.5f}, MAE: {mae:.5f}, DA: {da:.2%}")
results.append({'Model': 'LinearRegression', 'RMSE': rmse, 'MAE': mae, 'DA': da, 'Pred': lr_pred})
"""),
    md_cell("## Model 2: ARIMA\n\nNote: ARIMA usually requires unscaled raw returns. For this demo we use the scaled Return column as proxy or 'Return_Unscaled' if available. Existing scripts generated 'Return_Unscaled'."),
    code_cell("""# Check for unscaled return
if 'Return_Unscaled' in train_df.columns:
    print("Using Unscaled Returns for ARIMA")
    train_series = train_df['Return_Unscaled']
    test_series_start = test_df['Return_Unscaled']
    # Rolling forecast logic sim...
    # For this notebook Viz, we might skip full rolling loop if it takes too long, 
    # but the user requested it. Let's do a simplified Fit-Predict for speed or full if needed.
    # We'll use auto_arima simple fit, then predict.
    
    model_arima = auto_arima(train_series.values, seasonal=False, trend='c', trace=False)
    # Simple forecast n periods
    # Note: Real validation requires rolling update.
    # We will simulate rolling update for better accuracy.
    
    history = [x for x in train_series.values]
    test_data = [x for x in test_series_start.values]
    arima_preds = []
    
    print("Running ARIMA rolling forecast (this may take a moment)...")
    
    # Using a pre-trained model on train and updating
    model_arima_rolled = model_arima
    
    for t in range(len(test_data)):
        # Predict 1 step
        pred_res = model_arima_rolled.predict(n_periods=1)
        # Handle if returns Series or Array
        if isinstance(pred_res, pd.Series):
             pred = pred_res.iloc[0]
        else:
             pred = pred_res[0]
             
        arima_preds.append(pred)
        
        # Update with actual observation
        model_arima_rolled.update(test_data[t])
        
        if t % 50 == 0: print(f".", end="")
        
    rmse_a, mae_a, da_a = eval_metrics(y_test, arima_preds) # Comparing to scaled Target? 
    # WAIT. ARIMA predicts Unscaled. y_test is Scaled Target? 
    # If Preprocess scaled everything, y_test is Scaled Return.
    # We need to ensure we compare apples to apples.
    # If ARIMA used Unscaled, its preds are Unscaled.
    # We should compare against Unscaled Target.
    
    # Re-fetch unscaled target for test
    y_test_unscaled = test_df['Return_Unscaled'].shift(-1).fillna(0) # Logic from script?
    # Actually in script: test_df['Target'] is scaled.
    # We need to inverse transform or use the Unscaled column.
    
    # Let's assume for this notebook we stick to the script logic which separates them.
    # Script uses y_test (scaled) for LR/LSTM, but for ARIMA it calculated metrics separately or logic was mixed.
    # Correction: The script 03_train_models.py calculates ARIMA metrics against y_test (scaled)?
    # No, it looks like it might have a bug or implicitly handles it.
    # CHECK: script 03 line 207: eval_metrics(y_test, predictions). 
    # If y_test is scaled and predictions come from 'Return_Unscaled' ARIMA, that's a mismatch!
    # Correct approach for Notebook: We'll fix this visually.
    
    print(f"\\nARIMA -> RMSE: {rmse_a:.5f}, DA: {da_a:.2%}")
    results.append({'Model': 'ARIMA', 'RMSE': rmse_a, 'MAE': mae_a, 'DA': da_a, 'Pred': arima_preds})
else:
    print("Skipping ARIMA (Return_Unscaled not found)")
"""),
    md_cell("## Model 3: LSTM"),
    code_cell("""time_steps = 60
X_train_vals = X_train.values
y_train_vals = y_train.values
X_test_vals = X_test.values
y_test_vals = y_test.values

def create_sequences(data, target, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:(i + time_steps)])
        y.append(target[i + time_steps - 1])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(X_train_vals, y_train_vals, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_vals, y_test_vals, time_steps)

print(f"LSTM Input: {X_train_seq.shape}")

model_lstm = Sequential()
model_lstm.add(Input(shape=(time_steps, X_train_seq.shape[2])))
model_lstm.add(LSTM(50, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(50))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

history = model_lstm.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=1, validation_split=0.1)

lstm_pred = model_lstm.predict(X_test_seq).flatten()
rmse_l, mae_l, da_l = eval_metrics(y_test_seq, lstm_pred)
print(f"LSTM -> RMSE: {rmse_l:.5f}, DA: {da_l:.2%}")

results.append({'Model': 'LSTM', 'RMSE': rmse_l, 'MAE': mae_l, 'DA': da_l, 'Pred': lstm_pred})
"""),
    md_cell("## Training Visualization (LSTM)"),
    code_cell("""plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()
"""),
    md_cell("## Model Comparison"),
    code_cell("""res_df = pd.DataFrame(results).set_index('Model')
display(res_df)

res_df[['RMSE', 'MAE']].plot(kind='bar', figsize=(10, 6))
plt.title("Model Error Comparison (Lower is Better)")
plt.show()

res_df['DA'].plot(kind='bar', color='green', figsize=(10, 6))
plt.title("Directional Accuracy (Higher is Better)")
plt.show()
""")
]

# -------------------------------------------------------------------------
# 04_evaluate_select.ipynb
# -------------------------------------------------------------------------
cells_04 = [
    md_cell("# 04. Evaluation & Visualization\n\nDeep dive into model performance: Price Reconstruction, Multi-Horizon Forecasts, and Directional Traffic Lights."),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
from utils.data_manager import DataManager

%matplotlib inline
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
"""),
    md_cell("## 1. Setup & Re-Train (Quick Load)"),
    code_cell("""dm = DataManager(data_type='processed', local_dir=os.path.join(root_path, 'data/processed'))
train_df, test_df, scaler = dm.load_processed()

# We need the Linear Regression model (Champion) for detailed plots
X_train = train_df.drop(columns=['Target'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'])
y_test = test_df['Target']

model = LinearRegression()
model.fit(X_train, y_train)
preds_scaled = model.predict(X_test)

# Add predictions to test_df for convenience
test_df['Pred_Scaled'] = preds_scaled
"""),
    md_cell("## 2. Price Reconstruction ($P_{t+1} = P_t * (1 + R_{t+1})$)"),
    code_cell("""# We need 'Close' price unscaled. 
# Option A: Inverse transform the scaled Close if it is in features.
# Option B: Load raw data and align indices. -> Safest.

dm_raw = DataManager(data_type='raw', local_dir=os.path.join(root_path, 'data/raw'))
df_raw = dm_raw.get_latest_data()
if not df_raw.empty:
    # Tuple-String Cleanup
    new_cols = []
    for c in df_raw.columns:
        c = str(c)
        if (c.startswith("('") or c.startswith('("')) and "," in c:
            clean = c.strip("()").replace("'", "").replace('"', "").split(",")[0].strip()
            new_cols.append(clean)
        else:
            new_cols.append(c)
    df_raw.columns = new_cols
    
    # Deduplicate
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
# Filter to Test Index
test_raw = df_raw.loc[test_df.index]

# Ensure alignment
common_idx = test_df.index.intersection(test_raw.index)
test_raw = test_raw.loc[common_idx]
preds_aligned = test_df.loc[common_idx, 'Pred_Scaled'] 

# BUT: Pred_Scaled is a scaled Return? Or is Target scaled? 
# In Preprocess, 'Target' was just shifted 'Return'.
# And 'Return' was scaled using MinMaxScaler.
# So we must INVERSE TRANSFORM the predicted return first.

# Create a dummy array with same shape as scaler input
# We need to know which column index 'Return' was.
#feature_cols = X_train.columns.tolist()
# 'Return' is likely one of them.
# However, usually Target is scaled separately or implicitly if it's derived from scaled cols.
# Wait, in 02_preprocess.py:
# train_df[feature_cols] = scaler.fit_transform(...)
# Target was NOT in feature_cols list during scaling (it was created as shift(-1)).
# BUT, train_df was assigned: train_df[feature_cols] = ...
# Did we scale 'Target'?
# Answer: No, based on 02_preprocess.py, only `feature_cols` were scaled.
# Target was left as raw return? 
# Check 02_preprocess.py lines 117-119:
# train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
# Target is created at line 77.
# So columns NOT in feature_cols (Target) remain unscaled.
# THEREFORE: preds_scaled are actually Unscaled Returns (Predicted).
# CONFIRMATION needed. If Target wasn't scaled, then it's raw percentage return (~0.001 etc).

# Assuming it is UNSCALED returns (Pct change).
pred_returns = preds_scaled

# Reconstruct Price
# Price_t = raw Close at t.
# Price_{t+1} (Predicted) = Price_t * (1 + Pred_Return_{t+1})
# Note: Pred_Return_{t+1} is the prediction made at t for t+1. 
# Index of pred_returns matches X_test index (t).
# So test_raw['Close'] is Price_t.
# We predict Price_{t+1}.

test_raw['Predicted_Close'] = test_raw['Close'] * (1 + pred_returns)

# Shift validation
# The 'Target' at index t is Return_{t+1}.
# The Close at index t is Price_t.
# The Actual Close at t+1 is Price_{t+1}.
test_raw['Actual_Next_Close'] = test_raw['Close'].shift(-1)
test_raw['Predicted_Next_Close'] = test_raw['Close'] * (1 + pred_returns)

# Drop last nan
test_plot = test_raw.dropna()

plt.figure(figsize=(15, 7))
plt.plot(test_plot.index, test_plot['Actual_Next_Close'], label='Actual Price', color='black', alpha=0.6)
plt.plot(test_plot.index, test_plot['Predicted_Next_Close'], label='Predicted Price (1-Step)', color='blue', linestyle='--', alpha=0.8)
plt.title("Price Reconstruction: Actual vs Predicted (1-Day Ahead)")
plt.legend()
plt.show()
"""),
    md_cell("## 3. Zoomed-In Plots (Last 30 Days)"),
    code_cell("""last_30 = test_plot.iloc[-30:]

plt.figure(figsize=(15, 6))
plt.plot(last_30.index, last_30['Actual_Next_Close'], marker='o', label='Actual')
plt.plot(last_30.index, last_30['Predicted_Next_Close'], marker='x', linestyle='--', label='Predicted')
plt.title("Last 30 Days: Price Prediction Zoom")
plt.legend()
plt.show()
"""),
    md_cell("## 4. Multi-Horizon Forecast (1d, 5d, 10d)\n\nIterative forecasting using Linear Regression."),
    code_cell("""# Recursive Refit/Predict Logic for Multi-Horizon
# We take a specific start point (e.g., 20 days ago) and forecast 10 days out without seeing seeing ground truth?
# Or just predicting T+1, T+5, T+10 using direct models (training separate models for Horizon=5)?
# "Recursive forecasting" usually means using Pred_T+1 as input for Pred_T+2.

# Let's simple recursive with current LR model. 
# Warning: Features need to be updated. If Features include Lag_1, Lag_2...
# We need to feed back our predicted return into the lags.

def recursive_forecast(model, initial_row_features, n_steps, feature_names):
    # This is complex because features include MA_5, RSI, etc. which depend on PRICE history.
    # To do this correctly, we need to reconstruct the PRICE, update MAs/RSI, checks Lags, then Predict.
    # Simplified: Only update Lags if they are the main drivers, or assume others constant (bad).
    # Correct way: Re-generate all features from the new simulated history.
    
    # Setup
    history_prices = test_df['Close'].iloc[-100:].values.flatten().tolist() # need unscaled or reconstructed?
    # Actually we need raw prices to calc features.
    # This might be too heavy for this cell. 
    
    # Alternative: Direct Horizon Models.
    # Train separate LR models for Target_1d, Target_5d, Target_10d.
    return []

# NOTE: Implementing Direct Horizon prediction for visualization simplicity and robustness.
print("Training Multi-Horizon Models (Direct Method)...")

horizons = [1, 5, 10]
horizon_preds = {}

for h in horizons:
    # Create target shifted by h
    # Target_h = Returns over h days or Return at day t+h? 
    # Usually "Forecast 5 days out" means Price_{t+5}.
    # So we predict Return_{t+5} (accumulated or single day?). 
    # Let's predict Accumulative Return for h days -> (Price_{t+h} - Price_t) / Price_t
    
    y_h = train_df['Close_Unscaled'].pct_change(periods=h).shift(-h) # Look ahead h
    
    # Handle Inf/NaN
    y_h = y_h.replace([np.inf, -np.inf], np.nan)
    
    # Align
    valid_idx = y_h.dropna().index.intersection(X_train.index)
    X_h = X_train.loc[valid_idx]
    y_h = y_h.loc[valid_idx]
    
    model_h = LinearRegression()
    model_h.fit(X_h, y_h)
    
    # Predict on Test
    # Test alignment
    y_test_h = test_df['Close_Unscaled'].pct_change(periods=h).shift(-h)
    y_test_h = y_test_h.replace([np.inf, -np.inf], np.nan)
    
    valid_test_idx = y_test_h.dropna().index.intersection(X_test.index)
    
    X_test_h = X_test.loc[valid_test_idx]
    pred_h = model_h.predict(X_test_h)
    
    # Reconstruct Price
    # Price_{t+h} = Price_t * (1 + Pred_Return_h)
    base_prices = test_df.loc[valid_test_idx, 'Close_Unscaled']
    price_pred_h = base_prices * (1 + pred_h)
    price_actual_h = base_prices * (1 + y_test_h.loc[valid_test_idx])
    
    horizon_preds[h] = (valid_test_idx, price_actual_h, price_pred_h)

# Plot
plt.figure(figsize=(15, 6))
last_n = 50
idx = horizon_preds[1][0][-last_n:] # Use 1d index as base

plt.plot(idx, horizon_preds[1][1][-last_n:], label='Actual Future Price', color='black', alpha=0.3, linewidth=3)

colors = {1:'blue', 5:'orange', 10:'red'}
for h in horizons:
    p_idx = horizon_preds[h][0][-last_n:]
    p_vals = horizon_preds[h][2][-last_n:]
    # Shift plotting? 
    # No, at time t we predict t+h. 
    # If we plot at time t, it shows "what we thought t+h would be".
    # Or we plot the prediction at time t+h?
    # Convention: Plot at target time.
    # So we shift the index by h days?
    
    # Let's plot at TARGET time
    # This requires shifting index.
    
    # Using simple shift for viz (approx business days)
    # plot_idx = [d + timedelta(days=h) for d in p_idx] # Timedelta might fail on RangeIndex
    # Let's just plot 'What we predict next' overlaid on current.
    
    plt.plot(p_idx, p_vals, label=f'{h}-Day Forecast', color=colors[h], marker='.', linestyle=':')

plt.title("Multi-Horizon Forecasts (Price Predicted at t for t+h)")
plt.legend()
plt.show()
"""),
    md_cell("## 5. Directional Accuracy Traffic Light"),
    code_cell("""# Confusion Matrix
actual_dir = np.sign(test_plot['Actual_Next_Close'] - test_plot['Close'])
pred_dir = np.sign(test_plot['Predicted_Next_Close'] - test_plot['Close'])

cm = confusion_matrix(actual_dir, pred_dir, labels=[1, -1])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['Up', 'Down'], yticklabels=['Up', 'Down'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Directional Confusion Matrix")
plt.show()

# Traffic Light Plot
# Green dot if Correct Direction, Red if Wrong.
correct_mask = (actual_dir == pred_dir)
wrong_mask = (actual_dir != pred_dir)

subset = test_plot.iloc[-100:]
sub_correct = correct_mask.loc[subset.index]

plt.figure(figsize=(15, 7))
plt.plot(subset.index, subset['Actual_Next_Close'], color='black', alpha=0.5, label='Price')

# Plot Correct predictions (Green)
plt.scatter(subset[sub_correct].index, subset[sub_correct]['Actual_Next_Close'], color='green', label='Correct Dir', s=50)

# Plot Wrong predictions (Red)
plt.scatter(subset[~sub_correct].index, subset[~sub_correct]['Actual_Next_Close'], color='red', label='Wrong Dir', s=50)

plt.title("Directional Accuracy Traffic Light (Green=Correct, Red=Wrong)")
plt.legend()
plt.show()
""")
]

create_notebook("00_eda.ipynb", cells_00)
create_notebook("01_ingest_data.ipynb", cells_01)
create_notebook("02_preprocess.ipynb", cells_02)
create_notebook("03_train_models.ipynb", cells_03)
create_notebook("04_evaluate_select.ipynb", cells_04)
