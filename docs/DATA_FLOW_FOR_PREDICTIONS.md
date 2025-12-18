# Data Flow for Flask API Predictions

## How the Flask App Gets Data for Forecasting

### Quick Answer:
**The Flask app uses RAW data** (`data/raw/eurusd_raw.csv`) and performs **feature engineering on-the-fly** during each prediction request.

---

## Complete Data Flow

```
Docker Container Starts
        ↓
1. Try to load data from S3 (if USE_S3=true)
   └─> Download: s3://bucket/data/raw/eurusd_raw.csv
        ↓
2. Fallback to local files (priority order):
   ├─> data/raw/eurusd_raw.csv        ← PRIMARY SOURCE
   ├─> data/processed/test.csv         ← Fallback 1
   └─> data/processed/train.csv        ← Fallback 2
        ↓
3. Load CSV into memory as df_data
   - Only needs 'Close' column (or 'close')
   - Skips metadata rows [1, 2]
   - Converts to numeric
        ↓
4. Wait for prediction request
        ↓
5. When /predict is called:
   └─> FeatureEngineer.preprocess(df_data)
        ↓
6. Feature Engineering (ON-THE-FLY):
   ├─> Calculate Return = Close.pct_change()
   ├─> Calculate MA_5 = Close.rolling(5).mean()
   ├─> Calculate Lag_1, Lag_2, Lag_3, Lag_5
   ├─> Create Return_Unscaled (for ARIMA)
   ├─> Drop NaN rows
   ├─> Select LAST row (most recent data)
   └─> Scale features using scaler.pkl
        ↓
7. Return preprocessed X and current_price
        ↓
8. Make prediction
        ↓
9. Return predicted price
```

---

## Key Insights

### 1. **RAW Data is Used**
The Flask app loads **RAW** CSV data with just the 'Close' price column:
```python
# From app_cloud.py lines 288-310
if USE_S3:
    boto3.client('s3').download_file(
        S3_BUCKET, 
        'data/raw/eurusd_raw.csv',  # ← RAW DATA
        'data/raw/eurusd_raw.csv'
    )

# Fallback priority:
local_data_paths = [
    'data/raw/eurusd_raw.csv',      # ← FIRST CHOICE (RAW)
    'data/processed/test.csv',       # ← Fallback
    'data/processed/train.csv'       # ← Last resort
]
```

### 2. **Feature Engineering Happens at Prediction Time**
The `FeatureEngineer.preprocess()` method calculates features on-the-fly:
```python
# From app_cloud.py lines 195-210
data['Return'] = data['Close'].pct_change()
data['MA_5'] = data['Close'].rolling(window=5).mean()
for lag in [1, 2, 3, 5]:
    data[f'Lag_{lag}'] = data['Return'].shift(lag)
```

### 3. **Only the LAST Row is Used**
For prediction, only the most recent data point is needed:
```python
# From app_cloud.py line 220
last_row = data.iloc[[-1]][available_features]
current_price = data.iloc[-1]['Close']
```

### 4. **Scaling Happens at Prediction Time**
The scaler (trained during preprocessing) is applied to the features:
```python
# From app_cloud.py lines 224-233
if self.scaler:
    X_scaled = self.scaler.transform(last_row)
    X = pd.DataFrame(X_scaled, columns=available_features)
```

---

## For Data Ingestion Automation

### ✅ **Which File Should Your Automation Update?**

**Answer: `data/raw/eurusd_raw.csv`**

### Why?
1. **Primary Source**: This is the first file the Flask app tries to load
2. **Simple Format**: Only needs Date and Close columns
3. **No Preprocessing Required**: Feature engineering happens automatically
4. **Consistent with Training**: Your training pipeline also starts with raw data

### Automation Workflow

```
Scheduled Job (e.g., daily at market close)
        ↓
1. Fetch latest EUR/USD data
   (e.g., from yfinance, Alpha Vantage, etc.)
        ↓
2. Append new rows to eurusd_raw.csv
   Format: Date,Close
        ↓
3. Upload to S3 (if using cloud deployment)
   └─> s3://bucket/data/raw/eurusd_raw.csv
        ↓
4. Restart Flask container (optional)
   OR: Implement hot-reload endpoint
```

### Example Data Format

**eurusd_raw.csv:**
```csv
Date,Open,High,Low,Close,Volume
Ticker,EUR=X,,,,,
Date,Open,High,Low,Close,Volume
2024-01-01,1.1050,1.1080,1.1040,1.1070,0
2024-01-02,1.1070,1.1090,1.1060,1.1085,0
2024-01-03,1.1085,1.1100,1.1075,1.1095,0
...
2024-12-18,1.1720,1.1730,1.1710,1.1726,0  ← Latest
```

**Minimum Required Format:**
```csv
Date,Close
2024-01-01,1.1070
2024-01-02,1.1085
2024-01-03,1.1095
...
2024-12-18,1.1726  ← Latest
```

---

## Important Considerations

### 1. **Data Freshness**
The Flask app loads data **once at startup**. To get new predictions with updated data:
- **Option A**: Restart the Docker container
- **Option B**: Add a `/reload-data` endpoint (recommended for production)

### 2. **Historical Data Requirements**
The app needs at least **10 rows** of historical data to calculate features:
```python
# From app_cloud.py line 276
if len(df) > 10:
    df_data = df
    return True
```

Why 10? Because:
- MA_5 needs 5 rows
- Lag_5 needs 5 more rows
- Plus buffer for pct_change() and dropna()

### 3. **S3 vs Local**
**Current behavior:**
```python
if USE_S3:
    # Download from S3
    download_file(S3_BUCKET, 'data/raw/eurusd_raw.csv', ...)
else:
    # Use local file
    load_data_file('data/raw/eurusd_raw.csv')
```

**For automation:**
- **Cloud deployment**: Update S3 file, restart container
- **Local deployment**: Update local file, restart app

---

## Recommended Automation Script

### Option 1: Simple Daily Update

```python
# scripts/update_data.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Fetch latest EUR/USD data
ticker = yf.Ticker("EURUSD=X")
end_date = datetime.now()
start_date = end_date - timedelta(days=7)  # Get last week

# Download data
df = ticker.history(start=start_date, end=end_date)

# Load existing data
existing_data = pd.read_csv('data/raw/eurusd_raw.csv', 
                            index_col=0, 
                            parse_dates=True, 
                            skiprows=[1, 2])

# Append new data (avoid duplicates)
combined = pd.concat([existing_data, df])
combined = combined[~combined.index.duplicated(keep='last')]

# Save
combined.to_csv('data/raw/eurusd_raw.csv')

# Upload to S3 (if using cloud)
if os.getenv('USE_S3') == 'true':
    import boto3
    s3 = boto3.client('s3')
    s3.upload_file(
        'data/raw/eurusd_raw.csv',
        os.getenv('S3_BUCKET'),
        'data/raw/eurusd_raw.csv'
    )
    print("✅ Data updated and uploaded to S3")
```

### Option 2: Add Hot-Reload Endpoint (Recommended)

Add this to `api/app_cloud.py`:

```python
@app.route('/reload-data', methods=['POST'])
def reload_data():
    """Reload data from S3 or local file without restarting container"""
    global df_data
    
    if USE_S3:
        try:
            boto3.client('s3').download_file(
                S3_BUCKET, 
                'data/raw/eurusd_raw.csv',
                'data/raw/eurusd_raw.csv'
            )
            load_data_file('data/raw/eurusd_raw.csv')
            return jsonify({
                'status': 'success',
                'message': 'Data reloaded from S3',
                'rows': len(df_data),
                'latest_date': df_data.index[-1].strftime('%Y-%m-%d')
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        load_data_file('data/raw/eurusd_raw.csv')
        return jsonify({
            'status': 'success',
            'message': 'Data reloaded from local file',
            'rows': len(df_data)
        })
```

Then your automation can:
```bash
# Update data
python scripts/update_data.py

# Trigger reload (no container restart needed!)
curl -X POST http://your-ec2-ip:8080/reload-data
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Which file does Flask use?** | `data/raw/eurusd_raw.csv` (RAW data) |
| **Does it use processed data?** | No, it calculates features on-the-fly |
| **Which file for automation?** | Update `data/raw/eurusd_raw.csv` |
| **Minimum columns needed?** | Date (index) and Close |
| **Minimum rows needed?** | 10+ rows for feature calculation |
| **How to refresh data?** | Restart container OR add `/reload-data` endpoint |

**Recommendation:** Implement the `/reload-data` endpoint for production to avoid container restarts!
