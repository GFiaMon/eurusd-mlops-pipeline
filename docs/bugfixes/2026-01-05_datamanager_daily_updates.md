# DataManager Bug Fix - Missing Daily Updates

## Issue Summary
**Date**: 2026-01-05  
**Reported By**: User  
**Severity**: High - Data pipeline was not loading recent daily updates

### Problem Description
The `DataManager` was only retrieving data up to 2025-12-24 from S3, despite the Lambda function successfully ingesting daily updates through 2026-01-02. This caused a 9-day data gap in the application.

### Root Cause
The `_load_from_s3()` method in `utils/data_manager.py` was using `skiprows=[1, 2]` for **all** CSV files, assuming they had yfinance's 3-line header format:
```
Date,Close,High,Low,Open,Volume
Price
Ticker,EURUSD=X
2020-12-28,1.220509...
```

However, the daily update files created by `DataManager._save_to_s3_partitioned()` have a **simple header format**:
```
Date,Close,High,Low,Open,Volume
2025-12-26,1.177302...
```

When reading these 2-line files (header + 1 data row) with `skiprows=[1, 2]`, pandas was skipping the actual data row, resulting in empty DataFrames that were excluded from the merge.

### Files Affected
- **Main file**: `data_2025-12-24_from_2020-12-28.csv` (114KB, 1298 rows) - Simple header
- **Daily files**: 
  - `data_2025-12-26_from_2025-12-26.csv` (120 bytes, 1 row) - Simple header ❌ Was being skipped
  - `data_2025-12-29_from_2025-12-29.csv` (120 bytes, 1 row) - Simple header ❌ Was being skipped
  - `data_2025-12-31_from_2025-12-30.csv` (209 bytes, 2 rows) - Simple header ❌ Was being skipped
  - `data_2026-01-02_from_2026-01-02.csv` (119 bytes, 1 row) - Simple header ❌ Was being skipped

### Solution
Modified `_load_from_s3()` to **intelligently detect the header format** before reading each CSV:

1. Download the file content as a string
2. Check the first 3 lines for the yfinance "Ticker" metadata row
3. If yfinance format detected: use `skiprows=[1, 2]`
4. If simple format detected: use no skiprows
5. Only append non-empty DataFrames to the merge list

### Code Changes
**File**: `utils/data_manager.py`  
**Method**: `_load_from_s3()` (lines 469-527)

**Before**:
```python
obj_data = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
df_temp = pd.read_csv(
    obj_data['Body'],
    index_col=0,
    parse_dates=True,
    skiprows=[1, 2]  # Always skip - WRONG!
)
dfs.append(df_temp)  # Appends empty DataFrames
```

**After**:
```python
# Download and detect format
obj_data = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
content = obj_data['Body'].read().decode('utf-8')
lines = content.split('\n')[:3]
has_yfinance_header = len(lines) >= 2 and 'Ticker' in lines[1]

# Read with appropriate skiprows
from io import StringIO
if has_yfinance_header:
    df_temp = pd.read_csv(StringIO(content), index_col=0, parse_dates=True, skiprows=[1, 2])
else:
    df_temp = pd.read_csv(StringIO(content), index_col=0, parse_dates=True)

if not df_temp.empty:
    dfs.append(df_temp)
```

### Verification
**Before Fix**:
- Rows loaded: 1298
- Date range: 2020-12-30 to 2025-12-24
- Missing: 7 days of data (2025-12-26, 12-29, 12-30, 12-31, 2026-01-02)

**After Fix**:
- Rows loaded: 1305 ✅
- Date range: 2020-12-28 to 2026-01-02 ✅
- All daily updates successfully merged ✅

### Deployment Steps
1. ✅ Fixed `utils/data_manager.py`
2. ✅ Synced to `aws_lambda/data_ingestion/data_manager.py`
3. ✅ Synced to `aws_lambda/ingest/data_manager.py`
4. ⚠️ **TODO**: Redeploy Lambda function using `scripts/deployment/deploy_lambda_ingest.sh`
5. ⚠️ **TODO**: Restart Flask API if it uses DataManager for data loading

### Testing
Created `test_datamanager.py` to verify:
- ✅ `get_latest_date()` returns 2026-01-02
- ✅ `get_latest_data()` loads all 1305 rows
- ✅ All 5 S3 files are detected and merged correctly

### Lessons Learned
1. **Don't assume uniform file formats** - Even within the same data pipeline, files may have different structures
2. **Test with real data** - The bug only manifested with actual daily update files
3. **Add defensive checks** - The `if not df_temp.empty` check prevents empty DataFrames from polluting the merge
4. **Log granularly** - Added debug logging to show which files use which header format

### Related Issues
- This fix also resolves potential issues with any future CSV files that don't follow yfinance's format
- The `_load_from_local()` method already had similar smart detection logic (lines 544-558)

### Impact
- **Data Freshness**: Application now has access to the latest market data
- **Lambda Function**: Will correctly detect existing data and avoid unnecessary backfills
- **API Predictions**: Will use up-to-date historical data for feature engineering
