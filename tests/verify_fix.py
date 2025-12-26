import pandas as pd
import io
import sys
# Mock awswrangler
from unittest.mock import MagicMock
sys.modules['awswrangler'] = MagicMock()
import awswrangler as wr

# Import the updated function
# We need to ensure the import works, so we append the path
import os
sys.path.append(os.path.abspath('.'))
from api.src.data_loader import load_data_from_s3

# Function to simulate wr.s3.read_csv
def mock_read_csv(path, dataset, **kwargs):
    print(f"Mock read_csv called with kwargs: {kwargs}")
    
    # Simulated content (like yfinance download)
    csv_content = """Price,Close,High,Low,Open,Volume
Ticker,EURUSD=X,EURUSD=X,EURUSD=X,EURUSD=X,EURUSD=X
Date,,,,,
2020-12-23,1.218665,1.221941,1.215775,1.218590,0
2020-12-24,1.219140,1.221747,1.217937,1.219393,0
"""
    # If skiprows is passed, pandas handles it.
    if 'skiprows' in kwargs:
        return pd.read_csv(io.StringIO(csv_content), **kwargs)
    else:
        # If no skiprows, return raw naive read which caused the bug
        return pd.read_csv(io.StringIO(csv_content))

# Patch the mock
wr.s3.read_csv = mock_read_csv

print("\n--- Verifying Updated Logic with skiprows ---")
try:
    # This simulates exactly what app_cloud.py does now
    df = load_data_from_s3("fake-bucket", "fake-prefix", skiprows=[1, 2])
    
    print("Result DataFrame Index:", df.index)
    if isinstance(df.index, pd.DatetimeIndex):
        print("SUCCESS: Index IS DatetimeIndex")
        print("Date Range:", f"{df.index[0]} to {df.index[-1]}")
    else:
        print("FAILURE: Index is NOT DatetimeIndex")
        
except Exception as e:
    print(f"Test Failed: {e}")
    import traceback
    traceback.print_exc()
