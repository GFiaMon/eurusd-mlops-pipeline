import pandas as pd
import io

# Simulated content of the CSV file as seen in data/raw/eurusd_raw.csv
csv_content = """Price,Close,High,Low,Open,Volume
Ticker,EURUSD=X,EURUSD=X,EURUSD=X,EURUSD=X,EURUSD=X
Date,,,,,
2020-12-23,1.218665,1.221941,1.215775,1.218590,0
2020-12-24,1.219140,1.221747,1.217937,1.219393,0
"""

print("--- Simulating naive read (like current S3 loader might do) ---")
# Simulating wr.s3.read_csv which behaves like pd.read_csv without specific args
df = pd.read_csv(io.StringIO(csv_content))
print("Columns:", df.columns.tolist())
print("Index:", df.index)
print("Head:\n", df.head())

# Reproducing the logic in load_data_from_s3
print("\n--- Applying load_data_from_s3 logic ---")
if 'Date' in df.columns:
    print("Found 'Date' column")
    # ... logic ...
elif df.index.name != 'Date':
    print("Date column NOT found. Trying to parse index...")
    try:
        # This acts on the RangeIndex (0, 1, 2...)
        df.index = pd.to_datetime(df.index, errors='coerce')
        print("New Index:", df.index)
        print("Index as int64 (unique):", df.index.astype('int64').unique())
        try:
             print("Formatted date range:", f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        except:
             print("Could not strftime")
    except Exception as e:
        print(f"Error: {e}")

print("\n--- Testing Fix (skip rows) ---")
# The local loader does skiprows=[1,2]. Let's try that but ensure we handle 'Price' as Date
df_fix = pd.read_csv(io.StringIO(csv_content), skiprows=[1, 2])
print("Columns:", df_fix.columns.tolist())
print("Head:\n", df_fix.head())
if 'Price' in df_fix.columns:
    print("Renaming Price -> Date")
    df_fix = df_fix.rename(columns={'Price': 'Date'})
    df_fix['Date'] = pd.to_datetime(df_fix['Date'])
    df_fix = df_fix.set_index('Date')
    print("Fixed Index:", df_fix.index)
