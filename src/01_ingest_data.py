import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Constants
TICKER = "EURUSD=X"
YEARS = 5
RAW_DATA_PATH = os.path.join("data", "raw", "eurusd_raw.csv")

def ingest_data():
    """
    Fetches 5 years of daily EUR/USD data and saves it to csv.
    """
    print(f"Fetching data for {TICKER}...")
    
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS*365)
    
    # Fetch data
    df = yf.download(TICKER, start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print("Error: No data fetched. Check ticker or internet connection.")
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(RAW_DATA_PATH)
    print(f"Data saved to {RAW_DATA_PATH}")
    print(df.head())

    # Upload to S3
    s3_bucket = os.getenv("S3_BUCKET")
    if s3_bucket:
        try:
            import boto3
            s3 = boto3.client('s3')
            s3_key = RAW_DATA_PATH # data/raw/eurusd_raw.csv
            print(f"Uploading to S3: s3://{s3_bucket}/{s3_key}...")
            s3.upload_file(RAW_DATA_PATH, s3_bucket, s3_key)
            print("  ✅ Upload successful.")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    else:
        print("  ⚠️ S3_BUCKET not set in .env, skipping upload.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_data()
