import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Add project root to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_manager import DataManager

# Constants
TICKER = "EURUSD=X"
YEARS = 5

def ingest_data():
    """
    Fetches 5 years of daily EUR/USD data using DataManager.
    """
    # Initialize DataManager
    dm = DataManager(data_type='raw')
    print(f"🚀 Starting ingestion (Environment: {dm.get_environment()})")
    
    # Check if data is already fresh (within 24h)
    if dm.is_data_current(max_age_hours=24):
        # Load and verify data volume
        df = dm.get_latest_data()
        if len(df) > 250:
            print("✅ Data is already up-to-date (< 24h old) and sufficient. Skipping download.")
            print(f"📊 Current data: {len(df)} rows, Last date: {df.index.max()}")
            return
        print(f"⚠️  Data is fresh but insufficient ({len(df)} rows). Forcing update.")

    print(f"Fetching data for {TICKER}...")
    
    # Calculate start date
    latest_date = dm.get_latest_date()
    existing_df = None
    force_full = False
    
    # Check if existing data is substantial enough (e.g. at least 1 year)
    if latest_date:
        existing_df = dm.get_latest_data()
        if len(existing_df) < 250: # Approx 1 trading year
            print("⚠️  Existing data seems incomplete (< 1 year). Forcing full backfill.")
            force_full = True
            latest_date = None
    
    end_date = datetime.now()
    
    if latest_date and not force_full:
        start_date = latest_date + timedelta(days=1)
        if start_date.date() >= end_date.date():
             print("✅ Data is up to date.")
             return
        print(f"📅 Updating from {start_date.date()}")
    else:
        start_date = end_date - timedelta(days=YEARS*365)
        print(f"📅 Full backfill from {start_date.date()}")
    
    # Fetch data
    df = yf.download(TICKER, start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print("⚠️ No new data fetched.")
        return

    print(f"✅ Fetched {len(df)} new rows.")
    
    # Merge with existing if needed
    if existing_df is not None and not force_full:
        print("🔗 Merging with existing data...")
        # Ensure proper concatenation
        df = pd.concat([existing_df, df])
        # Deduplicate
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        print(f"📊 Total rows after merge: {len(df)}")

    # Save using DataManager (handles local mirror + S3 sync)
    print("💾 Saving data...")
    success = dm.save_data(df, metadata={'source': 'local_ingest', 'ticker': TICKER})
    
    if success:
        print(f"✅ Data save successful. Range: {df.index.min().date()} to {df.index.max().date()}")
        if dm.is_cloud_available():
            print("☁️  Synced to S3.")
        else:
            print("params: Local only (S3 not configured or offline).")
    else:
        print("❌ Data save failed.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_data()
