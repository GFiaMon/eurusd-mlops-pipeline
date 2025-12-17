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

if __name__ == "__main__":
    ingest_data()
