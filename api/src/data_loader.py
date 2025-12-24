import awswrangler as wr
import pandas as pd
import os
import boto3

def load_data_from_s3(bucket_name: str, prefix: str = "data/raw/", **kwargs) -> pd.DataFrame:
    """
    Reads all CSV files from an S3 bucket with a given prefix using awswrangler.
    
    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The folder prefix (e.g., 'raw-data/').
        **kwargs: Additional arguments passed to wr.s3.read_csv
        
    Returns:
        pd.DataFrame: A single DataFrame containing all data, sorted by Date.
    """
    path = f"s3://{bucket_name}/{prefix}"
    
    print(f"Reading data from {path} using awswrangler...")
    
    try:
        # dataset=True tells wr to look for all files in the folder (partitioned dataset)
        df = wr.s3.read_csv(path=path, dataset=True, **kwargs)
        
        # Check if we have the specific yfinance double-header issue where 'Price' is the header for the index
        if 'Price' in df.columns and 'Date' not in df.columns:
            print("Detected 'Price' column in place of Date - renaming...")
            df = df.rename(columns={'Price': 'Date'})

        # Ensure proper data types for numeric columns
        numeric_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in Close (essential column)
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
        
        # Handle Date column and set as index
        if 'Date' in df.columns:
            # Convert to datetime first
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove any rows with invalid dates
            df = df.drop_duplicates(subset=['Date'], keep='last')
            df = df.sort_values('Date').set_index('Date')
        elif df.index.name != 'Date':
            # If Date is already the index but wasn't recognized
            try:
                # Only attempt if it looks like a date (heuristics or check first element if possible)
                # But here we just try-except
                temp_index = pd.to_datetime(df.index, errors='coerce')
                # Check if we successfully converted significant portion
                if temp_index.notna().mean() > 0.5:
                     df.index = temp_index
                     df = df.sort_index()
                else:
                     print("Index does not appear to be datetime-convertible.")
            except Exception as e:
                print(f"Warning: Could not convert index to datetime: {e}")
            
        return df
        
    except Exception as e:
        print(f"Error reading from S3: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "data/raw/")
    
    if bucket:
        df = load_data_from_s3(bucket, prefix)
        print(f"Loaded {len(df)} rows.")
        print(df.head())
        print(df.tail())
    else:
        print("S3_BUCKET not set in .env")
