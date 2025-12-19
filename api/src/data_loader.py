import awswrangler as wr
import pandas as pd
import os
import boto3

def load_data_from_s3(bucket_name: str, prefix: str = "data/raw/") -> pd.DataFrame:
    """
    Reads all CSV files from an S3 bucket with a given prefix using awswrangler.
    
    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The folder prefix (e.g., 'raw-data/').
        
    Returns:
        pd.DataFrame: A single DataFrame containing all data, sorted by Date.
    """
    path = f"s3://{bucket_name}/{prefix}"
    
    print(f"Reading data from {path} using awswrangler...")
    
    try:
        # dataset=True tells wr to look for all files in the folder (partitioned dataset)
        df = wr.s3.read_csv(path=path, dataset=True)
        
        # Ensure sorting if needed (though wr usually keeps order if partitioned, explicit sort is safer)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='last')
            df = df.sort_values('Date').set_index('Date')
            
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
