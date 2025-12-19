import yfinance as yf
import pandas as pd
import boto3
import os
import re
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Fix yfinance cache on Lambda (Read-only FS)
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception as e:
    logger.warning(f"Could not set yf cache: {e}")

def get_latest_date_from_s3(bucket, prefix):
    """
    Scans S3 bucket for files matching data_{end}_from_{start}.csv
    Returns the latest end_date found.
    """
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as e:
        logger.error(f"Error listing objects: {e}")
        return None

    if 'Contents' not in response:
        return None

    max_date = None
    # Regex to extract end_date from "data_{end}_from_{start}.csv"
    # Example: data_2024-12-19_from_2019-12-19.csv
    pattern = re.compile(r"data_(\d{4}-\d{2}-\d{2})_from_(\d{4}-\d{2}-\d{2})\.csv")

    for obj in response['Contents']:
        key = obj['Key']
        # Remove prefix to get filename
        filename = os.path.basename(key) 
        match = pattern.match(filename)
        if match:
            end_date_str = match.group(1)
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                if max_date is None or end_date > max_date:
                    max_date = end_date
            except ValueError:
                continue
    
    return max_date

def lambda_handler(event, context):
    try:
        logger.info(f"yfinance version: {yf.__version__}")
        
        # 1. Configuration
        s3_bucket = os.environ.get("S3_BUCKET")
        s3_prefix = os.environ.get("S3_PREFIX", "data/raw/")
        
        if not s3_bucket:
            raise ValueError("S3_BUCKET environment variable is not set")

        logger.info(f"Starting ingestion. Bucket: {s3_bucket}, Prefix: {s3_prefix}")
        
        # 2. Determine Start Date
        latest_date = get_latest_date_from_s3(s3_bucket, s3_prefix)
        today = datetime.now().date() # Current date
        
        if latest_date:
            start_date = latest_date + timedelta(days=1)
            logger.info(f"Found existing data up to {latest_date}. Start date: {start_date}")
        else:
            # Backfill 5 years
            start_date = today - timedelta(days=5*365)
            logger.info(f"No existing data found. Starting backfill from {start_date}")

        if start_date >= today:
            logger.info("Data is already up to date. Nothing to do.")
            return {
                'statusCode': 200,
                'body': 'Data is up to date'
            }

        # 3. Fetch Data
        ticker = "EURUSD=X"
        end_date = datetime.now() # Match src/01_ingest_data.py behavior
        
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        
        # Using exact same params as working script, just enabling progress=False for logs
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
        if df.empty:
            logger.warning("No data fetched.")
            return {
                'statusCode': 200,
                'body': 'No new data fetched'
            }

        # 4. Save Logic
        # Determine actual range fetched
        if 'Date' in df.columns:
            # Should not happen as Date is index in yfinance usually
            df['Date'] = pd.to_datetime(df['Date'])
            actual_end_date = df['Date'].max().date()
        else:
            actual_end_date = df.index.max().date()
            
        csv_filename = f"data_{actual_end_date}_from_{start_date}.csv"
        tmp_path = f"/tmp/{csv_filename}"
        s3_key = f"{s3_prefix.rstrip('/')}/{csv_filename}" # Ensure single slash
        
        logger.info(f"Saving {len(df)} rows to {tmp_path}")
        df.to_csv(tmp_path)
        
        # 5. Upload to S3
        logger.info(f"Uploading to s3://{s3_bucket}/{s3_key}")
        s3 = boto3.client('s3')
        s3.upload_file(tmp_path, s3_bucket, s3_key)
        
        return {
            'statusCode': 200,
            'body': f'Successfully uploaded {s3_key}'
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        return {
            'statusCode': 500,
            'body': f"Error: {str(e)}"
        }

if __name__ == "__main__":
    # Local Test
    from dotenv import load_dotenv
    load_dotenv()
    lambda_handler(None, None)
