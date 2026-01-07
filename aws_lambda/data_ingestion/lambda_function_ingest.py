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

# Import DataManager
# In Lambda, data_manager.py is at root. Locally, it's in utils/.
try:
    from data_manager import DataManager
except ImportError:
    import sys
    # Add project root to path for local execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from utils.data_manager import DataManager

def lambda_handler(event, context):
    # Setup Logging to file
    log_file = "/tmp/ingest.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    start_time = datetime.now()
    
    try:
        logger.info(f"yfinance version: {yf.__version__}")
        
        # 1. Configuration
        s3_bucket = os.environ.get("S3_BUCKET")
        s3_prefix = os.environ.get("S3_PREFIX", "data/raw/")
        
        if not s3_bucket:
             # Try local .env if not set (for local test)
             if os.path.exists('.env'):
                 from dotenv import load_dotenv
                 load_dotenv()
                 s3_bucket = os.environ.get("S3_BUCKET")
        
        if not s3_bucket:
            raise ValueError("S3_BUCKET environment variable is not set")

        # Initialize DataManager
        # Force 'lambda' mode if running in AWS, otherwise 'auto' or 'local'
        # But since we want to mimic Lambda behavior (partitioned save), we might want explicit mode if testing that logic
        # For now, let's behave like the environment we are in.
        mode = 'lambda' if os.environ.get('AWS_LAMBDA_FUNCTION_NAME') else 'auto'
        
        dm = DataManager(mode=mode, s3_bucket=s3_bucket, s3_prefix=s3_prefix)
        logger.info(f"DataManager initialized in {dm.get_environment()} mode")
        
        # 2. Determine Start Date
        latest_date = dm.get_latest_date()
        today = datetime.now().date()
        
        if latest_date:
            # DataManager returns timestamp, convert to date
            if isinstance(latest_date, datetime):
                latest_date = latest_date.date()
                
            start_date = latest_date + timedelta(days=1)
            logger.info(f"Found existing data up to {latest_date}. Start date: {start_date}")
        else:
            # Backfill 5 years
            start_date = today - timedelta(days=5*365)
            logger.info(f"No existing data found. Starting backfill from {start_date}")

        if start_date >= today:
            logger.info("Data is already up to date.")
            return {
                'statusCode': 200,
                'body': 'Data is up to date'
            }

        # 3. Fetch Data
        ticker = "EURUSD=X"
        end_date = datetime.now()
        
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
        if df.empty:
            logger.warning("No data fetched.")
            return {
                'statusCode': 200,
                'body': 'No new data fetched'
            }

        # 4. Save Logic via DataManager
        logger.info(f"Saving {len(df)} rows...")
        success = dm.save_data(df, metadata={'source': 'lambda_ingest', 'ticker': ticker})
        
        if success:
            logger.info("Save successful.")
            return {
                'statusCode': 200,
                'body': f'Successfully ingested {len(df)} rows'
            }
        else:
            raise Exception("DataManager.save_data failed")

    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': f"Error: {str(e)}"
        }
    finally:
        # Upload logs to S3
        try:
            # Flush handlers
            for handler in logger.handlers:
                handler.flush()
                
            if s3_bucket:
                s3 = boto3.client('s3')
                timestamp = start_time.strftime('%Y%m%d_%H%M%S')
                s3_key = f"logs/ingest_{timestamp}.log"
                
                logger.info(f"Uploading logs to s3://{s3_bucket}/{s3_key}")
                s3.upload_file(log_file, s3_bucket, s3_key)
                
        except Exception as log_e:
            # Fallback print if S3 logging fails
            print(f"Failed to upload logs to S3: {log_e}")

if __name__ == "__main__":
    # Local Test
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    print("🚀 Running Lambda locally...")
    result = lambda_handler(None, None)
    print(f"Result: {result}")
