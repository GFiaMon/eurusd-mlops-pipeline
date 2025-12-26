import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_s3():
    print("🚀 Debugging S3 Date Detection")
    print("==============================")
    
    # Check Env
    bucket = os.getenv('S3_BUCKET')
    prefix = os.getenv('S3_PREFIX', 'data/raw/')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print(f"Config:")
    print(f"  S3_BUCKET: {bucket}")
    print(f"  S3_PREFIX: {prefix}")
    print(f"  AWS_REGION: {region}")
    
    if not bucket:
        print("❌ S3_BUCKET not set. Please set it in .env or environment.")
        return

    try:
        dm = DataManager(mode='auto', s3_bucket=bucket, s3_prefix=prefix)
        
        if not dm.is_cloud_available():
            print("❌ Cloud/S3 not available (missing creds or boto3?)")
            return

        print("\nAttempting to list objects...")
        # This will now raise an exception if it fails, or print debug info
        latest_date = dm.get_latest_date()
        
        print(f"\n✅ Result: Latest Date = {latest_date}")
        
    except Exception as e:
        print(f"\n❌ ERROR CAUGHT: {e}")
        print("This error was previously being swallowed, causing the backfill.")
        print("Please check your AWS credentials and permissions.")

if __name__ == "__main__":
    # Load .env if present
    from dotenv import load_dotenv
    load_dotenv()
    debug_s3()
