import os
import sys
import boto3
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
LOCAL_FILE = "/Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/data/raw/data_2025-12-24_from_2020-12-28.csv"
S3_KEY = "data/raw/data_2025-12-24_from_2020-12-28.csv"

print(f"DEBUG: Restoring {LOCAL_FILE} -> s3://{S3_BUCKET}/{S3_KEY}")

if not S3_BUCKET:
    print("ERROR: S3_BUCKET env var not set")
    sys.exit(1)

if not os.path.exists(LOCAL_FILE):
    print(f"ERROR: Local file not found: {LOCAL_FILE}")
    sys.exit(1)

try:
    s3 = boto3.client('s3')
    
    # Upload
    s3.upload_file(LOCAL_FILE, S3_BUCKET, S3_KEY)
    print("SUCCESS: File uploaded.")

except Exception as e:
    print(f"FATAL ERROR: {e}")
