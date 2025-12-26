#!/usr/bin/env python3
"""
Clean up S3 and local raw data, then regenerate with fixed DataManager.
"""
import os
import sys
import boto3
from dotenv import load_dotenv

load_dotenv()

def cleanup_and_regenerate():
    """Clean S3 and local, then force fresh data ingestion."""
    
    # 1. Clean local files
    print("🧹 Cleaning local data/raw/...")
    raw_dir = "data/raw"
    for file in os.listdir(raw_dir):
        if file.endswith('.csv'):
            filepath = os.path.join(raw_dir, file)
            os.remove(filepath)
            print(f"   Deleted: {filepath}")
    
    # 2. Clean S3
    s3_bucket = os.getenv('S3_BUCKET')
    if not s3_bucket:
        print("⚠️  S3_BUCKET not set, skipping S3 cleanup")
    else:
        print(f"\n🧹 Cleaning S3 bucket: {s3_bucket}")
        s3 = boto3.client('s3')
        prefix = "data/raw/"
        
        try:
            response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv'):
                        s3.delete_object(Bucket=s3_bucket, Key=key)
                        print(f"   Deleted: s3://{s3_bucket}/{key}")
            else:
                print("   No files found in S3")
        except Exception as e:
            print(f"   ❌ S3 cleanup failed: {e}")
    
    # 3. Force fresh ingestion
    print("\n📥 Running fresh data ingestion...")
    os.system("python src/01_ingest_data.py")
    
    print("\n✅ Cleanup and regeneration complete!")
    print("   Check data/raw/eurusd_latest.csv to verify clean columns")

if __name__ == "__main__":
    cleanup_and_regenerate()
