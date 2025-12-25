#!/usr/bin/env python3
"""
Fix data end date: Remove 2025-12-25 row and update filenames to 2025-12-24
"""
import os
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv()

def fix_end_date():
    """Remove 2025-12-25 data point and update filenames."""
    
    # 1. Load and fix eurusd_latest.csv
    print("📥 Loading eurusd_latest.csv...")
    df = pd.read_csv('data/raw/eurusd_latest.csv', index_col=0, parse_dates=True)
    print(f"   Original: {len(df)} rows, last date: {df.index.max()}")
    
    # Remove 2025-12-25 if it exists
    df = df[df.index < '2025-12-25']
    print(f"   Fixed: {len(df)} rows, last date: {df.index.max()}")
    
    # Save back
    df.to_csv('data/raw/eurusd_latest.csv')
    print("✅ Updated eurusd_latest.csv")
    
    # 2. Handle partitioned file
    old_partitioned = 'data/raw/data_2025-12-25_from_2020-12-28.csv'
    new_partitioned = 'data/raw/data_2025-12-24_from_2020-12-28.csv'
    
    if os.path.exists(old_partitioned):
        # Load, fix, and save with new name
        df_part = pd.read_csv(old_partitioned, index_col=0, parse_dates=True)
        df_part = df_part[df_part.index < '2025-12-25']
        df_part.to_csv(new_partitioned)
        os.remove(old_partitioned)
        print(f"✅ Renamed and fixed: {new_partitioned}")
    
    # 3. Update S3
    s3_bucket = os.getenv('S3_BUCKET')
    if not s3_bucket:
        print("⚠️  S3_BUCKET not set, skipping S3 update")
        return
    
    print(f"\n☁️  Updating S3: {s3_bucket}")
    s3 = boto3.client('s3')
    
    # Delete old file from S3
    old_s3_key = 'data/raw/data_2025-12-25_from_2020-12-28.csv'
    try:
        s3.delete_object(Bucket=s3_bucket, Key=old_s3_key)
        print(f"   Deleted: s3://{s3_bucket}/{old_s3_key}")
    except Exception as e:
        print(f"   Note: {old_s3_key} not found in S3 (may not exist)")
    
    # Upload new file
    new_s3_key = 'data/raw/data_2025-12-24_from_2020-12-28.csv'
    try:
        s3.upload_file(new_partitioned, s3_bucket, new_s3_key)
        print(f"   Uploaded: s3://{s3_bucket}/{new_s3_key}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
    
    print("\n✅ All done! Data now ends at 2025-12-24")

if __name__ == "__main__":
    fix_end_date()
