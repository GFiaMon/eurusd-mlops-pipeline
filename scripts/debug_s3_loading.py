import os
import sys
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', 'data/raw/')

print(f"DEBUG: Connecting to S3 Bucket: {S3_BUCKET} Prefix: {S3_PREFIX}")

if not S3_BUCKET:
    print("ERROR: S3_BUCKET env var not set")
    sys.exit(1)

try:
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    if 'Contents' not in response:
        print("ERROR: No objects found in S3 bucket/prefix")
        sys.exit(0)

    total_rows = 0
    dfs = []

    print("\n--- Listing Files ---")
    for obj in response['Contents']:
        key = obj['Key']
        size = obj['Size']
        
        if not key.endswith('.csv'):
            continue
            
        print(f"\nScanning File: {key} (Size: {size} bytes)")
        
        try:
            # Download content
            obj_data = s3.get_object(Bucket=S3_BUCKET, Key=key)
            content = obj_data['Body'].read().decode('utf-8')
            
            # Inspect Header
            lines = content.split('\n')[:5]
            # print("  First 3 lines:")
            # for i, line in enumerate(lines[:3]):
            #     print(f"    Line {i}: {repr(line)}")
            
            has_yfinance_header = len(lines) >= 2 and 'Ticker' in lines[1]
            # print(f"  Detected yfinance header: {has_yfinance_header}")

            # Parse
            if has_yfinance_header:
                df_temp = pd.read_csv(
                    StringIO(content),
                    index_col=0,
                    parse_dates=True,
                    skiprows=[1, 2]
                )
            else:
                df_temp = pd.read_csv(
                    StringIO(content),
                    index_col=0,
                    parse_dates=True
                )
            
            rows = len(df_temp)
            print(f"  > Loaded {rows} rows")
            
            if rows > 0:
                print(f"  > Date Range: {df_temp.index.min()} to {df_temp.index.max()}")
                dfs.append(df_temp)
                total_rows += rows
            else:
                print("  > WARNING: Loaded 0 rows")

        except Exception as e:
            print(f"  ERROR processing {key}: {e}")

    print("\n--- Summary ---")
    print(f"Total Files Processed: {len(dfs)}")
    print(f"Total Rows Loaded: {total_rows}")
    
    if dfs:
        full_df = pd.concat(dfs)
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        print(f"Unique Rows after merge: {len(full_df)}")
        print(f"Final Date Range: {full_df.index.min()} to {full_df.index.max()}")
        
except Exception as e:
    print(f"FATAL ERROR: {e}")
