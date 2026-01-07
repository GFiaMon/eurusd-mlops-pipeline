
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from utils.data_manager import DataManager

def test_data_manager():
    print("Initializing DataManager...")
    # Pointing to the correct directory based on user info
    dm = DataManager(data_type='raw', local_dir='data/raw')
    
    print("Loading latest data...")
    df = dm.get_latest_data()
    
    print(f"Loaded {len(df)} rows.")
    print("Columns:", df.columns.tolist())
    print("\nTail of DataFrame:")
    print(df.tail())
    
    # Check for latest date
    if not df.empty:
        latest_date = df.index.max()
        print(f"\nLatest Date: {latest_date}")
        
    return df

if __name__ == "__main__":
    test_data_manager()
