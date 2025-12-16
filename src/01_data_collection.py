import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_data(ticker="EURUSD=X", start_date=None, end_date=None, save_path="data/raw/eurusd_data.csv"):
    """
    Fetches historical data from yfinance and saves it to a CSV file.
    
    Args:
        ticker (str): The ticker symbol to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format. If None, fetches max history.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_path (str): Path to save the raw data.
    """
    logging.info(f"Fetching data for {ticker}...")
    
    try:
        # Fetch data with maximum history if no start date provided
        if start_date:
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            # period="max" is the best way to get all available history
            data = yf.download(ticker, period="max")
            
        if data.empty:
            logging.error("No data fetched. Check ticker or internet connection.")
            return

        logging.info(f"Successfully fetched {len(data)} rows.")
        
        # Ensure 'Date' is a column (reset index if it's the index)
        # yfinance usually returns Date as index
        data.reset_index(inplace=True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        data.to_csv(save_path, index=False)
        logging.info(f"Data saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    # Define absolute paths based on project structure
    # Assuming script is run from project root or src/
    # We'll use relative paths assuming execution from project root
    
    TICKER = "EURUSD=X"
    OUTPUT_PATH = "data/raw/eurusd_data.csv"
    
    collect_data(ticker=TICKER, save_path=OUTPUT_PATH)
