"""
Data collection module for EUR/USD prediction
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_eurusd_data(years=5, interval="1d"):
    """
    Fetch EUR/USD historical data from Yahoo Finance
    
    Parameters:
    -----------
    years : int
        Number of years of historical data
    interval : str
        Data interval (1d, 1h, 1wk)
    
    Returns:
    --------
    pandas.DataFrame
        Historical EUR/USD data
    """
    logger.info(f"Fetching {years} years of EUR/USD data...")
    
    ticker = "EURUSD=X"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        
        logger.info(f"Successfully fetched {len(data)} records")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    df = fetch_eurusd_data(years=3)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
