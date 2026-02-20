import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def collect_market_data(ticker="BOVA11.SA", start_date="2025-01-01"):
    """
    Collects historical adjusted close data from Yahoo Finance.
    """
    print(f"Starting collection for {ticker} from {start_date}...")
    
    # Data collection
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        
        if data.empty:
            print(f"No data found for {ticker}.")
            return
            
        print(f"Data collected: {data.shape[0]} rows.")
        
        # Select only Adj Close (or Close if Adj Close is not available)
        # yfinance usually returns a MultiIndex DataFrame if downloading multiple,
        # but for a single one, it might be simple. Let's ensure consistency.
        
        if 'Adj Close' in data.columns:
            final_df = data[['Adj Close']].copy()
        elif 'Close' in data.columns:
            print("Adj Close not found, using Close.")
            final_df = data[['Close']].copy()
        else:
            print("Price columns not found.")
            print(data.columns)
            return

        # Reset index to have Date as a column
        final_df.reset_index(inplace=True)
        
        # Rename columns to standard format
        final_df.columns = ['Date', 'Adj Close']
        
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'market_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Filename
        today = datetime.now().strftime("%Y%m%d")
        filename = f"{ticker.replace('.SA', '')}_{today}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save
        final_df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        
    except Exception as e:
        print(f"Error during collection: {e}")

if __name__ == "__main__":
    collect_market_data()
