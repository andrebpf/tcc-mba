import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime

from src.config import MARKET_DATA_DIR


def calculate_log_returns(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calculates log-returns for BOVA11 market data.
    
    Log-return formula: R_t = ln(P_t / P_{t-1})
    
    Args:
        csv_path: Path to the CSV file with market data
        output_path: Optional path to save the output CSV
        
    Returns:
        DataFrame with Date, Adj Close, and Log Return columns
    """
    print(f"Loading market data from: {csv_path}")
    
    # Load the market data
    df = pd.read_csv(csv_path)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (ascending) to ensure correct order
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} trading days")
    print(f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Calculate log-returns: ln(P_t / P_{t-1})
    # Using numpy's log for natural logarithm
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # First row will be NaN since there's no previous price
    # This is expected and correct
    print(f"\nCalculated log-returns for {df['Log_Return'].notna().sum()} days")
    print(f"First day (NaN expected): {df.loc[0, 'Date'].date()}")
    
    # Show basic statistics
    print(f"\n=== Log-Return Statistics ===")
    print(f"Mean: {df['Log_Return'].mean():.6f}")
    print(f"Std Dev: {df['Log_Return'].std():.6f}")
    print(f"Min: {df['Log_Return'].min():.6f}")
    print(f"Max: {df['Log_Return'].max():.6f}")
    
    # Save to file if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")
    
    return df


def main():
    """
    Main execution: loads the most recent BOVA11 data and calculates log-returns.
    """
    if not MARKET_DATA_DIR.exists():
        print(f"ERROR: Market data directory not found: {MARKET_DATA_DIR}")
        return

    # Find the most recent BOVA11 CSV file
    csv_files = [
        f.name
        for f in MARKET_DATA_DIR.iterdir()
        if f.name.startswith('BOVA11_') and f.name.endswith('.csv')
    ]
    
    if not csv_files:
        print("ERROR: No BOVA11 CSV files found in market_data directory")
        return
    
    # Sort to get the most recent
    csv_files.sort(reverse=True)
    latest_file = csv_files[0]
    
    input_path = MARKET_DATA_DIR / latest_file
    
    # Create output filename with log_returns prefix
    today = datetime.now().strftime("%Y%m%d")
    output_filename = f"BOVA11_log_returns_{today}.csv"
    output_path = MARKET_DATA_DIR / output_filename
    
    # Calculate log-returns
    df = calculate_log_returns(input_path, output_path)
    
    # Display sample data
    print(f"\n=== First 10 rows ===")
    print(df[['Date', 'Adj Close', 'Log_Return']].head(10).to_string(index=False))
    
    print(f"\n=== Last 10 rows ===")
    print(df[['Date', 'Adj Close', 'Log_Return']].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
