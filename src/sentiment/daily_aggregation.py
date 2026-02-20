"""
Daily Sentiment Aggregation Module

This module provides functions to aggregate news sentiment by date,
create lagged features, and handle market hour adjustments.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Optional
from pandas.tseries.offsets import BusinessDay


def handle_market_hours(
    df: pd.DataFrame,
    date_column: str = 'date',
    market_close_hour: int = 18
) -> pd.DataFrame:
    """
    Adjusts news dates for market hour considerations.
    
    News published after market close (default: 18:00) are associated
    with the next business day's returns.
    
    Args:
        df: DataFrame with news data
        date_column: Column containing datetime
        market_close_hour: Hour when market closes (default: 18)
        
    Returns:
        DataFrame with adjusted 'trading_date' column
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create trading_date column initialized with original date
    df['trading_date'] = df[date_column]
    
    # Get time of day
    df['time_of_day'] = df[date_column].dt.time
    
    # Market close time
    close_time = time(hour=market_close_hour, minute=0)
    
    # If news is after market close, assign to next business day
    after_close_mask = df['time_of_day'] > close_time
    df.loc[after_close_mask, 'trading_date'] = df.loc[after_close_mask, 'trading_date'] + pd.Timedelta(days=1)
    
    # Convert to business day (skip weekends)
    # If Saturday (5) or Sunday (6), move to the next business day (Monday)
    # Using apply for explicit handling
    def adjust_weekend(dt):
        if dt.weekday() >= 5: # Sat or Sun
            return dt + BusinessDay()
        return dt

    df['trading_date'] = df['trading_date'].apply(adjust_weekend)
    
    # Drop temporary column
    df.drop(columns=['time_of_day'], inplace=True)
    
    return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_column: str = 'date',
    score_column: str = 'sentiment_score',
    prob_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Aggregates sentiment data by date with multiple metrics.
    
    Args:
        df: DataFrame with sentiment data
        date_column: Column containing dates
        score_column: Column with sentiment scores
        prob_columns: List of probability columns to aggregate
        
    Returns:
        DataFrame with daily aggregated sentiment metrics
    """
    # Apply date adjustment row by row (After-Market and Weekend)
    print("Applying date adjustments (After-Market and Weekend)...")
    # Call handle_market_hours to get 'trading_date'
    df = handle_market_hours(df, date_column)
    
    # Extract date only from TRADING DATE for aggregation
    df['date_only'] = df['trading_date'].dt.date
    
    # Default probability columns
    if prob_columns is None:
        prob_columns = ['prob_neg', 'prob_neu', 'prob_pos']
    
    # Aggregation metrics
    agg_dict = {
        score_column: ['mean', 'std', 'min', 'max'],
        'title': 'count'  # Count of news per day
    }
    
    # Add probability columns if they exist
    for col in prob_columns:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Group by date
    daily_df = df.groupby('date_only').agg(agg_dict).reset_index()
    
    # Flatten column names
    daily_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                        for col in daily_df.columns]
    
    # Rename columns for clarity
    rename_map = {
        'date_only': 'date',
        f'{score_column}_mean': 'sentiment_mean',
        f'{score_column}_std': 'sentiment_std',
        f'{score_column}_min': 'sentiment_min',
        f'{score_column}_max': 'sentiment_max',
        'title_count': 'news_count'
    }
    daily_df.rename(columns=rename_map, inplace=True)
    
    # Calculate additional metrics
    
    # 1. Volume of Sentiment: count positive, negative, neutral news
    df['sentiment_class'] = pd.cut(
        df[score_column],
        bins=[-np.inf, -0.2, 0.2, np.inf],
        labels=['negative', 'neutral', 'positive']
    )
    
    sentiment_counts = df.groupby(['date_only', 'sentiment_class']).size().unstack(fill_value=0)
    sentiment_counts.columns = [f'count_{col}' for col in sentiment_counts.columns]
    sentiment_counts = sentiment_counts.reset_index()
    sentiment_counts.rename(columns={'date_only': 'date'}, inplace=True)
    
    # Merge with daily_df
    daily_df = daily_df.merge(sentiment_counts, on='date', how='left')
    
    # 2. Sentiment Momentum: difference from 3-day moving average
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    daily_df['sentiment_ma3'] = daily_df['sentiment_mean'].rolling(window=3, min_periods=1).mean()
    daily_df['sentiment_momentum'] = daily_df['sentiment_mean'] - daily_df['sentiment_ma3']
    
    # 3. Sentiment Range (max - min)
    daily_df['sentiment_range'] = daily_df['sentiment_max'] - daily_df['sentiment_min']
    
    # Convert date back to datetime for compatibility
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    return daily_df


def create_lagged_features(
    df: pd.DataFrame,
    columns_to_lag: list,
    lags: list = [1, 2, 3],
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Creates lagged features for time series analysis.
    
    Args:
        df: DataFrame with time series data
        columns_to_lag: List of column names to create lags for
        lags: List of lag periods (e.g., [1, 2, 3] for t-1, t-2, t-3)
        date_column: Column containing dates
        
    Returns:
        DataFrame with original data + lagged columns
    """
    df = df.copy()
    df = df.sort_values(date_column).reset_index(drop=True)
    
    for col in columns_to_lag:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
        
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def merge_with_market_data(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    date_column: str = 'date',
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merges daily sentiment data with market returns.
    
    Args:
        sentiment_df: DataFrame with daily sentiment aggregates
        market_df: DataFrame with market data (e.g., log returns)
        date_column: Column name for date (must exist in both DataFrames)
        how: Type of merge ('inner', 'left', 'right', 'outer')
        
    Returns:
        Merged DataFrame
    """
    # Ensure both have datetime dates
    sentiment_df = sentiment_df.copy()
    market_df = market_df.copy()
    
    sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column]).dt.date
    market_df[date_column] = pd.to_datetime(market_df[date_column]).dt.date
    
    # Merge
    merged_df = pd.merge(
        market_df,
        sentiment_df,
        on=date_column,
        how=how,
        suffixes=('_market', '_sentiment')
    )
    
    # Convert date back to datetime
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])
    
    return merged_df


def calculate_sentiment_correlation(
    df: pd.DataFrame,
    sentiment_col: str = 'sentiment_mean',
    return_col: str = 'Log_Return',
    max_lag: int = 5
) -> pd.DataFrame:
    """
    Calculates correlation between sentiment and returns at different lags.
    
    Args:
        df: DataFrame with both sentiment and return data
        sentiment_col: Column name for sentiment
        return_col: Column name for returns
        max_lag: Maximum lag to test
        
    Returns:
        DataFrame with correlation results
    """
    results = []
    
    # Current day correlation (t vs t)
    corr_t0 = df[[sentiment_col, return_col]].corr().iloc[0, 1]
    results.append({
        'lag': 0,
        'description': f'{sentiment_col}(t) vs {return_col}(t)',
        'correlation': corr_t0,
        'sample_size': df[[sentiment_col, return_col]].dropna().shape[0]
    })
    
    # Lagged correlations
    for lag in range(1, max_lag + 1):
        # Sentiment at t, return at t+lag
        temp_df = df.copy()
        temp_df[f'return_lead{lag}'] = temp_df[return_col].shift(-lag)
        
        corr = temp_df[[sentiment_col, f'return_lead{lag}']].corr().iloc[0, 1]
        results.append({
            'lag': lag,
            'description': f'{sentiment_col}(t) vs {return_col}(t+{lag})',
            'correlation': corr,
            'sample_size': temp_df[[sentiment_col, f'return_lead{lag}']].dropna().shape[0]
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    print("Daily Aggregation Module - Testing with sample data")
    
    # Create sample data
    # Using '6h' instead of '6H' for pandas compatibility
    dates = pd.date_range('2025-01-01', '2025-01-31', freq='6h')
    sample_df = pd.DataFrame({
        'date': dates,
        'title': [f'News {i}' for i in range(len(dates))],
        'sentiment_score': np.random.uniform(-0.5, 0.5, len(dates)),
        'prob_neg': np.random.uniform(0, 0.5, len(dates)),
        'prob_neu': np.random.uniform(0.2, 0.6, len(dates)),
        'prob_pos': np.random.uniform(0, 0.5, len(dates))
    })
    
    print(f"\nSample data: {len(sample_df)} news items")
    print(sample_df.head())
    
    # Test aggregation
    daily = aggregate_daily_sentiment(sample_df)
    print(f"\nDaily aggregation: {len(daily)} days")
    print(daily.head())
    
    # Test lagged features
    daily_lagged = create_lagged_features(daily, ['sentiment_mean'], lags=[1, 2])
    print(f"\nWith lagged features:")
    print(daily_lagged[['date', 'sentiment_mean', 'sentiment_mean_lag1', 'sentiment_mean_lag2']].head())
