"""Sentiment analysis package for financial news."""

from .sentiment_analyzer import (
    setup_sentiment_model,
    predict_sentiment,
    predict_batch,
    calculate_sentiment_score,
    analyze_news_file
)

__all__ = [
    'setup_sentiment_model',
    'predict_sentiment',
    'predict_batch',
    'calculate_sentiment_score',
    'analyze_news_file'
]
