# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MBA thesis (TCC) researching the correlation between financial news sentiment and BOVA11 (Brazilian stock market ETF) returns. Period analyzed: 01/01/2025 – 31/12/2025.

**Language:** Portuguese (Brazilian) — comments, variable names, and logs may be in PT-BR.

## Setup

```bash
# Create and activate virtual environment, install deps, register Jupyter kernel
.\setup_env.ps1

# Or manually:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline Execution Order

The pipeline runs in this sequence:

1. **Collect market data** → `src/cotation/collect_market_data.py`
2. **Calculate log returns** → `src/cotation/calculate_log_returns.py`
3. **Scrape news** → `src/scraper/scraper.py` (or `notebooks/01_news_scraper.ipynb`)
4. **Run sentiment analysis** → `src/sentiment/sentiment_analyzer.py` (or `notebooks/02_sentiment_analysis.ipynb`)
5. **Aggregate & merge** → `src/sentiment/daily_aggregation.py` (or `notebooks/03_sentiment_market_merge.ipynb`)

Each module can be run directly for quick testing via its `if __name__ == "__main__"` block:

```bash
python src/cotation/collect_market_data.py
python src/cotation/calculate_log_returns.py
python src/scraper/scraper.py
python src/sentiment/sentiment_analyzer.py
python src/sentiment/daily_aggregation.py
```

## Architecture

```
News (InfoMoney API) ──► Scraper ──► CSV
                                      │
                                      ▼
                              Sentiment Analyzer (FinBERT-PT-BR)
                                      │
                                      ▼
                              Daily Aggregation ◄── Market Data (BOVA11 via yfinance)
                                      │
                                      ▼
                              Correlation Analysis (notebooks/03)
```

**Data storage:** All intermediate and final datasets are saved as CSV files under `src/dataset/` (excluded from git).

## Key Domain Rules

- **Market hours rule:** News published after 18:00 is assigned to the **next business day** (not the current one), since markets are already closed.
- **Sentiment score formula:** `score = prob_positive - prob_negative` (range: -1 to 1)
- **Log return formula:** `R_t = ln(P_t / P_{t-1})`
- **Lagged features:** The aggregation module creates t-1, t-2, t-3 lag columns for time-series regression.
- **Model:** `lucas-leme/FinBERT-PT-BR` loaded via HuggingFace Transformers. Automatically uses CUDA if available.
- **Date format used throughout:** `DD/MM/YYYY` — use `src/utils/date_parser.py` for parsing.

## Important Files

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Master project specification — read this for full methodology details |
| `src/sentiment/sentiment_analyzer.py` | Core NLP inference, batch processing, GPU support |
| `src/sentiment/daily_aggregation.py` | Market-hour-aware aggregation + lag features + merge |
| `src/scraper/scraper.py` | InfoMoney WordPress REST API pagination |
| `src/utils/logger.py` | Shared logger — use `setup_logger(__name__)` in new modules |
