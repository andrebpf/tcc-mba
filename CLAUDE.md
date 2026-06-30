# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MBA thesis project studying the relationship between Brazilian financial news headline sentiment and daily BOVA11 ETF log returns during 2025.

- **Thesis title:** Correlation between financial headline sentiment and BOVA11 ETF returns
- **Research period:** January 1, 2025 to December 31, 2025
- **Text sources:** InfoMoney, MoneyTimes, and Exame
- **Market proxy:** BOVA11, using adjusted close prices from Yahoo Finance via `yfinance`
- **Sentiment model:** `lucas-leme/FinBERT-PT-BR`

## Language Rules

Use English for documentation, comments, logs, notebook explanations, plot labels, and filenames.

Keep Portuguese text when it is part of the research domain or model input:

- search terms and thematic filters
- scraped headlines and URLs
- manual labels or validation samples
- FinBERT-PT-BR input examples
## Setup

```powershell
.\setup_env.ps1
```

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=mba-tcc --display-name "Python (mba-tcc)"
```

## Notebook Workflow

The notebooks are the primary workflow. They call reusable Python modules under `src/`.

1. **Scrape and consolidate news:** `notebooks/01_news_scraper.ipynb`
2. **Run sentiment analysis:** `notebooks/02_sentiment_analysis.ipynb`
3. **Merge with market returns and analyze:** `notebooks/03_sentiment_market_merge.ipynb`

Selected modules can still be run directly for quick testing or artifact regeneration:

```powershell
python src/cotation/collect_market_data.py
python src/cotation/calculate_log_returns.py
python src/scraper/scraper.py
python src/sentiment/sentiment_analyzer.py
python src/sentiment/daily_aggregation.py
```

## Architecture

```text
Financial news APIs ──► Scraper ──► CSV
                                      │
                                      ▼
                              Sentiment Analyzer
                              FinBERT-PT-BR
                                      │
                                      ▼
                              Daily Aggregation ◄── BOVA11 market data
                                      │
                                      ▼
                              Correlation, OLS, ADF, Granger
```

## Data and Artifact Layout

- `data/market_data/`: BOVA11 raw adjusted prices and log returns
- `data/scraper/`: raw and consolidated news CSV files by run date and source
- `data/sentiment/`: article-level and daily sentiment outputs
- `data/result/`: final sentiment-return merged dataset
- `reports/figures/`: generated figures and chart source tables
- `notebooks/`: exploratory and final analysis notebooks

`data/` and generated report figures are excluded from git.

## Key Domain Rules

- **Market-hours rule:** headlines published after 18:00 are assigned to the next business day.
- **Weekend rule:** weekend-adjusted trading dates roll forward to the next business day.
- **Sentiment score:** `sentiment_score = prob_pos - prob_neg`, range `[-1, +1]`.
- **Log return:** `R_t = ln(P_t / P_{t-1})`.
- **Lagged features:** create t-1, t-2, and t-3 sentiment lag columns for time-series analysis.
- **Date parsing:** use `src/utils/date_parser.py` for `DD/MM/YYYY` portal dates.
- **Portuguese filters:** do not translate search keywords such as `Ações`, `Dólar`, `Inflação`, `Câmbio`, and `Itaú`.

## Current Thesis Results

- **Corpus:** 19,021 headlines from InfoMoney, MoneyTimes, and Exame after deduplication and thematic filtering.
- **Manual validation:** stratified audit with n = 155; accuracy 76.77%; macro F1 = 0.77.
- **Merged sample:** 247 paired trading-day observations after aligning sentiment and market returns.
- **Contemporaneous Pearson correlation:** r = +0.36, p < 0.001.
- **Lagged Pearson correlations:** lags 1 to 5 are not statistically significant.
- **Simple OLS:** beta = +0.031, p < 0.001, R2 = 0.128.
- **Multiple OLS:** R2 = 0.157; contemporaneous sentiment remains significant; t-2 sentiment is negative and significant.
- **Granger causality:** partial asymmetric bidirectionality. Return -> sentiment is significant at lags 1 to 4; sentiment -> return is significant at lags 2 to 4.
- **Interpretation:** sentiment is mainly a coincident indicator, with suggestive multi-source evidence of limited predictive information at short lags.

## Important Files

| File | Purpose |
|------|---------|
| `src/config.py` | Central project paths |
| `src/scraper/scraper.py` | WordPress REST API collection helpers for supported news sources |
| `src/sentiment/sentiment_analyzer.py` | FinBERT-PT-BR inference and scoring |
| `src/sentiment/daily_aggregation.py` | Market-hour-aware daily aggregation, lags, merge, and correlation helpers |
| `src/cotation/collect_market_data.py` | BOVA11 market data download |
| `src/cotation/calculate_log_returns.py` | BOVA11 log-return calculation |
| `src/utils/logger.py` | Shared logger setup |
