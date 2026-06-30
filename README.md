# MBA Thesis: Financial News Sentiment and BOVA11 Returns

This repository contains the data pipeline and analysis notebooks for an MBA thesis on the relationship between Brazilian financial news headline sentiment and daily BOVA11 ETF log returns.

The project collects headlines from Brazilian financial news portals, classifies sentiment with **FinBERT-PT-BR**, aggregates daily sentiment, merges it with BOVA11 market returns, and evaluates contemporaneous and lagged relationships with correlation, OLS regression, stationarity tests, and Granger causality.

**Research period:** January 1, 2025 to December 31, 2025  
**Author:** André Baconcelo Prado Furlanetti

## Research Design

- **Text sources:** InfoMoney, MoneyTimes, and Exame
- **Text unit:** headlines only, not full article bodies
- **Market proxy:** BOVA11 ETF adjusted close prices from Yahoo Finance through `yfinance`
- **Sentiment model:** `lucas-leme/FinBERT-PT-BR`
- **Sentiment score:** `sentiment_score = prob_pos - prob_neg`
- **Market-hours rule:** headlines published after 18:00 are assigned to the next business day
- **Return formula:** `R_t = ln(P_t / P_{t-1})`

Portuguese search terms and scraped headlines are intentionally kept in Portuguese because the corpus, filters, and FinBERT-PT-BR model depend on Brazilian Portuguese text.

## Current Thesis Results

| Metric | Result |
|--------|--------|
| Final headline corpus | 19,021 headlines |
| Sources | InfoMoney, MoneyTimes, Exame |
| Manual validation sample | 155 headlines |
| FinBERT-PT-BR validation | 76.77% accuracy, 0.77 macro F1 |
| Final paired trading-day sample | 247 observations |
| Pearson correlation, lag 0 | r = +0.36, p < 0.001 |
| Pearson correlations, lags 1-5 | Not statistically significant |
| Simple OLS | beta = +0.031, p < 0.001, R2 = 0.128 |
| Multiple OLS | R2 = 0.157; contemporaneous sentiment remains significant |
| Granger causality | Return -> sentiment significant at lags 1-4; sentiment -> return significant at lags 2-4 |

Interpretation: headline sentiment is mainly a coincident indicator of BOVA11 returns, consistent with the media feedback hypothesis. The multi-source corpus also shows suggestive, limited evidence that aggregate sentiment may contain short-lag predictive information.

## Repository Layout

```text
tcc-mba/
├── data/                 # Generated datasets, excluded from git
│   ├── market_data/      # BOVA11 adjusted prices and log returns
│   ├── scraper/          # Raw and consolidated news CSVs
│   ├── sentiment/        # Article-level and daily sentiment outputs
│   └── result/           # Final sentiment-return merged dataset
├── notebooks/            # Collection, sentiment, and statistical analysis notebooks
├── reports/
│   └── figures/          # Generated figures and chart source tables, excluded from git
├── src/
│   ├── config.py         # Central project paths
│   ├── cotation/         # BOVA11 market data collection and log-return calculation
│   ├── scraper/          # WordPress REST API collection helpers
│   ├── sentiment/        # FinBERT inference, daily aggregation, merge helpers
│   └── utils/            # Shared utilities
├── AGENTS.md             # Codex guidance
├── CLAUDE.md             # Claude Code guidance
├── requirements.txt
└── setup_env.ps1
```

## Setup

### Automatic Setup on Windows

```powershell
.\setup_env.ps1
```

This creates the virtual environment, installs dependencies, and registers the Jupyter kernel.

### Manual Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=mba-tcc --display-name "Python (mba-tcc)"
```

## Notebook Workflow

The notebooks are the primary workflow. They call reusable Python modules under `src/` for scraping, sentiment processing, aggregation, and market-data handling.

### 1. Scrape and Consolidate News

Run `notebooks/01_news_scraper.ipynb`.

The notebook collects headlines from InfoMoney, MoneyTimes, and Exame through their WordPress REST APIs, deduplicates by URL, applies thematic keyword filtering, and writes source-level and combined CSV files.

Typical outputs:

- `data/scraper/[run_date]/[source]/news_[term]_[date].csv`
- `data/scraper/[run_date]/consolidated_[source].csv`
- `data/scraper/[run_date]/combined_news.csv`
- `data/scraper/[run_date]/combined_news_filtered.csv`

### 2. Run Sentiment Analysis

Run `notebooks/02_sentiment_analysis.ipynb`.

The notebook loads FinBERT-PT-BR, validates the model against a manually labeled sample, performs batch inference, computes `sentiment_score`, and aggregates sentiment by adjusted trading day.

Typical outputs:

- `data/sentiment/news_with_sentiment.csv`
- `data/sentiment/news_with_sentiment_combined_[run_date].csv`
- `data/sentiment/daily_sentiment.csv`
- validation and sentiment figures under `reports/figures/`

### 3. Merge With Market Returns and Analyze

Run `notebooks/03_sentiment_market_merge.ipynb`.

The notebook merges daily sentiment with BOVA11 log returns, creates lagged sentiment variables, runs stationarity tests, Pearson correlations, OLS regressions, Granger causality tests, and rolling-correlation visualizations.

Typical outputs:

- `data/result/sentiment_returns_merged.csv`
- correlation, regression, time-series, and rolling-correlation figures under `reports/figures/`

## Helper Scripts

The notebooks are preferred for the full research flow, but selected modules can be run directly for testing or regenerating specific artifacts:

```powershell
python src/cotation/collect_market_data.py
python src/cotation/calculate_log_returns.py
python src/scraper/scraper.py
python src/sentiment/sentiment_analyzer.py
python src/sentiment/daily_aggregation.py
```

Market-data outputs are saved under `data/market_data/`.

## Running Notebooks

- **VS Code:** open the notebooks in `notebooks/` and select the `Python (mba-tcc)` kernel.
- **Jupyter:** run `jupyter lab` or `jupyter notebook` from the project root.

## GPU Notes

Sentiment inference uses CUDA automatically when an NVIDIA GPU is available. Adjust `batch_size` in notebook 2 based on available VRAM.

| VRAM | Recommended batch size |
|------|------------------------|
| 4 GB | 32 |
| 8 GB | 64 |
| 16 GB+ | 128 |

If CUDA runs out of memory, reduce `batch_size` to 8 or 16.

## References

- **Model:** [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR)
- **Market data:** [yfinance](https://github.com/ranaroussi/yfinance)
- **News sources:** [InfoMoney](https://www.infomoney.com.br), [MoneyTimes](https://www.moneytimes.com.br), [Exame](https://exame.com)
