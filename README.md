# MBA TCC: Sentiment Analysis and Market Correlation

This project collects Brazilian financial news, analyzes their sentiment using **FinBERT-PT-BR**, and studies the correlation between daily sentiment scores and BOVA11 log-returns.

**Research period:** January 1, 2025 – December 31, 2025
**Author:** André Furlanetti

---

## Setup

### Automatic (Windows / PowerShell)

```powershell
.\setup_env.ps1
```

This creates the virtual environment, installs all dependencies, and registers the Jupyter kernel.

### Manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=mba-tcc --display-name "Python (mba-tcc)"
```

---

## Pipeline Execution Order

### Step 0 — Collect market data (run once)

```powershell
python src/cotation/collect_market_data.py
python src/cotation/calculate_log_returns.py
```

These scripts download BOVA11 historical prices from Yahoo Finance and compute log-returns, saving CSVs to `src/dataset/market_data/`.

---

### Notebook 1 — `01_news_scraper.ipynb`

Collects financial news from InfoMoney via their WordPress REST API.

- Searches by term (Ibovespa, BOVA11, Petrobras, Vale, Itaú, etc.)
- Filters by date range (2025-01-01 to 2025-12-31)
- Deduplicates and keyword-filters the consolidated dataset

**Output files:**
- `src/dataset/scraper/search/news_[term]_[date].csv` — per-term raw results
- `src/dataset/scraper/consolidated_news_[date].csv` — deduplicated, filtered dataset

---

### Notebook 2 — `02_sentiment_analysis.ipynb`

Runs sentiment inference on all collected news using FinBERT-PT-BR.

- Loads the model with GPU (CUDA) support if available
- Manual validation against labeled samples (classification report + confusion matrix)
- Batch inference producing `prob_neg`, `prob_neu`, `prob_pos`, `sentiment_score`
- Exploratory analysis of score distribution
- Daily aggregation with market-hour adjustment (news after 18:00 → next trading day)

**Output files:**
- `src/dataset/sentiment/news_with_sentiment.csv` — all news with individual scores
- `src/dataset/sentiment/daily_sentiment.csv` — daily aggregated sentiment metrics

---

### Notebook 3 — `03_sentiment_market_merge.ipynb`

Merges sentiment data with BOVA11 returns and runs statistical analysis.

- Inner join between daily sentiment and trading days
- Creates lagged features (t-1, t-2, t-3) for predictive analysis
- OLS regression: simple (sentiment_mean → Log_Return) and multiple (with lags)
- Pearson correlation tests with significance levels
- Augmented Dickey-Fuller stationarity tests
- Granger causality tests (both directions)
- Rolling 30-day correlation visualization

**Output file:**
- `src/dataset/result/sentiment_returns_merged.csv` — final dataset ready for statistical modeling

---

## Running the Notebooks

- **VS Code**: Open notebooks in `notebooks/`, select the `Python (mba-tcc)` kernel, and run cells.
- **Jupyter**: Run `jupyter lab` or `jupyter notebook` from the project root.

---

## Data Folder Structure

```
src/dataset/
├── market_data/    # BOVA11 raw prices and log-returns
├── scraper/        # Raw and consolidated news CSVs
├── sentiment/      # Per-article and daily aggregated sentiment
└── result/         # Final merged dataset
```

> The `src/dataset/` directory is excluded from git (see `.gitignore`).

---

## GPU Configuration

Sentiment inference uses CUDA automatically when an NVIDIA GPU is available. Adjust `batch_size` in notebook 2 based on available VRAM:

| VRAM   | Recommended batch_size |
|--------|------------------------|
| 4 GB   | 32                     |
| 8 GB   | 64                     |
| 16 GB+ | 128                    |

If you get a `CUDA out of memory` error, reduce `batch_size` to 8 or 16.

---

## References

- **Model**: [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR)
- **Paper**: Santos et al. (2023) — FinBERT-PT-BR
- **Market data**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance)
- **News source**: [InfoMoney](https://www.infomoney.com.br) WordPress REST API
