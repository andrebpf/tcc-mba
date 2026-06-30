"""
Central project paths.

Keep filesystem locations here so scripts and notebooks do not depend on
hardcoded relative paths such as ../data.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

MARKET_DATA_DIR = DATA_DIR / "market_data"
SCRAPER_DATA_DIR = DATA_DIR / "scraper"
SENTIMENT_DATA_DIR = DATA_DIR / "sentiment"
RESULT_DATA_DIR = DATA_DIR / "result"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
