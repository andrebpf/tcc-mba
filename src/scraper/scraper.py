import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import html
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("mba_tcc.scraper")

INFOMONEY_API_URL = "https://www.infomoney.com.br/wp-json/wp/v2/posts"
MONEYTIMES_API_URL = "https://www.moneytimes.com.br/wp-json/wp/v2/posts"
EXAME_API_URL     = "https://exame.com/wp-json/wp/v2/posts/"  # trailing slash avoids 308 redirect


def fetch_posts_from_api(
    term: str,
    page: int = 1,
    base_url: str = INFOMONEY_API_URL,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetches posts from a WordPress REST API endpoint.
    Returns a tuple containing the list of posts and the total number of pages.
    Retries up to max_retries times with exponential backoff on transient errors.
    """
    params = {
        "search": term,
        "page": page,
        "per_page": 100,
        "orderby": "date",
        "order": "desc"
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=15)  # noqa: E501

            if response.status_code == 400:
                # Bad Request usually means invalid page (e.g., page=9999)
                logger.info("Pagination limit reached (400 Bad Request).")
                return [], 0

            response.raise_for_status()

            total_pages = int(response.headers.get('X-WP-TotalPages', 0))
            return response.json(), total_pages

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"API error on page {page} (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"API Request Error after {max_retries} attempts (page {page}): {e}")
                return [], 0

def get_news_from_period(
    term: str = "ibovespa",
    start_date: datetime = datetime(2025, 1, 1),
    end_date: Optional[datetime] = None,
    base_url: str = INFOMONEY_API_URL,
    source_name: str = "infomoney",
    sleep_seconds: float = 0.1,
) -> pd.DataFrame:
    """
    Orchestrates the news collection from a WordPress REST API within the specified period.

    Args:
        term: Search term to query.
        start_date: Earliest date to collect (inclusive). Stops when reached.
        end_date: Latest date to collect (inclusive). Skips newer posts.
        base_url: WordPress REST API endpoint. Defaults to InfoMoney.
        source_name: Label written to the 'source' column of every row.
        sleep_seconds: Delay between paginated requests. Use 0.5 for MoneyTimes.
    """
    all_news = []
    page = 1
    continue_scraping = True
    total_pages_known = 0

    log_msg = f"Starting API collection for '{term}' from {start_date.date()}"
    if end_date:
        log_msg += f" up to {end_date.date()}"
    logger.info(log_msg)

    while continue_scraping:
        if total_pages_known > 0:
            logger.info(f"Fetching API page {page} of {total_pages_known}...")
        else:
            logger.info(f"Fetching API page {page}...")
        
        posts, total_pages = fetch_posts_from_api(term, page, base_url)
        
        if not posts:
            break
            
        if total_pages_known == 0:
            total_pages_known = total_pages
            logger.info(f"Total pages available in API: {total_pages}")

        for post in posts:
            try:
                date_str = post.get("date")
                if not date_str:
                    continue
                    
                post_date = datetime.fromisoformat(date_str)
                
                # If post is newer than end_date, just skip it (continue to next post)
                if end_date and post_date > end_date:
                    continue

                # If post is older than start_date, we can stop scraping entirely
                if post_date < start_date:
                    logger.info(f"Start date reached ({post_date.date()}). Stopping collection.")
                    continue_scraping = False
                    break
                
                # Rendered title might have HTML entities
                title_raw = post.get("title", {}).get("rendered", "")
                title_clean = html.unescape(title_raw)
                
                all_news.append({
                    'date': post_date,
                    'title': title_clean,
                    'link': post.get("link"),
                    'source': source_name,
                })
                
            except ValueError as e:
                logger.warning(f"Error processing post date {post.get('id')}: {e}")
            
        if continue_scraping:
            # Extra safety check: stop if we reach the reported total pages
            if total_pages > 0 and page >= total_pages:
                logger.info(f"Reached the last page ({page}/{total_pages}).")
                break

            page += 1
            # WP API is robust, but let's be gentle
            time.sleep(sleep_seconds)
            
    df = pd.DataFrame(all_news)
    if not df.empty:
        df = df.sort_values(by='date', ascending=False)
        logger.info(f"Collection finished. Total of {len(df)} news collected.")
    else:
        logger.warning("Collection finished with no results.")
        
    return df

def scrape_terms_parallel(
    terms: List[str],
    start_date: datetime,
    end_date: Optional[datetime],
    output_dir: str,
    base_url: str = INFOMONEY_API_URL,
    source_name: str = "infomoney",
    sleep_seconds: float = 0.1,
    max_workers: int = 3,
) -> List[str]:
    """
    Scrape multiple search terms in parallel using a thread pool.
    Each thread paginates sequentially and respects sleep_seconds between pages.
    Returns a list of CSV file paths that were saved.

    Recommended max_workers: InfoMoney=5, MoneyTimes=3, Exame=3.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    def _scrape_one(term: str):
        df = get_news_from_period(
            term=term,
            start_date=start_date,
            end_date=end_date,
            base_url=base_url,
            source_name=source_name,
            sleep_seconds=sleep_seconds,
        )
        if df.empty:
            return None, term, 0
        safe = term.replace(" ", "_").lower()
        ts = datetime.now().strftime("%Y%m%d")
        path = os.path.join(output_dir, f"news_{safe}_{ts}.csv")
        df.to_csv(path, index=False)
        return path, term, len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scrape_one, t): t for t in terms}
        for future in as_completed(futures):
            path, term, count = future.result()
            if path:
                saved_paths.append(path)
                logger.info(f"[{source_name}] '{term}': {count} headlines saved → {path}")
            else:
                logger.warning(f"[{source_name}] '{term}': no results")

    logger.info(f"[{source_name}] Parallel scraping done. {len(saved_paths)}/{len(terms)} terms saved.")
    return saved_paths


if __name__ == "__main__":
    # Quick test
    print("Testing API scraper...")
    df = get_news_from_period(term="ibovespa", start_date=datetime(2026, 1, 1)) 
    print(df.head())
    if not df.empty:
        df.to_csv("test_api_scraper.csv", index=False)
        print("Test file saved: test_api_scraper.csv")
